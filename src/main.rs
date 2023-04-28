// Copyright 2019-2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod database;
mod group;
mod groups;
mod image;
mod thumbnailer;
mod vec;
mod view;

use crate::groups::Groups;
use boolinator::Boolinator;
use log::*;
use piston_window::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;
use thumbnailer::Thumbnailer;
use vec::*;

#[derive(Debug, Error)]
pub enum E {
    #[error("database error: {:?}", 0)]
    DatabaseError(sled::Error),

    #[error("decode error {:?}", 0)]
    DecodeError(bincode::Error),

    #[error("encode error {:?}", 0)]
    EncodeError(bincode::Error),

    #[error("missing data for key {:?}", 0)]
    MissingData(String),

    #[error("image error: {:?}", 0)]
    ImageError(::image::ImageError),
}

type R<T> = std::result::Result<T, E>;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Pow2(u8);

impl Pow2 {
    fn from(i: u32) -> Self {
        assert!(i.is_power_of_two());
        Pow2((32 - i.leading_zeros() - 1) as u8)
    }

    #[allow(unused)]
    fn u32(&self) -> u32 {
        1 << self.0
    }
}

#[test]
fn size_conversions() {
    assert_eq!(Pow2::from(128), Pow2(7));
    assert_eq!(Pow2(7).u32(), 128);
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Default,
)]
pub struct TileRef(u64);

impl TileRef {
    fn new(size: Pow2, index: u64, chunk: u16) -> Self {
        Self((chunk as u64) | ((index % (1u64 << 40)) << 16) | ((size.0 as u64) << 56))
    }

    #[cfg(test)]
    fn deconstruct(&self) -> (Pow2, u64, u16) {
        let size = ((self.0 & 0xFF00_0000_0000_0000u64) >> 56) as u8;
        let index = (self.0 & 0x00FF_FFFF_FFFF_0000u64) >> 16;
        let chunk = (self.0 & 0x0000_0000_0000_FFFFu64) as u16;
        (Pow2(size), index, chunk)
    }
}

#[test]
fn tile_ref_test() {
    assert_eq!(
        TileRef::new(Pow2(0xFFu8), 0u64, 0u16),
        TileRef(0xFF00_0000_0000_0000u64)
    );
    assert_eq!(
        TileRef::new(Pow2(0xFFu8), 0u64, 0u16).deconstruct(),
        (Pow2(0xFFu8), 0u64, 0u16)
    );
    assert_eq!(
        TileRef::new(Pow2(0xFFu8), 0u64, 0u16).0.to_be_bytes(),
        [0xFF, 0, 0, 0, 0, 0, 0, 0]
    );

    assert_eq!(
        TileRef::new(Pow2(0u8), 0x00FF_FFFF_FFFFu64, 0u16),
        TileRef(0x00FF_FFFF_FFFF_0000_u64)
    );
    assert_eq!(
        TileRef::new(Pow2(0u8), 0x00FF_FFFF_FFFFu64, 0u16).deconstruct(),
        (Pow2(0u8), 0x00FF_FFFF_FFFFu64, 0u16)
    );

    assert_eq!(
        TileRef::new(Pow2(0u8), 0u64, 0xFFFFu16),
        TileRef(0x0000_0000_0000_FFFFu64)
    );
    assert_eq!(
        TileRef::new(Pow2(0u8), 0u64, 0xFFFFu16).deconstruct(),
        (Pow2(0u8), 0u64, 0xFFFFu16)
    )
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Thumb {
    img_size: [u32; 2],
    tile_refs: Vec<TileRef>,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Metadata {
    thumbs: Vec<Thumb>,
}

impl Metadata {
    fn nearest(&self, target_size: u32) -> usize {
        let mut found = None;

        let ts_zeros = target_size.leading_zeros() as i16;

        for (i, thumb) in self.thumbs.iter().enumerate() {
            let size = thumb.size();
            let size_zeros = size.leading_zeros() as i16;
            let dist = (ts_zeros - size_zeros).abs();
            if let Some((found_dist, found_i)) = found.take() {
                if dist < found_dist {
                    found = Some((dist, i));
                } else {
                    found = Some((found_dist, found_i));
                }
            } else {
                found = Some((dist, i));
            }
        }

        let (_, i) = found.unwrap();
        i
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TileSpec {
    img_size: [u32; 2],

    // Grid width and height (in number of tiles).
    grid_size: [u32; 2],

    // Tile width and height in pixels.
    tile_size: [u32; 2],
}

impl TileSpec {
    fn ranges(img_size: u32, grid_size: u32, tile_size: u32) -> impl Iterator<Item = (u32, u32)> {
        (0..grid_size).map(move |i| {
            let min = i * tile_size;
            let max = std::cmp::min(img_size, min + tile_size);
            (min, max)
        })
    }

    fn x_ranges(&self) -> impl Iterator<Item = (u32, u32)> {
        Self::ranges(self.img_size[0], self.grid_size[0], self.tile_size[0])
    }

    fn y_ranges(&self) -> impl Iterator<Item = (u32, u32)> {
        Self::ranges(self.img_size[1], self.grid_size[1], self.tile_size[1])
    }
}

impl Thumb {
    fn max_dimension(&self) -> u32 {
        let [w, h] = self.img_size;
        std::cmp::max(w, h)
    }

    fn size(&self) -> u32 {
        self.max_dimension().next_power_of_two()
    }

    fn tile_spec(&self) -> TileSpec {
        let img_size = vec2_f64(self.img_size);
        let tile_size = vec2_scale(vec2_log(img_size, 8.0), 128.0);
        let grid_size = vec2_ceil(vec2_div(img_size, tile_size));
        let tile_size = vec2_ceil(vec2_div(img_size, grid_size));
        TileSpec {
            img_size: self.img_size,
            grid_size: vec2_u32(grid_size),
            tile_size: vec2_u32(tile_size),
        }
    }

    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        view: &view::View,
        tiles: &BTreeMap<TileRef, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool {
        let img = piston_window::image::Image::new();

        let max_dimension = self.max_dimension() as f64;

        let trans = trans.zoom(view.zoom / max_dimension);

        // Center the image within the grid square.
        let [x_offset, y_offset] = {
            let img_size = vec2_f64(self.img_size);
            let gaps = vec2_sub([max_dimension, max_dimension], img_size);
            vec2_scale(gaps, 0.5)
        };

        let tile_spec = self.tile_spec();

        let mut it = self.tile_refs.iter();
        for (y, _) in tile_spec.y_ranges() {
            for (x, _) in tile_spec.x_ranges() {
                let tile_ref = it.next().unwrap();
                if let Some(texture) = tiles.get(tile_ref) {
                    let trans = trans.trans(x_offset + x as f64, y_offset + y as f64);
                    img.draw(texture, draw_state, trans, g);
                }
            }
        }

        true
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MetadataState {
    Missing,
    Some(Metadata),
    Errored,
}

pub type TileMap<T> = BTreeMap<TileRef, T>;

struct App {
    db: Arc<database::Database>,

    groups: groups::Groups,

    thumbnailer: Thumbnailer,

    // Graphics state
    window_settings: WindowSettings,
    window: PistonWindow,
    texture_context: G2dTextureContext,

    // Movement state & modes.
    view: view::View,
    panning: bool,
    zooming: Option<f64>,
    cursor_captured: bool,

    // Mouse distance calculations are relative to this point.
    focus: Option<Vector2<f64>>,

    shift_held: bool,
}

pub struct Stopwatch {
    start: std::time::Instant,
    duration: std::time::Duration,
}

impl Stopwatch {
    fn from_millis(millis: u64) -> Self {
        Self {
            start: std::time::Instant::now(),
            duration: std::time::Duration::from_millis(millis),
        }
    }

    pub fn done(&self) -> bool {
        self.start.elapsed() >= self.duration
    }
}

impl App {
    fn new(
        images: Vec<image::Image>,
        db: Arc<database::Database>,
        thumbnailer: Thumbnailer,
    ) -> Self {
        let view = view::View::new(images.len());

        let groups = Groups::from(images, vec2_u32(view.grid_size));

        let window_settings = WindowSettings::new("pix", [800.0, 600.0])
            .exit_on_esc(true)
            .fullscreen(false);

        let mut window: PistonWindow = window_settings.build().expect("window build");

        let texture_context = window.create_texture_context();

        Self {
            db,

            groups,

            thumbnailer,

            window_settings,
            window,
            texture_context,

            view,
            panning: false,
            zooming: None,
            cursor_captured: false,

            shift_held: false,

            focus: None,
        }
    }

    fn update(&mut self, args: UpdateArgs) {
        let stopwatch = Stopwatch::from_millis(10);

        let grid_size = vec2_u32(self.view.grid_size);
        if grid_size != self.groups.grid_size() {
            self.groups.regroup(grid_size);
        }

        if let Some(z) = self.zooming {
            self.zoom(z.mul_add(args.dt, 1.0));
        }

        if self.focus.is_none() {
            self.groups.recheck(&self.view);
            self.focus = Some(self.view.mouse_dist([0, 0]));
        }

        self.recv_thumbs();

        self.groups.make_thumbs(&mut self.thumbnailer);

        self.groups
            .load_cache(&self.view, &*self.db, &mut self.texture_context, &stopwatch);
    }

    pub fn recv_thumbs(&mut self) {
        for (i, metadata_res) in self.thumbnailer.recv() {
            self.groups.update_metadata(i, metadata_res);
        }
    }

    fn resize(&mut self, win_size: Vector2<u32>) {
        self.view.resize_to(win_size);
        self.focus = None;
    }

    fn force_refocus(&mut self) {
        self.focus = None;
    }

    fn maybe_refocus(&mut self) {
        if let Some(old) = self.focus {
            let new = self.view.mouse_dist([0, 0]);
            let delta = vec2_sub(new, old);
            if vec2_square_len(delta) > 500.0 {
                self.force_refocus();
            }
        }
    }

    fn mouse_move(&mut self, loc: Vector2<f64>) {
        self.view.mouse_to(loc);
        self.maybe_refocus();
    }

    fn mouse_zoom(&mut self, v: f64) {
        for _ in 0..(v as isize) {
            self.zoom(1.0 + self.zoom_increment());
        }
        for _ in (v as isize)..0 {
            self.zoom(1.0 - self.zoom_increment());
        }
    }

    fn mouse_pan(&mut self, delta: Vector2<f64>) {
        if self.panning {
            if self.cursor_captured {
                self.view.center_mouse();
            }
            self.trans(vec2_scale(delta, 4.0));
        }
    }

    fn shift_increment(&self) -> f64 {
        if self.shift_held {
            // snap to zoom
            if self.view.zoom > 100.0 {
                self.view.zoom
            } else {
                100.0
            }
        } else {
            20.0
        }
    }

    fn zoom_increment(&self) -> f64 {
        if self.shift_held {
            0.5
        } else {
            0.1
        }
    }

    fn trans(&mut self, trans: Vector2<f64>) {
        self.view.trans_by(trans);
        self.maybe_refocus();
    }

    fn zoom(&mut self, ratio: f64) {
        self.view.zoom_by(ratio);
        self.maybe_refocus();
    }

    fn reset(&mut self) {
        self.view.reset();
        self.force_refocus();
    }

    fn button(&mut self, b: ButtonArgs) {
        match (b.state, b.button) {
            (ButtonState::Press, Button::Keyboard(Key::Z)) => {
                self.reset();
            }

            //(ButtonState::Press, Button::Keyboard(Key::F)) => {
            //    // TODO: this crashes with: thread 'main' panicked at 'Creating EventLoop multiple
            //    // times is not supported.',
            //    self.window.set_should_close(true);
            //    self.window_settings.set_fullscreen(true);
            //    self.window = self.window_settings.build().unwrap();

            //    self.groups.reset();
            //    self.focus = None;
            //    self.panning = false;
            //    self.cursor_captured = false;
            //    self.zooming = None;
            //}
            (ButtonState::Press, Button::Keyboard(Key::T)) => {
                self.cursor_captured = !self.cursor_captured;
                self.window.set_capture_cursor(self.cursor_captured);
                self.panning = self.cursor_captured;
                self.view.center_mouse();
            }

            (ButtonState::Press, Button::Keyboard(Key::Up)) => {
                self.trans([0.0, self.shift_increment()]);
            }

            (ButtonState::Press, Button::Keyboard(Key::Down)) => {
                self.trans([0.0, -self.shift_increment()]);
            }

            (ButtonState::Press, Button::Keyboard(Key::Left)) => {
                self.trans([self.shift_increment(), 0.0]);
            }

            (ButtonState::Press, Button::Keyboard(Key::Right)) => {
                self.trans([-self.shift_increment(), 0.0]);
            }

            (ButtonState::Press, Button::Keyboard(Key::PageUp)) => {
                self.view.center_mouse();
                self.zoom(1.0 - self.zoom_increment());
            }

            (ButtonState::Press, Button::Keyboard(Key::PageDown)) => {
                self.view.center_mouse();
                self.zoom(1.0 + self.zoom_increment());
            }

            (state, Button::Keyboard(Key::LShift)) | (state, Button::Keyboard(Key::RShift)) => {
                self.shift_held = state == ButtonState::Press;
            }

            (state, Button::Mouse(MouseButton::Middle)) => {
                self.panning = state == ButtonState::Press;
            }

            (state, Button::Mouse(MouseButton::Left)) => {
                self.zooming = (state == ButtonState::Press).as_some(5.0);
            }

            (state, Button::Mouse(MouseButton::Right)) => {
                self.zooming = (state == ButtonState::Press).as_some(-5.0);
            }

            _ => {}
        }
    }

    fn draw_2d(e: &Event, c: Context, g: &mut G2d, view: &view::View, groups: &Groups) {
        clear([0.0, 0.0, 0.0, 1.0], g);

        let args = e.render_args().expect("render args");
        let draw_state = DrawState::default().scissor([0, 0, args.draw_size[0], args.draw_size[1]]);

        let _black = color::hex("000000");
        let _missing_color = color::hex("888888");
        let _op_color = color::hex("222222");

        groups.draw(c.transform, view, &draw_state, g);
    }

    fn run(&mut self) {
        loop {
            if let Some(e) = self.window.next() {
                e.update(|args| {
                    self.update(*args);
                });

                e.resize(|args| {
                    self.resize(args.draw_size);
                });

                e.mouse_scroll(|[_, v]| {
                    self.mouse_zoom(v);
                });

                e.mouse_cursor(|loc| {
                    self.mouse_move(loc);
                });

                e.mouse_relative(|delta| {
                    self.mouse_pan(delta);
                });

                e.button(|b| self.button(b));

                // borrowck
                let v = &self.view;
                let groups = &self.groups;
                self.window.draw_2d(&e, |c, g, _device| {
                    Self::draw_2d(&e, c, g, v, groups);
                });
            } else {
                break;
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct File {
    path: String,
    modified: u64,
    file_size: u64,
}

fn find_images(dirs: Vec<PathBuf>) -> Vec<Arc<File>> {
    let mut ret = Vec::new();

    for dir in dirs {
        for entry in walkdir::WalkDir::new(&dir) {
            let i = ret.len();
            if i > 0 && i % 1000 == 0 {
                info!("Found {} images...", i);
            }

            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    error!("Walkdir error: {:?}", e);
                    continue;
                }
            };

            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(e) => {
                    error!("Metadata lookup error: {:?}: {:?}", entry, e);
                    continue;
                }
            };

            if metadata.is_dir() {
                info!("Searching in {:?}", entry.path());
                continue;
            }

            let file_size = metadata.len();

            let modified: u64 = metadata
                .modified()
                .expect("metadata modified")
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("duration since unix epoch")
                .as_secs();

            let path = entry.path();

            let path = match path.canonicalize() {
                Ok(path) => path,
                Err(e) => {
                    error!("unable to canonicalize: {:?} {:?}", path, e);
                    continue;
                }
            };

            let path = if let Some(path) = path.to_str() {
                path.to_owned()
            } else {
                error!("Skipping non-utf8 path: {:?}", path);
                continue;
            };

            let file = File {
                path,
                modified,
                file_size,
            };

            ret.push(Arc::new(file));
        }
    }

    ret.sort();
    ret
}

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Set number of background thumbnailer threads.
    #[arg(long, value_name = "COUNT")]
    threads: Option<usize>,

    /// Set database path.
    #[arg(long, value_name = "PATH")]
    db_path: Option<PathBuf>,

    /// Images or directories to open.
    #[arg(value_name = "PATH", default_value = ".")]
    paths: Vec<PathBuf>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let thumbnailer_threads: usize = if let Some(threads) = args.threads {
        threads
    } else {
        num_cpus::get()
    };
    info!("Thumbnailer threads {}", thumbnailer_threads);

    let db_path: PathBuf = if let Some(db_path) = args.db_path {
        db_path
    } else {
        let mut db_path = dirs_next::cache_dir().expect("cache dir");
        db_path.push("pix/thumbs.db");
        db_path
    };
    info!("Database path: {:?}", db_path);

    info!("Paths: {:?}", args.paths);
    let files = find_images(args.paths);
    if files.is_empty() {
        error!("No files found, exiting.");
        std::process::exit(1);
    } else {
        info!("Found {} files", files.len());
    }

    let db = Arc::new(database::Database::open(&db_path).expect("db open"));

    let images: Vec<image::Image> = {
        files
            .into_par_iter()
            .enumerate()
            .map(|(i, file)| {
                let metadata = match db.get_metadata(&file) {
                    Ok(Some(metadata)) => MetadataState::Some(metadata),
                    Ok(None) => MetadataState::Missing,
                    Err(e) => {
                        error!("error loading metadata for: {:?}: {:?}", file, e);
                        MetadataState::Errored
                    }
                };
                image::Image::from(i, file, metadata)
            })
            .collect()
    };

    let uid_base = db.reserve(images.len());

    let thumbnailer = Thumbnailer::new(Arc::clone(&db), uid_base, thumbnailer_threads);

    App::new(images, Arc::clone(&db), thumbnailer).run();
}
