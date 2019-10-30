// Copyright 2019 Google LLC
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

#![feature(arbitrary_self_types)]
#![recursion_limit = "1000"]
#![feature(vec_remove_item)]
#![feature(drain_filter)]
#![feature(stmt_expr_attributes)]
#![allow(unused_imports)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate failure;

use crate::stats::ScopedDuration;
use ::image::GenericImage;
use ::image::GenericImageView;
use boolinator::Boolinator;
use clap::Arg;
use futures::future::Fuse;
use futures::future::FutureExt;
use futures::future::RemoteHandle;
use futures::select;
use futures::task::SpawnExt;
use piston_window::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::ops::Bound::*;
use std::sync::Arc;
use std::time::Instant;
use std::time::SystemTime;

#[macro_use]
extern crate lazy_static;

mod database;
mod stats;

#[derive(Debug, Fail)]
pub enum E {
    #[fail(display = "rocksdb error: {:?}", 0)]
    RocksError(rocksdb::Error),

    #[fail(display = "decode error {:?}", 0)]
    DecodeError(bincode::Error),

    #[fail(display = "encode error {:?}", 0)]
    EncodeError(bincode::Error),

    #[fail(display = "missing data for key {:?}", 0)]
    MissingData(String),

    #[fail(display = "image error: {:?}", 0)]
    ImageError(::image::ImageError),
}

type R<T> = std::result::Result<T, E>;

#[derive(Debug, Default)]
struct View {
    num_tiles: usize,

    // Window dimensions.
    w: f64,
    h: f64,

    // Logical dimensions.
    lw: i64,
    lh: i64,

    // View offsets.
    x: f64,
    y: f64,

    // Scale from logical to physical coordinates.
    zoom: f64,

    // Mouse coordinates.
    mx: f64,
    my: f64,

    // Has the user panned or zoomed?
    auto: bool,
}

impl View {
    fn new(num_tiles: usize) -> Self {
        Self {
            num_tiles,
            w: 800.0,
            h: 600.0,
            lw: 1,
            lh: 1,
            auto: true,
            ..Default::default()
        }
    }

    fn center_mouse(&mut self) {
        self.mx = self.w / 2.0;
        self.my = self.h / 2.0;
    }

    fn reset(&mut self) {
        self.auto = true;
        self.x = 0.0;
        self.y = 0.0;

        let pixels_per_image = (self.w * self.h) / self.num_tiles as f64;
        self.zoom = pixels_per_image.sqrt().floor();

        self.lw = std::cmp::max(1, (self.w / self.zoom).floor() as i64);
        self.lh = (self.num_tiles as f64 / self.lw as f64).ceil() as i64;

        // Numer of rows takes the overflow, rescale to ensure the grid fits the window.
        let gh = self.lh as f64 * self.zoom;
        if gh > self.h {
            self.zoom *= self.h / gh;
        }

        // Add a black border.
        self.zoom *= 0.95;

        // Recenter the grid.
        let gh = self.lh as f64 * self.zoom;
        let gw = self.lw as f64 * self.zoom;
        self.x = (self.w - gw) / 2.0;
        self.y = (self.h - gh) / 2.0;
    }

    fn resize(&mut self, w: f64, h: f64, num_tiles: usize) {
        self.w = w;
        self.h = h;
        self.num_tiles = num_tiles;
        if self.auto {
            self.reset();
        }
    }

    fn move_by(&mut self, x: f64, y: f64) {
        self.auto = false;
        self.x += x;
        self.y += y;
    }

    fn zoom_by(&mut self, r: f64) {
        self.auto = false;

        let z = self.zoom;
        self.zoom *= 1.0 + r;

        // min size
        if self.zoom < 8.0 {
            self.zoom = 8.0;
        }

        let zd = self.zoom - z;

        let grid_size = self.lw as f64 * z;
        let x_bias = (self.mx - self.x) / grid_size;
        let y_bias = (self.my - self.y) / grid_size;

        let pd = self.lw as f64 * zd;
        self.x -= pd * x_bias;
        self.y -= pd * y_bias;
    }

    fn dims(&self) -> [f64; 2] {
        [self.w, self.h]
    }

    // TODO: Separate coordinates from visible check.
    // TODO: Convert visibility check into visible ratio.
    fn coords(&self, i: usize) -> (f64, f64, bool) {
        let i = i as f64;

        let (x_min, y_min) = (
            (i as i64 % self.lw as i64) as f64 * self.zoom,
            (i as i64 / self.lw as i64) as f64 * self.zoom,
        );

        let (x_max, y_max) = (x_min + self.zoom, y_min + self.zoom);

        let is_visible = ((self.x + x_max) > 0.0 && (self.x + x_min) < self.w)
            && ((self.y + y_max) > 0.0 && (self.y + y_min) < self.h);

        (x_min, y_min, is_visible)
    }
}

fn u32u8(size: u32) -> u8 {
    assert!(size.is_power_of_two());
    (32 - size.leading_zeros() - 1) as u8
}

fn u8u32(size: u8) -> u32 {
    1 << size
}

#[test]
fn size_conversions() {
    assert_eq!(u32u8(128), 7);
    assert_eq!(u8u32(7), 128);
}

#[derive(Debug, Serialize, Deserialize)]
struct Tiles {
    tile_size: u32,
    wc: u8,
    hc: u8,
    refs: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Refs {
    One(u64),
    Many(Box<Tiles>),
}

#[derive(Debug, Serialize, Deserialize)]
struct Thumb {
    w: u32,
    h: u32,
    refs: Refs,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    thumbs: Vec<Thumb>,
}

impl Thumb {
    fn max_dimension(&self) -> u32 {
        std::cmp::max(self.w, self.h)
    }

    fn size(&self) -> u32 {
        self.max_dimension().next_power_of_two()
    }
}

impl Draw for Thumb {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<u64, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool {
        let img = image::Image::new();

        let max_dimension = self.max_dimension();

        let zoom = zoom / (max_dimension as f64);
        let trans = trans.zoom(zoom);

        let (xo, yo) = (
            (max_dimension - self.w) as f64 / 2.0,
            (max_dimension - self.h) as f64 / 2.0,
        );

        match &self.refs {
            Refs::One(tile) => {
                if let Some(texture) = tiles.get(&tile) {
                    let trans = trans.trans(xo, yo);
                    img.draw(texture, &draw_state, trans, g);
                }
            }
            Refs::Many(t) => {
                let mut it = t.refs.iter();
                for ty in 0..(t.hc as u32) {
                    let ty = yo + (ty * t.tile_size) as f64;

                    for tx in 0..(t.wc as u32) {
                        let tx = xo + (tx * t.tile_size) as f64;

                        let tile = it.next().unwrap();

                        if let Some(texture) = tiles.get(tile) {
                            let trans = trans.trans(tx, ty);
                            img.draw(texture, &draw_state, trans, g);
                        }
                    }
                }
            }
        }

        true
    }
}

trait Nearest<T> {
    fn nearest(&self, target_size: u32) -> usize;
}

impl Nearest<Thumb> for Vec<Thumb> {
    fn nearest(&self, target_size: u32) -> usize {
        let mut found = None;

        for (i, thumb) in self.iter().enumerate() {
            let size = thumb.size();
            let dist = (target_size as i64 - size as i64).abs();
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

fn tile_spec_for_image_size(size: u32, w: u32, h: u32) -> (u32, u8, u8) {
    // overflow check
    assert!(size.is_power_of_two());

    let wc = (w + size - 1) / size;
    let hc = (h + size - 1) / size;

    if wc < 256 && hc < 256 {
        (size, wc as u8, hc as u8)
    } else {
        tile_spec_for_image_size(size << 1, w, h)
    }
}

#[test]
fn tile_counts_test() {
    assert_eq!(tile_spec_for_image_size(256, 1, 1), (256, 1, 1));
    assert_eq!(tile_spec_for_image_size(256, 256, 256), (256, 1, 1));
    assert_eq!(tile_spec_for_image_size(256, 257, 256), (256, 2, 1));
    assert_eq!(tile_spec_for_image_size(256, 257, 257), (256, 2, 2));
}

type ThumbRet = R<Metadata>;

fn make_thumb(db: Arc<database::Database>, file: Arc<File>, uid: u64) -> ThumbRet {
    let _s = ScopedDuration::new("make_thumb");

    let mut image = ::image::open(&file.path).map_err(E::ImageError)?;

    let (w, h) = image.dimensions();

    let target_tile_size = TARGET_TILE_SIZE;

    let orig_bucket = std::cmp::max(w, h).next_power_of_two();

    let min_bucket = std::cmp::min(MIN_SIZE, orig_bucket);

    let mut bucket = orig_bucket;

    let mut thumbs: Vec<Thumb> = Vec::new();

    let mut tiles: BTreeMap<u64, Vec<u8>> = BTreeMap::new();

    while min_bucket <= bucket {
        let current_bucket = {
            let (w, h) = image.dimensions();
            std::cmp::max(w, h).next_power_of_two()
        };

        // Downsample if needed.
        if bucket < current_bucket {
            image = image.thumbnail(bucket, bucket);
        }

        let lossy = bucket != orig_bucket;

        let (w, h) = image.dimensions();

        let (tile_size, wc, hc) = tile_spec_for_image_size(target_tile_size, w, h);

        let mut chunk_id = 0u16;

        let mut refs: Vec<u64> = Vec::new();

        for y in 0..(hc as u32) {
            let (min_y, max_y) = (y * tile_size, std::cmp::min((y + 1) * tile_size, h));

            for x in 0..(wc as u32) {
                let (min_x, max_x) = (x * tile_size, std::cmp::min((x + 1) * tile_size, w));

                let sub_image = ::image::DynamicImage::ImageRgba8(
                    image
                        .sub_image(min_x, min_y, max_x - min_x, max_y - min_y)
                        .to_image(),
                );

                let format = if lossy {
                    ::image::ImageOutputFormat::JPEG(70)
                } else {
                    ::image::ImageOutputFormat::JPEG(100)
                };

                let mut buf = Vec::with_capacity((2 * tile_size * tile_size) as usize);
                sub_image.write_to(&mut buf, format).expect("write_to");

                let tile_id = make_tile_id(u32u8(bucket), uid, chunk_id);
                chunk_id += 1;

                tiles.insert(tile_id, buf);

                refs.push(tile_id);
            }
        }

        let refs = if refs.len() == 1 {
            Refs::One(*refs.first().unwrap())
        } else {
            let many = Tiles {
                tile_size,
                wc,
                hc,
                refs,
            };
            Refs::Many(Box::new(many))
        };

        let thumb = Thumb { w, h, refs };

        thumbs.push(thumb);

        bucket >>= 1;
    }

    thumbs.reverse();

    let metadata = Metadata { thumbs };

    // Do before metadata write to prevent invalid metadata references.
    for (id, tile) in tiles {
        db.set(id, &tile).expect("db set");
    }

    db.set_metadata(&*file, &metadata).expect("set metadata");

    Ok(metadata)
}

static UPS: u64 = 100;

// TODO: make flag
static MIN_SIZE: u32 = 8;

// TODO: make flag
static TARGET_TILE_SIZE: u32 = 128;

static UPSIZE_FACTOR: f64 = 1.5;

trait Draw {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<u64, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool;
}

#[derive(Default)]
struct Image {
    filepath: Arc<File>,
    metadata: Option<R<Metadata>>,
    size: Option<usize>,
}

impl Image {
    fn from(filepath: Arc<File>, metadata: Option<Metadata>) -> Self {
        Image {
            filepath,
            metadata: metadata.map(Ok),
            ..Default::default()
        }
    }
}

impl Draw for Image {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<u64, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool {
        let metadata = if let Some(Ok(metadata)) = &self.metadata {
            metadata
        } else {
            return false;
        };

        for thumb in &metadata.thumbs {
            // TODO: only draw the active thumb.
            thumb.draw(trans, zoom, tiles, draw_state, g);
        }

        true
    }
}

type Handle<T> = Fuse<RemoteHandle<T>>;

struct App {
    db: Arc<database::Database>,

    images: Vec<Image>,

    // Files ordered by distance from the mouse.
    mouse_order_xy: (f64, f64),

    // Graphics state
    new_window_settings: Option<WindowSettings>,
    window_settings: WindowSettings,
    window: PistonWindow,
    texture_context: G2dTextureContext,

    tiles: BTreeMap<u64, G2dTexture>,

    // Movement state & modes.
    view: View,
    panning: bool,
    zooming: Option<f64>,
    cursor_captured: bool,

    cache_todo: [VecDeque<usize>; 2],

    thumb_todo: [VecDeque<usize>; 2],
    thumb_handles: BTreeMap<usize, Handle<ThumbRet>>,
    thumb_executor: futures::executor::ThreadPool,
    thumb_threads: usize,

    shift_held: bool,

    should_recalc: Option<()>,

    base_id: u64,
}

impl App {
    fn new(
        images: Vec<Image>,
        db: Arc<database::Database>,
        thumbnailer_threads: usize,
        base_id: u64,
    ) -> Self {
        let view = View::new(images.len());

        let window_settings = WindowSettings::new("pix", view.dims())
            .exit_on_esc(true)
            .fullscreen(false);

        let mut window: PistonWindow = window_settings.build().expect("window build");
        window.set_ups(UPS);

        let texture_context = window.create_texture_context();

        Self {
            db,

            mouse_order_xy: (0.0, 0.0),

            new_window_settings: None,
            window_settings,
            window,
            texture_context,

            tiles: BTreeMap::new(),

            view,
            panning: false,
            zooming: None,
            cursor_captured: false,

            cache_todo: [
                VecDeque::with_capacity(images.len()),
                VecDeque::with_capacity(images.len()),
            ],

            thumb_handles: BTreeMap::new(),
            thumb_executor: futures::executor::ThreadPool::builder()
                .pool_size(thumbnailer_threads)
                .name_prefix("thumbnailer")
                .create()
                .unwrap(),
            thumb_threads: thumbnailer_threads,

            thumb_todo: [
                VecDeque::with_capacity(images.len()),
                VecDeque::with_capacity(images.len()),
            ],

            shift_held: false,

            should_recalc: Some(()),

            base_id,

            images,
        }
    }

    fn rebuild_window(&mut self) {
        if let Some(new) = self.new_window_settings.take() {
            // force reload of images
            for s in &mut self.images {
                s.size = None;
            }

            self.window_settings = new.clone();
            self.window = new.build().expect("window build");
            self.tiles.clear();

            self.should_recalc = Some(());

            self.panning = false;
            self.cursor_captured = false;
            self.zooming = None;
        }
    }

    fn target_size(&self) -> u32 {
        ((self.view.zoom * UPSIZE_FACTOR) as u32).next_power_of_two()
    }

    fn load_tile_from_db(&mut self, now: &SystemTime) -> bool {
        let _s = ScopedDuration::new("load_tile_from_db");

        let target_size = self.target_size();

        let texture_settings = TextureSettings::new();

        // visible first
        for p in 0..self.cache_todo.len() {
            // at most one pass through the list.
            // TODO
            for _ in 0..self.cache_todo[p].len() {
                let i = self.cache_todo[p].pop_front().unwrap();

                let image = &self.images[i];

                let metadata = match &image.metadata {
                    Some(Ok(metadata)) => metadata,
                    Some(_) => continue,
                    None => {
                        // No metadata found, thus no thumbnails to load. Move it into the thumb
                        // queue to be thumbnailed.
                        self.thumb_todo[p].push_back(i);
                        continue;
                    }
                };

                let thumbs = &metadata.thumbs;

                // If visible
                let n = if p == 0 {
                    thumbs.nearest(target_size)
                } else {
                    0
                };

                let current_size = image.size.unwrap_or(0);

                // Progressive resizing.
                let n = match n.cmp(&current_size) {
                    Ordering::Less => current_size - 1,
                    Ordering::Equal => {
                        // Already loaded target size.
                        continue;
                    }
                    Ordering::Greater => current_size + 1,
                };

                let thumb = &thumbs[n];

                let tiles = match &thumb.refs {
                    Refs::One(r) => vec![*r],
                    Refs::Many(tiles) => tiles.refs.clone(),
                };

                // Load new tiles.
                for &tile in &tiles {
                    // Already loaded.
                    if self.tiles.contains_key(&tile) {
                        continue;
                    }

                    // load the tile from the cache
                    let _s3 = ScopedDuration::new("load_tile");

                    let data = self.db.get(tile).expect("db get").expect("missing tile");

                    let image = ::image::load_from_memory(&data).expect("load image");

                    // TODO: Would be great to move off thread.
                    let image = Texture::from_image(
                        &mut self.texture_context,
                        &image.to_rgba(),
                        &texture_settings,
                    )
                    .expect("texture");

                    self.tiles.insert(tile, image);

                    // Check if we've exhausted our time budget (we are in the main
                    // thread).
                    if now.elapsed().unwrap() > std::time::Duration::from_millis(10) {
                        // There might still be work to be done, resume from here next
                        // time.
                        self.cache_todo[p].push_front(i);
                        return true;
                    }
                }

                // Unload old tiles.
                for (j, thumb) in thumbs.iter().enumerate() {
                    if j == n {
                        continue;
                    }
                    match &thumb.refs {
                        Refs::One(r) => {
                            self.tiles.remove(r);
                        }
                        Refs::Many(t) => {
                            for tile in &t.refs {
                                self.tiles.remove(tile);
                            }
                        }
                    };
                }

                self.images[i].size = Some(n);

                self.cache_todo[p].push_back(i);
            }
        }

        false
    }

    fn enqueue(&mut self, i: usize) {
        let (_, _, is_visible) = self.view.coords(i);
        let p = (!is_visible) as usize;
        self.cache_todo[p].push_front(i);
    }

    fn recv_thumbs(&mut self) {
        let _s = ScopedDuration::new("recv_thumbs");

        let mut done: Vec<usize> = Vec::new();

        let mut handles = BTreeMap::new();
        std::mem::swap(&mut handles, &mut self.thumb_handles);

        for (&i, mut handle) in &mut handles {
            select! {
                thumb_res = handle => {
                    self.images[i].metadata = match thumb_res {
                        Ok(metadata) => {
                            // re-trigger cache lookup
                            self.enqueue(i);
                            Some(Ok(metadata))
                        }
                        Err(e) => {
                            error!("make_thumb: {}", e);
                            Some(Err(e))
                        }
                    };

                    done.push(i);
                }

                default => {}
            }
        }

        for i in &done {
            handles.remove(i);
        }

        std::mem::swap(&mut handles, &mut self.thumb_handles);
    }

    fn make_thumbs(&mut self) {
        let _s = ScopedDuration::new("make_thumbs");

        for p in 0..self.thumb_todo.len() {
            for _ in 0..self.thumb_todo[p].len() {
                if self.thumb_handles.len() > self.thumb_threads {
                    return;
                }

                let i = self.thumb_todo[p].pop_front().unwrap();

                let image = &self.images[i];
                if image.metadata.is_some() {
                    continue;
                }

                // Already fetching.
                if self.thumb_handles.contains_key(&i) {
                    continue;
                }

                let tile_id_index = self.base_id + i as u64;
                let file = Arc::clone(&image.filepath);
                let db = Arc::clone(&self.db);

                let fut = async move { make_thumb(db, file, tile_id_index) };

                let handle = self.thumb_executor.spawn_with_handle(fut).unwrap().fuse();

                self.thumb_handles.insert(i, handle);
            }
        }
    }

    fn update(&mut self, args: UpdateArgs) {
        let now = SystemTime::now();

        if let Some(z) = self.zooming {
            self.zoom_by(z * args.dt);
            return;
        }

        if let Some(()) = self.should_recalc.take() {
            self.recalc_visible();
            return;
        }

        if self.load_tile_from_db(&now) {
            return;
        }

        self.recv_thumbs();
        self.make_thumbs();
    }

    fn resize(&mut self, w: f64, h: f64) {
        self.view.resize(w, h, self.images.len());
        self.should_recalc = Some(());
    }

    fn recalc_visible(&mut self) {
        let _s = ScopedDuration::new("recalc_visible");

        for q in &mut self.cache_todo {
            q.clear();
        }

        for q in &mut self.thumb_todo {
            q.clear();
        }

        let mut mouse_distance: Vec<usize> = self
            .images
            .iter()
            .enumerate()
            .filter_map(|(i, image)| match image.metadata {
                Some(Err(_)) => None,
                _ => Some(i),
            })
            .collect();

        let v = &self.view;
        mouse_distance.sort_by_key(|&i| {
            let (x, y, _) = v.coords(i);
            let (dx, dy) = ((v.x + x - v.mx), (v.y + y - v.my));
            ((dx * dx) + (dy * dy)) as usize
        });

        let (hi, lo): (Vec<usize>, Vec<usize>) =
            mouse_distance.into_iter().partition(|&i| v.coords(i).2);

        self.cache_todo[0].extend(hi.iter());
        self.cache_todo[1].extend(lo.iter());
    }

    fn mouse_cursor(&mut self, x: f64, y: f64) {
        self.view.mx = x;
        self.view.my = y;

        let (ox, oy) = self.mouse_order_xy;
        let dist = ((x - ox) as u64).checked_pow(2).unwrap_or(0)
            + ((y - oy) as u64).checked_pow(2).unwrap_or(0);
        let trigger_dist = std::cmp::max(50, self.view.zoom as u64);
        if dist > trigger_dist {
            self.should_recalc = Some(());
            self.mouse_order_xy = (x, y);
        }
    }

    fn mouse_scroll(&mut self, _h: f64, v: f64) {
        for _ in 0..(v as i64) {
            self.zoom_by(self.zoom_increment());
        }
        for _ in (v as i64)..0 {
            self.zoom_by(-self.zoom_increment());
        }
    }

    fn mouse_relative(&mut self, dx: f64, dy: f64) {
        if self.panning {
            if self.cursor_captured {
                self.view.center_mouse();
            }
            self.move_by(dx * 4.0, dy * 4.0);
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

    fn move_by(&mut self, x: f64, y: f64) {
        self.view.move_by(x, y);
        self.should_recalc = Some(());
    }

    fn zoom_by(&mut self, r: f64) {
        self.view.zoom_by(r);
        self.should_recalc = Some(());
    }

    fn reset(&mut self) {
        self.view.reset();
        self.should_recalc = Some(());
    }

    fn button(&mut self, b: ButtonArgs) {
        match (b.state, b.button) {
            (ButtonState::Press, Button::Keyboard(Key::Z)) => {
                self.reset();
            }

            (ButtonState::Press, Button::Keyboard(Key::F)) => {
                let mut settings = self.window_settings.clone();
                settings.set_fullscreen(!settings.get_fullscreen());
                self.new_window_settings = Some(settings);
            }

            (ButtonState::Press, Button::Keyboard(Key::T)) => {
                self.cursor_captured = !self.cursor_captured;
                self.window.set_capture_cursor(self.cursor_captured);
                self.panning = self.cursor_captured;
                self.view.center_mouse();
            }

            (ButtonState::Press, Button::Keyboard(Key::Up)) => {
                self.move_by(0.0, self.shift_increment());
            }

            (ButtonState::Press, Button::Keyboard(Key::Down)) => {
                self.move_by(0.0, -self.shift_increment());
            }

            (ButtonState::Press, Button::Keyboard(Key::Left)) => {
                self.move_by(self.shift_increment(), 0.0);
            }

            (ButtonState::Press, Button::Keyboard(Key::Right)) => {
                self.move_by(-self.shift_increment(), 0.0);
            }

            (ButtonState::Press, Button::Keyboard(Key::PageUp)) => {
                self.view.center_mouse();
                self.zoom_by(-self.zoom_increment());
            }

            (ButtonState::Press, Button::Keyboard(Key::PageDown)) => {
                self.view.center_mouse();
                self.zoom_by(self.zoom_increment());
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

    fn draw_2d(
        thumb_handles: &BTreeMap<usize, Handle<ThumbRet>>,
        e: &Event,
        c: Context,
        g: &mut G2d,
        view: &View,
        tiles: &BTreeMap<u64, G2dTexture>,
        images: &[Image],
    ) {
        clear([0.0, 0.0, 0.0, 1.0], g);

        let c = c.trans(view.x, view.y);

        let args = e.render_args().expect("render args");
        let draw_state = DrawState::default().scissor([0, 0, args.draw_size[0], args.draw_size[1]]);

        let black = color::hex("000000");
        let missing_color = color::hex("888888");
        let op_color = color::hex("222222");

        let zoom = (view.zoom * view.zoom) / (view.zoom + 1.0);

        for (i, image) in images.iter().enumerate() {
            let (x, y, is_visible) = view.coords(i);
            if !is_visible {
                continue;
            }

            let trans = c.transform.trans(x, y);

            if image.draw(trans, zoom, tiles, &draw_state, g) {
                continue;
            }

            if thumb_handles.contains_key(&i) {
                rectangle(op_color, [0.0, 0.0, zoom, zoom], trans, g);
                rectangle(black, [1.0, 1.0, zoom - 2.0, zoom - 2.0], trans, g);
            } else {
                rectangle(missing_color, [zoom / 2.0, zoom / 2.0, 1.0, 1.0], trans, g);
            }
        }
    }

    fn run(&mut self) {
        loop {
            let _s = ScopedDuration::new("run_loop");

            self.rebuild_window();

            if let Some(e) = self.window.next() {
                let _s = ScopedDuration::new("run_loop_next");

                e.update(|args| {
                    let _s = ScopedDuration::new("update");
                    self.update(*args);
                });

                e.resize(|args| {
                    let _s = ScopedDuration::new("resize");
                    self.resize(args.draw_size[0] as f64, args.draw_size[1] as f64)
                });

                e.mouse_scroll(|hv| {
                    let _s = ScopedDuration::new("mouse_scroll");
                    self.mouse_scroll(hv[0], hv[1]);
                });

                e.mouse_cursor(|xy| {
                    let _s = ScopedDuration::new("mouse_cursor");
                    self.mouse_cursor(xy[0], xy[1]);
                });

                e.mouse_relative(|dxdy| {
                    let _s = ScopedDuration::new("mouse_relative");
                    self.mouse_relative(dxdy[0], dxdy[1]);
                });

                e.button(|b| {
                    let _s = ScopedDuration::new("button");
                    self.button(b)
                });

                // borrowck
                let v = &self.view;
                let t = &self.tiles;
                let images = &self.images;
                let thumb_handles = &self.thumb_handles;
                self.window.draw_2d(&e, |c, g, _device| {
                    let _s = ScopedDuration::new("draw_2d");
                    Self::draw_2d(thumb_handles, &e, c, g, v, t, images);
                });
            } else {
                break;
            }
        }

        self.thumb_handles.clear();
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct File {
    // TODO: Use bstring?
    path: String,
    modified: u64,
    file_size: u64,
}

fn find_images(dirs: Vec<String>) -> Vec<Arc<File>> {
    let _s = ScopedDuration::new("find_images");

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
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("duration since unix epoch")
                .as_secs();

            let path = entry.path().canonicalize().expect("canonicalize");
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

fn make_tile_id(size: u8, index: u64, chunk: u16) -> u64 {
    assert!(index < (1u64 << 40));
    (chunk as u64) | (index << 16) | ((size as u64) << 56)
}

#[allow(unused)]
fn deconstruct_tile_id(tile_id: u64) -> (u8, u64, u16) {
    let size = ((tile_id & 0xFF00_0000_0000_0000u64) >> 56) as u8;
    let index = (tile_id & 0x00FF_FFFF_FFFF_0000u64) >> 16;
    let chunk = (tile_id & 0x0000_0000_0000_FFFFu64) as u16;
    (size, index, chunk)
}

#[test]
fn tile_id_test() {
    assert_eq!(make_tile_id(0xFFu8, 0u64, 0u16), 0xFF00_0000_0000_0000u64);
    assert_eq!(
        make_tile_id(0u8, 0x00FF_FFFF_FFFFu64, 0u16),
        0x00F_FFFFF_FFFF_0000u64
    );
    assert_eq!(make_tile_id(0u8, 0u64, 0xFFFFu16), 0x0000_0000_0000_FFFFu64);
    assert_eq!(
        make_tile_id(0xFFu8, 0u64, 0u16).to_be_bytes(),
        [0xFF, 0, 0, 0, 0, 0, 0, 0]
    );
}

fn main() {
    env_logger::init();

    /////////////////
    // PARSE FLAGS //
    /////////////////

    let matches = clap::App::new("pix")
        .version("1.0")
        .author("Mason Larobina <mason.larobina@gmail.com>")
        .arg(
            Arg::with_name("paths")
                .value_name("PATHS")
                .multiple(true)
                .help("Images or directories of images to view."),
        )
        .arg(
            Arg::with_name("threads")
                .long("--threads")
                .value_name("COUNT")
                .takes_value(true)
                .required(false)
                .help("Set number of background thumbnailer threads."),
        )
        .arg(
            Arg::with_name("db_path")
                .long("--db_path")
                .value_name("PATH")
                .takes_value(true)
                .help("Alternate thumbnail database path."),
        )
        .get_matches();

    let paths = matches
        .values_of_lossy("paths")
        .unwrap_or_else(|| vec![String::from(".")]);
    info!("Paths: {:?}", paths);

    let thumbnailer_threads: usize = if let Some(threads) = matches.value_of("threads") {
        threads.parse().expect("not an int")
    } else {
        num_cpus::get()
    };
    info!("Thumbnailer threads {}", thumbnailer_threads);

    let db_path: String = if let Some(db_path) = matches.value_of("db_path") {
        db_path.to_owned()
    } else {
        let mut db_path = dirs::cache_dir().expect("cache dir");
        db_path.push("pix/thumbs.db");
        db_path.to_str().expect("db path as str").to_owned()
    };
    info!("Database path: {}", db_path);

    /////////
    // RUN //
    /////////

    let files = find_images(paths);

    assert!(!files.is_empty());
    info!("Found {} images", files.len());

    let db = database::Database::open(&db_path).expect("db open");
    let base_id = db.reserve(files.len());

    let images: Vec<Image> = files
        .into_iter()
        .map(|file| {
            let metadata = match db.get_metadata(&*file) {
                Ok(Some(metadata)) => Some(metadata),
                Err(e) => {
                    error!("get metadata error: {:?}", e);
                    None
                }
                _ => None,
            };

            Image::from(file, metadata)
        })
        .collect();

    {
        let _s = ScopedDuration::new("uptime");
        App::new(images, Arc::new(db), thumbnailer_threads, base_id).run();
    }

    stats::dump();
}
