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
#[macro_use]
extern crate lazy_static;

mod database;
mod stats;

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
use std::time::{Duration, Instant, SystemTime};
use vecmath::{vec2_add, vec2_mul, vec2_scale, vec2_sub, Vector2};

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
    num_images: f64,

    // Window dimensions.
    win_size: Vector2<f64>,

    // Logical dimensions.
    grid_size: Vector2<f64>,

    // View offsets.
    trans: Vector2<f64>,

    // Scale from logical to physical coordinates.
    zoom: f64,
    min_zoom: f64,

    // Mouse coordinates.
    mouse: Vector2<f64>,

    // Has the user panned or zoomed?
    auto: bool,
}

impl View {
    fn new(num_images: usize) -> Self {
        Self {
            num_images: num_images as f64,
            win_size: [800., 600.],
            grid_size: [1.0, 1.0],
            auto: true,
            ..Default::default()
        }
    }

    fn center_mouse(&mut self) {
        self.mouse = vec2_scale(self.win_size, 0.5);
    }

    fn reset(&mut self) {
        self.auto = true;

        let [w, h] = self.win_size;

        self.zoom = {
            let px_per_image = (w * h) / self.num_images;
            px_per_image.sqrt()
        };

        self.grid_size = {
            let grid_w = f64::max(1.0, (w / self.zoom).floor());
            let grid_h = (self.num_images / grid_w).ceil();
            [grid_w, grid_h]
        };

        // Numer of rows takes the overflow, rescale to ensure the grid fits the window.
        let grid_px = vec2_scale(self.grid_size, self.zoom);
        if h < grid_px[1] {
            self.zoom *= h / grid_px[1];
        }

        // Add black border.
        self.zoom *= 0.95;

        self.min_zoom = self.zoom * 0.5;

        self.trans = {
            let grid_px = vec2_scale(self.grid_size, self.zoom);
            let border_px = vec2_sub(self.win_size, grid_px);
            vec2_scale(border_px, 0.5)
        };
    }

    fn resize(&mut self, win_size: Vector2<f64>) {
        self.win_size = win_size;
        if self.auto {
            self.reset();
        }
    }

    fn trans(&mut self, trans: Vector2<f64>) {
        self.auto = false;
        self.trans = vec2_add(self.trans, trans);
    }

    fn zoom(&mut self, ratio: f64) {
        self.auto = false;

        let zoom = self.zoom;
        self.zoom = f64::max(self.min_zoom, zoom * ratio);

        let bias = {
            let grid_pos = vec2_sub(self.mouse, self.trans);
            let grid_px = vec2_scale(self.grid_size, zoom);
            vec2_div(grid_pos, grid_px)
        };

        let trans = {
            let grid_delta = vec2_scale(self.grid_size, self.zoom - zoom);
            vec2_mul(grid_delta, bias)
        };

        self.trans = vec2_sub(self.trans, trans);
    }

    fn coords(&self, i: usize) -> Vector2<f64> {
        let grid_w = self.grid_size[0] as usize;
        let coords = [(i % grid_w) as f64, (i / grid_w) as f64];
        vec2_add(self.trans, vec2_scale(coords, self.zoom))
    }

    fn is_visible(&self, min: Vector2<f64>) -> bool {
        let max = vec2_add(min, [self.zoom, self.zoom]);
        let [w, h] = self.win_size;
        (max[0] > 0.0 && min[0] < w) && (max[1] > 0.0 && min[1] < h)
    }

    fn visible_ratio(&self, [x_min, y_min]: Vector2<f64>) -> f64 {
        let [x_max, y_max] = vec2_add([x_min, y_min], [self.zoom, self.zoom]);
        let [w, h] = self.win_size;
        f64::max(
            f64::min(((x_max / w) - 0.5).abs(), ((x_min / w) - 0.5).abs()),
            f64::min(((y_max / h) - 0.5).abs(), ((y_min / h) - 0.5).abs()),
        ) + 0.5
    }
}

#[test]
fn view_vis_test() {
    let view = View {
        win_size: [200.0, 100.0],
        grid_size: [20.0, 10.0],
        zoom: 10.0,
        ..Default::default()
    };

    assert_eq!(view.coords(0), [0.0, 0.0]);
    assert_eq!(view.coords(1), [10.0, 0.0]);
    assert_eq!(view.coords(20), [0.0, 10.0]);

    assert_eq!(view.visible_ratio([0.0, 0.0]), 0.95);
    assert_eq!(view.visible_ratio([190.0, 0.0]), 0.95);
    assert_eq!(view.visible_ratio([190.0, 90.0]), 0.95);
    assert_eq!(view.visible_ratio([0.0, 90.0]), 0.95);

    assert_eq!(view.visible_ratio([-20.0, 0.0]), 1.05);
    assert_eq!(view.visible_ratio([210.0, 0.0]), 1.05);

    assert_eq!(view.visible_ratio([0.0, -20.0]), 1.1);
    assert_eq!(view.visible_ratio([0.0, 110.0]), 1.1);
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Pow2(u8);

impl Pow2 {
    fn from(i: u32) -> Self {
        assert!(i.is_power_of_two());
        Pow2((32 - i.leading_zeros() - 1) as u8)
    }

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
        TileRef(0x00F_FFFFF_FFFF_0000u64)
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

#[derive(Debug, Serialize, Deserialize)]
struct Thumb {
    img_size: [u32; 2],
    tile_refs: Vec<TileRef>,
}

#[derive(Debug, Serialize, Deserialize)]
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
}

impl Draw for Thumb {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<TileRef, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool {
        let img = image::Image::new();

        let max_dimension = self.max_dimension() as f64;

        let trans = trans.zoom(zoom / max_dimension);

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
                    img.draw(texture, &draw_state, trans, g);
                }
            }
        }

        true
    }
}

type ThumbRet = R<Metadata>;

fn make_thumb(db: Arc<database::Database>, file: Arc<File>, uid: u64) -> ThumbRet {
    let _s = ScopedDuration::new("make_thumb");

    let mut image = ::image::open(&file.path).map_err(E::ImageError)?;

    let (w, h) = image.dimensions();

    let orig_bucket = std::cmp::max(w, h).next_power_of_two();

    let min_bucket = std::cmp::min(MIN_SIZE, orig_bucket);

    let mut bucket = orig_bucket;

    let mut thumbs: Vec<Thumb> = Vec::new();

    let mut tiles: BTreeMap<TileRef, Vec<u8>> = BTreeMap::new();

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

        let mut chunk_id = 0u16;

        let mut thumb = Thumb {
            img_size: [w, h],
            tile_refs: Vec::new(),
        };

        let spec = thumb.tile_spec();

        for (min_y, max_y) in spec.y_ranges() {
            let y_range = max_y - min_y;

            for (min_x, max_x) in spec.x_ranges() {
                let x_range = max_x - min_x;

                let sub_image = ::image::DynamicImage::ImageRgba8(
                    image.sub_image(min_x, min_y, x_range, y_range).to_image(),
                );

                let format = if lossy {
                    ::image::ImageOutputFormat::JPEG(70)
                } else {
                    ::image::ImageOutputFormat::JPEG(100)
                };

                let mut buf = Vec::with_capacity((2 * x_range * y_range) as usize);
                sub_image.write_to(&mut buf, format).expect("write_to");

                let tile_id = TileRef::new(Pow2::from(bucket), uid, chunk_id);
                chunk_id += 1;

                tiles.insert(tile_id, buf);

                thumb.tile_refs.push(tile_id);
            }
        }

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

static UPSIZE_FACTOR: f64 = 1.5;

trait Draw {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<TileRef, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool;
}

#[derive(Debug)]
enum MetadataState {
    Missing,
    Some(Metadata),
    Errored,
}

impl std::default::Default for MetadataState {
    fn default() -> Self {
        MetadataState::Missing
    }
}

#[derive(Default)]
struct Image {
    file: Arc<File>,
    metadata: MetadataState,
    size: Option<usize>,
}

impl Image {
    fn from(file: Arc<File>, metadata: Option<Metadata>) -> Self {
        Image {
            file,
            metadata: match metadata {
                Some(metadata) => MetadataState::Some(metadata),
                None => MetadataState::Missing,
            },
            ..Default::default()
        }
    }
}

impl Draw for Image {
    fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        tiles: &BTreeMap<TileRef, G2dTexture>,
        draw_state: &DrawState,
        g: &mut G2d,
    ) -> bool {
        if let Some(n) = self.size {
            let metadata = match &self.metadata {
                MetadataState::Some(metadata) => metadata,
                _ => unreachable!(),
            };
            let thumb = &metadata.thumbs[n];
            thumb.draw(trans, zoom, tiles, draw_state, g);
            true
        } else {
            false
        }
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

    tiles: BTreeMap<TileRef, G2dTexture>,

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

#[inline(always)]
fn vec2_div<T>(a: Vector2<T>, b: Vector2<T>) -> Vector2<T>
where
    T: Copy + std::ops::Div<T, Output = T>,
{
    [a[0] / b[0], a[1] / b[1]]
}

#[inline(always)]
fn vec2_u32(a: Vector2<f64>) -> Vector2<u32> {
    [a[0] as u32, a[1] as u32]
}

#[inline(always)]
fn vec2_f64(a: Vector2<u32>) -> Vector2<f64> {
    [a[0] as f64, a[1] as f64]
}

#[inline(always)]
fn vec2_ceil(a: Vector2<f64>) -> Vector2<f64> {
    [a[0].ceil(), a[1].ceil()]
}

#[inline(always)]
fn vec2_log(a: Vector2<f64>, base: f64) -> Vector2<f64> {
    [a[0].log(base), a[1].log(base)]
}

impl App {
    fn new(
        images: Vec<Image>,
        db: Arc<database::Database>,
        thumbnailer_threads: usize,
        base_id: u64,
    ) -> Self {
        let view = View::new(images.len());

        let window_settings = WindowSettings::new("pix", view.win_size)
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
            while let Some(i) = self.cache_todo[p].pop_front() {
                let image = &self.images[i];

                let metadata = match &image.metadata {
                    MetadataState::Missing => {
                        self.thumb_todo[p].push_back(i);
                        continue;
                    }
                    MetadataState::Some(metadata) => metadata,
                    MetadataState::Errored => {
                        unreachable!();
                    }
                };

                // If visible
                let n = if p == 0 {
                    metadata.nearest(target_size)
                } else {
                    let ratio = self.view.visible_ratio(self.view.coords(i));
                    let shift = f64::max(0.0, ratio - 1.0).floor() as usize;
                    metadata.nearest(target_size >> shift)
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

                // Load new tiles.
                for tile_ref in &metadata.thumbs[n].tile_refs {
                    // Already loaded.
                    if self.tiles.contains_key(tile_ref) {
                        continue;
                    }

                    // load the tile from the cache
                    let _s3 = ScopedDuration::new("load_tile");

                    let data = self
                        .db
                        .get(*tile_ref)
                        .expect("db get")
                        .expect("missing tile");

                    let image = ::image::load_from_memory(&data).expect("load image");

                    // TODO: Would be great to move off thread.
                    let image = Texture::from_image(
                        &mut self.texture_context,
                        &image.to_rgba(),
                        &texture_settings,
                    )
                    .expect("texture");

                    self.tiles.insert(*tile_ref, image);

                    // Check if we've exhausted our time budget (we are in the main
                    // thread).
                    if now.elapsed().unwrap() > Duration::from_millis(10) {
                        // Resume processing this image on the next call.
                        self.cache_todo[p].push_front(i);
                        return true;
                    }
                }

                // Unload old tiles.
                for (j, thumb) in metadata.thumbs.iter().enumerate() {
                    if j == n {
                        continue;
                    }
                    for tile_ref in &thumb.tile_refs {
                        self.tiles.remove(tile_ref);
                    }
                }

                self.images[i].size = Some(n);

                self.cache_todo[p].push_back(i);
            }
        }

        false
    }

    fn enqueue(&mut self, i: usize) {
        let is_visible = self.view.is_visible(self.view.coords(i));
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
                            MetadataState::Some(metadata)
                        }
                        Err(e) => {
                            error!("make_thumb: {}", e);
                            MetadataState::Errored
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
            while let Some(i) = self.thumb_todo[p].pop_front() {
                if self.thumb_handles.len() > self.thumb_threads {
                    return;
                }

                let image = &self.images[i];

                match image.metadata {
                    MetadataState::Missing => {}
                    _ => continue,
                }

                if self.thumb_handles.contains_key(&i) {
                    continue;
                }

                let tile_id_index = self.base_id + i as u64;
                let file = Arc::clone(&image.file);
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
            self.zoom(1.0 + (z * args.dt));
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

    fn resize(&mut self, win_size: Vector2<f64>) {
        self.view.resize(win_size);
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
                MetadataState::Errored => None,
                _ => Some(i),
            })
            .collect();

        let v = &self.view;
        mouse_distance.sort_by_key(|&i| {
            let coords = v.coords(i);
            let [dx, dy] = vec2_sub(coords, v.mouse);
            ((dx * dx) + (dy * dy)) as usize
        });

        let (hi, lo): (Vec<usize>, Vec<usize>) = mouse_distance
            .into_iter()
            .partition(|&i| v.is_visible(v.coords(i)));

        self.cache_todo[0].extend(hi.iter());
        self.cache_todo[1].extend(lo.iter());
    }

    fn mouse_cursor(&mut self, x: f64, y: f64) {
        self.view.mouse = [x, y];

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
            self.zoom(1.0 + self.zoom_increment());
        }
        for _ in (v as i64)..0 {
            self.zoom(1.0 - self.zoom_increment());
        }
    }

    fn mouse_relative(&mut self, delta: Vector2<f64>) {
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
        self.view.trans(trans);
        self.should_recalc = Some(());
    }

    fn zoom(&mut self, ratio: f64) {
        self.view.zoom(ratio);
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

    fn draw_2d(
        thumb_handles: &BTreeMap<usize, Handle<ThumbRet>>,
        e: &Event,
        c: Context,
        g: &mut G2d,
        view: &View,
        tiles: &BTreeMap<TileRef, G2dTexture>,
        images: &[Image],
    ) {
        clear([0.0, 0.0, 0.0, 1.0], g);

        let args = e.render_args().expect("render args");
        let draw_state = DrawState::default().scissor([0, 0, args.draw_size[0], args.draw_size[1]]);

        let black = color::hex("000000");
        let missing_color = color::hex("888888");
        let op_color = color::hex("222222");

        let zoom = (view.zoom * view.zoom) / (view.zoom + 1.0);

        for (i, image) in images.iter().enumerate() {
            let [x, y] = view.coords(i);

            if !view.is_visible([x, y]) {
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
                    let [w, h] = args.draw_size;
                    self.resize([w as f64, h as f64]);
                });

                e.mouse_scroll(|hv| {
                    let _s = ScopedDuration::new("mouse_scroll");
                    self.mouse_scroll(hv[0], hv[1]);
                });

                e.mouse_cursor(|xy| {
                    let _s = ScopedDuration::new("mouse_cursor");
                    self.mouse_cursor(xy[0], xy[1]);
                });

                e.mouse_relative(|delta| {
                    let _s = ScopedDuration::new("mouse_relative");
                    self.mouse_relative(delta);
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
