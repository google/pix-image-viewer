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

#![feature(async_await)]
#![feature(fnbox)]
#![feature(await_macro)]
#![feature(arbitrary_self_types)]
#![recursion_limit = "1000"]
#![feature(vec_remove_item)]
#![feature(drain_filter)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate failure;

use crate::stats::ScopedDuration;
use ::image::GenericImageView;
use boolinator::Boolinator;
use clap::Arg;
use piston_window::*;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use std::time::SystemTime;

#[macro_use]
extern crate lazy_static;

mod database;
mod queue;
mod stats;

fn npow2(i: u32) -> u32 {
    if i.is_power_of_two() {
        i
    } else {
        i.next_power_of_two()
    }
}

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

        self.lw = (self.w / self.zoom).floor() as i64;
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
        if self.zoom < 4.0 {
            self.zoom = 4.0;
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

#[derive(Debug)]
struct Image {
    w: u32,
    h: u32,
    data: G2dTexture,
    is_original: bool,
}

impl Image {
    fn from_vec(data: &[u8], factory: &mut GfxFactory) -> Self {
        let _s = ScopedDuration::new("image_from");
        let image = ::image::load_from_memory(data).expect("load image");
        Self::from_image(false, image, factory)
    }

    fn from_image(
        is_original: bool,
        image: ::image::DynamicImage,
        factory: &mut GfxFactory,
    ) -> Self {
        let _s = ScopedDuration::new("image_from");

        let (w, h) = image.dimensions();

        let data = Texture::from_image(factory, &image.to_rgba(), &TextureSettings::new())
            .expect("texture");

        Self {
            w,
            h,
            data,
            is_original,
        }
    }

    fn max_dimension(&self) -> u32 {
        std::cmp::max(self.w, self.h)
    }

    fn cur_size(&self) -> u32 {
        npow2(self.max_dimension())
    }
}

struct Thumb {
    // cache fill
    thumbs: Vec<(u32, Vec<u8>)>,
    // metadata
    dimensions: (u32, u32),
    // use directly
    image: ::image::DynamicImage,
    is_original: bool,
}

fn make_thumb(file: Arc<File>, make_size: u32) -> Option<Thumb> {
    let _s = ScopedDuration::new("make_thumb");

    assert!(make_size.is_power_of_two());

    match ::image::open(&file.path) {
        Ok(original) => {
            let (w, h) = original.dimensions();
            let max_dimension = npow2(std::cmp::max(w, h));
            let make_size = std::cmp::min(make_size, max_dimension);

            let mut make_sizes: BTreeSet<u32> = [32, 64, 128, 256]
                .iter()
                .cloned()
                .filter(|i| !file.cache_sizes.contains(i))
                .collect();

            make_sizes.insert(make_size);

            let mut thumbs = Vec::new();

            let mut image = Some((true, original));

            let mut image_ret = None;

            for &size in make_sizes.iter().rev() {
                let mut data = Vec::new();

                if size < max_dimension {
                    let (_, img) = image
                        .as_ref()
                        .or(image_ret.as_ref())
                        .expect("image or image_ret");
                    image = Some((false, img.thumbnail(size, size)));
                }

                image
                    .as_ref()
                    .or(image_ret.as_ref())
                    .expect("img")
                    .1
                    .write_to(&mut data, ::image::ImageFormat::JPEG)
                    .expect("write_to");

                thumbs.push((size, data));

                if size == make_size {
                    image_ret = image.take();
                }
            }

            let (is_original, image) = image_ret.expect("image ret");

            Some(Thumb {
                thumbs,
                dimensions: (w, h),
                image,
                is_original,
            })
        }

        Err(e) => {
            error!("Open image error: {:?}: {:?}", file.path, e);
            None
        }
    }
}

static UPS: u64 = 100;

static MIN_SIZE: u32 = 32;

static UPSIZE_FACTOR: f64 = 1.5;

struct App<'a> {
    db: &'a database::Database,

    files: &'a mut [Arc<File>],

    // TODO: Embed in self.textures?
    sizes: Vec<u32>,

    // Files ordered by distance from the mouse.
    mouse_order_xy: (f64, f64),

    skip: HashSet<usize>,

    // Graphics state
    new_window_settings: Option<WindowSettings>,
    window_settings: WindowSettings,
    window: PistonWindow,
    textures: BTreeMap<usize, Image>,

    // Movement state & modes.
    view: View,
    panning: bool,
    zooming: Option<f64>,
    cursor_captured: bool,

    cache_todo: [VecDeque<usize>; 2],

    thumb_todo: [VecDeque<usize>; 2],
    thumb_runner: queue::Queue<Option<Thumb>>,

    shift_held: bool,

    should_recalc: Option<()>,
}

impl<'a> App<'a> {
    fn new(
        files: &'a mut [Arc<File>],
        db: &'a database::Database,
        thumbnailer_threads: u32,
    ) -> Self {
        let view = View::new(files.len());

        let window_settings = WindowSettings::new("pix", view.dims())
            .exit_on_esc(true)
            .opengl(OpenGL::V3_2)
            .fullscreen(false);

        let mut window: PistonWindow = window_settings.build().expect("window build");
        window.set_ups(UPS);

        Self {
            db,

            sizes: (0..files.len()).map(|_| 0).collect(),

            mouse_order_xy: (0.0, 0.0),

            skip: HashSet::new(),

            new_window_settings: None,
            window_settings,
            window,
            textures: BTreeMap::new(),

            view,
            panning: false,
            zooming: None,
            cursor_captured: false,

            cache_todo: [
                VecDeque::with_capacity(files.len()),
                VecDeque::with_capacity(files.len()),
            ],

            thumb_todo: [
                VecDeque::with_capacity(files.len()),
                VecDeque::with_capacity(files.len()),
            ],

            thumb_runner: queue::Queue::new(thumbnailer_threads),

            shift_held: false,

            should_recalc: Some(()),

            files,
        }
    }

    fn rebuild_window(&mut self) {
        if let Some(new) = self.new_window_settings.take() {
            // force reload of images
            for s in &mut self.sizes {
                *s = 0;
            }

            self.window_settings = new.clone();
            self.window = new.build().expect("window build");
            self.textures.clear();

            self.should_recalc = Some(());

            self.panning = false;
            self.cursor_captured = false;
            self.zooming = None;
        }
    }

    fn zoom_size(&self) -> u32 {
        std::cmp::max(MIN_SIZE, npow2((self.view.zoom * UPSIZE_FACTOR) as u32))
    }

    fn next_cache_op(&mut self) -> Option<(usize, u32)> {
        let _s = ScopedDuration::new("next_cache_op");

        for p in 0..self.cache_todo.len() {
            for _ in 0..self.cache_todo[p].len() {
                if let Some(i) = self.cache_todo[p].pop_front() {
                    self.thumb_todo[p].push_back(i);

                    let file = &self.files[i];
                    if let Some(max_dimension) = file.max_dimension() {
                        let max_size = npow2(max_dimension);

                        // TODO: move into function
                        let target_size = if p == 0 {
                            std::cmp::min(self.zoom_size(), max_size)
                        } else {
                            1
                        };

                        let cur_size = self.textures.get(&i).map(Image::cur_size);

                        if let Some(cache_size) = file.nearest_cache_size(cur_size, target_size) {
                            return Some((i, cache_size));
                        }
                    }
                }
            }
        }

        return None;
    }

    fn next_thumb_op(&mut self) -> Option<(usize, u32)> {
        let _s = ScopedDuration::new("next_thumb_ob");

        for p in 0..self.thumb_todo.len() {
            for _ in 0..self.thumb_todo[p].len() {
                if let Some(i) = self.thumb_todo[p].pop_front() {
                    // Skip for now, it is re-inserted later.
                    if self.thumb_runner.inflight(i) {
                        continue;
                    }

                    let file = &self.files[i];

                    let zoom_size = self.zoom_size();

                    if let Some(max_dimension) = file.max_dimension() {
                        let max_size = npow2(max_dimension);
                        let target_size = std::cmp::min(zoom_size, max_size);

                        let img = self.textures.get(&i);

                        if p == 0 {
                            // Should load original?
                            if target_size == max_size {
                                if img.map(|i| i.is_original).unwrap_or(false) {
                                    assert_eq!(Some(max_dimension), img.map(Image::max_dimension));
                                    continue;
                                }
                                return Some((i, max_size));
                            }

                            if Some(target_size) == img.map(Image::cur_size) {
                                continue;
                            }

                            return Some((i, target_size));
                        }

                        assert!(!file.cache_sizes.is_empty());
                        continue;
                    }

                    let target_size = if p == 0 { zoom_size } else { MIN_SIZE };
                    return Some((i, target_size));
                }
            }
        }

        return None;
    }

    fn enqueue(&mut self, i: usize) {
        let (_, _, is_visible) = self.view.coords(i);
        let p = (!is_visible) as usize;
        self.cache_todo[p].push_front(i);
    }

    fn request_cache(&mut self) -> bool {
        if let Some((i, cache_size)) = self.next_cache_op() {
            let s = ScopedDuration::new("request_cache");

            let data = match self.db.get(&self.files[i], cache_size) {
                Ok(data) => data,
                Err(e) => {
                    error!("db get error {:?}", e);
                    return false;
                }
            };

            // TODO: Load large thumbs offthread?
            let texture = Image::from_vec(&data, &mut self.window.factory);
            self.set_texture(i, texture);

            assert_eq!(self.sizes[i], cache_size);
            stats::record(&format!("request_cache_{:05}", cache_size), s.elapsed());

            true
        } else {
            false
        }
    }

    fn request_thumb(&mut self) -> bool {
        if self.thumb_runner.is_full() {
            return false;
        }

        if let Some((i, target_size)) = self.next_thumb_op() {
            let _s = ScopedDuration::new("request_thumb");

            let files = Arc::clone(&self.files[i]);
            let work_fn = Box::new(move || make_thumb(files, target_size));

            self.thumb_runner.send(i, work_fn);

            true
        } else {
            false
        }
    }

    fn recv_thumb(&mut self) -> bool {
        if let Some((i, thumb_opt)) = self.thumb_runner.recv() {
            let s = ScopedDuration::new("recv_thumb");

            if let Some(thumb) = thumb_opt {
                let mut f = Arc::get_mut(&mut self.files[i]).expect("get mut");

                f.dimensions = Some(thumb.dimensions);

                for (size, data) in thumb.thumbs {
                    if let Err(e) = self.db.set(f, size, &data) {
                        error!("db set error: {:?}", e);
                    }
                }

                let texture =
                    Image::from_image(thumb.is_original, thumb.image, &mut self.window.factory);
                self.set_texture(i, texture);

                // Recheck incase target size has changed.
                self.enqueue(i);

                stats::record(&format!("recv_thumb_{:05}", self.sizes[i]), s.elapsed());
            } else {
                self.skip.insert(i);
            }
            true
        } else {
            false
        }
    }

    fn set_texture(&mut self, i: usize, texture: Image) {
        let size = npow2(texture.max_dimension());
        self.sizes[i] = size;
        self.textures.insert(i, texture);
    }

    fn update(&mut self, args: &UpdateArgs) {
        let _s = ScopedDuration::new("update");

        if let Some(z) = self.zooming {
            self.zoom_by(z * args.dt);
            return;
        }

        if let Some(()) = self.should_recalc.take() {
            self.recalc_visible();
            return;
        }

        let time = Instant::now();
        loop {
            // TODO: Or use estimator to ensure computations fit within budget.
            if time.elapsed().as_millis() > (1000 / UPS as u128) / 2 {
                break;
            }

            // TODO: And unload images no longer visible to reduce memory usage.
            if self.request_cache() || self.recv_thumb() || self.request_thumb() {
                continue;
            } else {
                break;
            }
        }
    }

    fn resize(&mut self, w: f64, h: f64) {
        self.view.resize(w, h, self.files.len());
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

        let skip = &self.skip;
        let mut mouse_distance: Vec<usize> = (0..self.files.len())
            .filter(|i| !skip.contains(i))
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
        let dist = ((x - ox) as u64).checked_pow(2).unwrap_or(0) + ((y - oy) as u64).checked_pow(2).unwrap_or(0);
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
        thumb_runner: &queue::Queue<Option<Thumb>>,
        e: &Event,
        c: Context,
        g: &mut G2d,
        view: &View,
        textures: &BTreeMap<usize, Image>,
    ) {
        let _s = ScopedDuration::new("draw_2d");

        clear([0.0, 0.0, 0.0, 1.0], g);

        let c = c.trans(view.x, view.y);

        let image = image::Image::new();

        let args = e.render_args().expect("render args");
        let draw_state =
            DrawState::default().scissor([0, 0, args.draw_width as u32, args.draw_height as u32]);

        let missing_color = color::hex("888888");
        let op_color = color::hex("444444");

        let og_color = color::hex("FF0000");

        for i in 0..view.num_tiles {
            let (x, y, is_visible) = view.coords(i);
            if !is_visible {
                continue;
            }

            let gap_px = 2.5;
            let zoom = view.zoom * (view.zoom / (view.zoom + gap_px));

            let loading = thumb_runner.inflight(i);

            if let Some(img) = textures.get(&i) {
                let max_dimension = img.max_dimension();

                let scale = zoom / (max_dimension as f64);

                let (x_offset, y_offset) = (
                    (max_dimension - img.w) as f64 / 2.0,
                    (max_dimension - img.h) as f64 / 2.0,
                );
                let cx = x + (x_offset * scale);
                let cy = y + (y_offset * scale);

                let trans = c.transform.trans(cx, cy).scale(scale, scale);
                image.draw(&img.data, &draw_state, trans, g);

                if img.is_original {
                    let trans = c.transform.trans(x, y);
                    rectangle(og_color, [0.0, 0.0, 1.0, zoom], trans, g);
                    rectangle(og_color, [0.0, 0.0, zoom, 1.0], trans, g);
                    rectangle(og_color, [zoom - 1.0, 0.0, 1.0, zoom], trans, g);
                    rectangle(og_color, [0.0, zoom - 1.0, zoom, 1.0], trans, g);
                }
            } else if !loading {
                let trans = c.transform.trans(x, y);
                rectangle(missing_color, [zoom / 2.0, zoom / 2.0, 1.0, 1.0], trans, g);
            }

            if loading {
                let trans = c.transform.trans(x, y);
                rectangle(op_color, [0.0, 0.0, 1.0, zoom], trans, g);
                rectangle(op_color, [0.0, 0.0, zoom, 1.0], trans, g);
                rectangle(op_color, [zoom - 1.0, 0.0, 1.0, zoom], trans, g);
                rectangle(op_color, [0.0, zoom - 1.0, zoom, 1.0], trans, g);
            }
        }
    }

    fn run(&mut self) {
        let mut between_updates = None;

        loop {
            let _s = ScopedDuration::new("run_loop");

            self.rebuild_window();

            if let Some(e) = self.window.next() {
                e.update(|args| {
                    between_updates = None;
                    self.update(args);
                    between_updates = Some(ScopedDuration::new("between_updates"));
                });

                e.resize(|w, h| self.resize(w, h));

                e.mouse_scroll(|h, v| {
                    self.mouse_scroll(h, v);
                });

                e.mouse_cursor(|x, y| {
                    self.mouse_cursor(x, y);
                });

                e.mouse_relative(|dx, dy| {
                    self.mouse_relative(dx, dy);
                });

                e.button(|b| self.button(b));

                // borrowck
                let v = &self.view;
                let t = &self.textures;
                let thumb_runner = &self.thumb_runner;
                self.window.draw_2d(&e, |c, g| {
                    Self::draw_2d(thumb_runner, &e, c, g, v, t);
                });
            } else {
                between_updates.take();
                break;
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct File {
    path: String,
    last_modified_secs: u64,
    byte_size: u64,
    dimensions: Option<(u32, u32)>,
    cache_sizes: BTreeSet<u32>,
}

impl File {
    fn max_dimension(&self) -> Option<u32> {
        if let Some((w, h)) = self.dimensions {
            Some(std::cmp::max(w, h))
        } else {
            None
        }
    }

    fn nearest_cache_size(&self, cur_size: Option<u32>, target_size: u32) -> Option<u32> {
        let _s = ScopedDuration::new("nearest_cache_size");

        let zeros = |i: u32| -> i32 { i.leading_zeros() as i32 };

        assert!(target_size.is_power_of_two());
        let target_zeros = zeros(target_size);

        self.cache_sizes
            .iter()
            .fold(cur_size, |a, &b| {
                if let Some(a) = a {
                    if (target_zeros - zeros(a)).abs() < (target_zeros - zeros(b)).abs() {
                        Some(a)
                    } else {
                        Some(b)
                    }
                } else {
                    Some(b)
                }
            })
            .filter(|&s| Some(s) != cur_size)
    }
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

            let byte_size = metadata.len();

            let last_modified_secs: u64 = metadata
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
                byte_size,
                last_modified_secs,
                dimensions: None,
                cache_sizes: BTreeSet::new(),
            };

            ret.push(Arc::new(file));
        }
    }

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

    let thumbnailer_threads: u32 = if let Some(threads) = matches.value_of("threads") {
        threads.parse().expect("not an int")
    } else {
        num_cpus::get() as u32
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

    let mut files = find_images(paths);
    files.sort();

    assert!(!files.is_empty());
    info!("Found {} images", files.len());

    let db = database::Database::open(&db_path).expect("db open");

    for file in &mut files {
        let f = Arc::get_mut(file).expect("file get mut");
        if let Err(e) = db.restore_file_metadata(f) {
            error!("failed to restore metadata {:?}", e);
        }
    }

    {
        let _s = ScopedDuration::new("uptime");
        App::new(&mut files, &db, thumbnailer_threads).run();
    }

    if let Some(stats) = db.get_statistics() {
        info!("rocksdb stats:\n{}", stats);
    }

    stats::dump();
}
