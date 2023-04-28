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

use crate::database::Database;
use crate::image::Image;
use crate::vec::*;
use crate::view::View;
use crate::Stopwatch;
use crate::TileRef;
use crate::R;
use crate::{Metadata, MetadataState};
use log::*;
use piston_window::{
    color, rectangle, DrawState, G2d, G2dTexture, G2dTextureContext, Texture, TextureSettings,
    Transformed,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug)]
pub struct Group {
    pub extents: [Vector2<u32>; 2],
    pub tiles: BTreeMap<TileRef, G2dTexture>,
    pub images: BTreeMap<Vector2<u32>, Image>,
    pub cache_todo: [VecDeque<Vector2<u32>>; 2],
    pub thumb_todo: [VecDeque<Vector2<u32>>; 2],
}

impl Group {
    pub fn new(extents: [Vector2<u32>; 2]) -> Self {
        Self {
            extents,
            tiles: BTreeMap::new(),
            images: BTreeMap::new(),
            cache_todo: [VecDeque::new(), VecDeque::new()],
            thumb_todo: [VecDeque::new(), VecDeque::new()],
        }
    }

    pub fn insert(&mut self, coords: Vector2<u32>, image: Image) {
        self.images.insert(coords, image);
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        for image in self.images.values_mut() {
            image.reset();
        }
        self.tiles.clear();

        for queue in &mut self.cache_todo {
            queue.clear();
        }

        for queue in &mut self.thumb_todo {
            queue.clear();
        }
    }

    pub fn recheck(&mut self, view: &View) {
        for queue in &mut self.thumb_todo {
            queue.clear();
        }

        for queue in &mut self.cache_todo {
            queue.clear();
        }

        let mut mouse_dist: Vec<(&Vector2<u32>, &Image)> = Vec::with_capacity(self.images.len());
        mouse_dist.extend(self.images.iter());
        mouse_dist.sort_by_key(|(&coords, _)| vec2_square_len(view.mouse_dist(coords)) as isize);

        for (&coords, image) in &mouse_dist {
            let p = !view.is_visible(view.trans(coords)) as usize;

            match image.metadata {
                MetadataState::Some(_) => {
                    self.cache_todo[p].push_back(coords);
                }
                MetadataState::Missing => {
                    self.thumb_todo[p].push_back(coords);
                }
                MetadataState::Errored => continue,
            }
        }
    }

    pub fn load_cache(
        &mut self,
        p: usize,
        view: &View,
        db: &Database,
        texture_context: &mut G2dTextureContext,
        stopwatch: &Stopwatch,
    ) -> bool {
        let target_size = view.target_size();

        let texture_settings = TextureSettings::new();

        while let Some(coords) = self.cache_todo[p].pop_front() {
            let image = self.images.get_mut(&coords).unwrap();

            let metadata = image.get_metadata().expect("Image::get_metadata");

            let view_coords = view.trans(coords);

            let shift = if p == 0 {
                0
            } else {
                let ratio = view.visible_ratio(view_coords);
                f64::max(0.0, ratio - 1.0).floor() as usize
            };

            let new_size = metadata.nearest(target_size >> shift);

            let current_size = image.size.unwrap_or(0);

            // Progressive resizing.
            let new_size = match new_size.cmp(&current_size) {
                Ordering::Less => current_size - 1,
                Ordering::Equal => {
                    // Already loaded target size.
                    continue;
                }
                Ordering::Greater => current_size + 1,
            };

            // Load new tiles.
            for tile_ref in &metadata.thumbs[new_size].tile_refs {
                // Already loaded.
                if self.tiles.contains_key(tile_ref) {
                    continue;
                }

                if stopwatch.done() {
                    self.cache_todo[p].push_front(coords);
                    return false;
                }

                let data = db.get(*tile_ref).expect("db get").expect("missing tile");

                let image = ::image::load_from_memory(&data).expect("load image");

                // TODO: Would be great to move off thread.
                let image =
                    Texture::from_image(texture_context, &image.to_rgba8(), &texture_settings)
                        .expect("texture");

                self.tiles.insert(*tile_ref, image);
            }

            // Unload old tiles.
            for (j, thumb) in metadata.thumbs.iter().enumerate() {
                if j == new_size {
                    continue;
                }
                for tile_ref in &thumb.tile_refs {
                    self.tiles.remove(tile_ref);
                }
            }

            image.size = Some(new_size);
            self.cache_todo[p].push_back(coords);
        }

        true
    }

    pub fn make_thumbs(&mut self, p: usize, thumbnailer: &mut crate::Thumbnailer) -> bool {
        loop {
            if thumbnailer.is_full() {
                return false;
            }

            if let Some(coords) = self.thumb_todo[p].pop_front() {
                let image = self.images.get(&coords).unwrap();
                if !thumbnailer.make_thumbs(image) {
                    return false;
                }
            } else {
                break true;
            }
        }
    }

    pub fn update_metadata(&mut self, coords: Vector2<u32>, metadata_res: R<Metadata>) {
        let image = self.images.get_mut(&coords).unwrap();
        image.metadata = match metadata_res {
            Ok(metadata) => {
                self.cache_todo[0].push_front(coords);
                MetadataState::Some(metadata)
            }
            Err(e) => {
                error!("make_thumb: {}", e);
                MetadataState::Errored
            }
        };
    }

    pub fn draw(&self, trans: [[f64; 3]; 2], view: &View, draw_state: &DrawState, g: &mut G2d) {
        //{
        //    let [min, max] = self.extents;
        //    let op_color = color::hex("FF0000");
        //    let [x, y] = view.trans(min);
        //    let [w, h] = vec2_scale(vec2_f64(vec2_sub(max, min)), view.zoom);
        //    let trans = trans.trans(x, y);
        //    rectangle(op_color, [0.0, 0.0, w, 1.0], trans, g);
        //    rectangle(op_color, [0.0, 0.0, 1.0, h], trans, g);
        //    rectangle(op_color, [w, 0.0, 1.0, h], trans, g);
        //    rectangle(op_color, [0.0, h, w, 1.0], trans, g);
        //}

        let dot_color = color::hex("444444");
        let mid_zoom = view.zoom * 0.5;

        for (&coords, image) in &self.images {
            let coords = view.trans(coords);

            if !view.is_visible(coords) {
                continue;
            }

            let trans = trans.trans(coords[0], coords[1]);

            if image.draw(trans, view, &self.tiles, draw_state, g) {
                continue;
            } else {
                rectangle(dot_color, [mid_zoom, mid_zoom, 1.0, 1.0], trans, g);
            }
        }
    }

    pub fn mouse_dist(&self, view: &View) -> usize {
        let midpoint = vec2_div(vec2_add(self.extents[0], self.extents[1]), [2, 2]);
        let mouse_dist = view.mouse_dist(midpoint);
        vec2_square_len(mouse_dist) as usize
    }
}
