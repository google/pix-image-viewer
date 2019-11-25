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

use crate::database::Database;
use crate::image::Image;
use crate::stats::ScopedDuration;
use crate::vec::*;
use crate::view::View;
use crate::Draw;
use crate::TileRef;
use crate::R;
use crate::{Metadata, MetadataState};
use piston_window::Transformed;
use piston_window::{
    color, rectangle, DrawState, G2d, G2dTexture, G2dTextureContext, Texture, TextureSettings,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug)]
pub struct Group {
    pub extents: [Vector2<u32>; 2],
    pub tiles: BTreeMap<TileRef, G2dTexture>,
    pub images: BTreeMap<Vector2<u32>, Image>,
    pub cache_todo: VecDeque<Vector2<u32>>,
    pub thumb_todo: VecDeque<Vector2<u32>>,
}

impl Group {
    pub fn new(extents: [Vector2<u32>; 2]) -> Self {
        Self {
            extents,
            tiles: BTreeMap::new(),
            images: BTreeMap::new(),
            cache_todo: VecDeque::new(),
            thumb_todo: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, coords: Vector2<u32>, image: Image) {
        self.images.insert(coords, image);
    }

    pub fn reset(&mut self) {
        for image in self.images.values_mut() {
            image.reset();
        }
        self.tiles.clear();
        self.thumb_todo.clear();
        self.cache_todo.clear();
    }

    pub fn recheck(&mut self, view: &View) {
        self.thumb_todo.clear();
        self.cache_todo.clear();

        let mut mouse_dist: Vec<Vector2<u32>> = self
            .images
            .iter()
            .filter_map(|(&coords, image)| {
                if image.is_loadable() {
                    Some(coords)
                } else {
                    None
                }
            })
            .collect();

        mouse_dist.sort_by_key(|&i| vec2_square_len(view.mouse_dist(i)) as isize);

        self.cache_todo.extend(mouse_dist);
    }

    pub fn load_cache(
        &mut self,
        view: &View,
        db: &Database,
        target_size: u32,
        texture_settings: &TextureSettings,
        texture_context: &mut G2dTextureContext,
    ) {
        for coords in self.cache_todo.pop_front() {
            let image = self.images.get_mut(&coords).unwrap();

            if image.metadata == MetadataState::Unknown {
                image.metadata = match db.get_metadata(&*image.file) {
                    Ok(Some(metadata)) => MetadataState::Some(metadata),
                    Ok(None) => MetadataState::Missing,
                    Err(e) => {
                        error!("get metadata error: {:?}", e);
                        MetadataState::Errored
                    }
                };
            }

            let metadata = match &image.metadata {
                MetadataState::Unknown => unreachable!(),
                MetadataState::Missing => {
                    self.thumb_todo.push_back(coords);
                    continue;
                }
                MetadataState::Some(metadata) => metadata,
                MetadataState::Errored => continue,
            };

            let is_visible = view.is_visible(view.coords(image.i));

            let shift = if is_visible {
                0
            } else {
                let ratio = view.visible_ratio(view.coords(image.i));
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

                // load the tile from the cache
                let _s3 = ScopedDuration::new("load_tile");

                let data = db.get(*tile_ref).expect("db get").expect("missing tile");

                let image = ::image::load_from_memory(&data).expect("load image");

                // TODO: Would be great to move off thread.
                let image =
                    Texture::from_image(texture_context, &image.to_rgba(), texture_settings)
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
            self.cache_todo.push_back(coords);
        }
    }

    pub fn make_thumbs(&mut self, thumbnailer: &mut crate::Thumbnailer) -> bool {
        let _s = ScopedDuration::new("make_thumbs");
        loop {
            if thumbnailer.is_full() {
                return false;
            }

            if let Some(coords) = self.thumb_todo.pop_front() {
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
                self.cache_todo.push_front(coords);
                MetadataState::Some(metadata)
            }
            Err(e) => {
                error!("make_thumb: {}", e);
                MetadataState::Errored
            }
        };
    }

    pub fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        view: &View,
        draw_state: &DrawState,
        g: &mut G2d,
    ) {
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

        for (&coords, image) in &self.images {
            let coords = view.trans(coords);

            if !view.is_visible(coords) {
                continue;
            }

            let trans = trans.trans(coords[0], coords[1]);

            if image.draw(trans, zoom, &self.tiles, &draw_state, g) {
                continue;
            }
        }
    }
}
