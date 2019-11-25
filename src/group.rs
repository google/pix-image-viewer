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
use crate::stats::ScopedDuration;
use crate::vec::*;
use crate::view::View;
use crate::TileRef;
use crate::R;
use crate::{Metadata, MetadataState};
use piston_window::{G2dTexture, G2dTextureContext, Texture, TextureSettings};
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Default)]
pub struct Group {
    pub min_extent: Vector2<u32>,
    pub max_extent: Vector2<u32>,
    pub tiles: BTreeMap<TileRef, G2dTexture>,
    pub images: BTreeMap<Vector2<u32>, crate::image::Image>,
    pub cache_todo: VecDeque<Vector2<u32>>,
    pub thumb_todo: VecDeque<Vector2<u32>>,
}

impl Group {
    pub fn insert(&mut self, coords: Vector2<u32>, image: crate::image::Image) {
        self.min_extent = vec2_min(self.min_extent, coords);
        self.max_extent = vec2_max(self.max_extent, vec2_add(coords, [1, 1]));
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

    pub fn recheck(&mut self) {
        self.thumb_todo.clear();
        self.cache_todo.clear();
        self.cache_todo.extend(self.images.keys());
        // TODO: reorder by mouse distance.
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

    pub fn make_thumbs(&mut self, thumbnailer: &mut crate::Thumbnailer) {
        let _s = ScopedDuration::new("make_thumbs");
        loop {
            if thumbnailer.is_full() {
                return;
            }

            if let Some(coords) = self.thumb_todo.pop_front() {
                let image = self.images.get(&coords).unwrap();
                thumbnailer.make_thumbs(image);
            } else {
                break;
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
}
