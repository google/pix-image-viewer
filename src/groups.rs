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
use crate::group::Group;
use crate::image::Image;
use crate::stats::ScopedDuration;
use crate::thumbnailer::Thumbnailer;
use crate::vec::*;
use crate::view::View;
use crate::{Metadata, Stopwatch, R};
use piston_window::{DrawState, G2d, G2dTextureContext};
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Groups {
    grid_size: Vector2<u32>,
    group_size: Vector2<u32>,
    groups: Vec<(Vector2<u32>, Group)>,
}

impl Groups {
    fn group_size_from_grid_size(grid_size: Vector2<u32>) -> Vector2<u32> {
        vec2_max(vec2_u32(vec2_log(vec2_f64(grid_size), 2.0)), [1, 1])
    }

    pub fn from(images: Vec<Image>, grid_size: Vector2<u32>) -> Self {
        let mut ret = Groups {
            grid_size,
            group_size: Self::group_size_from_grid_size(grid_size),
            ..Default::default()
        };

        let mut group_map: BTreeMap<Vector2<u32>, Group> = BTreeMap::new();
        for image in images.into_iter() {
            ret.insert(&mut group_map, image);
        }
        ret.groups.extend(group_map.into_iter());

        ret
    }

    pub fn grid_size(&self) -> Vector2<u32> {
        self.grid_size
    }

    fn image_coords(&self, i: usize) -> Vector2<u32> {
        let w = self.grid_size[0] as usize;
        [(i % w) as u32, (i / w) as u32]
    }

    fn group_coords(&self, image_coords: Vector2<u32>) -> Vector2<u32> {
        vec2_div(image_coords, self.group_size)
    }

    // Only used while building or re-grouping.
    fn insert(&mut self, group_map: &mut BTreeMap<Vector2<u32>, Group>, image: Image) {
        let image_coords = self.image_coords(image.i);
        let group_coords = self.group_coords(image_coords);
        let group = group_map.entry(group_coords).or_insert_with(|| {
            let min = vec2_mul(group_coords, self.group_size);
            let max = vec2_add(min, self.group_size);
            Group::new([min, max])
        });
        group.insert(image_coords, image);
    }

    pub fn update_metadata(&mut self, i: usize, metadata_res: R<Metadata>) {
        let _s = ScopedDuration::new("Groups::update_metadata");

        let image_coords = self.image_coords(i);
        let group_coords = self.group_coords(image_coords);

        // This looks horrible and O(n), but it's likely O(1) for thumbnails close to the mouse
        // cursor.
        for (coords, group) in &mut self.groups {
            if coords == &group_coords {
                group.update_metadata(image_coords, metadata_res);
                return;
            }
        }
    }

    pub fn regroup(&mut self, grid_size: Vector2<u32>) {
        let _s = ScopedDuration::new("Groups::regroup");

        self.grid_size = grid_size;
        self.group_size = Self::group_size_from_grid_size(grid_size);

        let mut groups: Vec<(Vector2<u32>, Group)> = Vec::with_capacity(self.groups.len());
        std::mem::swap(&mut groups, &mut self.groups);

        let mut group_map: BTreeMap<Vector2<u32>, Group> = BTreeMap::new();
        for (_, group) in groups {
            for (_, image) in group.images.into_iter() {
                self.insert(&mut group_map, image);
            }
        }

        self.groups.extend(group_map.into_iter());
    }

    pub fn recheck(&mut self, view: &View) {
        let _s = ScopedDuration::new("Groups::recheck");

        for (_, group) in &mut self.groups {
            group.recheck(view);
        }

        self.groups.sort_by_key(|(_, g)| g.mouse_dist(view));
    }

    pub fn reset(&mut self) {
        for (_, group) in &mut self.groups {
            group.reset();
        }
    }

    pub fn load_cache(
        &mut self,
        view: &View,
        db: &Database,
        texture_context: &mut G2dTextureContext,
        stopwatch: &Stopwatch,
    ) {
        let _s = ScopedDuration::new("Groups::load_cache");

        for p in 0..2 {
            for (_, group) in &mut self.groups {
                if !group.load_cache(p, view, db, texture_context, stopwatch) {
                    return;
                }
            }
        }
    }

    pub fn make_thumbs(&mut self, thumbnailer: &mut Thumbnailer) {
        for p in 0..2 {
            for (_, group) in &mut self.groups {
                if !group.make_thumbs(p, thumbnailer) {
                    return;
                }
            }
        }
    }

    pub fn draw(&self, trans: [[f64; 3]; 2], view: &View, draw_state: &DrawState, g: &mut G2d) {
        let _s = ScopedDuration::new("Groups::draw");

        for (_, group) in &self.groups {
            group.draw(trans, view, draw_state, g);
        }
    }
}
