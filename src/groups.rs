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
use crate::stats::ScopedDuration;
use crate::thumbnailer::Thumbnailer;
use crate::vec::*;
use crate::view::View;
use crate::Metadata;
use crate::Stopwatch;
use crate::R;
use piston_window::{DrawState, G2d, G2dTextureContext, TextureSettings};
use std::collections::BTreeMap;

fn i2c(i: usize, [grid_w, _]: Vector2<u32>) -> Vector2<u32> {
    [(i % grid_w as usize) as u32, (i / grid_w as usize) as u32]
}

#[derive(Debug, Default)]
pub struct Groups {
    pub grid_size: Vector2<u32>,
    group_size: Vector2<u32>,
    groups: BTreeMap<Vector2<u32>, Group>,
    mouse_dist: Vec<Vector2<u32>>,
}

impl Groups {
    fn group_size_from_grid_size(grid_size: Vector2<u32>) -> Vector2<u32> {
        vec2_max(vec2_u32(vec2_log(vec2_f64(grid_size), 2.0)), [1, 1])
    }

    pub fn from(images: Vec<crate::image::Image>, grid_size: Vector2<u32>) -> Self {
        let mut ret = Groups {
            grid_size,
            group_size: Self::group_size_from_grid_size(grid_size),
            ..Default::default()
        };

        for image in images.into_iter() {
            ret.insert(image);
        }

        ret
    }

    fn group_coords(&self, coords: Vector2<u32>) -> Vector2<u32> {
        vec2_div(coords, self.group_size)
    }

    fn image_coords(&self, i: usize) -> Vector2<u32> {
        i2c(i, self.grid_size)
    }

    fn insert(&mut self, image: crate::image::Image) {
        let image_coords = self.image_coords(image.i);
        let group_coords = self.group_coords(image_coords);
        let group_size = self.group_size;
        let group = self.groups.entry(group_coords).or_insert_with(|| {
            let min = vec2_mul(group_coords, group_size);
            let max = vec2_add(min, group_size);
            Group::new([min, max])
        });
        group.insert(image_coords, image);
    }

    pub fn update_metadata(&mut self, i: usize, metadata_res: R<Metadata>) {
        let image_coords = self.image_coords(i);
        let group_coords = self.group_coords(image_coords);
        let group = self.groups.get_mut(&group_coords).unwrap();
        group.update_metadata(image_coords, metadata_res);
    }

    pub fn regroup(&mut self, grid_size: Vector2<u32>) {
        let _s = ScopedDuration::new("regroup");

        let mut groups = BTreeMap::new();
        std::mem::swap(&mut groups, &mut self.groups);

        self.grid_size = grid_size;
        self.group_size = Self::group_size_from_grid_size(grid_size);

        for (_, group) in groups.into_iter() {
            for (_, image) in group.images.into_iter() {
                self.insert(image);
            }
        }
    }

    pub fn recheck(&mut self, view: &View) {
        for group in self.groups.values_mut() {
            group.recheck(view);
        }

        let mut mouse_dist: Vec<(&Vector2<u32>, &Group)> = self.groups.iter().collect();
        mouse_dist.sort_by_key(|(_, g)| g.mouse_dist(view));

        self.mouse_dist.clear();
        self.mouse_dist
            .extend(mouse_dist.into_iter().map(|(k, _)| k));
    }

    pub fn reset(&mut self) {
        for group in self.groups.values_mut() {
            group.reset();
        }
    }

    pub fn load_cache(
        &mut self,
        view: &View,
        db: &Database,
        target_size: u32,
        texture_settings: &TextureSettings,
        texture_context: &mut G2dTextureContext,
        stopwatch: &Stopwatch,
    ) {
        for coords in &self.mouse_dist {
            let group = self.groups.get_mut(coords).unwrap();
            if !group.load_cache(
                view,
                db,
                target_size,
                texture_settings,
                texture_context,
                stopwatch,
            ) {
                return;
            }
        }
    }

    pub fn make_thumbs(&mut self, thumbnailer: &mut Thumbnailer) {
        let _s = ScopedDuration::new("App::make_thumbs");
        for coords in &self.mouse_dist {
            let group = self.groups.get_mut(coords).unwrap();
            if !group.make_thumbs(thumbnailer) {
                return;
            }
        }
    }

    pub fn recv_thumbs(&mut self, thumbnailer: &mut Thumbnailer) {
        let _s = ScopedDuration::new("App::recv_thumbs");
        for (i, metadata_res) in thumbnailer.recv() {
            self.update_metadata(i, metadata_res);
        }
    }

    pub fn draw(
        &self,
        trans: [[f64; 3]; 2],
        zoom: f64,
        view: &View,
        draw_state: &DrawState,
        g: &mut G2d,
    ) {
        for group in self.groups.values() {
            group.draw(trans, zoom, view, draw_state, g);
        }
    }
}
