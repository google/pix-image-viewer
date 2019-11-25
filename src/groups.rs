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

use crate::group::Group;
use crate::stats::ScopedDuration;
use crate::vec::*;
use std::collections::BTreeMap;

fn i2c(i: usize, [grid_w, _]: Vector2<u32>) -> Vector2<u32> {
    [(i % grid_w as usize) as u32, (i / grid_w as usize) as u32]
}

#[derive(Debug, Default)]
pub struct Groups {
    pub grid_size: Vector2<u32>,
    pub group_size: Vector2<u32>,
    pub groups: BTreeMap<[u32; 2], Group>,
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

    fn insert(&mut self, image: crate::image::Image) {
        let coords = i2c(image.i, self.grid_size);
        let group_coords = self.group_coords(coords);
        let group = self.groups.entry(group_coords).or_insert(Group::default());
        group.insert(coords, image);
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

    pub fn reset(&mut self) {
        for group in self.groups.values_mut() {
            group.reset();
        }
    }
}
