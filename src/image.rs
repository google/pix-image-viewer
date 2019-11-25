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

use crate::{Draw, File, MetadataState, TileRef};
use piston_window::{DrawState, G2d, G2dTexture};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default, Debug)]
pub struct Image {
    pub i: usize,
    pub file: Arc<File>,
    pub metadata: MetadataState,
    pub size: Option<usize>,
}

impl Image {
    pub fn from(i: usize, file: Arc<File>) -> Self {
        Image {
            i,
            file,
            ..Default::default()
        }
    }

    pub fn is_loadable(&self) -> bool {
        self.metadata != MetadataState::Errored
    }

    pub fn is_missing(&self) -> bool {
        self.metadata == MetadataState::Missing
    }

    pub fn reset(&mut self) {
        self.size = None;
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
                _ => unreachable!("image draw unreachable"),
            };
            let thumb = &metadata.thumbs[n];
            thumb.draw(trans, zoom, tiles, draw_state, g);
            true
        } else {
            false
        }
    }
}
