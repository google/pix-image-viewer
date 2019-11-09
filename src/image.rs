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

use crate::{Draw, File, Metadata, MetadataState, TileRef};
use ::image::GenericImage;
use ::image::GenericImageView;
use piston_window::{DrawState, G2d, G2dTexture};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub struct Image {
    pub file: Arc<File>,
    pub metadata: MetadataState,
    pub size: Option<usize>,
}

impl Image {
    pub fn from(file: Arc<File>) -> Self {
        Image {
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

    pub fn make_thumb(
        &self,
        tile_id_index: u64,
    ) -> impl std::future::Future<Output = crate::R<(Arc<File>, Metadata, crate::TileMap<Vec<u8>>)>>
    {
        let file = Arc::clone(&self.file);
        async move { make_thumb(file, tile_id_index) }
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

// TODO: make flag
static MIN_SIZE: u32 = 8;

pub type ThumbRet = crate::R<Metadata>;

fn make_thumb(
    file: Arc<File>,
    uid: u64,
) -> crate::R<(Arc<File>, Metadata, crate::TileMap<Vec<u8>>)> {
    let _s = crate::stats::ScopedDuration::new("make_thumb");

    let mut image = ::image::open(&file.path).map_err(crate::E::ImageError)?;

    let (w, h) = image.dimensions();

    let orig_bucket = std::cmp::max(w, h).next_power_of_two();

    let min_bucket = std::cmp::min(MIN_SIZE, orig_bucket);

    let mut bucket = orig_bucket;

    let mut thumbs: Vec<crate::Thumb> = Vec::new();

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

        let mut thumb = crate::Thumb {
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

                let tile_id = crate::TileRef::new(crate::Pow2::from(bucket), uid, chunk_id);
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

    Ok((file, metadata, tiles))
}
