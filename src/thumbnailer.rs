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
use crate::image;
use crate::File;
use crate::Metadata;
use crate::TileMap;
use crate::TileRef;
use crate::R;
use ::image::GenericImage;
use ::image::GenericImageView;
use futures::future::Fuse;
use futures::future::FutureExt;
use futures::future::RemoteHandle;
use futures::select;
use futures::task::SpawnExt;
use std::collections::BTreeMap;
use std::sync::Arc;

type Handle<T> = Fuse<RemoteHandle<T>>;

pub type MakeThumbRet = R<Metadata>;

pub struct Thumbnailer {
    db: Arc<Database>,
    threads: usize,
    uid_base: u64,
    executor: futures::executor::ThreadPool,
    handles: BTreeMap<usize, Handle<MakeThumbRet>>,
}

impl Thumbnailer {
    pub fn new(db: Arc<Database>, uid_base: u64, threads: usize) -> Self {
        Self {
            db,
            threads,
            uid_base,
            executor: futures::executor::ThreadPool::builder()
                .pool_size(threads)
                .name_prefix("thumbnailer")
                .create()
                .unwrap(),
            handles: BTreeMap::new(),
        }
    }

    pub fn is_full(&self) -> bool {
        self.handles.len() > self.threads
    }

    pub fn contains(&self, i: usize) -> bool {
        self.handles.contains_key(&i)
    }

    async fn update_db(
        res: R<(Arc<File>, Metadata, TileMap<Vec<u8>>)>,
        db: Arc<Database>,
    ) -> R<Metadata> {
        match res {
            Ok((file, metadata, tiles)) => {
                // Do before metadata write to prevent invalid metadata references.
                for (id, tile) in tiles {
                    db.set(id, &tile).expect("db set");
                }

                db.set_metadata(&*file, &metadata).expect("set metadata");

                Ok(metadata)
            }
            Err(e) => Err(e),
        }
    }

    pub fn recv(&mut self) -> Vec<(usize, R<Metadata>)> {
        let mut ret = Vec::new();

        // TODO: make more efficient
        for (&i, mut handle) in &mut self.handles {
            select! {
                thumb_res = handle => {
                    ret.push((i, thumb_res));
                }
                default => {}
            }
        }

        for (i, _) in &ret {
            self.handles.remove(i);
        }

        ret
    }

    pub fn make_thumbs(&mut self, image: &image::Image) -> bool {
        assert!(!self.is_full());

        if !image.is_missing() || self.contains(image.i) {
            return false;
        }

        let uid = self.uid_base + image.i as u64;

        let db = Arc::clone(&self.db);

        let fut =
            Self::make_thumb(Arc::clone(&image.file), uid).then(move |r| Self::update_db(r, db));

        let handle = self.executor.spawn_with_handle(fut).unwrap().fuse();

        self.handles.insert(image.i, handle);

        true
    }

    async fn make_thumb(file: Arc<File>, uid: u64) -> R<(Arc<File>, Metadata, TileMap<Vec<u8>>)> {
        let _s = crate::stats::ScopedDuration::new("Thumbnailer::make_thumb");

        let mut image = ::image::open(&file.path).map_err(crate::E::ImageError)?;

        let (w, h) = image.dimensions();

        let orig_bucket = std::cmp::max(w, h).next_power_of_two();

        let min_bucket = std::cmp::min(8, orig_bucket);

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
}
