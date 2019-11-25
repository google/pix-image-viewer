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
use crate::R;
use futures::future::Fuse;
use futures::future::FutureExt;
use futures::future::RemoteHandle;
use futures::select;
use futures::task::SpawnExt;
use std::collections::BTreeMap;
use std::sync::Arc;

type Handle<T> = Fuse<RemoteHandle<T>>;

pub struct Thumbnailer {
    db: Arc<Database>,
    threads: usize,
    base_id: u64,
    executor: futures::executor::ThreadPool,
    handles: BTreeMap<usize, Handle<image::ThumbRet>>,
}

impl Thumbnailer {
    pub fn new(db: Arc<Database>, base_id: u64, threads: usize) -> Self {
        Self {
            db,
            threads,
            base_id,
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

    pub fn make_thumbs(&mut self, image: &image::Image) {
        assert!(!self.is_full());

        if !image.is_missing() || self.contains(image.i) {
            return;
        }

        let tile_id_index = self.base_id + image.i as u64;

        let db = Arc::clone(&self.db);

        let fut = image
            .make_thumb(tile_id_index)
            .then(move |x| Self::update_db(x, db));

        let handle = self.executor.spawn_with_handle(fut).unwrap().fuse();

        self.handles.insert(image.i, handle);
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
}
