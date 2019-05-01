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

extern crate bincode;
use crate::stats::ScopedDuration;
use crate::File;
use bincode::{deserialize, serialize};
use std::ops::Deref;

static FILE_PREFIX: &str = "F";

static THUMB_PREFIX: &str = "T";

// Mixed into all keys, bump when making breaking database format changes.
static DB_VERSION: u32 = 1;

#[derive(Debug, Fail)]
pub enum E {
    #[fail(display = "rocksdb error: {:?}", 0)]
    RocksError(rocksdb::Error),

    #[fail(display = "decode error {:?} for key {:?}", 0, 1)]
    DecodeError(bincode::Error, String),

    #[fail(display = "encode error {:?} for file {:?}", 0, 1)]
    EncodeError(bincode::Error, File),

    #[fail(display = "missing data for key {:?}", 0)]
    MissingData(String),
}

type R<T> = std::result::Result<T, E>;

#[derive(Debug)]
struct Key(String);

impl Key {
    fn hash_file(file: &File) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let k = (
            &file.path,
            file.last_modified_secs,
            file.byte_size,
            DB_VERSION,
        );
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn for_file(file: &File) -> Key {
        Self(format!(
            "{}:{}:{}",
            FILE_PREFIX,
            file.path,
            Self::hash_file(file)
        ))
    }

    fn for_thumb(file: &File, size: u32) -> Key {
        Self(format!(
            "{}:{:04}:{}:{}",
            THUMB_PREFIX,
            size,
            file.path,
            Self::hash_file(file),
        ))
    }
}

impl Deref for Key {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.0.as_bytes()
    }
}

#[test]
fn key_for_file() {
    assert_eq!(
        Key::for_file(&File {
            path: String::from("/here"),
            last_modified_secs: 1234,
            byte_size: 456,
            ..Default::default()
        })
        .0,
        "F:/here:16246155260862624421"
    );
}

#[test]
fn key_for_thumb() {
    assert_eq!(
        Key::for_thumb(
            &File {
                path: String::from("/here"),
                last_modified_secs: 1234,
                byte_size: 456,
                ..Default::default()
            },
            128
        )
        .0,
        "T:0128:/here:16246155260862624421"
    );
}

// Wrap rocksdb types.
pub struct Data(rocksdb::DBVector);

impl Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

pub struct Database {
    db: rocksdb::DB,
    opts: rocksdb::Options,
}

impl Database {
    pub fn open(path: &str) -> Result<Self, rocksdb::Error> {
        info!("Database path: {}", path);

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.increase_parallelism(num_cpus::get() as i32);
        opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
        opts.enable_statistics();
        opts.set_stats_dump_period_sec(60);

        let mut block_options = rocksdb::BlockBasedOptions::default();
        block_options.set_block_size(1024 * 1024);
        block_options.set_lru_cache(1024 * 1024 * 1024);
        // TODO: Still not sure if this works.
        block_options.set_bloom_filter(10, false);
        opts.set_block_based_table_factory(&block_options);

        let db = rocksdb::DB::open(&opts, path)?;

        Ok(Self { db, opts })
    }

    pub fn get_statistics(&self) -> Option<String> {
        self.opts.get_statistics()
    }

    pub fn restore_file_metadata(&self, file: &mut File) -> R<()> {
        let k = Key::for_file(file);
        if let Some(v) = self.db.get(k.as_ref()).map_err(E::RocksError)? {
            let f: File = deserialize(&*v).map_err(move |e| E::DecodeError(e, k.0))?;
            file.dimensions = f.dimensions;
            file.cache_sizes = f.cache_sizes;
        }
        Ok(())
    }

    pub fn set(&self, file: &mut File, size: u32, data: &[u8]) -> R<()> {
        let _s = ScopedDuration::new("database_set");

        assert!(size.is_power_of_two());
        file.cache_sizes.insert(size);

        {
            let k = Key::for_thumb(file, size);
            self.db.put(k.as_ref(), data).map_err(E::RocksError)?;
        }

        {
            let k = Key::for_file(file);
            let v: Vec<u8> = serialize(file).map_err(move |e| E::EncodeError(e, file.clone()))?;
            self.db.put(k.as_ref(), &v).map_err(E::RocksError)?;
        }

        Ok(())
    }

    pub fn get(&self, file: &File, size: u32) -> R<Data> {
        let _s = ScopedDuration::new("database_get");

        assert!(size.is_power_of_two());

        assert!(file.cache_sizes.contains(&size));

        let k = Key::for_thumb(file, size);

        Ok(Data(
            self.db
                .get(k.as_ref())
                .map_err(E::RocksError)?
                .ok_or_else(move || E::MissingData(k.0))?,
        ))
    }
}
