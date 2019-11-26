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

use crate::stats;
use crate::{File, Metadata, TileRef, E, R};
use bincode::{deserialize, serialize};
use std::ops::Deref;

static MAX_ID: &[u8] = b"_MAX_ID";
static METADATA_PREFIX: char = 'M';
static TILE_PREFIX: char = 'T';

// Mixed into all keys, bump when making breaking database format changes.
static DB_VERSION: u32 = 2;

#[derive(Debug)]
struct Key(String);

impl Key {
    fn hash_file(file: &File) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let k = (&file.path, file.modified, file.file_size, DB_VERSION);
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn for_file(file: &File) -> Key {
        Self(format!(
            "{}{}:{}",
            METADATA_PREFIX,
            file.path,
            Self::hash_file(file)
        ))
    }

    fn for_thumb(tile_ref: TileRef) -> [u8; 9] {
        let mut k: [u8; 9] = [TILE_PREFIX as u8; 9];
        (&mut k[1..9]).copy_from_slice(&tile_ref.0.to_be_bytes());
        k
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
            modified: 1234,
            file_size: 456,
            ..Default::default()
        })
        .0,
        "M/here:5289273993602405726"
    );
}

// Wrap database types.
pub struct Data(sled::IVec);

impl Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

pub struct Database {
    db: sled::Db,
}

impl Database {
    pub fn open(path: &str) -> R<Self> {
        info!("database path: {}", path);

        let db = sled::Db::open(path).map_err(E::DatabaseError)?;

        Ok(Self { db })
    }

    pub fn get_metadata(&self, file: &File) -> R<Option<Metadata>> {
        let _s = stats::ScopedDuration::new("Database::get_metadata");

        let k = Key::for_file(file);

        if let Some(v) = self.db.get(k.as_ref()).map_err(E::DatabaseError)? {
            stats::record(
                "metadata_size_bytes",
                std::time::Duration::from_micros(v.len() as u64),
            );

            let metadata: Metadata = deserialize(&*v).map_err(E::DecodeError)?;

            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    pub fn set_metadata(&self, file: &File, metadata: &Metadata) -> R<()> {
        let _s = stats::ScopedDuration::new("Database::set_metadata");

        let k = Key::for_file(file);

        let encoded: Vec<u8> = serialize(metadata).map_err(E::EncodeError)?;

        stats::record(
            "metadata_size_bytes",
            std::time::Duration::from_micros(encoded.len() as u64),
        );

        self.db
            .insert(k.as_ref(), encoded)
            .map_err(E::DatabaseError)?;

        Ok(())
    }

    pub fn set(&self, tile_ref: TileRef, data: &[u8]) -> R<()> {
        let _s = stats::ScopedDuration::new("Database::set");

        let k = Key::for_thumb(tile_ref);
        self.db.insert(&k, data).map_err(E::DatabaseError)?;

        Ok(())
    }

    pub fn get(&self, tile_ref: TileRef) -> R<Option<Data>> {
        let _s = stats::ScopedDuration::new("Database::get");

        let k = Key::for_thumb(tile_ref);
        if let Some(v) = self.db.get(k.as_ref()).map_err(E::DatabaseError)? {
            Ok(Some(Data(v)))
        } else {
            Ok(None)
        }
    }

    // TODO: recycle old keys
    pub fn reserve(&self, count: usize) -> u64 {
        let max_id = self
            .db
            .get(MAX_ID)
            .unwrap()
            .map(|v| std::str::from_utf8(&v).unwrap().parse::<u64>().unwrap())
            .unwrap_or(0);

        let next_max_id = max_id + count as u64;
        assert!(next_max_id < (1u64 << 40));

        self.db
            .insert(MAX_ID, format!("{}", next_max_id).as_bytes())
            .unwrap();

        std::dbg!(max_id)
    }
}
