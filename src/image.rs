use crate::{Draw, Metadata, MetadataState, File, TileRef};
use std::collections::BTreeMap;
use std::sync::Arc;
use piston_window::{G2dTexture,DrawState, G2d};

#[derive(Default)]
pub struct Image {
    pub file: Arc<File>,
    pub metadata: MetadataState,
    pub size: Option<usize>,
}

impl Image {
    pub fn from(file: Arc<File>, metadata: Option<Metadata>) -> Self {
        Image {
            file,
            metadata: match metadata {
                Some(metadata) => MetadataState::Some(metadata),
                None => MetadataState::Missing,
            },
            ..Default::default()
        }
    }
    
    pub fn loadable(&self) -> bool {
        match self.metadata {
            MetadataState::Errored => false,
            _ => true,
        }
    }

    pub fn is_missing(&self) -> bool {
        match self.metadata {
            MetadataState::Missing => true,
            _ => false,
        }
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
