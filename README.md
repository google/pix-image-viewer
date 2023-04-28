# Pix Image Viewer

Explore thousands of images in a zoomable and pannable grid.

Heavily inspired by [Galapix](https://github.com/Galapix/galapix) but without
the segfaults.

Screenshots: https://imgur.com/a/ENyh2NF

*Disclaimer:* This is not an official Google product.

# Installing

You'll need the Rust package manager `cargo` which can be installed by
https://rustup.rs/ or your distributions package management system.

The crate is published (occasionally) to
https://crates.io/crates/pix-image-viewer and can be installed with:

    cargo install pix-image-viewer

Or from github head:

    cargo install --git=https://github.com/google/pix-image-viewer.git

Or from within the source directory:

    cargo install --path=.

# Controls

## Mouse

| Button | Action |
| ------ | ------ |
| Left/Right | Zoom in/out. |
| Middle | Press and move to pan. |

## Keyboard

| Key | Action |
| ------ | ------ |
| Up/Down/Left/Right | Move the viewport. |
| PageUp/PageDown | Zoom in/out. |
| T | Toggle panning mode (capture the mouse & cursor moves the viewport). |
| F | Toggle fullscreen mode. (2023-04-29: Temporarily disabled due to piston window changes) |
| Shift | Hold to zoom and pan in larger increments. |

# Limitations

*   SledDB only allows a single process to manage the database at a time. Due to
    this only a single instance can run at a time *per database path*. A simple
    workaround could be to use ephemeral `--db_path=...` locations.

# Tech

*   Rust stable.
*   [Sled](https://github.com/spacejam/sled) for the image thumbnail cache.
*   [Piston](https://www.piston.rs/) game framework + OpenGL.
*   Rust [image](https://github.com/image-rs/image) decoding/resizing.

# Future direction

*   Vulkan or gfx-rs? Allows more work off the render & event handling thread.
*   Efficient handling of large images? Tiling? [DONE]
*   Efficient handling of millions of small images?
*   Sort images by directory/size/time?
*   Cluster images by directory/size/time?
*   Image curation commands (delete, select, etc)?
*   Seamless image loading/fetching/thumbnailing. [DONE]
*   Command-line thumbnailing mode?
*   Push more magic numbers / consts into flags.
*   Selecting image(s).
*   Running commands on selected image(s).

And there are many TODOs in the code itself.

# Developing

Please use the provided pre-commit hook to keep source code rustfmt clean and
the tests passing on rust stable.

# Naming conflict

I'm now aware of a naming conflict with https://github.com/linuxmint/pix so will
likely be renaming the project soon. Ideas welcome!

# Source Code Headers

Every file containing source code must include copyright and license
information.

Apache header:

    Copyright 2019 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
