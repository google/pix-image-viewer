# Pix Image Viewer

Explore tens of thousands of images in a grid. Use mouse or keyboard to zoom or
pan around the image grid.

Heavily inspired by [Galapix](https://github.com/Galapix/galapix).

*Disclaimer:* This is not an official Google product.

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
| F | Toggle fullscreen mode. |
| Shift | Hold to zoom and pan in larger increments. |

# Limitations

*   RocksDB only allows a single process to write to a single database at a
    time. Due to this, only a single instance will be able to run at a time with
    the default flags. Use the `--db_path=...` flag to point new pix instances
    at unique database locations.

# Tech

*   Rust nightly & friends.
*   RocksDB for the image thumbnail cache.
*   [Piston](https://www.piston.rs/) game framework + OpenGL.
*   Rust image decoding/resizing.

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

And there are many TODOs in the code itself.

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
