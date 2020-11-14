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

use crate::vec::*;

#[derive(Debug, Default)]
pub struct View {
    num_images: f64,

    // Window dimensions.
    win_size: Vector2<f64>,

    // Logical dimensions.
    pub grid_size: Vector2<f64>,

    // View offsets.
    pub trans: Vector2<f64>,

    // Scale from logical to physical coordinates.
    pub zoom: f64,

    min_zoom: f64,

    // Mouse coordinates.
    mouse: Vector2<f64>,

    // Has the user panned or zoomed?
    auto: bool,
}

impl View {
    pub fn new(num_images: usize) -> Self {
        let mut ret = Self {
            num_images: num_images as f64,
            win_size: [800., 600.],
            grid_size: [1.0, 1.0],
            auto: true,
            ..Default::default()
        };
        ret.reset();
        ret
    }

    pub fn target_size(&self) -> u32 {
        ((self.zoom * 1.5) as u32).next_power_of_two()
    }

    pub fn center_mouse(&mut self) {
        self.mouse = vec2_scale(self.win_size, 0.5);
    }

    pub fn reset(&mut self) {
        self.auto = true;

        let [w, h] = self.win_size;

        self.zoom = {
            let px_per_image = (w * h) / self.num_images;
            px_per_image.sqrt()
        };

        self.grid_size = {
            let grid_w = f64::max(1.0, (w / self.zoom).floor());
            let grid_h = (self.num_images / grid_w).ceil();
            [grid_w, grid_h]
        };

        // Numer of rows takes the overflow, rescale to ensure the grid fits the window.
        let grid_px = vec2_scale(self.grid_size, self.zoom);
        if h < grid_px[1] {
            self.zoom *= h / grid_px[1];
        }

        // Add black border.
        self.zoom *= 0.95;

        self.min_zoom = self.zoom * 0.5;

        self.trans = {
            let grid_px = vec2_scale(self.grid_size, self.zoom);
            let border_px = vec2_sub(self.win_size, grid_px);
            vec2_scale(border_px, 0.5)
        };
    }

    pub fn resize_to(&mut self, win_size: Vector2<u32>) {
        self.win_size = vec2_f64(win_size);
        if self.auto {
            self.reset();
        }
    }

    pub fn _mouse(&self) -> Vector2<f64> {
        self.mouse
    }

    pub fn mouse_to(&mut self, mouse: Vector2<f64>) {
        self.mouse = mouse;
    }

    pub fn trans_by(&mut self, trans: Vector2<f64>) {
        self.auto = false;
        self.trans = vec2_add(self.trans, trans);
    }

    pub fn zoom_by(&mut self, ratio: f64) {
        self.auto = false;

        let zoom = self.zoom;
        self.zoom = f64::max(self.min_zoom, zoom * ratio);

        let bias = {
            let grid_pos = vec2_sub(self.mouse, self.trans);
            let grid_px = vec2_scale(self.grid_size, zoom);
            vec2_div(grid_pos, grid_px)
        };

        let trans = {
            let grid_delta = vec2_scale(self.grid_size, self.zoom - zoom);
            vec2_mul(grid_delta, bias)
        };

        self.trans = vec2_sub(self.trans, trans);
    }

    pub fn trans(&self, image_coords: Vector2<u32>) -> Vector2<f64> {
        vec2_add(self.trans, vec2_scale(vec2_f64(image_coords), self.zoom))
    }

    pub fn mouse_dist(&self, image_coords: Vector2<u32>) -> Vector2<f64> {
        let mid = self.zoom / 2.0;
        let mid_coords = vec2_add(self.trans(image_coords), [mid, mid]);
        vec2_sub(mid_coords, self.mouse)
    }

    pub fn is_visible(&self, min: Vector2<f64>) -> bool {
        let max = vec2_add(min, [self.zoom, self.zoom]);
        let [w, h] = self.win_size;
        (max[0] > 0.0 && min[0] < w) && (max[1] > 0.0 && min[1] < h)
    }

    pub fn visible_ratio(&self, [x_min, y_min]: Vector2<f64>) -> f64 {
        let [x_max, y_max] = vec2_add([x_min, y_min], [self.zoom, self.zoom]);
        let [w, h] = self.win_size;
        f64::max(
            f64::min(((x_max / w) - 0.5).abs(), ((x_min / w) - 0.5).abs()),
            f64::min(((y_max / h) - 0.5).abs(), ((y_min / h) - 0.5).abs()),
        ) + 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::View;

    #[test]
    fn is_visible() {
        let view = View {
            win_size: [200.0, 100.0],
            grid_size: [20.0, 10.0],
            zoom: 10.0,
            ..Default::default()
        };

        assert!(view.is_visible([0.0, 0.0]));
        assert!(view.is_visible([190.0, 0.0]));
        assert!(view.is_visible([0.0, 90.0]));

        assert!(!view.is_visible([-20.0, 0.0]));
        assert!(!view.is_visible([210.0, 0.0]));

        assert!(!view.is_visible([0.0, -20.0]));
        assert!(!view.is_visible([0.0, 110.0]));
    }

    #[test]
    fn visible_ratio() {
        let view = View {
            win_size: [200.0, 100.0],
            grid_size: [20.0, 10.0],
            zoom: 10.0,
            ..Default::default()
        };

        assert_eq!(view.visible_ratio([0.0, 0.0]), 0.95);
        assert_eq!(view.visible_ratio([190.0, 0.0]), 0.95);
        assert_eq!(view.visible_ratio([0.0, 90.0]), 0.95);

        assert_eq!(view.visible_ratio([-20.0, 0.0]), 1.05);
        assert_eq!(view.visible_ratio([210.0, 0.0]), 1.05);

        assert_eq!(view.visible_ratio([0.0, -20.0]), 1.1);
        assert_eq!(view.visible_ratio([0.0, 110.0]), 1.1);
    }
}
