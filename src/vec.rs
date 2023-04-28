// Copyright 2019-2023 Google LLC
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

pub use vecmath::{vec2_add, vec2_mul, vec2_scale, vec2_square_len, vec2_sub, Vector2};

#[inline(always)]
pub fn vec2_div<T>(a: Vector2<T>, b: Vector2<T>) -> Vector2<T>
where
    T: Copy + std::ops::Div<T, Output = T>,
{
    [a[0] / b[0], a[1] / b[1]]
}

#[inline(always)]
pub fn vec2_u32(a: Vector2<f64>) -> Vector2<u32> {
    [a[0] as u32, a[1] as u32]
}

#[inline(always)]
pub fn vec2_f64(a: Vector2<u32>) -> Vector2<f64> {
    [a[0] as f64, a[1] as f64]
}

#[inline(always)]
pub fn vec2_ceil(a: Vector2<f64>) -> Vector2<f64> {
    [a[0].ceil(), a[1].ceil()]
}

#[inline(always)]
pub fn vec2_log(a: Vector2<f64>, base: f64) -> Vector2<f64> {
    [a[0].log(base), a[1].log(base)]
}

#[inline(always)]
pub fn _vec2_min(a: Vector2<u32>, b: Vector2<u32>) -> Vector2<u32> {
    [std::cmp::min(a[0], b[0]), std::cmp::min(a[1], b[1])]
}

#[inline(always)]
pub fn vec2_max(a: Vector2<u32>, b: Vector2<u32>) -> Vector2<u32> {
    [std::cmp::max(a[0], b[0]), std::cmp::max(a[1], b[1])]
}
