pub use vecmath::{vec2_add, vec2_scale, vec2_square_len, vec2_sub, Vector2, vec2_mul};

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
