//! Portable kernels written for LLVM's auto-vectoriser.
//!
//! Reductions accumulate into a fixed `[f32; LANES]` array: each lane is an
//! independent dependency chain, so LLVM maps the array onto SIMD registers
//! (4 NEON/SSE vectors, 2 AVX vectors) without needing permission to
//! reassociate float additions. The final tree reduction is fixed, so results
//! are deterministic for a given build. Multiplies and adds stay separate (no
//! `mul_add`): on x86 baseline builds `f32::mul_add` is a libm call.

use super::{diff_sq_padded, head, valid_lags};

/// Accumulator width (floats) of the reductions.
const LANES: usize = 16;

/// Pairwise tree sum of an accumulator array (fixed order).
#[inline(always)]
fn reduce(acc: [f32; LANES]) -> f32 {
    let mut v = [0.0f32; LANES / 2];
    for (o, (a, b)) in v.iter_mut().zip(acc[..LANES / 2].iter().zip(&acc[LANES / 2..])) {
        *o = a + b;
    }
    let s4 = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (s4[0] + s4[2]) + (s4[1] + s4[3])
}

/// `Σ a[i]·b[i]` over `min(a.len(), b.len())`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for ((s, &xi), &yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }
    let mut tail = 0.0f32;
    for (&xi, &yi) in ar.iter().zip(br) {
        tail += xi * yi;
    }
    reduce(acc) + tail
}

/// `Σ a[i]²`.
pub fn sum_sq(a: &[f32]) -> f32 {
    let (ac, ar) = a.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for x in ac {
        for (s, &xi) in acc.iter_mut().zip(x) {
            *s += xi * xi;
        }
    }
    let mut tail = 0.0f32;
    for &xi in ar {
        tail += xi * xi;
    }
    reduce(acc) + tail
}

/// `y[i] += a * x[i]` over `min(y.len(), x.len())`.
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// `y[i] += x[i] * w[i]` over the shortest length.
pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
    for (yi, (&xi, &wi)) in y.iter_mut().zip(x.iter().zip(w)) {
        *yi += xi * wi;
    }
}

/// `Σ (a[j] − b[j])²` over `min(a.len(), b.len())`.
#[inline]
pub(crate) fn diff_one(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for ((s, &xi), &yi) in acc.iter_mut().zip(x).zip(y) {
            let d = xi - yi;
            *s += d * d;
        }
    }
    let mut tail = 0.0f32;
    for (&xi, &yi) in ar.iter().zip(br) {
        let d = xi - yi;
        tail += d * d;
    }
    reduce(acc) + tail
}

/// Direct type-I difference function; see [`super::diff_sq`].
pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    let valid = valid_lags(x.len(), w, tau0, out.len());
    let (body, rest) = out.split_at_mut(valid);
    let a = head(x, w);
    for (k, o) in body.iter_mut().enumerate() {
        let tau = tau0 + k;
        *o = x.get(tau..tau + w).map_or(0.0, |b| diff_one(a, b));
    }
    diff_sq_padded(x, w, tau0.saturating_add(valid), rest);
}
