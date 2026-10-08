//! x86_64 AVX2 + FMA kernels.
//!
//! Every public function is `unsafe` and compiled with
//! `#[target_feature(enable = "avx2,fma")]`: callers must have verified the
//! features at runtime (the dispatcher in [`super`] does, via
//! `is_x86_feature_detected!`, caching the answer in an atomic). Structure
//! mirrors the NEON backend with 256-bit vectors: four FMA accumulators of 8
//! lanes per reduction, and four lags per pass in `diff_sq`.
//!
//! The only unsafe operations inside are unaligned loads/stores of whole
//! `[f32; N]` arrays produced by `as_chunks`, so offsets are in bounds by
//! construction.

use core::arch::x86_64::*;

use super::{diff_sq_padded, head, valid_lags};

/// Loads 8 floats from `p[off..off + 8]`.
///
/// # Safety
/// `off + 8 <= N`; AVX must be available.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ld<const N: usize>(p: &[f32; N], off: usize) -> __m256 {
    debug_assert!(off + 8 <= N);
    // SAFETY: the caller guarantees `off + 8 <= N`.
    unsafe { _mm256_loadu_ps(p.as_ptr().add(off)) }
}

/// Stores 8 floats to `p[off..off + 8]`.
///
/// # Safety
/// `off + 8 <= N`; AVX must be available.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn st<const N: usize>(p: &mut [f32; N], off: usize, v: __m256) {
    debug_assert!(off + 8 <= N);
    // SAFETY: the caller guarantees `off + 8 <= N`.
    unsafe { _mm256_storeu_ps(p.as_mut_ptr().add(off), v) }
}

/// Horizontal sum of one vector.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s);
    let s2 = _mm_add_ps(s, sh);
    let hi2 = _mm_movehl_ps(sh, s2);
    _mm_cvtss_f32(_mm_add_ss(s2, hi2))
}

/// Tree sum of four accumulators.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn reduce4(acc: [__m256; 4]) -> f32 {
    hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
}

/// `Σ a[i]·b[i]` over `min(a.len(), b.len())`.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<32>();
    let (bc, br) = b.as_chunks::<32>();
    let mut acc = [_mm256_setzero_ps(); 4];
    for (x, y) in ac.iter().zip(bc) {
        // SAFETY: offsets 0, 8, 16, 24 of 32-element arrays.
        unsafe {
            acc[0] = _mm256_fmadd_ps(ld(x, 0), ld(y, 0), acc[0]);
            acc[1] = _mm256_fmadd_ps(ld(x, 8), ld(y, 8), acc[1]);
            acc[2] = _mm256_fmadd_ps(ld(x, 16), ld(y, 16), acc[2]);
            acc[3] = _mm256_fmadd_ps(ld(x, 24), ld(y, 24), acc[3]);
        }
    }
    let (a8, a1) = ar.as_chunks::<8>();
    let (b8, b1) = br.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        // SAFETY: offset 0 of 8-element arrays.
        unsafe {
            acc[0] = _mm256_fmadd_ps(ld(x, 0), ld(y, 0), acc[0]);
        }
    }
    let mut sum = reduce4(acc);
    for (&x, &y) in a1.iter().zip(b1) {
        sum += x * y;
    }
    sum
}

/// `Σ a[i]²`.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sum_sq(a: &[f32]) -> f32 {
    let (ac, ar) = a.as_chunks::<32>();
    let mut acc = [_mm256_setzero_ps(); 4];
    for x in ac {
        // SAFETY: offsets 0, 8, 16, 24 of 32-element arrays.
        unsafe {
            let v0 = ld(x, 0);
            let v1 = ld(x, 8);
            let v2 = ld(x, 16);
            let v3 = ld(x, 24);
            acc[0] = _mm256_fmadd_ps(v0, v0, acc[0]);
            acc[1] = _mm256_fmadd_ps(v1, v1, acc[1]);
            acc[2] = _mm256_fmadd_ps(v2, v2, acc[2]);
            acc[3] = _mm256_fmadd_ps(v3, v3, acc[3]);
        }
    }
    let (a8, a1) = ar.as_chunks::<8>();
    for x in a8 {
        // SAFETY: offset 0 of an 8-element array.
        let v = unsafe { ld(x, 0) };
        acc[0] = _mm256_fmadd_ps(v, v, acc[0]);
    }
    let mut sum = reduce4(acc);
    for &x in a1 {
        sum += x * x;
    }
    sum
}

/// `y[i] += a * x[i]` over `min(y.len(), x.len())`.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    let n = y.len().min(x.len());
    let (y, _) = y.split_at_mut(n);
    let x = head(x, n);
    let (yc, yr) = y.as_chunks_mut::<16>();
    let (xc, xr) = x.as_chunks::<16>();
    let av = _mm256_set1_ps(a);
    for (yv, xv) in yc.iter_mut().zip(xc) {
        // SAFETY: offsets 0 and 8 of 16-element arrays.
        unsafe {
            let r0 = _mm256_fmadd_ps(av, ld(xv, 0), ld(yv, 0));
            let r1 = _mm256_fmadd_ps(av, ld(xv, 8), ld(yv, 8));
            st(yv, 0, r0);
            st(yv, 8, r1);
        }
    }
    for (yi, &xi) in yr.iter_mut().zip(xr) {
        *yi += a * xi;
    }
}

/// `y[i] += x[i] * w[i]` over the shortest length.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
    let n = y.len().min(x.len()).min(w.len());
    let (y, _) = y.split_at_mut(n);
    let (x, w) = (head(x, n), head(w, n));
    let (yc, yr) = y.as_chunks_mut::<16>();
    let (xc, xr) = x.as_chunks::<16>();
    let (wc, wr) = w.as_chunks::<16>();
    for ((yv, xv), wv) in yc.iter_mut().zip(xc).zip(wc) {
        // SAFETY: offsets 0 and 8 of 16-element arrays.
        unsafe {
            let r0 = _mm256_fmadd_ps(ld(xv, 0), ld(wv, 0), ld(yv, 0));
            let r1 = _mm256_fmadd_ps(ld(xv, 8), ld(wv, 8), ld(yv, 8));
            st(yv, 0, r0);
            st(yv, 8, r1);
        }
    }
    for ((yi, &xi), &wi) in yr.iter_mut().zip(xr).zip(wr) {
        *yi += xi * wi;
    }
}

/// `Σ (a[j] − b[j])²` for one lag.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn diff_one(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<32>();
    let (bc, br) = b.as_chunks::<32>();
    let mut acc = [_mm256_setzero_ps(); 4];
    for (x, y) in ac.iter().zip(bc) {
        // SAFETY: offsets 0, 8, 16, 24 of 32-element arrays.
        unsafe {
            let d0 = _mm256_sub_ps(ld(x, 0), ld(y, 0));
            let d1 = _mm256_sub_ps(ld(x, 8), ld(y, 8));
            let d2 = _mm256_sub_ps(ld(x, 16), ld(y, 16));
            let d3 = _mm256_sub_ps(ld(x, 24), ld(y, 24));
            acc[0] = _mm256_fmadd_ps(d0, d0, acc[0]);
            acc[1] = _mm256_fmadd_ps(d1, d1, acc[1]);
            acc[2] = _mm256_fmadd_ps(d2, d2, acc[2]);
            acc[3] = _mm256_fmadd_ps(d3, d3, acc[3]);
        }
    }
    let (a8, a1) = ar.as_chunks::<8>();
    let (b8, b1) = br.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        // SAFETY: offset 0 of 8-element arrays.
        let d = unsafe { _mm256_sub_ps(ld(x, 0), ld(y, 0)) };
        acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
    }
    let mut sum = reduce4(acc);
    for (&x, &y) in a1.iter().zip(b1) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Four consecutive lags at once (see the NEON version).
#[inline]
#[target_feature(enable = "avx2,fma")]
fn diff_four(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = b.iter().fold(a.len(), |n, s| n.min(s.len()));
    let a = head(a, n);
    let (ac, ar) = a.as_chunks::<16>();
    let (b0c, b0r) = head(b[0], n).as_chunks::<16>();
    let (b1c, b1r) = head(b[1], n).as_chunks::<16>();
    let (b2c, b2r) = head(b[2], n).as_chunks::<16>();
    let (b3c, b3r) = head(b[3], n).as_chunks::<16>();
    let mut acc = [_mm256_setzero_ps(); 8];
    for ((((x, y0), y1), y2), y3) in ac.iter().zip(b0c).zip(b1c).zip(b2c).zip(b3c) {
        // SAFETY: offsets 0 and 8 of 16-element arrays.
        unsafe {
            let x0 = ld(x, 0);
            let x1 = ld(x, 8);
            let d = _mm256_sub_ps(x0, ld(y0, 0));
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
            let d = _mm256_sub_ps(x1, ld(y0, 8));
            acc[1] = _mm256_fmadd_ps(d, d, acc[1]);
            let d = _mm256_sub_ps(x0, ld(y1, 0));
            acc[2] = _mm256_fmadd_ps(d, d, acc[2]);
            let d = _mm256_sub_ps(x1, ld(y1, 8));
            acc[3] = _mm256_fmadd_ps(d, d, acc[3]);
            let d = _mm256_sub_ps(x0, ld(y2, 0));
            acc[4] = _mm256_fmadd_ps(d, d, acc[4]);
            let d = _mm256_sub_ps(x1, ld(y2, 8));
            acc[5] = _mm256_fmadd_ps(d, d, acc[5]);
            let d = _mm256_sub_ps(x0, ld(y3, 0));
            acc[6] = _mm256_fmadd_ps(d, d, acc[6]);
            let d = _mm256_sub_ps(x1, ld(y3, 8));
            acc[7] = _mm256_fmadd_ps(d, d, acc[7]);
        }
    }
    let mut r = [
        hsum(_mm256_add_ps(acc[0], acc[1])),
        hsum(_mm256_add_ps(acc[2], acc[3])),
        hsum(_mm256_add_ps(acc[4], acc[5])),
        hsum(_mm256_add_ps(acc[6], acc[7])),
    ];
    for (rl, br) in r.iter_mut().zip([b0r, b1r, b2r, b3r]) {
        for (&x, &y) in ar.iter().zip(br) {
            let d = x - y;
            *rl += d * d;
        }
    }
    r
}

/// Direct type-I difference function; see [`super::diff_sq`].
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    let valid = valid_lags(x.len(), w, tau0, out.len());
    let (body, rest) = out.split_at_mut(valid);
    let a = head(x, w);
    let (blocks, singles) = body.as_chunks_mut::<4>();
    for (bi, o) in blocks.iter_mut().enumerate() {
        let tau = tau0 + 4 * bi;
        let win = |l: usize| x.get(tau + l..tau + l + w).unwrap_or(&[]);
        *o = diff_four(a, [win(0), win(1), win(2), win(3)]);
    }
    let done = 4 * blocks.len();
    for (k, o) in singles.iter_mut().enumerate() {
        let tau = tau0 + done + k;
        // (A `match`, not `map_or` with a closure: calls to target-feature
        // functions are only safe from code that has those features.)
        *o = match x.get(tau..tau + w) {
            Some(b) => diff_one(a, b),
            None => 0.0,
        };
    }
    diff_sq_padded(x, w, tau0.saturating_add(valid), rest);
}
