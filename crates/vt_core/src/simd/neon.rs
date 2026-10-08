//! aarch64 NEON kernels (`core::arch::aarch64`).
//!
//! Reductions keep 8 independent `float32x4_t` FMA accumulators (32 floats per
//! iteration): FMA latency is 4 cycles and the M-series cores issue up to 4
//! FP ops per cycle, so fewer chains leave the pipes idle (the perf research
//! measured 84 ns for a 2048-element dot vs 98 ns for the auto-vectorised
//! 4×4 accumulator version). `diff_sq` blocks four lags per pass so each
//! `x[j]` vector is loaded once and reused by four lags (8 accumulators: 4
//! lags × 2 vectors).
//!
//! Safety: the implementations are `#[target_feature(enable = "neon")]`
//! functions (Rust requires the attribute for the intrinsics to be callable
//! even though NEON is enabled for the whole build), wrapped by safe public
//! functions at the bottom of the file. The module is only compiled with
//! `cfg(target_feature = "neon")` (part of the baseline of every standard
//! aarch64 target), so those calls are always sound. Inside, the only
//! `unsafe` operations are vector loads/stores of whole `[f32; N]` arrays
//! produced by `as_chunks`, so every pointer offset is in bounds by
//! construction.

use core::arch::aarch64::*;

use super::{diff_sq_padded, head, valid_lags};

/// Loads 4 floats from `p[off..off + 4]`.
///
/// # Safety
/// `off + 4 <= N`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn ld<const N: usize>(p: &[f32; N], off: usize) -> float32x4_t {
    debug_assert!(off + 4 <= N);
    // SAFETY: the caller guarantees `off + 4 <= N`, so the 16-byte read stays
    // inside the array.
    unsafe { vld1q_f32(p.as_ptr().add(off)) }
}

/// Stores 4 floats to `p[off..off + 4]`.
///
/// # Safety
/// `off + 4 <= N`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn st<const N: usize>(p: &mut [f32; N], off: usize, v: float32x4_t) {
    debug_assert!(off + 4 <= N);
    // SAFETY: the caller guarantees `off + 4 <= N`.
    unsafe { vst1q_f32(p.as_mut_ptr().add(off), v) }
}

/// Tree sum of 8 accumulators into one scalar.
#[inline]
#[target_feature(enable = "neon")]
fn reduce8(acc: [float32x4_t; 8]) -> f32 {
    let s0 = vaddq_f32(acc[0], acc[1]);
    let s1 = vaddq_f32(acc[2], acc[3]);
    let s2 = vaddq_f32(acc[4], acc[5]);
    let s3 = vaddq_f32(acc[6], acc[7]);
    vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
}

/// `Σ a[i]·b[i]` over `min(a.len(), b.len())`.
#[target_feature(enable = "neon")]
fn dot_impl(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<32>();
    let (bc, br) = b.as_chunks::<32>();
    let mut acc = [vdupq_n_f32(0.0); 8];
    for (x, y) in ac.iter().zip(bc) {
        // SAFETY: offsets 0, 4, …, 28 of 32-element arrays.
        unsafe {
            acc[0] = vfmaq_f32(acc[0], ld(x, 0), ld(y, 0));
            acc[1] = vfmaq_f32(acc[1], ld(x, 4), ld(y, 4));
            acc[2] = vfmaq_f32(acc[2], ld(x, 8), ld(y, 8));
            acc[3] = vfmaq_f32(acc[3], ld(x, 12), ld(y, 12));
            acc[4] = vfmaq_f32(acc[4], ld(x, 16), ld(y, 16));
            acc[5] = vfmaq_f32(acc[5], ld(x, 20), ld(y, 20));
            acc[6] = vfmaq_f32(acc[6], ld(x, 24), ld(y, 24));
            acc[7] = vfmaq_f32(acc[7], ld(x, 28), ld(y, 28));
        }
    }
    // Up to 31 leftovers: whole vectors into the first accumulator, then
    // scalar FMAs (aarch64 always has a hardware fmadd).
    let (a4, a1) = ar.as_chunks::<4>();
    let (b4, b1) = br.as_chunks::<4>();
    for (x, y) in a4.iter().zip(b4) {
        // SAFETY: offset 0 of 4-element arrays.
        unsafe {
            acc[0] = vfmaq_f32(acc[0], ld(x, 0), ld(y, 0));
        }
    }
    let mut sum = reduce8(acc);
    for (&x, &y) in a1.iter().zip(b1) {
        sum = x.mul_add(y, sum);
    }
    sum
}

/// `Σ a[i]²`.
#[target_feature(enable = "neon")]
fn sum_sq_impl(a: &[f32]) -> f32 {
    // One load per FMA here (vs two in `dot`), so the loop is bounded by FMA
    // latency rather than loads: 16 chains are needed to keep 4 pipes busy
    // (measured 91 ns vs 124 ns with 8 chains at 4096 on the M4 Max).
    let (ac, ar) = a.as_chunks::<64>();
    let mut acc = [vdupq_n_f32(0.0); 16];
    for x in ac {
        for (k, s) in acc.iter_mut().enumerate() {
            // SAFETY: offsets 0, 4, …, 60 of 64-element arrays.
            let v = unsafe { ld(x, 4 * k) };
            *s = vfmaq_f32(*s, v, v);
        }
    }
    let (a4, a1) = ar.as_chunks::<4>();
    for (i, x) in a4.iter().enumerate() {
        // SAFETY: offset 0 of a 4-element array.
        let v = unsafe { ld(x, 0) };
        // Spread leftovers over independent chains too.
        if let Some(s) = acc.get_mut(i) {
            *s = vfmaq_f32(*s, v, v);
        }
    }
    let lo = [acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]];
    let hi = [acc[8], acc[9], acc[10], acc[11], acc[12], acc[13], acc[14], acc[15]];
    let mut sum = reduce8(lo) + reduce8(hi);
    for &x in a1 {
        sum = x.mul_add(x, sum);
    }
    sum
}

/// `y[i] += a * x[i]` over `min(y.len(), x.len())`.
#[target_feature(enable = "neon")]
fn axpy_impl(y: &mut [f32], a: f32, x: &[f32]) {
    let n = y.len().min(x.len());
    let (y, _) = y.split_at_mut(n);
    let x = head(x, n);
    let (yc, yr) = y.as_chunks_mut::<16>();
    let (xc, xr) = x.as_chunks::<16>();
    let av = vdupq_n_f32(a);
    for (yv, xv) in yc.iter_mut().zip(xc) {
        // SAFETY: offsets 0, 4, 8, 12 of 16-element arrays.
        unsafe {
            let r0 = vfmaq_f32(ld(yv, 0), av, ld(xv, 0));
            let r1 = vfmaq_f32(ld(yv, 4), av, ld(xv, 4));
            let r2 = vfmaq_f32(ld(yv, 8), av, ld(xv, 8));
            let r3 = vfmaq_f32(ld(yv, 12), av, ld(xv, 12));
            st(yv, 0, r0);
            st(yv, 4, r1);
            st(yv, 8, r2);
            st(yv, 12, r3);
        }
    }
    for (yi, &xi) in yr.iter_mut().zip(xr) {
        *yi = a.mul_add(xi, *yi);
    }
}

/// `y[i] += x[i] * w[i]` over the shortest length.
#[target_feature(enable = "neon")]
fn mul_acc_impl(y: &mut [f32], x: &[f32], w: &[f32]) {
    let n = y.len().min(x.len()).min(w.len());
    let (y, _) = y.split_at_mut(n);
    let (x, w) = (head(x, n), head(w, n));
    let (yc, yr) = y.as_chunks_mut::<16>();
    let (xc, xr) = x.as_chunks::<16>();
    let (wc, wr) = w.as_chunks::<16>();
    for ((yv, xv), wv) in yc.iter_mut().zip(xc).zip(wc) {
        // SAFETY: offsets 0, 4, 8, 12 of 16-element arrays.
        unsafe {
            let r0 = vfmaq_f32(ld(yv, 0), ld(xv, 0), ld(wv, 0));
            let r1 = vfmaq_f32(ld(yv, 4), ld(xv, 4), ld(wv, 4));
            let r2 = vfmaq_f32(ld(yv, 8), ld(xv, 8), ld(wv, 8));
            let r3 = vfmaq_f32(ld(yv, 12), ld(xv, 12), ld(wv, 12));
            st(yv, 0, r0);
            st(yv, 4, r1);
            st(yv, 8, r2);
            st(yv, 12, r3);
        }
    }
    for ((yi, &xi), &wi) in yr.iter_mut().zip(xr).zip(wr) {
        *yi = xi.mul_add(wi, *yi);
    }
}

/// `Σ (a[j] − b[j])²` for one lag (`a`, `b` of equal length).
#[inline]
#[target_feature(enable = "neon")]
fn diff_one(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (head(a, n), head(b, n));
    let (ac, ar) = a.as_chunks::<32>();
    let (bc, br) = b.as_chunks::<32>();
    let mut acc = [vdupq_n_f32(0.0); 8];
    for (x, y) in ac.iter().zip(bc) {
        // SAFETY: offsets 0, 4, …, 28 of 32-element arrays.
        unsafe {
            for (k, s) in acc.iter_mut().enumerate() {
                let d = vsubq_f32(ld(x, 4 * k), ld(y, 4 * k));
                *s = vfmaq_f32(*s, d, d);
            }
        }
    }
    let (a4, a1) = ar.as_chunks::<4>();
    let (b4, b1) = br.as_chunks::<4>();
    for (x, y) in a4.iter().zip(b4) {
        // SAFETY: offset 0 of 4-element arrays.
        let d = unsafe { vsubq_f32(ld(x, 0), ld(y, 0)) };
        acc[0] = vfmaq_f32(acc[0], d, d);
    }
    let mut sum = reduce8(acc);
    for (&x, &y) in a1.iter().zip(b1) {
        let d = x - y;
        sum = d.mul_add(d, sum);
    }
    sum
}

/// Four consecutive lags at once: `b[l]` is the window for lag τ + l.
#[inline]
#[target_feature(enable = "neon")]
fn diff_four(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
    let n = b.iter().fold(a.len(), |n, s| n.min(s.len()));
    let a = head(a, n);
    let (ac, ar) = a.as_chunks::<8>();
    let (b0c, b0r) = head(b[0], n).as_chunks::<8>();
    let (b1c, b1r) = head(b[1], n).as_chunks::<8>();
    let (b2c, b2r) = head(b[2], n).as_chunks::<8>();
    let (b3c, b3r) = head(b[3], n).as_chunks::<8>();
    let mut acc = [vdupq_n_f32(0.0); 8];
    for ((((x, y0), y1), y2), y3) in ac.iter().zip(b0c).zip(b1c).zip(b2c).zip(b3c) {
        // SAFETY: offsets 0 and 4 of 8-element arrays.
        unsafe {
            let x0 = ld(x, 0);
            let x1 = ld(x, 4);
            let d = vsubq_f32(x0, ld(y0, 0));
            acc[0] = vfmaq_f32(acc[0], d, d);
            let d = vsubq_f32(x1, ld(y0, 4));
            acc[1] = vfmaq_f32(acc[1], d, d);
            let d = vsubq_f32(x0, ld(y1, 0));
            acc[2] = vfmaq_f32(acc[2], d, d);
            let d = vsubq_f32(x1, ld(y1, 4));
            acc[3] = vfmaq_f32(acc[3], d, d);
            let d = vsubq_f32(x0, ld(y2, 0));
            acc[4] = vfmaq_f32(acc[4], d, d);
            let d = vsubq_f32(x1, ld(y2, 4));
            acc[5] = vfmaq_f32(acc[5], d, d);
            let d = vsubq_f32(x0, ld(y3, 0));
            acc[6] = vfmaq_f32(acc[6], d, d);
            let d = vsubq_f32(x1, ld(y3, 4));
            acc[7] = vfmaq_f32(acc[7], d, d);
        }
    }
    let mut r = [
        vaddvq_f32(vaddq_f32(acc[0], acc[1])),
        vaddvq_f32(vaddq_f32(acc[2], acc[3])),
        vaddvq_f32(vaddq_f32(acc[4], acc[5])),
        vaddvq_f32(vaddq_f32(acc[6], acc[7])),
    ];
    for (rl, br) in r.iter_mut().zip([b0r, b1r, b2r, b3r]) {
        for (&x, &y) in ar.iter().zip(br) {
            let d = x - y;
            *rl = d.mul_add(d, *rl);
        }
    }
    r
}

/// Direct type-I difference function; see [`super::diff_sq`].
#[target_feature(enable = "neon")]
fn diff_sq_impl(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    let valid = valid_lags(x.len(), w, tau0, out.len());
    let (body, rest) = out.split_at_mut(valid);
    let a = head(x, w);
    let (blocks, singles) = body.as_chunks_mut::<4>();
    for (bi, o) in blocks.iter_mut().enumerate() {
        let tau = tau0 + 4 * bi;
        // All four lags are valid, so every window lies inside `x`.
        let win = |l: usize| x.get(tau + l..tau + l + w).unwrap_or(&[]);
        *o = diff_four(a, [win(0), win(1), win(2), win(3)]);
    }
    let done = 4 * blocks.len();
    for (k, o) in singles.iter_mut().enumerate() {
        let tau = tau0 + done + k;
        *o = match x.get(tau..tau + w) {
            Some(b) => diff_one(a, b),
            None => 0.0,
        };
    }
    diff_sq_padded(x, w, tau0.saturating_add(valid), rest);
}

// ---------------------------------------------------------------------------
// Safe public entry points
// ---------------------------------------------------------------------------
//
// SAFETY (all wrappers): this module is compiled only with
// `cfg(target_feature = "neon")`, so every CPU the binary can run on has NEON
// and calling the `#[target_feature(enable = "neon")]` implementations is
// sound.

/// `Σ a[i]·b[i]` over `min(a.len(), b.len())`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: see above.
    unsafe { dot_impl(a, b) }
}

/// `Σ a[i]²`.
#[inline]
pub fn sum_sq(a: &[f32]) -> f32 {
    // SAFETY: see above.
    unsafe { sum_sq_impl(a) }
}

/// `y[i] += a * x[i]` over `min(y.len(), x.len())`.
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    // SAFETY: see above.
    unsafe { axpy_impl(y, a, x) }
}

/// `y[i] += x[i] * w[i]` over the shortest length.
#[inline]
pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
    // SAFETY: see above.
    unsafe { mul_acc_impl(y, x, w) }
}

/// Direct type-I difference function; see [`super::diff_sq`].
#[inline]
pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    // SAFETY: see above.
    unsafe { diff_sq_impl(x, w, tau0, out) }
}
