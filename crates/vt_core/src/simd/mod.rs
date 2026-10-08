//! Hot kernels with three implementations each: [`scalar`] (portable, written
//! so LLVM auto-vectorises it: fixed-size chunks and multi-lane accumulator
//! arrays), [`neon`] (aarch64, `core::arch` intrinsics) and `avx2` (x86_64,
//! `#[target_feature(enable = "avx2,fma")]`, selected at runtime with
//! `is_x86_feature_detected!`). The top-level functions dispatch to the best
//! available implementation.
//!
//! Dispatch: on aarch64 NEON is part of the baseline target, so the choice is
//! made at compile time (zero overhead). On x86_64 the CPUID result is cached
//! in an atomic on first use. Everything else uses `scalar`.
//!
//! Why hand-written NEON: the perf research on the M4 Max measured a naive
//! iterator dot product at 909 ns for 2048 elements (a serial `fadd` chain,
//! since float addition is not reassociated), auto-vectorised accumulator
//! arrays at 98 ns, and NEON with 8 independent FMA accumulators at 84 ns.
//! The scalar backend deliberately uses plain multiply + add: on x86 baseline
//! `f32::mul_add` becomes a libm `fmaf` call (28× slower).
//!
//! Measured on the M4 Max (`benches/kernels.rs`, n = 4096): `dot` naive
//! iterator 1.96 µs, scalar 178 ns, NEON 171 ns (both at the L1 load limit);
//! `sum_sq` scalar 170 ns, NEON 91 ns; `axpy`/`mul_acc` scalar = NEON (memory
//! bound, LLVM already vectorises the scalar loop); `diff_sq` with W = 1024
//! and 512 lags: scalar 25.7 µs, NEON 16.3 µs. For W ≳ 512 the FFT route
//! ([`crate::fft::Correlator::diff_sq`], 4.2 µs with vDSP at W = 1024) is
//! faster; the direct kernel stays exact for near-periodic frames, where the
//! FFT route's `E₀ + E_τ − 2r` loses relative precision to cancellation.
//!
//! Semantics shared by all backends:
//! * Lengths: binary kernels run over the shortest of their slices.
//! * Results differ between backends only by rounding (different summation
//!   order; NEON/AVX2 fuse multiply-adds). A given backend is deterministic.
//! * [`diff_sq`] never panics: lags whose window would run past the end of `x`
//!   are computed with `x` zero-padded (the same "missing samples are zero"
//!   rule as [`crate::ring::Ring`]).

pub mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use neon as best;

#[cfg(not(any(all(target_arch = "aarch64", target_feature = "neon"), target_arch = "x86_64")))]
use scalar as best;

/// Runtime-dispatched x86_64 entry points.
#[cfg(target_arch = "x86_64")]
mod best {
    use super::{avx2, scalar};
    use core::sync::atomic::{AtomicU8, Ordering};

    /// 0 = not yet probed, 1 = scalar, 2 = AVX2 + FMA.
    static LEVEL: AtomicU8 = AtomicU8::new(0);

    #[inline]
    pub(super) fn has_avx2() -> bool {
        match LEVEL.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let ok = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
                LEVEL.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
                ok
            }
        }
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        if has_avx2() {
            // SAFETY: AVX2 and FMA support was verified at runtime.
            unsafe { avx2::dot(a, b) }
        } else {
            scalar::dot(a, b)
        }
    }

    #[inline]
    pub fn sum_sq(a: &[f32]) -> f32 {
        if has_avx2() {
            // SAFETY: AVX2 and FMA support was verified at runtime.
            unsafe { avx2::sum_sq(a) }
        } else {
            scalar::sum_sq(a)
        }
    }

    #[inline]
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        if has_avx2() {
            // SAFETY: AVX2 and FMA support was verified at runtime.
            unsafe { avx2::axpy(y, a, x) }
        } else {
            scalar::axpy(y, a, x)
        }
    }

    #[inline]
    pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
        if has_avx2() {
            // SAFETY: AVX2 and FMA support was verified at runtime.
            unsafe { avx2::mul_acc(y, x, w) }
        } else {
            scalar::mul_acc(y, x, w)
        }
    }

    #[inline]
    pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
        if has_avx2() {
            // SAFETY: AVX2 and FMA support was verified at runtime.
            unsafe { avx2::diff_sq(x, w, tau0, out) }
        } else {
            scalar::diff_sq(x, w, tau0, out)
        }
    }
}

/// `Σ a[i]·b[i]` over `min(a.len(), b.len())`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    best::dot(a, b)
}

/// `Σ a[i]²`.
#[inline]
pub fn sum_sq(a: &[f32]) -> f32 {
    best::sum_sq(a)
}

/// `y[i] += a * x[i]` over `min(y.len(), x.len())`.
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    best::axpy(y, a, x)
}

/// `y[i] += x[i] * w[i]` (windowed overlap-add) over the shortest length.
#[inline]
pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
    best::mul_acc(y, x, w)
}

/// Direct (type-I) difference function (R§8.4):
/// `out[k] = Σ_{j<w} (x[j] − x[j+τ])²`, τ = tau0 + k.
///
/// Requires `x.len() >= w + tau0 + out.len() − 1`; lags beyond that are
/// computed as if `x` were zero-padded (never panics, work stays bounded by
/// `out.len() · min(w, x.len())`). NEON/AVX2 process four lags per pass
/// (lag blocking) so each loaded `x[j]` vector is reused four times.
#[inline]
pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    best::diff_sq(x, w, tau0, out)
}

/// For tests/benches: which backend the dispatcher picked (`"neon"`,
/// `"avx2"` or `"scalar"`).
pub fn active_backend() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        "neon"
    }
    #[cfg(target_arch = "x86_64")]
    {
        if best::has_avx2() { "avx2" } else { "scalar" }
    }
    #[cfg(not(any(all(target_arch = "aarch64", target_feature = "neon"), target_arch = "x86_64")))]
    {
        "scalar"
    }
}

// ---------------------------------------------------------------------------
// Helpers shared by the backends
// ---------------------------------------------------------------------------

/// First `n` elements of `s` (all of `s` if shorter). Never panics.
#[inline(always)]
pub(crate) fn head(s: &[f32], n: usize) -> &[f32] {
    s.get(..n).unwrap_or(s)
}

/// Number of leading lags (from `tau0`) whose full window lies inside `x`:
/// the largest `v <= n_out` with `w + tau0 + v − 1 <= len`.
#[inline]
pub(crate) fn valid_lags(len: usize, w: usize, tau0: usize, n_out: usize) -> usize {
    match w.checked_add(tau0) {
        Some(need) if need <= len => n_out.min(len - need + 1),
        _ => 0,
    }
}

/// Zero-padded difference for lags `tau0 + k` (slow path for out-of-range
/// lags only). Work per lag is bounded by `min(w, x.len())`.
pub(crate) fn diff_sq_padded(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
    let jmax = w.min(x.len());
    for (k, o) in out.iter_mut().enumerate() {
        let tau = tau0.saturating_add(k);
        let mut acc = 0.0f32;
        for (j, &a) in x.iter().enumerate().take(jmax) {
            let b = j.checked_add(tau).and_then(|i| x.get(i)).copied().unwrap_or(0.0);
            let d = a - b;
            acc += d * d;
        }
        *o = acc;
    }
}

#[cfg(test)]
mod tests;
