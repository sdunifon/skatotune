//! FFT-based correlation and spectra for the pitch detector (R§8.4).
//!
//! [`Correlator`] wraps one fixed real FFT size with all scratch
//! preallocated, exposing exactly what the NSDF / type-I CMNDF verifier needs:
//! linear cross-correlation, autocorrelation, power spectra and the type-I
//! difference function. Two backends:
//!
//! * [`FftBackend::RealFft`]: the `realfft` crate (rustfft underneath, NEON /
//!   AVX kernels), portable.
//! * [`FftBackend::Vdsp`]: Apple Accelerate `vDSP_DFT_zrop` through a small
//!   hand-written FFI (macOS only), 1.4-2.4× faster than realfft per
//!   transform on the M4 Max and bit-reproducible across instances (its split
//!   buffers are cache-line aligned; see the `vdsp` module). Elsewhere, or for
//!   lengths vDSP does not implement, it silently falls back to `RealFft`
//!   ([`Correlator::backend`] reports what is used).
//!
//! Transform sizes ([`FftSizing`]): [`Correlator::new`] rounds up to a power
//! of two; [`Correlator::with_sizing`] with [`FftSizing::Fast`] picks the
//! smallest `f·2^k`, f ∈ {1, 3, 5, 15}, k ≥ 4, that covers the request
//! ([`fast_len`]; P§3.4.1). Those lengths are native to both backends
//! (vDSP's zrop DFT implements exactly 2^k and f·2^k; realfft runs them on
//! radix-2/3/5 kernels), so a detector buffer of W + τ_max samples is
//! transformed at e.g. 1536 instead of 2048 or 3072 instead of 4096. Linear
//! correlation only needs `n ≥ a.len() + out.len() − 1`, so a smaller n is
//! exact; see `docs/PERFORMANCE.md` for the measured gain per size.
//!
//! Both backends produce results on the same (true DFT) scale; tests check
//! them against a direct O(n²) correlation at power-of-two and fast sizes.
//!
//! Measured on the M4 Max (`benches/kernels.rs`, best-of-rounds medians;
//! full tables in `docs/PERFORMANCE.md`): autocorrelation (forward + inverse)
//! at n = 2048: realfft 3.56 µs, vDSP 2.02 µs; power spectrum (forward only)
//! at 2048: 1.85 µs vs 1.07 µs; type-I difference for W = 1024 / τ = 512:
//! realfft 5.85 µs at n = 2048 and 4.47 µs at n = 1536, vDSP 3.64 / 3.42 µs
//! (direct NEON: 16.4 µs); W = 2048 / τ = 1024: realfft 12.7 / 9.7 µs at
//! n = 4096 / 3072, vDSP 7.5 / 5.9 µs (direct NEON: 66.4 µs).

#[cfg(target_os = "macos")]
mod vdsp;

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Which FFT implementation a [`Correlator`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftBackend {
    /// `realfft` / `rustfft` (portable).
    RealFft,
    /// macOS Accelerate (vDSP). Falls back to `RealFft` elsewhere.
    Vdsp,
}

impl FftBackend {
    /// The fastest backend for this platform: `Vdsp` on macOS, `RealFft`
    /// otherwise.
    pub fn best() -> Self {
        if cfg!(target_os = "macos") { FftBackend::Vdsp } else { FftBackend::RealFft }
    }
}

/// How a [`Correlator`] turns a requested minimum length into its transform
/// size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FftSizing {
    /// The next power of two (minimum 2): the [`Correlator::new`] behaviour.
    #[default]
    PowerOfTwo,
    /// The smallest `f·2^k ≥ min_len` with f ∈ {1, 3, 5, 15} and k ≥ 4
    /// ([`fast_len`]): never larger than the power of two, native to vDSP and
    /// fast in realfft. E.g. 1424 → 1536, 560 → 640, 3072 → 3072.
    Fast,
}

impl FftSizing {
    /// The transform size this policy picks for a request of `min_len`
    /// samples. Always even and ≥ `min_len` (for any realistic length).
    pub fn len_for(self, min_len: usize) -> usize {
        match self {
            FftSizing::PowerOfTwo => pow2_len(min_len),
            FftSizing::Fast => fast_len(min_len),
        }
    }
}

/// Smallest power of two ≥ `min_len` (minimum 2), saturating at the largest
/// representable power of two.
fn pow2_len(min_len: usize) -> usize {
    min_len.max(2).checked_next_power_of_two().unwrap_or(1 << (usize::BITS - 1))
}

/// Smallest length `f·2^k ≥ min_len` with f ∈ {1, 3, 5, 15} and k ≥ 4
/// (minimum 16): the sizes Apple's `vDSP_DFT_zrop` implements, and ones
/// realfft decomposes into radix-2/3/5 passes (P§3.4.1, P§3.5). Never larger
/// than the next power of two. Examples: 979 → 1024, 1424 → 1536,
/// 1536 → 1536, 653 → 768, 560 → 640, 1056 → 1280.
///
/// Requests beyond the largest representable candidate get the power-of-two
/// answer of [`FftSizing::PowerOfTwo`].
pub fn fast_len(min_len: usize) -> usize {
    let pow2 = pow2_len(min_len.max(16));
    [3usize, 5, 15]
        .into_iter()
        .filter_map(|f| {
            // Smallest f·2^k ≥ min_len with k ≥ 4, if it fits in usize.
            let mut n = f << 4;
            while n < min_len {
                n = n.checked_mul(2)?;
            }
            Some(n)
        })
        .fold(pow2, usize::min)
}

/// realfft plans and scratch.
struct RealPlan {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    /// Time-domain buffer (n); realfft uses its input as scratch, so every
    /// transform copies into it.
    time: Box<[f32]>,
    spec_a: Box<[Complex<f32>]>,
    spec_b: Box<[Complex<f32>]>,
    scratch: Box<[Complex<f32>]>,
}

impl RealPlan {
    fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd.get_scratch_len().max(inv.get_scratch_len());
        let zc = |len: usize| vec![Complex::new(0.0f32, 0.0); len].into_boxed_slice();
        Self {
            time: vec![0.0; n].into_boxed_slice(),
            spec_a: zc(n / 2 + 1),
            spec_b: zc(n / 2 + 1),
            scratch: zc(scratch_len),
            fwd,
            inv,
        }
    }

    /// Copies `x` (zero-padded / truncated to n) into `time`, transforms it
    /// into `spec_a` (or `spec_b`). Returns false on an (impossible) length
    /// error.
    fn forward(&mut self, x: &[f32], into_b: bool) -> bool {
        let n = self.time.len();
        let x = x.get(..n).unwrap_or(x);
        let (head, tail) = self.time.split_at_mut(x.len());
        head.copy_from_slice(x);
        tail.fill(0.0);
        let spec = if into_b { &mut self.spec_b } else { &mut self.spec_a };
        self.fwd.process_with_scratch(&mut self.time, spec, &mut self.scratch).is_ok()
    }

    /// Inverse of `spec_a` into `time`, then `out[τ] = time[τ] / n`.
    fn inverse_a_to(&mut self, out: &mut [f32]) -> bool {
        // A real signal's spectrum has exactly real DC and (for even n)
        // Nyquist bins; enforce it so realfft does not report invalid input.
        if let Some(c) = self.spec_a.first_mut() {
            c.im = 0.0;
        }
        if self.time.len().is_multiple_of(2)
            && let Some(c) = self.spec_a.last_mut()
        {
            c.im = 0.0;
        }
        if self.inv.process_with_scratch(&mut self.spec_a, &mut self.time, &mut self.scratch).is_err() {
            return false;
        }
        let scale = 1.0 / self.time.len() as f32;
        for (o, &t) in out.iter_mut().zip(self.time.iter()) {
            *o = t * scale;
        }
        true
    }

    fn cross_correlate(&mut self, a: &[f32], b: &[f32], out: &mut [f32]) -> bool {
        if !(self.forward(a, false) && self.forward(b, true)) {
            return false;
        }
        // conj(A)·B: the inverse gives Σ_j a[j]·b[j+τ].
        for (x, y) in self.spec_a.iter_mut().zip(self.spec_b.iter()) {
            *x = x.conj() * y;
        }
        self.inverse_a_to(out)
    }

    fn autocorrelate(&mut self, x: &[f32], out: &mut [f32]) -> bool {
        if !self.forward(x, false) {
            return false;
        }
        for c in self.spec_a.iter_mut() {
            *c = Complex::new(c.norm_sqr(), 0.0);
        }
        self.inverse_a_to(out)
    }

    fn power_spectrum(&mut self, x: &[f32], out: &mut [f32]) -> bool {
        if !self.forward(x, false) {
            return false;
        }
        for (o, c) in out.iter_mut().zip(self.spec_a.iter()) {
            *o = c.norm_sqr();
        }
        true
    }
}

enum Plan {
    Real(RealPlan),
    #[cfg(target_os = "macos")]
    Vdsp(vdsp::Plan),
}

/// FFT-based linear correlation and spectra for a fixed transform size (a
/// power of two from [`Correlator::new`], or any [`FftSizing`] from
/// [`Correlator::with_sizing`]). All scratch is preallocated at construction;
/// every method is allocation-free and never panics (out-of-contract lengths
/// are truncated and unfilled outputs are zeroed).
pub struct Correlator {
    n: usize,
    backend: FftBackend,
    plan: Plan,
}

#[allow(clippy::len_without_is_empty)] // a transform size is never empty
impl Correlator {
    /// Plans transforms of size `n` (rounded up to a power of two, minimum 2)
    /// with the requested backend. `Vdsp` falls back to `RealFft` off macOS or
    /// if vDSP rejects the length.
    pub fn new(n: usize, backend: FftBackend) -> Self {
        Self::with_sizing(n, backend, FftSizing::PowerOfTwo)
    }

    /// Plans transforms of size `sizing.len_for(min_len)` with the requested
    /// backend (same fallback rule as [`Correlator::new`]). For the detector's
    /// type-I frame, `min_len = W + τ_max + 1` makes every lag exact;
    /// [`FftSizing::Fast`] then gives e.g. 1536 where `new` gives 2048.
    /// Allocates: call it off the audio thread.
    pub fn with_sizing(min_len: usize, backend: FftBackend, sizing: FftSizing) -> Self {
        Self::with_exact_len(sizing.len_for(min_len), backend)
    }

    /// Plans exactly `n` (an even length from [`FftSizing::len_for`]).
    fn with_exact_len(n: usize, backend: FftBackend) -> Self {
        #[cfg(target_os = "macos")]
        if backend == FftBackend::Vdsp
            && let Some(p) = vdsp::Plan::new(n)
        {
            return Self { n, backend: FftBackend::Vdsp, plan: Plan::Vdsp(p) };
        }
        let _ = backend;
        Self { n, backend: FftBackend::RealFft, plan: Plan::Real(RealPlan::new(n)) }
    }

    /// Transform size n.
    pub fn len(&self) -> usize {
        self.n
    }

    /// The backend actually in use.
    pub fn backend(&self) -> FftBackend {
        self.backend
    }

    /// Linear cross-correlation `out[τ] = Σ_j a[j]·b[j+τ]` (zero-padded) for τ
    /// in `0..out.len()`.
    ///
    /// Exact (no circular wrap) for every τ with `a.len() + τ <= n`, i.e. for
    /// all outputs when `a.len() + out.len() <= n + 1`; `b.len()` only needs
    /// to be ≤ n. Inputs longer than n are truncated; outputs past n are 0.
    pub fn cross_correlate(&mut self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let (body, rest) = split_out(out, self.n);
        rest.fill(0.0);
        let ok = match &mut self.plan {
            Plan::Real(p) => p.cross_correlate(a, b, body),
            #[cfg(target_os = "macos")]
            Plan::Vdsp(p) => {
                p.cross_correlate(a, b, body);
                true
            }
        };
        if !ok {
            body.fill(0.0);
        }
    }

    /// Autocorrelation `out[τ] = Σ_j x[j]·x[j+τ]`; exact for
    /// `x.len() + out.len() <= n + 1`.
    pub fn autocorrelate(&mut self, x: &[f32], out: &mut [f32]) {
        let (body, rest) = split_out(out, self.n);
        rest.fill(0.0);
        let ok = match &mut self.plan {
            Plan::Real(p) => p.autocorrelate(x, body),
            #[cfg(target_os = "macos")]
            Plan::Vdsp(p) => {
                p.autocorrelate(x, body);
                true
            }
        };
        if !ok {
            body.fill(0.0);
        }
    }

    /// `|X[k]|²` for k in 0..=n/2 of zero-padded `x` (`out.len() == n/2 + 1`;
    /// extra outputs are 0, missing ones are skipped). True DFT scale: a unit
    /// sine of N/m cycles over n samples has `|X|² = (n/2)²` at its bin.
    pub fn power_spectrum(&mut self, x: &[f32], out: &mut [f32]) {
        let (body, rest) = split_out(out, self.n / 2 + 1);
        rest.fill(0.0);
        let ok = match &mut self.plan {
            Plan::Real(p) => p.power_spectrum(x, body),
            #[cfg(target_os = "macos")]
            Plan::Vdsp(p) => {
                p.power_spectrum(x, body);
                true
            }
        };
        if !ok {
            body.fill(0.0);
        }
    }

    /// Type-I difference function via FFT (the fast path for
    /// [`crate::simd::diff_sq`], same semantics):
    /// `out[k] = Σ_{j<w} (x[j] − x[j+τ])²`, τ = tau0 + k, computed as
    /// `Σx[j]² + Σx[j+τ]² − 2·r(τ)` with r one cross-correlation of the
    /// `w`-sample head against `x[tau0..]` and the energies from a running
    /// f64 sum (R§8.4 "keep a W + τ_max buffer and compute the type-I
    /// difference").
    ///
    /// Needs `w + out.len() − 1 <= n`; otherwise it falls back to the direct
    /// kernel. Values are clamped at 0 (FFT rounding is ~1e-6 of the window
    /// energy, so near-zero differences are less precise than the direct
    /// kernel's).
    pub fn diff_sq(&mut self, x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
        let n_out = out.len();
        if n_out == 0 {
            return;
        }
        if w.saturating_add(n_out - 1) > self.n {
            crate::simd::diff_sq(x, w, tau0, out);
            return;
        }
        let a = x.get(..w).unwrap_or(x);
        let b_end = tau0.saturating_add(w + n_out - 1).min(x.len());
        let b = x.get(tau0.min(x.len())..b_end).unwrap_or(&[]);
        self.cross_correlate(a, b, out);
        let sq = |i: usize| {
            let v = x.get(i).copied().unwrap_or(0.0) as f64;
            v * v
        };
        // Window energies: the initial sums use the vectorised f32 kernel (its
        // ~1e-7 relative error is far below the FFT's); the per-lag sliding
        // update runs in f64 so it cannot drift over hundreds of lags.
        let e0 = crate::simd::sum_sq(a) as f64;
        // Energy of the lagged window x[τ .. τ + w), slid one lag at a time.
        let mut et = crate::simd::sum_sq(b.get(..w).unwrap_or(b)) as f64;
        for (k, o) in out.iter_mut().enumerate() {
            let tau = tau0.saturating_add(k);
            let d = e0 + et - 2.0 * *o as f64;
            *o = d.max(0.0) as f32;
            et += sq(tau.saturating_add(w)) - sq(tau);
            // Guard against drift producing tiny negative energies.
            et = et.max(0.0);
        }
    }
}

impl Clone for Correlator {
    /// A fresh correlator of the same size and backend (scratch contents are
    /// never observable, so this is equivalent). Allocates: not for the audio
    /// thread.
    fn clone(&self) -> Self {
        Self::with_exact_len(self.n, self.backend)
    }
}

impl core::fmt::Debug for Correlator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Correlator").field("n", &self.n).field("backend", &self.backend).finish_non_exhaustive()
    }
}

/// Splits `out` into the part a transform of `limit` outputs can fill and the
/// remainder.
fn split_out(out: &mut [f32], limit: usize) -> (&mut [f32], &mut [f32]) {
    let m = out.len().min(limit);
    out.split_at_mut(m)
}

#[cfg(test)]
mod tests;
