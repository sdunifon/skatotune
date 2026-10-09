//! Per-hop FFT verifier (R§8.4, R§8.5).
//!
//! Each hop one frame of the decimated verifier stream (16 kHz in Eco, 24 kHz
//! in HQ/Max) is analysed with FFT correlations:
//!
//! * **Type-I difference** over a fixed window W = 2.2/f_min and lags up to
//!   τ_max ≈ 0.48·W (R§8.4 "keep τ_max ≤ W/2 to 0.6·W"):
//!   `d(τ) = Σ_{j<W} x_j² + Σ_{j<W} x_{j+τ}² − 2 Σ_{j<W} x_j x_{j+τ}`.
//!   The cross term is one correlation of the W-sample head against the
//!   W + τ_max buffer; the energies are f64 sliding sums. Type-I (fixed W)
//!   rather than MPM's shrinking type-II window, because pYIN's threshold
//!   prior was calibrated on it and type-II reports false periodicity at long
//!   lags (R§8.4, R§11.2 #4).
//! * **CMNDF** `cmndf[τ] = d[τ]·τ / Σ_{j=1..τ} d[j]`, whose troughs feed the
//!   pYIN probabilities.
//! * **Normalised correlation** `nsdf[τ] = 1 − d[τ]/(e₀ + e_τ) ∈ [−1, 1]`
//!   (the type-I form of McLeod's NSDF; same scale as the E/H ratio), used
//!   for clarity and octave comparisons.
//!
//! The frame is stored newest-first, so the W-sample head is the most recent
//! audio and the lagged copy reaches back in time (the correlation values are
//! unchanged by the reversal).
//!
//! HQ/Max add a **short weighted frame** (W/2, Hann weights on the head,
//! McLeod's windowed-NSDF idea for ramps; R§7.2.2 two-resolution verifier)
//! and a **prime-harmonic score** from the power spectrum of the long buffer
//! (SWIPE′-like, with negative lobes at half-harmonics; R§8.5).

use vt_core::fft::{Correlator, FftBackend, FftSizing};
use vt_core::simd;

use crate::eh::{EhTrough, find_troughs};
use crate::hist::Mirror;

/// FFT backend for the verifier: realfft. It was chosen over
/// `FftBackend::best()` because vDSP once gave instance-dependent last-bit
/// differences (DESIGN §2.1 rule 6); vt_core has since fixed that with
/// line-aligned split buffers (docs/PERFORMANCE.md §3.5), so switching to
/// vDSP (≈ 2× faster correlations at ≥ 1024 points, slower at ≤ 512) is now
/// a measured option rather than a correctness risk.
const FFT_BACKEND: FftBackend = FftBackend::RealFft;
/// Transform size for the lag-domain correlation of a `len`-sample frame:
/// 3·2^k when that still holds `len` (it is then ¾ of the next power of two),
/// else the power of two. Every lag stays exact (linear correlation) at any
/// size ≥ `len`. realfft is 25-27 % faster at 3·2^k than at the power of two
/// above it (docs/PERFORMANCE.md §3.4), while the 5·2^k and 15·2^k sizes of
/// `vt_core::fft::fast_len` gain nothing or lose (15·2^k = 960 for the Eco
/// Low Male frame measured +6 % per hop against 1024), so only 3·2^k is used.
/// At 48 kHz: HQ Low Male / Wide long frame 2048 → 1536 and short frame
/// 1024 → 768, Eco Alto/Tenor 1024 → 768, Eco Soprano 512 → 384, HQ Soprano
/// 1024 → 768; HQ Alto/Tenor stays 1024 (979 samples).
pub(crate) fn correlation_len(len: usize) -> usize {
    let p = len.max(2).next_power_of_two();
    let three = p / 4 * 3;
    if p >= 64 && three >= len { three } else { p }
}

/// Plans the correlation of a `len`-sample frame at [`correlation_len`].
fn correlation_plan(len: usize) -> Correlator {
    // `correlation_len` is a power of two or 3·2^k with k ≥ 4, both of which
    // `FftSizing::Fast` plans exactly.
    Correlator::with_sizing(correlation_len(len), FFT_BACKEND, FftSizing::Fast)
}
/// Long frame: W in periods of f_min (R§8.4).
pub(crate) const LONG_WINDOW_PERIODS: f32 = 2.2;
/// CMNDF troughs above this never receive pYIN mass (thresholds stop at 1.0).
const MAX_CMNDF_TROUGH: f32 = 1.0;

/// Type-I frame analysis at one verifier rate.
pub(crate) struct LongFrame {
    /// Window length W (verifier samples).
    w: usize,
    tau_min: usize,
    tau_max: usize,
    corr: Correlator,
    /// Newest-first frame, `w + tau_max + 1` samples.
    frame: Box<[f32]>,
    /// Cross term, then reused as `d`.
    d: Box<[f32]>,
    nsdf: Box<[f32]>,
    /// Lagged-window energy `e_τ` per lag and the head energy `e₀` of the
    /// last analysis (the attack-compensated correlation of the voicing-onset
    /// check is evaluated from them on demand, near one lag).
    et_lag: Box<[f64]>,
    e0: f64,
    cmndf: Box<[f32]>,
    /// Power-of-two plan for the power spectrum (HQ/Max harmonic score). The
    /// correlations use the exact-size `corr`; the spectrum keeps the
    /// power-of-two zero padding the harmonic score's linear bin
    /// interpolation and thresholds were calibrated with. `None` for a frame
    /// built without a spectrum (Eco).
    spec_corr: Option<Correlator>,
    /// Hann window over the whole buffer (power spectrum).
    hann: Box<[f32]>,
    windowed: Box<[f32]>,
    /// Power spectrum, `n/2 + 1` bins.
    spec: Box<[f32]>,
    spec_valid: bool,
    /// Mean amplitude (√power) of the spectrum, the harmonic-score scale.
    spec_mean_amp: f32,
}

impl LongFrame {
    /// Frame of `w` samples with lags `tau_min ..= tau_max` (verifier samples).
    /// `with_spectrum` plans the power spectrum of the harmonic score (HQ/Max);
    /// without it [`LongFrame::analyze`] never computes one.
    pub(crate) fn new(w: usize, tau_min: usize, tau_max: usize, with_spectrum: bool) -> Self {
        let w = w.max(8);
        let tau_max = tau_max.max(tau_min + 2);
        let len = w + tau_max + 1;
        let corr = correlation_plan(len);
        let spec_corr = with_spectrum.then(|| Correlator::new(len, FFT_BACKEND));
        let n = spec_corr.as_ref().map_or(0, Correlator::len);
        let mut hann = vec![0.0f32; if with_spectrum { len } else { 0 }].into_boxed_slice();
        vt_core::window::hann(&mut hann);
        Self {
            w,
            tau_min,
            tau_max,
            corr,
            spec_corr,
            frame: vec![0.0; len].into_boxed_slice(),
            d: vec![0.0; tau_max + 1].into_boxed_slice(),
            nsdf: vec![0.0; tau_max + 1].into_boxed_slice(),
            et_lag: vec![0.0; tau_max + 1].into_boxed_slice(),
            e0: 0.0,
            cmndf: vec![1.0; tau_max + 1].into_boxed_slice(),
            hann,
            windowed: vec![0.0; if with_spectrum { len } else { 0 }].into_boxed_slice(),
            spec: vec![0.0; if with_spectrum { n / 2 + 1 } else { 0 }].into_boxed_slice(),
            spec_valid: false,
            spec_mean_amp: 0.0,
        }
    }

    /// Buffer length the frame reads (`W + τ_max + 1`).
    pub(crate) fn span(&self) -> usize {
        self.frame.len()
    }

    pub(crate) fn window_len(&self) -> usize {
        self.w
    }

    pub(crate) fn tau_max(&self) -> usize {
        self.tau_max
    }

    /// Analyses the frame whose newest sample is stream index `end − 1`.
    /// Returns false (and leaves every curve flat) when the head is silent.
    pub(crate) fn analyze(&mut self, hist: &Mirror, end: i64, with_spectrum: bool, silence: f64) -> bool {
        let len = self.frame.len();
        hist.copy_reversed(end, len, &mut self.frame);
        let w = self.w;
        let head = &self.frame[..w];
        let e0 = simd::sum_sq(head) as f64;
        self.spec_valid = false;
        if e0 <= silence * w as f64 {
            self.nsdf.fill(0.0);
            self.et_lag.fill(0.0);
            self.e0 = 0.0;
            self.cmndf.fill(1.0);
            self.d.fill(0.0);
            return false;
        }
        // Cross term r[τ] = Σ_{j<W} x_j x_{j+τ} (exact: W + τ_max + 1 ≤ n + 1).
        self.corr.cross_correlate(head, &self.frame, &mut self.d);
        self.e0 = e0;
        // d, nsdf with f64 sliding lagged energy e_τ = Σ_{j<W} x_{j+τ}².
        let mut et = e0;
        let mut dsum = 0.0f64;
        let sq = |v: f32| (v as f64) * (v as f64);
        for tau in 0..=self.tau_max {
            let (Some(dt), Some(nt), Some(ot), Some(ct)) =
                (self.d.get_mut(tau), self.nsdf.get_mut(tau), self.et_lag.get_mut(tau), self.cmndf.get_mut(tau))
            else {
                break;
            };
            let r = *dt as f64;
            let dv = (e0 + et - 2.0 * r).max(0.0);
            *dt = dv as f32;
            *nt = (1.0 - dv / (e0 + et).max(f64::MIN_POSITIVE)) as f32;
            *ot = et;
            if tau == 0 {
                *ct = 1.0;
            } else {
                dsum += dv;
                *ct = if dsum > 0.0 { (dv * tau as f64 / dsum) as f32 } else { 1.0 };
            }
            let out = self.frame.get(tau).copied().unwrap_or(0.0);
            let inn = self.frame.get(tau + w).copied().unwrap_or(0.0);
            et = (et - sq(out) + sq(inn)).max(0.0);
        }
        if let (true, Some(spec_corr)) = (with_spectrum, self.spec_corr.as_mut()) {
            for ((o, &x), &h) in self.windowed.iter_mut().zip(self.frame.iter()).zip(self.hann.iter()) {
                *o = x * h;
            }
            spec_corr.power_spectrum(&self.windowed, &mut self.spec);
            let sum: f32 = self.spec.iter().map(|&p| p.max(0.0).sqrt()).sum();
            self.spec_mean_amp = sum / self.spec.len().max(1) as f32;
            self.spec_valid = true;
        }
        true
    }

    /// CMNDF troughs over `[τ_min, τ_max]` (lag order, parabolic on the raw
    /// difference d, "never cmndf", R§8.4). Heights are the CMNDF values.
    pub(crate) fn troughs(&self, out: &mut [EhTrough]) -> usize {
        let lo = self.tau_min;
        let Some(c) = self.cmndf.get(lo..=self.tau_max) else {
            return 0;
        };
        let n = find_troughs(c, lo, MAX_CMNDF_TROUGH, true, out);
        // Re-centre each trough on the parabolic minimum of d.
        for t in out.iter_mut().take(n) {
            let k = t.lag.round() as usize;
            if let (Some(&a), Some(&b), Some(&cc)) = (self.d.get(k.wrapping_sub(1)), self.d.get(k), self.d.get(k + 1))
                && k > lo
            {
                let (off, _) = vt_core::math::parabolic_peak(a, b, cc);
                t.lag = k as f32 + off;
            }
        }
        n
    }

    /// Interpolated NSDF maximum near fractional lag `tau` (±`radius` lags).
    /// Returns (lag, value); `None` outside the lag range or for a silent frame.
    pub(crate) fn nsdf_near(&self, tau: f32, radius: usize) -> Option<(f32, f32)> {
        peak_near(&self.nsdf, tau, radius, self.tau_min.max(1), self.tau_max)
    }

    /// Attack-compensated normalised correlation maximum near `tau`
    /// (`eh::attack_corr`; for the voicing-onset check, where the frame
    /// still reaches into the silence before the note).
    pub(crate) fn onset_nsdf_near(&self, tau: f32, radius: usize) -> Option<(f32, f32)> {
        if !tau.is_finite() || tau < 0.0 || self.e0 <= 0.0 {
            return None;
        }
        // The same search as `nsdf_near`, on the attack-compensated values of
        // the ±(radius + 1) lags around `tau` only.
        const R: usize = 4;
        let radius = radius.min(R - 1);
        let c = tau.round() as usize;
        let lo = c.saturating_sub(radius + 1).max(self.tau_min.max(1));
        let hi = (c + radius + 1).min(self.tau_max).min(self.d.len().checked_sub(1)?);
        if lo > hi {
            return None;
        }
        let mut v = [f32::NAN; 2 * R + 1];
        for (k, slot) in (lo..=hi).zip(v.iter_mut()) {
            let (Some(&d), Some(&et)) = (self.d.get(k), self.et_lag.get(k)) else { break };
            *slot = crate::eh::attack_corr(self.e0, et, 0.5 * (self.e0 + et - d as f64)) as f32;
        }
        let n = hi - lo + 1;
        let y = v.get(..n)?;
        // Search ±radius around c inside the window, keep one neighbour each side.
        let a = c.saturating_sub(radius).max(lo) - lo;
        let b = (c + radius).min(hi) - lo;
        if a > b {
            return None;
        }
        let mut best = a;
        for k in a..=b {
            if y[k] > y[best] {
                best = k;
            }
        }
        let (off, val) = match (best.checked_sub(1).and_then(|i| y.get(i)), y.get(best + 1)) {
            (Some(&l), Some(&r)) => vt_core::math::parabolic_peak(l, y[best], r),
            _ => (0.0, y[best]),
        };
        Some(((lo + best) as f32 + off, val.clamp(-1.0, 1.0)))
    }

    /// Prime-harmonic score of `f_norm` (= f / verifier rate, cycles per
    /// sample), or `None` when no spectrum is available or the frame cannot
    /// resolve half-harmonics of `f_norm`.
    ///
    /// `S(f) = Σ_k k^−½ [A(kf) − ½(A((k−½)f) + A((k+½)f))] / Σ_k k^−½` over
    /// k ∈ {1} ∪ primes, kf below 0.45 of the rate, with A = √(power): the
    /// SWIPE′ prime-harmonic sum with its negative half-harmonic lobes, which
    /// punish both sub-harmonic (missing odd harmonics) and octave-up (energy
    /// between the candidate's harmonics) candidates (R§8.5).
    pub(crate) fn harmonic_score(&self, f_norm: f32) -> Option<f32> {
        if !self.spec_valid || f_norm.is_nan() || f_norm <= 0.0 {
            return None;
        }
        // Hann main lobe half-width is 2/len cycles/sample; half-harmonics
        // (f/2 away) must clear it with margin.
        let len = self.frame.len() as f32;
        if 0.5 * f_norm < 2.5 / len {
            return None;
        }
        let n = self.spec_corr.as_ref()?.len() as f32;
        let amp = |f: f32| -> f32 {
            let pos = f * n;
            let i = pos.floor();
            let t = pos - i;
            let i = i as usize;
            let a = self.spec.get(i).copied().unwrap_or(0.0).max(0.0).sqrt();
            let b = self.spec.get(i + 1).copied().unwrap_or(0.0).max(0.0).sqrt();
            a + t * (b - a)
        };
        const KS: [f32; 7] = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for &k in &KS {
            if (k + 0.5) * f_norm > 0.45 {
                break;
            }
            let wgt = 1.0 / k.sqrt();
            num += wgt * (amp(k * f_norm) - 0.5 * (amp((k - 0.5) * f_norm) + amp((k + 0.5) * f_norm)));
            den += wgt;
        }
        if den <= 0.0 {
            return None;
        }
        // Normalise by the mean amplitude so scores are comparable across frames.
        Some(num / den / self.spec_mean_amp.max(1e-20))
    }
}

/// Parabolic maximum of `y` near fractional index `tau` (searching ±`radius`
/// samples, restricted to `[lo, hi]`).
fn peak_near(y: &[f32], tau: f32, radius: usize, lo: usize, hi: usize) -> Option<(f32, f32)> {
    if !tau.is_finite() || tau < 0.0 {
        return None;
    }
    let c = tau.round() as usize;
    let a = c.saturating_sub(radius).max(lo);
    let b = (c + radius).min(hi).min(y.len().checked_sub(1)?);
    if a > b {
        return None;
    }
    let mut best = a;
    for k in a..=b {
        if y[k] > y[best] {
            best = k;
        }
    }
    let (off, val) = match (y.get(best.wrapping_sub(1)), y.get(best + 1)) {
        (Some(&l), Some(&r)) if best > 0 => vt_core::math::parabolic_peak(l, y[best], r),
        _ => (0.0, y[best]),
    };
    Some((best as f32 + off, val.clamp(-1.0, 1.0)))
}

/// Short, Hann-weighted type-I frame (HQ/Max second resolution).
///
/// `d_w(τ) = Σ_{j<W} w_j (x_j − x_{j+τ})² = Σ w_j x_j² + Σ w_j x_{j+τ}² − 2 Σ w_j x_j x_{j+τ}`
/// with the weights on the head positions. The cross term is
/// `xcorr(w·x, x)` and the lagged energy `xcorr(w, x²)`: two FFT
/// correlations. Normalised like the long frame:
/// `nsdf_w = 1 − d_w / (e₀ʷ + e_τʷ)`. Tapering the head removes the edge
/// sensitivity of a rectangular window, which makes the measure more
/// consistent on vibrato and glides (McLeod's WSNAC, R§8.4).
pub(crate) struct ShortFrame {
    w: usize,
    tau_min: usize,
    tau_max: usize,
    corr: Correlator,
    frame: Box<[f32]>,
    sq: Box<[f32]>,
    wx: Box<[f32]>,
    win: Box<[f32]>,
    cross: Box<[f32]>,
    lagged: Box<[f32]>,
    ratio: Box<[f32]>,
    nsdf: Box<[f32]>,
}

impl ShortFrame {
    pub(crate) fn new(w: usize, tau_min: usize, tau_max: usize) -> Self {
        let w = w.max(8);
        let tau_max = tau_max.max(tau_min + 2);
        let len = w + tau_max + 1;
        let corr = correlation_plan(len);
        let mut win = vec![0.0f32; w].into_boxed_slice();
        // Symmetric Hann (no zero-weight padding sample wasted at the end).
        for (i, v) in win.iter_mut().enumerate() {
            let t = (i as f64 + 0.5) / w as f64;
            *v = (0.5 - 0.5 * (std::f64::consts::TAU * t).cos()) as f32;
        }
        Self {
            w,
            tau_min,
            tau_max,
            corr,
            frame: vec![0.0; len].into_boxed_slice(),
            sq: vec![0.0; len].into_boxed_slice(),
            wx: vec![0.0; w].into_boxed_slice(),
            win,
            cross: vec![0.0; tau_max + 1].into_boxed_slice(),
            lagged: vec![0.0; tau_max + 1].into_boxed_slice(),
            ratio: vec![1.0; tau_max + 1].into_boxed_slice(),
            nsdf: vec![0.0; tau_max + 1].into_boxed_slice(),
        }
    }

    pub(crate) fn span(&self) -> usize {
        self.frame.len()
    }

    pub(crate) fn window_len(&self) -> usize {
        self.w
    }

    pub(crate) fn tau_max(&self) -> usize {
        self.tau_max
    }

    /// Analyses the frame ending at stream index `end` (exclusive). Returns
    /// false for a silent head.
    pub(crate) fn analyze(&mut self, hist: &Mirror, end: i64, silence: f64) -> bool {
        let len = self.frame.len();
        hist.copy_reversed(end, len, &mut self.frame);
        let w = self.w;
        for (s, &x) in self.sq.iter_mut().zip(self.frame.iter()) {
            *s = x * x;
        }
        for ((o, &x), &h) in self.wx.iter_mut().zip(self.frame.iter()).zip(self.win.iter()) {
            *o = x * h;
        }
        let e0: f64 = self.wx.iter().zip(self.frame.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
        if e0 <= silence * 0.5 * w as f64 {
            self.ratio.fill(1.0);
            self.nsdf.fill(0.0);
            return false;
        }
        self.corr.cross_correlate(&self.wx, &self.frame, &mut self.cross);
        self.corr.cross_correlate(&self.win, &self.sq, &mut self.lagged);
        for (((r, n), &c), &et) in
            self.ratio.iter_mut().zip(self.nsdf.iter_mut()).zip(self.cross.iter()).zip(self.lagged.iter())
        {
            let et = (et as f64).max(0.0);
            let den = (e0 + et).max(f64::MIN_POSITIVE);
            let dv = (e0 + et - 2.0 * c as f64).max(0.0);
            *r = (dv / den) as f32;
            *n = 1.0 - *r;
        }
        true
    }

    /// Minima of `1 − nsdf_w` in lag order (heights on the E/H scale).
    pub(crate) fn troughs(&self, out: &mut [EhTrough]) -> usize {
        let Some(r) = self.ratio.get(self.tau_min..=self.tau_max) else {
            return 0;
        };
        // Interior only: 1 − nsdf is not cumulative-mean normalised (see eh.rs).
        find_troughs(r, self.tau_min, 0.8, false, out)
    }

    /// Interpolated weighted-NSDF maximum near `tau`.
    #[cfg(test)]
    pub(crate) fn nsdf_near(&self, tau: f32, radius: usize) -> Option<(f32, f32)> {
        peak_near(&self.nsdf, tau, radius, self.tau_min.max(1), self.tau_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hist: &mut Mirror, n: usize, period: f32, harmonics: &[(f32, f32)]) {
        for i in 0..n {
            let ph = std::f32::consts::TAU * i as f32 / period;
            let x: f32 = harmonics.iter().enumerate().map(|(k, &(a, p))| a * ((k + 1) as f32 * ph + p).sin()).sum();
            hist.push(x);
        }
    }

    #[test]
    fn correlation_plans_use_fast_sizes_only_where_they_pay() {
        // (frame length W + τ_max + 1 at 48 kHz, expected transform size)
        for (len, n) in [
            (1424, 1536), // HQ Low Male / Wide long frame (2048 before)
            (768, 768),   // HQ Low Male / Wide short frame (1024 before)
            (653, 768),   // Eco Alto/Tenor (1024 before)
            (374, 384),   // Eco Soprano (512 before)
            (560, 768),   // HQ Soprano (1024 before)
            (979, 1024),  // HQ Alto/Tenor: 768 is too short
            (950, 1024),  // Eco Low Male: 15·2^k = 960 would be slower than 1024
            (512, 512),   // HQ Alto/Tenor short frame
            (600, 768),   // Eco Alto/Tenor at 44.1 kHz: not 5·2^k = 640
        ] {
            assert_eq!(correlation_len(len), n, "len {len}");
            assert_eq!(correlation_plan(len).len(), n, "len {len}");
        }
        // The harmonic-score spectrum keeps the power-of-two size.
        let f = LongFrame::new(960, 20, 463, true);
        assert_eq!((f.corr.len(), f.spec_corr.as_ref().map(Correlator::len)), (1536, Some(2048)));
        assert!(LongFrame::new(640, 13, 309, false).spec_corr.is_none());
    }

    #[test]
    fn long_frame_finds_period_and_type_i_is_unbiased() {
        let mut hist = Mirror::new(4096);
        let period = 91.3f32;
        tone(&mut hist, 3000, period, &[(1.0, 0.0), (0.6, 0.3), (0.3, 1.1), (0.2, 2.0)]);
        let mut f = LongFrame::new(660, 20, 318, true);
        assert!(f.analyze(&hist, hist.len(), true, 1e-12));
        let mut t = [EhTrough::default(); 32];
        let n = f.troughs(&mut t);
        // First deep trough is the period (its multiples are deep too).
        let best = t[..n].iter().find(|t| t.height < 0.01).unwrap();
        assert!((best.lag - period).abs() < 0.05, "{:?}", &t[..n]);
        let (lag, v) = f.nsdf_near(91.0, 2).unwrap();
        assert!((lag - period).abs() < 0.05 && v > 0.99, "{lag} {v}");
        // Harmonic score prefers the true f over its octave and sub-octave.
        let f0 = 1.0 / period;
        let s1 = f.harmonic_score(f0).unwrap();
        let s2 = f.harmonic_score(2.0 * f0).unwrap();
        let s_half = f.harmonic_score(0.5 * f0);
        assert!(s1 > s2, "{s1} vs {s2}");
        if let Some(sh) = s_half {
            assert!(s1 > sh, "{s1} vs {sh}");
        }
        assert!(f.harmonic_score(f32::NAN).is_none());
    }

    #[test]
    fn white_noise_has_no_deep_troughs() {
        let mut hist = Mirror::new(4096);
        let mut s = 0x1234_5678u32;
        for _ in 0..3000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            hist.push((s as f32 / u32::MAX as f32) - 0.5);
        }
        let mut f = LongFrame::new(660, 20, 318, true);
        assert!(f.analyze(&hist, hist.len(), false, 1e-12));
        let mut t = [EhTrough::default(); 32];
        let n = f.troughs(&mut t);
        let min = t[..n].iter().map(|t| t.height).fold(f32::INFINITY, f32::min);
        // Type-I CMNDF on noise stays high (R§8.4: median ≈ 0.89 at long lags).
        assert!(min > 0.6, "noise trough {min}");
        assert!(f.harmonic_score(0.01).is_none(), "no spectrum requested");
    }

    #[test]
    fn silent_frame_reports_nothing() {
        let hist = Mirror::new(4096);
        let mut f = LongFrame::new(400, 10, 200, true);
        assert!(!f.analyze(&hist, 0, true, 1e-12));
        let mut t = [EhTrough::default(); 8];
        assert_eq!(f.troughs(&mut t), 0);
        let mut s = ShortFrame::new(200, 10, 120);
        assert!(!s.analyze(&hist, 0, 1e-12));
        assert_eq!(s.troughs(&mut t), 0);
    }

    #[test]
    fn short_frame_tracks_a_glide_better_than_its_rectangular_peak_drop() {
        // Glide: period 60 → 54 samples over the frame. The weighted measure
        // must still show a clear trough near the current period.
        let mut hist = Mirror::new(4096);
        let mut ph = 0.0f32;
        for i in 0..2000 {
            let p = 60.0 - 6.0 * (i as f32 / 2000.0);
            ph += std::f32::consts::TAU / p;
            hist.push(ph.sin() + 0.5 * (2.0 * ph).sin());
        }
        let mut s = ShortFrame::new(330, 15, 198);
        assert!(s.analyze(&hist, hist.len(), 1e-12));
        let mut t = [EhTrough::default(); 16];
        let n = s.troughs(&mut t);
        let best = t[..n].iter().min_by(|a, b| a.height.total_cmp(&b.height)).unwrap();
        assert!((best.lag - 54.3).abs() < 1.0, "{:?}", &t[..n]);
        let (_, v) = s.nsdf_near(best.lag, 1).unwrap();
        assert!(v > 0.9, "{v}");
    }
}
