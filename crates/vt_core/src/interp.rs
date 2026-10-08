//! Fractional-position reads from a [`Ring`] (R§8.16).
//!
//! PSOLA grains, the Classic read pointer and formant resampling all read the
//! input at fractional absolute positions (`f64`, never `f32`: at 48 kHz an f32
//! position loses sub-sample resolution after ~6 minutes). Two families:
//!
//! * [`cubic`]: 4-point cubic Hermite (Catmull-Rom). Cheap, used in Eco.
//! * [`SincBank`]: Kaiser-windowed sinc, polyphase tables with linear
//!   interpolation between adjacent phases. Several cutoffs are tabulated so a
//!   read that advances `rate > 1` source samples per output sample (formant
//!   read with γ > 1, Classic shifting up) is low-passed to the output Nyquist
//!   (R§8.16 "when γ > 1, low-pass the source by γ").
//!
//! ### Measured accuracy
//!
//! Worst absolute error (dB re. full scale) reading unit sines at random
//! fractional positions, maximised over every frequency in the band
//! 0 < f ≤ f_max (dense sweep):
//!
//! | kernel | f_max = 0.1 fs | 0.2 fs | 0.25 fs | 0.3 fs | 0.35 fs | 0.4 fs |
//! |---|---|---|---|---|---|---|
//! | Sinc16 (β 9.25, 512 phases) | −90 dB | −86 dB | −86 dB | −83 dB | −44 dB | −21 dB |
//! | Sinc8 (β 5.0, 256 phases)   | −53 dB | −53 dB | −48 dB | −47 dB | −23 dB | −12 dB |
//! | Cubic                       | −47 dB | −26 dB | −19 dB | −13 dB | −9 dB  | −5 dB  |
//!
//! No 16-tap FIR can do much better near 0.4 fs: an optimised (iteratively
//! reweighted least-squares minimax) 16-tap fractional-delay design reaches
//! only −51 dB at 0.4 fs, and −80 dB there needs ≈ 26 taps. β is therefore
//! chosen to maximise accuracy over the band that carries vocal energy
//! (≤ 0.3 fs = 14.4 kHz at 48 kHz) and lets the response roll off above it,
//! like any short interpolator.

use crate::ring::Ring;
use crate::window::kaiser_at;

/// Interpolation kernel choice for the engines (per quality tier).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpQuality {
    /// 4-point cubic Hermite (no anti-aliasing for `rate > 1`).
    Cubic,
    /// 8-tap Kaiser-windowed sinc.
    Sinc8,
    /// 16-tap Kaiser-windowed sinc.
    Sinc16,
}

/// Splits a fractional absolute position into (floor index, fraction in [0, 1)).
/// Returns `None` for NaN/Inf positions.
#[inline]
fn split_pos(pos: f64) -> Option<(i64, f32)> {
    if !pos.is_finite() {
        return None;
    }
    let fl = pos.floor();
    // `as` saturates; positions are sample counts far below 2^63.
    Some((fl as i64, (pos - fl) as f32))
}

/// 4-point cubic Hermite (Catmull-Rom) read at fractional absolute `pos`.
///
/// Uses samples `floor(pos) − 1 ..= floor(pos) + 2`, so it needs 2 samples of
/// lookahead past `floor(pos)`. Unavailable samples read as 0; a NaN/Inf
/// position returns 0.
#[inline]
pub fn cubic(ring: &Ring, pos: f64) -> f32 {
    let Some((i, t)) = split_pos(pos) else {
        return 0.0;
    };
    let [xm1, x0, x1, x2] = ring.fetch::<4>(i.saturating_sub(1));
    let c1 = 0.5 * (x1 - xm1);
    let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
    let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
    ((c3 * t + c2) * t + c1) * t + x0
}

/// Number of half-octave-ish cutoff tables: cutoff `0.5·2^(−k/4)·fs` for
/// k = 0..9, i.e. anti-aliased reads up to `rate = 4` in quarter-octave steps.
const RATE_TABLES: usize = 9;

/// Kaiser β per kernel length, chosen numerically (see module docs): best
/// worst-case error over f ≤ 0.3·fs.
fn beta_for_taps(taps: usize) -> f64 {
    match taps {
        8 => 5.0,
        16 => 9.25,
        // Interpolated rule for other lengths (≈ the two tuned points).
        t => 0.53 * t as f64 + 0.75,
    }
}

/// Kaiser-windowed polyphase sinc tables, built in `new()` and read without
/// allocation.
///
/// For a read at `pos = i + t` the kernel taps are samples
/// `i − taps/2 + 1 ..= i + taps/2` (so `half_width()` samples of lookahead past
/// `floor(pos)`). Phases are tabulated at `t = p / phases` for p = 0..=phases
/// and linearly interpolated, which adds < −110 dB of error at 512 phases.
/// Every phase row is normalised to unit DC gain, so a constant input reads
/// back exactly constant at every fractional position (no amplitude
/// modulation at the grain rate).
#[derive(Clone)]
pub struct SincBank {
    taps: usize,
    phases: usize,
    /// `RATE_TABLES` tables of `(phases + 1) × taps` coefficients, tap-major
    /// within a phase row.
    coefs: Box<[f32]>,
}

impl core::fmt::Debug for SincBank {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The tables themselves are hundreds of KB; show the shape only.
        f.debug_struct("SincBank").field("taps", &self.taps).field("phases", &self.phases).finish_non_exhaustive()
    }
}

impl SincBank {
    /// Builds the tables. `taps` should be 8 or 16 (any even value in 2..=64 is
    /// accepted; others are rounded/clamped into that range) and `phases`
    /// ≥ 256 (smaller values are raised to 256).
    pub fn new(taps: usize, phases: usize) -> Self {
        let taps = (taps.clamp(2, 64) + 1) & !1;
        let phases = phases.max(256);
        let beta = beta_for_taps(taps);
        let half = (taps / 2) as f64;
        let row = taps;
        let table_len = (phases + 1) * row;
        let mut coefs = vec![0.0f32; RATE_TABLES * table_len].into_boxed_slice();
        let mut h = vec![0.0f64; taps];
        for k in 0..RATE_TABLES {
            // Cutoff as a fraction of fs: 0.5 (full band) for k = 0, then
            // quarter-octave steps down for faster reads.
            let fc = 0.5 * (-(k as f64) / 4.0).exp2();
            for p in 0..=phases {
                let t = p as f64 / phases as f64;
                for (j, hj) in h.iter_mut().enumerate() {
                    // Tap j sits at u = (j − taps/2 + 1) − t samples from pos.
                    let u = (j as f64 - half + 1.0) - t;
                    let x = 2.0 * fc * u;
                    let sinc = if x.abs() < 1e-12 {
                        1.0
                    } else {
                        (core::f64::consts::PI * x).sin() / (core::f64::consts::PI * x)
                    };
                    *hj = 2.0 * fc * sinc * kaiser_at(u / half, beta);
                }
                let sum: f64 = h.iter().sum();
                let base = k * table_len + p * row;
                for (dst, &hj) in coefs[base..base + row].iter_mut().zip(&h) {
                    *dst = (hj / sum) as f32;
                }
            }
        }
        Self { taps, phases, coefs }
    }

    /// Kernel length in samples.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Phase resolution of the tables.
    pub fn phases(&self) -> usize {
        self.phases
    }

    /// `taps / 2`: samples of lookahead needed past `floor(pos)` (and the read
    /// also reaches `taps/2 − 1` samples behind it).
    pub fn half_width(&self) -> usize {
        self.taps / 2
    }

    /// Table index for a read rate: `round(4·log2(rate))`, clamped to the
    /// tabulated range. Rates ≤ 1 (and NaN) use the full-band table.
    ///
    /// Counted against the exact rounding thresholds 2^((k+½)/4) rather than
    /// computing a logarithm: eight independent compares are cheaper than a
    /// `fast_log2` (whose division would sit on the coefficient-address path).
    #[inline]
    fn table_for_rate(rate: f32) -> usize {
        const THRESH: [f32; RATE_TABLES - 1] =
            [1.090_507_7, 1.296_839_6, 1.542_210_8, 1.834_008_1, 2.181_015_5, 2.593_679_2, 3.084_421_6, 3.668_016_2];
        // NaN compares false everywhere → table 0.
        THRESH.iter().map(|&t| usize::from(rate >= t)).sum()
    }

    /// Read at `pos` with a low-pass appropriate for reading at `rate` source
    /// samples per output sample (rate ≤ 1.0 → full band). Unavailable samples
    /// read as 0; NaN/Inf `pos` returns 0.
    #[inline]
    pub fn read(&self, ring: &Ring, pos: f64, rate: f32) -> f32 {
        match self.taps {
            8 => self.read_n::<8>(ring, pos, rate),
            16 => self.read_n::<16>(ring, pos, rate),
            _ => self.read_generic(ring, pos, rate),
        }
    }

    /// Two adjacent phase rows and the blend fraction for fractional part `t`.
    #[inline]
    fn rows(&self, t: f32, rate: f32) -> (usize, f32) {
        let ph = t * self.phases as f32;
        // t can round to 1.0; row `phases` exists, so clamp p0 to phases − 1
        // and let the blend reach it.
        let p0 = (ph as usize).min(self.phases - 1);
        let frac = ph - p0 as f32;
        let base = Self::table_for_rate(rate) * (self.phases + 1) * self.taps + p0 * self.taps;
        (base, frac)
    }

    #[inline]
    fn read_n<const T: usize>(&self, ring: &Ring, pos: f64, rate: f32) -> f32 {
        let Some((i, t)) = split_pos(pos) else {
            return 0.0;
        };
        let x: [f32; T] = ring.fetch::<T>(i.saturating_sub(T as i64 / 2 - 1));
        let (base, frac) = self.rows(t, rate);
        let Some(rows) = self.coefs.get(base..base + 2 * T) else {
            return 0.0;
        };
        let (r0, r1) = rows.split_at(T);
        let (Ok(r0), Ok(r1)) = (<&[f32; T]>::try_from(r0), <&[f32; T]>::try_from(r1)) else {
            return 0.0;
        };
        // Four partial sums per row so the dot products vectorise (fixed T
        // lets LLVM fully unroll into NEON/SSE lanes).
        let mut a0 = [0.0f32; 4];
        let mut a1 = [0.0f32; 4];
        for k in 0..T {
            a0[k % 4] += x[k] * r0[k];
            a1[k % 4] += x[k] * r1[k];
        }
        let s0 = (a0[0] + a0[1]) + (a0[2] + a0[3]);
        let s1 = (a1[0] + a1[1]) + (a1[2] + a1[3]);
        s0 + frac * (s1 - s0)
    }

    fn read_generic(&self, ring: &Ring, pos: f64, rate: f32) -> f32 {
        let Some((i, t)) = split_pos(pos) else {
            return 0.0;
        };
        let taps = self.taps;
        let start = i.saturating_sub(taps as i64 / 2 - 1);
        let (base, frac) = self.rows(t, rate);
        let Some(rows) = self.coefs.get(base..base + 2 * taps) else {
            return 0.0;
        };
        let (r0, r1) = rows.split_at(taps);
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        for (k, (&c0, &c1)) in r0.iter().zip(r1).enumerate() {
            let x = ring.get_signed(start.saturating_add(k as i64));
            s0 += x * c0;
            s1 += x * c1;
        }
        s0 + frac * (s1 - s0)
    }
}

/// Convenience dispatcher used by the engines: owns one 8-tap and one 16-tap
/// bank (≈ 370 KB of tables in total) and routes reads by [`InterpQuality`].
#[derive(Clone, Debug)]
pub struct Interpolator {
    sinc8: SincBank,
    sinc16: SincBank,
}

impl Default for Interpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpolator {
    /// Builds the Sinc8 (256 phases) and Sinc16 (512 phases) banks.
    pub fn new() -> Self {
        Self { sinc8: SincBank::new(8, 256), sinc16: SincBank::new(16, 512) }
    }

    /// Fractional read with kernel `q`. `rate` (source samples advanced per
    /// output sample) selects the anti-alias cutoff for the sinc kernels and
    /// is ignored by `Cubic`.
    #[inline]
    pub fn read(&self, q: InterpQuality, ring: &Ring, pos: f64, rate: f32) -> f32 {
        match q {
            InterpQuality::Cubic => cubic(ring, pos),
            InterpQuality::Sinc8 => self.sinc8.read(ring, pos, rate),
            InterpQuality::Sinc16 => self.sinc16.read(ring, pos, rate),
        }
    }

    /// Samples of lookahead past `floor(pos)` that a read with `q` needs
    /// (the newest written sample must be ≥ `floor(pos) + half_width(q)`).
    pub fn half_width(q: InterpQuality) -> usize {
        match q {
            InterpQuality::Cubic => 2,
            InterpQuality::Sinc8 => 4,
            InterpQuality::Sinc16 => 8,
        }
    }

    /// The 8-tap bank.
    pub fn sinc8(&self) -> &SincBank {
        &self.sinc8
    }

    /// The 16-tap bank.
    pub fn sinc16(&self) -> &SincBank {
        &self.sinc16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift for test positions.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Ring filled with `sin(2π f n + φ)` (f in cycles/sample), computed in f64.
    fn sine_ring(f: f64, phi: f64, len: usize) -> Ring {
        let mut r = Ring::new(len);
        for n in 0..len {
            r.push((2.0 * core::f64::consts::PI * f * n as f64 + phi).sin() as f32);
        }
        r
    }

    /// Worst-case absolute error (dB re. unit amplitude) of `read` on sines at
    /// frequency `f` over many random fractional positions.
    fn worst_error_db(f: f64, read: &dyn Fn(&Ring, f64) -> f32) -> f64 {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut worst = 0.0f64;
        for trial in 0..4 {
            let phi = trial as f64 * 0.7;
            let ring = sine_ring(f, phi, 4096);
            for _ in 0..2000 {
                let pos = 100.0 + rng.next_f64() * 3800.0;
                let exact = (2.0 * core::f64::consts::PI * f * pos + phi).sin();
                worst = worst.max((read(&ring, pos) as f64 - exact).abs());
            }
        }
        20.0 * worst.max(1e-12).log10()
    }

    #[test]
    fn sinc16_accuracy_band_limited_sines() {
        let it = Interpolator::new();
        let read = |r: &Ring, p: f64| it.read(InterpQuality::Sinc16, r, p, 1.0);
        // Dense sweep over the band (the worst case sits between grid points
        // of a coarse sweep).
        for i in 1..=60 {
            let f = i as f64 * 0.005;
            let e = worst_error_db(f, &read);
            assert!(e < -80.0, "Sinc16 error at f={f}·fs: {e:.1} dB");
        }
        // Above 0.3 fs a 16-tap kernel must roll off (see module docs); assert
        // the documented behaviour so regressions are caught.
        let e35 = worst_error_db(0.35, &read);
        let e40 = worst_error_db(0.4, &read);
        assert!(e35 < -40.0, "Sinc16 at 0.35 fs: {e35:.1} dB");
        assert!(e40 < -18.0, "Sinc16 at 0.4 fs: {e40:.1} dB");
    }

    #[test]
    fn sinc8_accuracy_band_limited_sines() {
        let it = Interpolator::new();
        let read = |r: &Ring, p: f64| it.read(InterpQuality::Sinc8, r, p, 1.0);
        for i in 1..=60 {
            let f = i as f64 * 0.005;
            let bound = if f <= 0.2 { -50.0 } else { -45.0 };
            let e = worst_error_db(f, &read);
            assert!(e < bound, "Sinc8 error at f={f}·fs: {e:.1} dB");
        }
    }

    #[test]
    fn cubic_accuracy_and_exactness() {
        // Cubic Hermite is exact at integer positions and accurate for low f.
        let ring = sine_ring(0.01, 0.3, 1024);
        for i in 10..1000u64 {
            assert_eq!(cubic(&ring, i as f64), ring.get(i));
        }
        let e = worst_error_db(0.01, &|r, p| cubic(r, p));
        assert!(e < -70.0, "cubic at 0.01 fs: {e:.1} dB");
        let e = worst_error_db(0.1, &|r, p| cubic(r, p));
        assert!(e < -35.0, "cubic at 0.1 fs: {e:.1} dB");
        // Reproduces quadratics exactly (Catmull-Rom has 2nd-order accuracy).
        let mut q = Ring::new(64);
        for n in 0..64 {
            q.push(0.01 * (n as f32) * (n as f32) - 0.3 * n as f32 + 1.0);
        }
        for k in 0..100 {
            let p = 5.0 + k as f64 * 0.37;
            let exact = 0.01 * p * p - 0.3 * p + 1.0;
            assert!((cubic(&q, p) as f64 - exact).abs() < 2e-5, "p={p}");
        }
    }

    #[test]
    fn sinc_constant_input_reads_constant() {
        let mut r = Ring::new(256);
        for _ in 0..256 {
            r.push(0.75);
        }
        let it = Interpolator::new();
        for q in [InterpQuality::Sinc8, InterpQuality::Sinc16] {
            for rate in [0.5, 1.0, 1.3, 2.0, 3.9, 10.0] {
                for k in 0..500 {
                    let p = 50.0 + k as f64 * 0.311;
                    let y = it.read(q, &r, p, rate);
                    assert!((y - 0.75).abs() < 2e-6, "{q:?} rate={rate} p={p} y={y}");
                }
            }
        }
    }

    #[test]
    fn sinc_exact_at_integer_positions() {
        let ring = sine_ring(0.123, 0.0, 512);
        let it = Interpolator::new();
        for i in 20..480u64 {
            for q in [InterpQuality::Sinc8, InterpQuality::Sinc16] {
                let y = it.read(q, &ring, i as f64, 1.0);
                assert!((y - ring.get(i)).abs() < 1e-6, "{q:?} i={i}");
            }
        }
    }

    #[test]
    fn rate_tables_attenuate_aliasing_band() {
        // Reading at rate 2 halves the output Nyquist: a tone at 0.4 fs (above
        // the new Nyquist of 0.25 fs) must be strongly attenuated, a tone at
        // 0.05 fs must pass.
        let it = Interpolator::new();
        let rms = |f: f64, rate: f32| {
            let ring = sine_ring(f, 0.1, 4096);
            let mut acc = 0.0f64;
            let n = 1500;
            for k in 0..n {
                let p = 200.0 + k as f64 * 2.0 + 0.37;
                let y = it.read(InterpQuality::Sinc16, &ring, p, rate) as f64;
                acc += y * y;
            }
            (acc / n as f64).sqrt() * core::f64::consts::SQRT_2
        };
        assert!(rms(0.05, 2.0) > 0.95);
        let stop = 20.0 * rms(0.4, 2.0).log10();
        assert!(stop < -40.0, "aliasing band at rate 2: {stop:.1} dB");
        // Full-band table (rate 1) passes 0.2 fs, rate-2 table removes it.
        assert!(rms(0.2, 1.0) > 0.99);
        assert_eq!(SincBank::table_for_rate(1.0), 0);
        assert_eq!(SincBank::table_for_rate(0.3), 0);
        assert_eq!(SincBank::table_for_rate(f32::NAN), 0);
        assert_eq!(SincBank::table_for_rate(2.0), 4);
        assert_eq!(SincBank::table_for_rate(4.0), 8);
        assert_eq!(SincBank::table_for_rate(100.0), 8);
        assert_eq!(SincBank::table_for_rate(f32::INFINITY), 8);
        // The thresholds implement round(4·log2(rate)) exactly.
        for i in 0..4000 {
            let rate = 1.0 + i as f32 * 0.001;
            let expect = ((4.0 * (rate as f64).log2() + 0.5).floor() as usize).min(RATE_TABLES - 1);
            let got = SincBank::table_for_rate(rate);
            // Allow disagreement only within float rounding of a threshold.
            let near =
                (4.0 * (rate as f64).log2() + 0.5).fract().min(1.0 - (4.0 * (rate as f64).log2() + 0.5).fract()) < 1e-5;
            assert!(got == expect || near, "rate={rate}: {got} vs {expect}");
        }
    }

    #[test]
    fn lookahead_and_missing_samples() {
        assert_eq!(Interpolator::half_width(InterpQuality::Cubic), 2);
        assert_eq!(Interpolator::half_width(InterpQuality::Sinc8), 4);
        assert_eq!(Interpolator::half_width(InterpQuality::Sinc16), 8);
        let it = Interpolator::new();
        assert_eq!(it.sinc16().half_width(), 8);
        assert_eq!(it.sinc8().taps(), 8);
        // A read whose taps end exactly at the newest sample is identical to a
        // read with more data available (no hidden dependence on the future).
        let mut a = Ring::new(1024);
        let mut b = Ring::new(1024);
        for n in 0..600 {
            let x = (n as f32 * 0.21).sin();
            a.push(x);
            b.push(x);
        }
        for n in 600..700 {
            b.push((n as f32 * 0.21).sin());
        }
        let newest = a.written() - 1;
        for q in [InterpQuality::Cubic, InterpQuality::Sinc8, InterpQuality::Sinc16] {
            let hw = Interpolator::half_width(q) as f64;
            let p = newest as f64 - hw + 0.999;
            assert_eq!(it.read(q, &a, p, 1.0), it.read(q, &b, p, 1.0), "{q:?}");
        }
        // NaN / Inf / negative positions are safe.
        for q in [InterpQuality::Cubic, InterpQuality::Sinc8, InterpQuality::Sinc16] {
            assert_eq!(it.read(q, &a, f64::NAN, 1.0), 0.0);
            assert_eq!(it.read(q, &a, f64::INFINITY, 1.0), 0.0);
            assert!(it.read(q, &a, -3.5, 1.0).is_finite());
            assert!(it.read(q, &a, 1e30, f32::NAN).is_finite());
        }
    }

    #[test]
    fn generic_tap_counts() {
        let bank = SincBank::new(12, 256);
        assert_eq!(bank.taps(), 12);
        let e = worst_error_db(0.05, &|r, p| bank.read(r, p, 1.0));
        assert!(e < -50.0, "12-tap: {e:.1} dB");
        assert_eq!(SincBank::new(7, 10).taps(), 8);
        assert_eq!(SincBank::new(7, 10).phases(), 256);
        assert_eq!(SincBank::new(1000, 256).taps(), 64);
    }
}
