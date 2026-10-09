//! Shared per-sample helpers for the engines: the fractional reader (tier
//! kernel, crossfaded tier switches, exact integer reads, causality counters)
//! and the raised-cosine table used for grain windows and fades.

use skato_core::interp::{InterpQuality, Interpolator};
use skato_core::ring::Ring;

/// Interpolation kernel for a quality tier (R§7.2.2 "Shifter interpolation"):
/// Eco reads with 4-point cubic Hermite, HQ and Max with the 16-tap
/// Kaiser-windowed sinc.
pub(crate) fn quality_for(tier: skato_core::QualityTier) -> InterpQuality {
    match tier {
        skato_core::QualityTier::Eco => InterpQuality::Cubic,
        skato_core::QualityTier::Hq | skato_core::QualityTier::Max => InterpQuality::Sinc16,
    }
}

/// Counters proving the engines never read input that has not been written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadStats {
    /// Fractional reads performed.
    pub reads: u64,
    /// Reads whose kernel reached past the newest written sample.
    pub past_newest: u64,
    /// Largest `floor(pos) + half_width − newest` seen (≤ 0 when causal).
    pub max_ahead: i64,
}

impl Default for ReadStats {
    fn default() -> Self {
        Self { reads: 0, past_newest: 0, max_ahead: i64::MIN }
    }
}

impl ReadStats {
    /// Records one read at `pos` of kernel half-width `hw` when `newest` is
    /// the newest written absolute index.
    #[inline]
    pub fn note(&mut self, pos: f64, hw: usize, newest: u64) {
        // `as` saturates; positions are far below 2^63.
        let ahead = pos.floor() as i64 + hw as i64 - newest as i64;
        self.reads += 1;
        if ahead > self.max_ahead {
            self.max_ahead = ahead;
        }
        if ahead > 0 {
            self.past_newest += 1;
        }
        debug_assert!(ahead <= 0, "read {ahead} samples past the newest input (pos {pos}, newest {newest})");
    }

    /// Folds another counter set into this one.
    pub fn merge(&mut self, o: &ReadStats) {
        self.reads += o.reads;
        self.past_newest += o.past_newest;
        self.max_ahead = self.max_ahead.max(o.max_ahead);
    }
}

/// Fractional reader handed to the engines for one output sample.
///
/// During a tier switch both kernels are read and linearly crossfaded
/// (R§7.2.2: "switching tiers crossfades over one hop"), so the change of
/// interpolation error never shows up as a step.
#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    pub interp: &'a Interpolator,
    /// Current kernel.
    pub q: InterpQuality,
    /// Kernel being faded out.
    pub q_prev: InterpQuality,
    /// Weight of `q` (1.0 = no crossfade in progress).
    pub blend: f32,
}

impl Reader<'_> {
    /// Largest half-width of the kernels this reader may use for a read.
    #[inline]
    pub fn half_width(&self) -> usize {
        let a = Interpolator::half_width(self.q);
        if self.blend >= 1.0 { a } else { a.max(Interpolator::half_width(self.q_prev)) }
    }

    /// Reads `ring` at fractional absolute `pos`, low-passed for a read rate
    /// of `rate` source samples per output sample. Integer positions read at
    /// rate ≤ 1 return the stored sample exactly (bit-exact delays).
    #[inline]
    pub fn read(&self, ring: &Ring, pos: f64, rate: f32) -> f32 {
        if rate <= 1.0 && pos >= 0.0 && pos.fract() == 0.0 {
            return ring.get(pos as u64);
        }
        let a = self.interp.read(self.q, ring, pos, rate);
        if self.blend >= 1.0 {
            a
        } else {
            let b = self.interp.read(self.q_prev, ring, pos, rate);
            b + self.blend * (a - b)
        }
    }
}

/// Number of table intervals of the raised-cosine table.
const RC_N: usize = 1024;

/// `0.5 − 0.5·cos(π x)` on x ∈ [0, 1], tabulated (1025 points, linearly
/// interpolated: max error 6e-7, ≈ −124 dB). Grain windows and fades are
/// evaluated once per output sample per grain, so the table replaces a `cos`
/// call in the hot loop (P§3.4.2).
pub(crate) struct RcTable {
    t: Box<[f32]>,
}

impl RcTable {
    pub fn new() -> Self {
        let t: Vec<f32> =
            (0..=RC_N).map(|i| (0.5 - 0.5 * (core::f64::consts::PI * i as f64 / RC_N as f64).cos()) as f32).collect();
        Self { t: t.into_boxed_slice() }
    }

    /// Equal-gain raised-cosine fade-in at `x` (clamped to [0, 1], NaN → 0).
    /// `rc(x) + (1 − rc(x)) = 1` is used for complementary halves; the table
    /// is also symmetric, `rc(1 − x) = 1 − rc(x)` to within 6e-7.
    #[inline]
    pub fn rc(&self, x: f64) -> f32 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let f = x * RC_N as f64;
        let i = (f as usize).min(RC_N - 1);
        let a = (f - i as f64) as f32;
        let (Some(&y0), Some(&y1)) = (self.t.get(i), self.t.get(i + 1)) else {
            return if x >= 1.0 { 1.0 } else { 0.0 };
        };
        y0 + a * (y1 - y0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc_table_accuracy_and_symmetry() {
        let t = RcTable::new();
        assert_eq!(t.rc(0.0), 0.0);
        assert_eq!(t.rc(1.0), 1.0);
        assert_eq!(t.rc(-1.0), 0.0);
        assert_eq!(t.rc(2.0), 1.0);
        assert_eq!(t.rc(f64::NAN), 0.0);
        let mut worst = 0.0f64;
        for i in 0..=100_000 {
            let x = i as f64 / 100_000.0;
            let exact = 0.5 - 0.5 * (core::f64::consts::PI * x).cos();
            worst = worst.max((t.rc(x) as f64 - exact).abs());
            assert!((t.rc(x) + t.rc(1.0 - x) - 1.0).abs() < 2e-6);
        }
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn reader_integer_positions_are_exact() {
        let interp = Interpolator::new();
        let mut ring = Ring::new(256);
        for i in 0..256 {
            ring.push(((i * 7919) % 101) as f32 * 0.013 - 0.6);
        }
        for q in [InterpQuality::Cubic, InterpQuality::Sinc8, InterpQuality::Sinc16] {
            let r = Reader { interp: &interp, q, q_prev: InterpQuality::Cubic, blend: 0.3 };
            for i in 20..200u64 {
                assert_eq!(r.read(&ring, i as f64, 1.0), ring.get(i));
            }
        }
    }

    #[test]
    fn reader_blends_kernels() {
        let interp = Interpolator::new();
        let mut ring = Ring::new(256);
        for i in 0..256 {
            ring.push((i as f32 * 0.9).sin());
        }
        let pos = 100.37;
        let a = interp.read(InterpQuality::Sinc16, &ring, pos, 1.0);
        let b = interp.read(InterpQuality::Cubic, &ring, pos, 1.0);
        let r = Reader { interp: &interp, q: InterpQuality::Sinc16, q_prev: InterpQuality::Cubic, blend: 0.25 };
        assert!((r.read(&ring, pos, 1.0) - (b + 0.25 * (a - b))).abs() < 1e-7);
        assert_eq!(r.half_width(), 8);
        let r1 = Reader { blend: 1.0, ..r };
        assert_eq!(r1.read(&ring, pos, 1.0), a);
    }

    #[test]
    fn read_stats_track_lookahead() {
        let mut s = ReadStats::default();
        s.note(91.5, 8, 100);
        s.note(92.0, 8, 100);
        assert_eq!(s.max_ahead, 0);
        assert_eq!(s.past_newest, 0);
        assert_eq!(s.reads, 2);
    }
}
