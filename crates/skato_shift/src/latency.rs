//! Reported latency D (R§7.2.1) and the timing constants it is built from.
//!
//! D is the single number the host compensates. Every engine in this crate
//! derives its scheduling limits from the same constants, so the formula and
//! the engines cannot drift apart:
//!
//! * **Live**: `D = L_rise + d_mark(T_max) + h_interp + margin`, where
//!   - `L_rise` (1 ms) is the rise of the asymmetric short-rise grain (R§8.15),
//!   - `d_mark(T) = 0.2·T + 2` is how long after a pitch mark the NCC mark
//!     tracker can publish it (search reach 0.1·T past the predicted mark plus
//!     the forward window W_f = 0.1·T, R§8.13, plus the parabolic neighbour and
//!     the rounding of the trigger index),
//!   - `h_interp` is the half-kernel of the largest interpolator any tier uses
//!     (Sinc16, 8 samples), so a tier switch never changes D,
//!   - `margin = 0.05·T_max + 0.1 ms` absorbs the period change from one
//!     period to the next (vibrato moves the period by 2-4 % per period), so the
//!     asymmetric grain's lazy boundary decision still sees the next mark.
//!
//!   At 48 kHz this gives 3.10 ms (Soprano), 4.44 ms (Alto/Tenor) and
//!   5.85 ms (Low Male, Wide), in line with the R§7.2.1 estimates (44.1 kHz:
//!   3.13 / 4.47 / 5.87 ms).
//! * **Live+**: the user's 5-10 ms, never below the Live value for the range.
//! * **Studio**: the user's 20-50 ms, never below the Live value, nor below
//!   `T_max + d_mark(T_max) + h_interp + margin`, the smallest D at which a
//!   symmetric 2·T grain fits for the lowest note of the range (the
//!   range-dependent Studio minimum of R§7.2.1; it only binds for Low Male and
//!   Wide below 23 ms).
//! * **Live-Min**: 1 ms. The Classic engine needs only the interpolation
//!   half-kernel plus room for its splice crossfade, and shortens the crossfade
//!   when a large upward ratio would otherwise overtake the input.
//!
//! **Heard latency.** D is the compensated latency. The Natural engine's
//! content delay wanders around it by up to a period (R§2.1), measured per
//! grain under a constant +35 c correction (Alto/Tenor, 48 kHz, mean / p95):
//!
//! | mode (D) | 110 Hz | 220 Hz | 440 Hz |
//! |---|---|---|---|
//! | Live (4.44 ms) | 8.7 / 12.8 ms | 6.5 / 8.5 ms | 4.4 / 5.4 ms |
//! | Live+ (10 ms) | 14.3 / 18.4 ms | 10.0 / 12.1 ms | 10.0 / 11.0 ms |
//! | Studio (40 ms) | 40.0 / 44.2 ms | 40.0 / 42.1 ms | 40.0 / 41.0 ms |
//!
//! Live (and Live+ below the pitch whose half period fits its look-ahead)
//! averages D + ≈ 0.45·T because grains can only use marks already
//! detected (see `psola::asym_rise` for the measured trade-off between that
//! and grain quality, DSP-9); Studio and the rest of Live+ are centred on D.
//! On the vocadito recordings (40 tracks, Alto/Tenor at 44.1 kHz) the mean
//! is 5.95 ms for Live (D 4.47), 10.3 ms for Live+ and 40.0 ms for Studio.
//!
//! The engines never rely on D being "large enough": every grain-source choice
//! and every Classic splice is checked against the newest written sample, so a
//! too-small D only costs content delay, never a read of unwritten input.

use skato_core::interp::{InterpQuality, Interpolator};
use skato_core::{LatencyMode, VoiceRange};

/// Rise (and complementary fall) of the asymmetric short-rise grain, ms
/// (R§8.15: 1-1.5 ms; the low end keeps Live latency down).
pub(crate) const RISE_MS: f64 = 1.0;

/// Half-width of the largest interpolation kernel used by any tier (Sinc16).
pub(crate) const MAX_HALF_WIDTH: usize = 8;

/// Locked mark search half-range, as a fraction of the period (R§8.13:
/// τ ∈ [0.9·T, 1.1·T]).
pub(crate) const SEARCH_LOCKED: f64 = 0.10;

/// Widened search half-range at onsets and after a weak period (R§8.13).
pub(crate) const SEARCH_WIDE: f64 = 0.20;

/// NCC window before the mark, as a fraction of the period (R§8.13: W_b =
/// 0.5-0.75·T).
pub(crate) const WB_FRAC: f64 = 0.6;

/// NCC window after the mark, as a fraction of the period (R§8.13: W_f =
/// 0.1·T). Sets the mark-detection delay together with `SEARCH_LOCKED`.
pub(crate) const WF_FRAC: f64 = 0.1;

/// Live-Min reported latency, ms (R§7.2.1: ≈ 1 ms).
const LIVE_MIN_MS: f64 = 1.0;

/// Sample rates outside this range (or non-finite) fall back to 48 kHz for the
/// purpose of sizing; the plugin never runs outside it.
pub(crate) fn sanitize_rate(sample_rate: f32) -> f64 {
    if sample_rate.is_finite() && (8_000.0..=768_000.0).contains(&sample_rate) { sample_rate as f64 } else { 48_000.0 }
}

/// Longest period of the voice range in samples (`fs / f_min`).
pub(crate) fn t_max(sr: f64, range: VoiceRange) -> f64 {
    sr / range.f_min() as f64
}

/// Shortest period of the voice range in samples (`fs / f_max`).
pub(crate) fn t_min(sr: f64, range: VoiceRange) -> f64 {
    sr / range.f_max() as f64
}

/// Mark-detection delay for period `t` (samples): the search reaches
/// `SEARCH_LOCKED·T` past the predicted mark and needs `WF_FRAC·T` after it,
/// plus one sample for the parabolic neighbour and one for rounding.
#[inline]
pub(crate) fn mark_delay(t: f64) -> f64 {
    (SEARCH_LOCKED + WF_FRAC) * t + 2.0
}

/// Rise length of the asymmetric grain in samples.
pub(crate) fn rise_samples(sr: f64) -> f64 {
    RISE_MS * 1e-3 * sr
}

/// Safety margin of the Live formula (see module docs).
fn margin(sr: f64, t_max: f64) -> f64 {
    0.05 * t_max + 1e-4 * sr
}

/// Live D in samples (fractional, before rounding up).
fn live(sr: f64, range: VoiceRange) -> f64 {
    let tm = t_max(sr, range);
    debug_assert_eq!(Interpolator::half_width(InterpQuality::Sinc16), MAX_HALF_WIDTH);
    rise_samples(sr) + mark_delay(tm) + MAX_HALF_WIDTH as f64 + margin(sr, tm)
}

/// Smallest Studio D at which a symmetric 2·T grain fits for the lowest note.
fn studio_min(sr: f64, range: VoiceRange) -> f64 {
    let tm = t_max(sr, range);
    tm + mark_delay(tm) + MAX_HALF_WIDTH as f64 + margin(sr, tm)
}

/// Milliseconds → samples, with a non-finite value replaced by `default_ms`
/// and the result clamped to `[lo, hi]` ms.
fn ms_param(sr: f64, ms: f32, default_ms: f64, lo: f64, hi: f64) -> f64 {
    let ms = if ms.is_finite() { (ms as f64).clamp(lo, hi) } else { default_ms };
    ms * 1e-3 * sr
}

/// Reported latency D in samples for a configuration (R§7.2.1). Single source
/// of truth: the engine and the plugin call this, and [`crate::Shifter`] is
/// built with its result.
///
/// It covers every engine's lookahead at every tier in that mode (the
/// interpolation term uses the largest kernel, Sinc16), so switching tier or
/// engine never changes D. `live_plus_ms` is clamped to 5-10 ms and
/// `studio_ms` to 20-50 ms (non-finite values give the defaults 10 and 40);
/// both are raised to the range-dependent minimum described in the module
/// docs. Never returns 0.
pub fn latency_samples(
    sample_rate: f32,
    mode: LatencyMode,
    range: VoiceRange,
    live_plus_ms: f32,
    studio_ms: f32,
) -> usize {
    let sr = sanitize_rate(sample_rate);
    let d = match mode {
        LatencyMode::LiveMin => LIVE_MIN_MS * 1e-3 * sr,
        LatencyMode::Live => live(sr, range),
        LatencyMode::LivePlus => ms_param(sr, live_plus_ms, 10.0, 5.0, 10.0).max(live(sr, range)),
        LatencyMode::Studio => {
            ms_param(sr, studio_ms, 40.0, 20.0, 50.0).max(live(sr, range)).max(studio_min(sr, range))
        }
    };
    (d.ceil() as usize).max(MAX_HALF_WIDTH + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANGES: [VoiceRange; 4] = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];

    fn ms(sr: f32, d: usize) -> f64 {
        d as f64 * 1000.0 / sr as f64
    }

    #[test]
    fn live_values_match_research_estimates() {
        // R§7.2.1: Soprano ≈ 2.5-3 ms + margin, Alto/Tenor ≈ 3.6-4.1 ms +
        // margin, Low Male ≈ 4.7-5.2 ms + margin.
        let sr = 48_000.0;
        let d = |r| ms(sr, latency_samples(sr, LatencyMode::Live, r, 10.0, 40.0));
        assert!((2.8..3.6).contains(&d(VoiceRange::Soprano)), "{}", d(VoiceRange::Soprano));
        assert!((4.0..5.0).contains(&d(VoiceRange::AltoTenor)), "{}", d(VoiceRange::AltoTenor));
        assert!((5.2..6.3).contains(&d(VoiceRange::LowMale)), "{}", d(VoiceRange::LowMale));
        assert_eq!(d(VoiceRange::Wide), d(VoiceRange::LowMale));
    }

    #[test]
    fn modes_are_ordered_and_honour_user_values() {
        for sr in [44_100.0f32, 48_000.0, 88_200.0, 96_000.0, 192_000.0] {
            for r in RANGES {
                let lm = latency_samples(sr, LatencyMode::LiveMin, r, 10.0, 40.0);
                let l = latency_samples(sr, LatencyMode::Live, r, 10.0, 40.0);
                let lp = latency_samples(sr, LatencyMode::LivePlus, r, 7.5, 40.0);
                let st = latency_samples(sr, LatencyMode::Studio, r, 10.0, 40.0);
                assert!(lm < l && l <= lp && lp < st, "sr={sr} {r:?}: {lm} {l} {lp} {st}");
                assert!((ms(sr, lm) - 1.0).abs() < 0.05);
                // The user's value is used whenever it is above the minimum.
                assert!((ms(sr, st) - 40.0).abs() < 0.03, "sr={sr} {r:?}: {}", ms(sr, st));
                if l < (7.5e-3 * sr as f64) as usize {
                    assert!((ms(sr, lp) - 7.5).abs() < 0.03);
                }
                // Live scales with the sample rate (same milliseconds up to
                // the fixed-sample interpolation and rounding terms).
                let l48 = latency_samples(48_000.0, LatencyMode::Live, r, 10.0, 40.0);
                assert!((ms(sr, l) - ms(48_000.0, l48)).abs() < 0.25);
            }
        }
    }

    #[test]
    fn clamps_and_minimums() {
        let sr = 48_000.0;
        let lp = |v| latency_samples(sr, LatencyMode::LivePlus, VoiceRange::Soprano, v, 40.0);
        assert_eq!(lp(1.0), lp(5.0));
        assert_eq!(lp(100.0), lp(10.0));
        assert_eq!(lp(f32::NAN), lp(10.0));
        let st = |r, v| latency_samples(sr, LatencyMode::Studio, r, 10.0, v);
        assert_eq!(st(VoiceRange::Soprano, 5.0), st(VoiceRange::Soprano, 20.0));
        assert_eq!(st(VoiceRange::Soprano, 500.0), st(VoiceRange::Soprano, 50.0));
        assert_eq!(st(VoiceRange::Soprano, f32::INFINITY), st(VoiceRange::Soprano, 40.0));
        // Low Male at 20 ms is raised so a symmetric grain fits a 55 Hz note.
        let low = st(VoiceRange::LowMale, 20.0);
        assert!(ms(sr, low) > 22.0 && ms(sr, low) < 24.5, "{}", ms(sr, low));
        // Live+ at 5 ms for Low Male is raised to the Live value.
        let l = latency_samples(sr, LatencyMode::Live, VoiceRange::LowMale, 10.0, 40.0);
        assert_eq!(latency_samples(sr, LatencyMode::LivePlus, VoiceRange::LowMale, 5.0, 40.0), l);
        // Garbage sample rates are sized as 48 kHz and never give 0.
        for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
            for m in [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
                let d = latency_samples(bad, m, VoiceRange::AltoTenor, 10.0, 40.0);
                assert_eq!(d, latency_samples(48_000.0, m, VoiceRange::AltoTenor, 10.0, 40.0));
                assert!(d > MAX_HALF_WIDTH);
            }
        }
    }

    #[test]
    fn live_covers_each_term() {
        // D_live ≥ rise + mark delay at the lowest note + interpolation.
        for sr in [44_100.0f32, 48_000.0, 96_000.0] {
            for r in RANGES {
                let s = sr as f64;
                let d = latency_samples(sr, LatencyMode::Live, r, 10.0, 40.0) as f64;
                assert!(d >= rise_samples(s) + mark_delay(t_max(s, r)) + MAX_HALF_WIDTH as f64);
            }
        }
    }
}
