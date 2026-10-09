//! Scalar maths helpers used on the audio thread.
//!
//! The `fast_*` functions replace `exp2`/`log2`/`tanh` from libm in per-sample
//! and per-hop code. They are branch-free polynomial / rational approximations
//! built on IEEE-754 exponent bit tricks, so they inline, auto-vectorise and
//! never call into libm (on x86 baseline builds libm `exp2f`/`log2f` are real
//! function calls). Error bounds are asserted by dense sweeps in the tests below
//! and speedups are measured in `benches/kernels.rs` (`fast_math` group).
//!
//! Everything here is total: NaN, ±Inf and out-of-range inputs produce a finite
//! (or documented) result and never panic.
//!
//! Pitch conventions follow [`crate::types`]: absolute cents are
//! `6900 + 1200·log2(f / 440)` (A4 = 6900 = MIDI note 69 × 100).

/// `log2(e)`, for turning natural exponentials into `exp2`.
const LOG2_E: f32 = core::f32::consts::LOG2_E;

/// `1.5 · 2^23`. Adding it to a float with |x| < 2^22 rounds x to the nearest
/// integer (ties to even) and leaves that integer in the low mantissa bits.
const ROUND_MAGIC: f32 = 12_582_912.0;

/// Minimax coefficients of `2^f` on f ∈ [-0.5, 0.5] (degree 6, Lawson-fitted
/// for relative error; fit error 1.9e-9, so f32 rounding dominates).
const EXP2_C: [f32; 7] = [
    1.0,
    core::f32::consts::LN_2, // the fitted 0.693_147_2 rounds to LN_2 in f32
    0.240_226_46,
    0.055_503_286,
    0.009_618_489,
    0.001_339_993_1,
    0.000_153_458_12,
];

/// 2^x.
///
/// Max relative error < 1e-6 (measured 9.7e-8) for x in [-60, 60], and the
/// same accuracy over the whole normal range [-126, 127]. Saturates outside
/// that range: x ≤ -126 → 2^-126, x ≥ 127 → 2^127. NaN → 2^-126 (≈ 0, i.e.
/// NaN is treated like silence).
#[inline]
pub fn fast_exp2(x: f32) -> f32 {
    // `max`/`min` (not `clamp`) on purpose: they drop NaN in favour of the
    // other operand, so NaN → -126, and compile to single fmaxnm/fminnm.
    #[allow(clippy::manual_clamp)]
    let x = x.max(-126.0).min(127.0);
    // Split x = k + f with k = round(x), f ∈ [-0.5, 0.5]. The magic-number add
    // performs the rounding without a libm call on any target.
    let t = x + ROUND_MAGIC;
    let k = t.to_bits().wrapping_sub(ROUND_MAGIC.to_bits()) as i32;
    let f = x - (t - ROUND_MAGIC);
    // Horner evaluation of 2^f (plain mul + add: no libm `fmaf` on x86).
    let c = &EXP2_C;
    let p = c[0] + f * (c[1] + f * (c[2] + f * (c[3] + f * (c[4] + f * (c[5] + f * c[6])))));
    // 2^k assembled directly in the exponent field; k ∈ [-126, 127] keeps the
    // biased exponent in [1, 254] (a normal float).
    let scale = f32::from_bits(((k + 127) as u32) << 23);
    p * scale
}

/// log2(x) for x > 0 in the normal range.
///
/// Max absolute error < 1e-5 (measured 3.9e-6 at the extremes of the
/// exponent range, where the f32 result spacing itself is 7.6e-6; 3.6e-7 for
/// x in [2^-8, 2^8]). Returns -126.0 for x ≤ 0, NaN and subnormal x; +Inf →
/// 128.0.
#[inline]
pub fn fast_log2(x: f32) -> f32 {
    // Clamp handles x <= 0, NaN (dropped by `max`) and subnormals in one op.
    let x = x.max(f32::MIN_POSITIVE);
    // Split x = 2^e · m with m ∈ [√½, √2) so that t = (m-1)/(m+1) stays small
    // (|t| ≤ 0.1716). Subtracting the bit pattern of √½ before extracting the
    // exponent does the re-centring without a branch.
    const SQRT_HALF_BITS: u32 = 0x3f35_04f3;
    let bits = x.to_bits().wrapping_sub(SQRT_HALF_BITS);
    let e = (bits as i32) >> 23;
    let m = f32::from_bits((bits & 0x007f_ffff).wrapping_add(SQRT_HALF_BITS));
    // ln(m) = 2·atanh(t) = 2(t + t³/3 + t⁵/5 + t⁷/7 + …). Truncating after t⁷
    // leaves 2·t⁹/9 / ln 2 ≈ 4e-8 of error, far below f32 resolution of the sum.
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    const C1: f32 = 2.0 * LOG2_E;
    const C3: f32 = 2.0 * LOG2_E / 3.0;
    const C5: f32 = 2.0 * LOG2_E / 5.0;
    const C7: f32 = 2.0 * LOG2_E / 7.0;
    let p = t * (C1 + t2 * (C3 + t2 * (C5 + t2 * C7)));
    e as f32 + p
}

/// tanh(x). Max absolute error < 1e-4 (measured 3.9e-7).
///
/// 13/6-degree odd/even rational minimax approximation (the coefficients used
/// by Eigen's `ptanh_float`), clamped at |x| = 7.905 where f32 tanh is 1.0.
/// NaN → 0.0.
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    const CLAMP: f32 = 7.905_311;
    const A1: f32 = 4.893_524_6e-3;
    const A3: f32 = 6.372_619_5e-4;
    const A5: f32 = 1.485_722_35e-5;
    const A7: f32 = 5.122_297_3e-8;
    const A9: f32 = -8.604_672e-11;
    const A11: f32 = 2.000_188e-13;
    const A13: f32 = -2.760_768_4e-16;
    const B0: f32 = 4.893_525e-3;
    const B2: f32 = 2.268_434_7e-3;
    const B4: f32 = 1.185_347_1e-4;
    const B6: f32 = 1.198_258_4e-6;
    // `max`/`min` would turn NaN into ±CLAMP (±1.0); map it to 0 explicitly.
    let x = if x.is_nan() { 0.0 } else { x.clamp(-CLAMP, CLAMP) };
    let x2 = x * x;
    let p = x * (A1 + x2 * (A3 + x2 * (A5 + x2 * (A7 + x2 * (A9 + x2 * (A11 + x2 * A13))))));
    let q = B0 + x2 * (B2 + x2 * (B4 + x2 * B6));
    p / q
}

/// Pitch ratio for an interval in cents: `2^(cents / 1200)`.
#[inline]
pub fn cents_to_ratio(cents: f32) -> f32 {
    fast_exp2(cents * (1.0 / 1200.0))
}

/// Interval in cents for a pitch ratio: `1200 · log2(ratio)`.
/// Ratios ≤ 0 return `1200 · -126` (a huge negative interval), never NaN.
#[inline]
pub fn ratio_to_cents(ratio: f32) -> f32 {
    1200.0 * fast_log2(ratio)
}

/// Frequency in Hz → absolute cents (A4 = 440 Hz = 6900). Error ≈ 1e-4 cents.
#[inline]
pub fn hz_to_cents_fast(hz: f32) -> f32 {
    6900.0 + 1200.0 * fast_log2(hz * (1.0 / 440.0))
}

/// Absolute cents → frequency in Hz. Relative error ≈ 1e-7.
#[inline]
pub fn cents_to_hz_fast(cents: f32) -> f32 {
    440.0 * fast_exp2((cents - 6900.0) * (1.0 / 1200.0))
}

/// One-pole smoothing coefficient `α = 1 − exp(−step/τ)` for a filter
/// `y += α (x − y)` updated every `step_samples` with a 63% time constant of
/// `tau_samples`. Returns 1.0 (instant) when τ ≤ 0 or NaN, 0.0 when τ = +Inf.
///
/// Uses `exp_m1` so that α stays accurate for τ ≫ step (where `1 − exp(−ε)`
/// would cancel catastrophically in f32).
#[inline]
pub fn onepole_alpha(tau_samples: f32, step_samples: f32) -> f32 {
    if tau_samples.is_nan() || tau_samples <= 0.0 {
        return 1.0;
    }
    let step = if step_samples.is_finite() { step_samples.max(0.0) } else { 0.0 };
    (-(-step / tau_samples).exp_m1()).clamp(0.0, 1.0)
}

/// Decibels → linear amplitude, `10^(db/20)`. Very negative values (and NaN)
/// give ≈ 0 (2^-126); very large values saturate at 2^127.
#[inline]
pub fn db_to_lin(db: f32) -> f32 {
    // 10^(db/20) = 2^(db · log2(10)/20).
    const K: f32 = core::f32::consts::LOG2_10 / 20.0;
    fast_exp2(db * K)
}

/// Linear amplitude → decibels, `20·log10(|lin|)`, floored at -160 dB.
/// The sign is ignored (a sample of -0.5 is -6 dB); 0, NaN → -160 dB.
#[inline]
pub fn lin_to_db(lin: f32) -> f32 {
    // 20·log10(x) = 20·log10(2)·log2(x).
    const K: f32 = 20.0 * core::f32::consts::LOG10_2;
    (K * fast_log2(lin.abs())).max(-160.0)
}

/// Parabolic interpolation of an extremum through the three points
/// (−1, ym1), (0, y0), (1, yp1).
///
/// Returns `(offset, value)`: the vertex offset clamped to [-1, 1] and the
/// parabola's value there. Works for maxima and minima alike. A degenerate
/// (flat or collinear) triple returns `(0.0, y0)`. When the vertex lies outside
/// [-1, 1], the parabola is evaluated at the clamped end, i.e. returns the
/// neighbour (`ym1` or `yp1`).
///
/// For the unclamped case this is the classic
/// `δ = (ym1 − yp1) / (2(ym1 − 2y0 + yp1))`, `peak = y0 − ¼(ym1 − yp1)δ`
/// (R§8.4).
#[inline]
pub fn parabolic_peak(ym1: f32, y0: f32, yp1: f32) -> (f32, f32) {
    let curv = ym1 - 2.0 * y0 + yp1; // 2nd difference = 2a
    let slope = 0.5 * (yp1 - ym1); // b
    // Collinear / flat points or NaN: no vertex to find.
    let tiny = f32::EPSILON * (ym1.abs() + y0.abs() + yp1.abs()).max(f32::MIN_POSITIVE);
    if curv.is_nan() || curv.abs() <= tiny {
        return (0.0, y0);
    }
    let delta = (-slope / curv).clamp(-1.0, 1.0);
    if !delta.is_finite() {
        return (0.0, y0);
    }
    // p(δ) = y0 + bδ + aδ², a = curv/2.
    (delta, y0 + delta * (slope + 0.5 * curv * delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense sweep helper: `n` evenly spaced points over [lo, hi].
    fn sweep(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f32> {
        (0..=n).map(move |i| (lo + (hi - lo) * i as f64 / n as f64) as f32)
    }

    #[test]
    fn exp2_relative_error_dense() {
        let mut worst = 0.0f64;
        for x in sweep(-60.0, 60.0, 2_000_000) {
            let exact = (x as f64).exp2();
            let rel = ((fast_exp2(x) as f64 - exact) / exact).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 1e-6, "fast_exp2 worst relative error {worst:e}");
        // Whole normal range, coarser.
        for x in sweep(-126.0, 127.0, 400_000) {
            let exact = (x as f64).exp2();
            let rel = ((fast_exp2(x) as f64 - exact) / exact).abs();
            assert!(rel < 1e-6, "x={x} rel={rel:e}");
        }
    }

    #[test]
    fn exp2_exact_at_integers_and_saturates() {
        for k in -126..=127 {
            let y = fast_exp2(k as f32);
            let exact = (k as f64).exp2();
            assert!(((y as f64 - exact) / exact).abs() < 2e-7, "k={k}");
        }
        assert_eq!(fast_exp2(1000.0), fast_exp2(127.0));
        assert_eq!(fast_exp2(f32::INFINITY), fast_exp2(127.0));
        assert_eq!(fast_exp2(-1000.0), f32::MIN_POSITIVE);
        assert_eq!(fast_exp2(f32::NEG_INFINITY), f32::MIN_POSITIVE);
        assert_eq!(fast_exp2(f32::NAN), f32::MIN_POSITIVE);
        assert!(fast_exp2(127.0).is_finite());
    }

    #[test]
    fn log2_abs_error_dense() {
        // Log-spaced sweep over the whole normal range plus a dense linear
        // sweep over the musically relevant range.
        let mut worst = 0.0f64;
        for e in sweep(-125.99, 127.99, 2_000_000) {
            let x = (e as f64).exp2() as f32;
            if !x.is_normal() {
                continue;
            }
            let err = (fast_log2(x) as f64 - (x as f64).log2()).abs();
            worst = worst.max(err);
        }
        assert!(worst < 1e-5, "fast_log2 worst abs error {worst:e}");
        let mut worst_mid = 0.0f64;
        for x in sweep(1.0 / 256.0, 256.0, 2_000_000) {
            let err = (fast_log2(x) as f64 - (x as f64).log2()).abs();
            worst_mid = worst_mid.max(err);
        }
        assert!(worst_mid < 1e-6, "fast_log2 mid-range abs error {worst_mid:e}");
    }

    #[test]
    fn log2_edge_cases() {
        assert_eq!(fast_log2(0.0), -126.0);
        assert_eq!(fast_log2(-1.0), -126.0);
        assert_eq!(fast_log2(f32::NAN), -126.0);
        assert_eq!(fast_log2(f32::NEG_INFINITY), -126.0);
        assert_eq!(fast_log2(1e-40), -126.0); // subnormal
        assert!((fast_log2(f32::INFINITY) - 128.0).abs() < 1e-4);
        for k in -126..=127 {
            let x = (k as f64).exp2() as f32;
            assert!((fast_log2(x) - k as f32).abs() < 1e-5, "k={k}");
        }
    }

    #[test]
    fn tanh_abs_error_dense() {
        let mut worst = 0.0f64;
        for x in sweep(-20.0, 20.0, 4_000_000) {
            let err = (fast_tanh(x) as f64 - (x as f64).tanh()).abs();
            worst = worst.max(err);
        }
        assert!(worst < 1e-4, "fast_tanh worst abs error {worst:e}");
        assert_eq!(fast_tanh(0.0), 0.0);
        assert_eq!(fast_tanh(f32::NAN), 0.0);
        assert!((fast_tanh(f32::INFINITY) - 1.0).abs() < 1e-6);
        assert!((fast_tanh(f32::NEG_INFINITY) + 1.0).abs() < 1e-6);
        // Odd symmetry and |tanh| <= 1.
        for x in sweep(0.0, 10.0, 10_000) {
            assert_eq!(fast_tanh(-x), -fast_tanh(x));
            assert!(fast_tanh(x).abs() <= 1.0);
        }
    }

    #[test]
    fn cents_and_ratio_conversions() {
        for c in sweep(-4800.0, 4800.0, 100_000) {
            let r = cents_to_ratio(c);
            let exact = (c as f64 / 1200.0).exp2();
            assert!(((r as f64 - exact) / exact).abs() < 1e-6);
            assert!((ratio_to_cents(r) - c).abs() < 0.01, "c={c}");
        }
        assert!((ratio_to_cents(2.0) - 1200.0).abs() < 1e-3);
        assert!(ratio_to_cents(0.0).is_finite());
        for hz in sweep(20.0, 20_000.0, 100_000) {
            let c = hz_to_cents_fast(hz);
            assert!((c - crate::hz_to_cents(hz)).abs() < 0.01, "hz={hz}");
            assert!(((cents_to_hz_fast(c) - hz) / hz).abs() < 1e-5);
        }
        assert!((hz_to_cents_fast(440.0) - 6900.0).abs() < 1e-3);
        assert!((cents_to_hz_fast(6900.0) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn onepole_alpha_definition() {
        // After τ samples of steps the response reaches 1 − e^-1 (63.2 %).
        let tau = 480.0;
        let a = onepole_alpha(tau, 1.0);
        let mut y = 0.0f64;
        for _ in 0..480 {
            y += a as f64 * (1.0 - y);
        }
        assert!((y - (1.0 - (-1.0f64).exp())).abs() < 1e-4, "y={y}");
        // Per-hop stepping gives the same time constant.
        let ah = onepole_alpha(tau, 64.0);
        let mut yh = 0.0f64;
        let hops = 20;
        for _ in 0..hops {
            yh += ah as f64 * (1.0 - yh);
        }
        let exact = 1.0 - (-(hops as f64 * 64.0) / 480.0f64).exp();
        assert!((yh - exact).abs() < 1e-4);
        assert_eq!(onepole_alpha(0.0, 64.0), 1.0);
        assert_eq!(onepole_alpha(-5.0, 64.0), 1.0);
        assert_eq!(onepole_alpha(f32::NAN, 64.0), 1.0);
        assert_eq!(onepole_alpha(f32::INFINITY, 64.0), 0.0);
        // Accurate for very long time constants (no cancellation to 0).
        let tiny = onepole_alpha(1e7, 1.0);
        assert!(((tiny as f64) - 1e-7).abs() / 1e-7 < 1e-4, "tiny={tiny}");
    }

    #[test]
    fn db_conversions() {
        for db in sweep(-150.0, 30.0, 100_000) {
            let lin = db_to_lin(db);
            let exact = 10f64.powf(db as f64 / 20.0);
            assert!(((lin as f64 - exact) / exact).abs() < 2e-6, "db={db}");
            assert!((lin_to_db(lin) - db).abs() < 1e-3, "db={db}");
        }
        assert_eq!(lin_to_db(0.0), -160.0);
        assert_eq!(lin_to_db(f32::NAN), -160.0);
        assert_eq!(lin_to_db(1e-12), -160.0);
        assert!((lin_to_db(-0.5) - lin_to_db(0.5)).abs() < 1e-6);
        assert!((lin_to_db(1.0)).abs() < 1e-5);
        assert!(db_to_lin(-1000.0) < 1e-37);
        assert!(db_to_lin(f32::NAN) < 1e-37);
    }

    #[test]
    fn parabolic_peak_recovers_vertex() {
        // Sample y = 3 − 2(x − d)² at −1, 0, 1 and recover d and the peak.
        for i in -90..=90 {
            let d = i as f32 / 100.0;
            let f = |x: f32| 3.0 - 2.0 * (x - d) * (x - d);
            let (off, val) = parabolic_peak(f(-1.0), f(0.0), f(1.0));
            assert!((off - d).abs() < 1e-5, "d={d} off={off}");
            assert!((val - 3.0).abs() < 1e-5);
        }
        // Minimum works too.
        let g = |x: f32| 1.0 + (x - 0.3) * (x - 0.3);
        let (off, val) = parabolic_peak(g(-1.0), g(0.0), g(1.0));
        assert!((off - 0.3).abs() < 1e-5 && (val - 1.0).abs() < 1e-5);
        // Degenerate and clamped cases.
        assert_eq!(parabolic_peak(1.0, 1.0, 1.0), (0.0, 1.0));
        assert_eq!(parabolic_peak(0.0, 1.0, 2.0), (0.0, 1.0)); // collinear
        let (off, val) = parabolic_peak(0.0, 1.0, 1.9); // vertex beyond +1
        assert_eq!(off, 1.0);
        assert!((val - 1.9).abs() < 1e-6);
        let (off, val) = parabolic_peak(f32::NAN, 1.0, 0.5);
        assert_eq!((off, val), (0.0, 1.0));
    }
}
