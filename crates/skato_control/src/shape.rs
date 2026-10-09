//! Error shaping, R§7.6.2 G: Amount, Snap Min (soft dead zone), Snap Max and
//! the Flex-Tune weight. All functions are pure so they can be unit tested
//! against the formulas directly.

/// Flex-Tune taper fraction k_t (R§7.6.2 G, "design choice, calibrate it").
/// The correction weight falls from 1 to 0 over the outer 40 % of the radius.
pub(crate) const FLEX_TAPER: f32 = 0.4;

/// Soft dead zone: `e2 = e1 · x² / (1 + x²)` with `x = |e1| / D0`.
///
/// Errors well below `D0` are attenuated quadratically (left mostly alone),
/// errors well above it pass almost unchanged. `D0 <= 0` (or NaN) disables it.
///
/// Evaluated as `e1 / (1 + 1/x²)`, which is the same function but total in
/// f32: a tiny `D0` (e.g. 1e-20 c from an automation tail) makes `x²`
/// overflow to infinity, and `e1·∞/(1 + ∞)` would be NaN, while
/// `e1/(1 + 0)` is the correct limit `e1`. `e1 = 0` gives `0/∞ = 0`.
#[inline]
pub(crate) fn snap_min(e1: f32, d0: f32) -> f32 {
    if d0.is_nan() || d0 <= 0.0 {
        return e1;
    }
    let x = e1 / d0;
    e1 / (1.0 + 1.0 / (x * x))
}

/// Flex-Tune weight `w(d)` for a deviation `d = |p_dec − pitch(n*)|`.
///
/// `gap` is the distance to the adjacent legal note on the side of the
/// deviation (100 c when chromatic) and `flex` the knob 0-100. The correction
/// region has radius `R = 0.5·G·(1 − F/100)`; inside `(1 − k_t)·R` the weight
/// is 1, it then follows a raised-cosine taper to 0 at `R`. `F = 0` gives
/// `w ≡ 1` (correct everything), `F = 100` gives `R = 0` (correct nothing).
#[inline]
pub(crate) fn flex_weight(d: f32, gap: f32, flex: f32) -> f32 {
    if flex <= 0.0 {
        return 1.0;
    }
    let r = 0.5 * gap.max(0.0) * (1.0 - (flex / 100.0).min(1.0));
    if d >= r {
        return 0.0;
    }
    let inner = (1.0 - FLEX_TAPER) * r;
    if d <= inner {
        return 1.0;
    }
    let t = (d - inner) / (FLEX_TAPER * r);
    0.5 * (1.0 + (core::f32::consts::PI * t).cos())
}

/// Snap Max gate: `true` when the decision pitch is too far from the target
/// to be corrected. `max_cents <= 0` disables it.
#[inline]
pub(crate) fn beyond_snap_max(deviation: f32, max_cents: f32) -> bool {
    max_cents > 0.0 && deviation.abs() > max_cents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_min_is_soft_dead_zone() {
        assert_eq!(snap_min(12.0, 0.0), 12.0);
        // At |e| = D0 exactly half the error survives.
        assert!((snap_min(10.0, 10.0) - 5.0).abs() < 1e-6);
        assert!((snap_min(-10.0, 10.0) + 5.0).abs() < 1e-6);
        // Small errors are strongly attenuated, large ones barely touched.
        assert!(snap_min(2.0, 20.0).abs() < 0.02);
        assert!((snap_min(200.0, 10.0) - 200.0).abs() < 0.6);
        assert_eq!(snap_min(0.0, 10.0), 0.0);
        assert_eq!(snap_min(5.0, f32::NAN), 5.0);
    }

    /// RT-1: tiny and subnormal Snap Min values (automation or modulation
    /// tails) used to overflow `x²` and return NaN, which then latched in the
    /// retune one-pole. The dead zone must be finite for every finite input
    /// and converge to the identity as `D0 → 0`.
    #[test]
    fn snap_min_is_total_for_tiny_dead_zones() {
        let d0s = [1e-45f32, 1e-40, 1e-38, 1e-30, 1e-20, 1e-17, 1e-12, 1e-3];
        let errs = [0.01f32, 0.1, 1.0, 5.0, 35.0, 100.0, 600.0, 1200.0, 2400.0];
        for d0 in d0s {
            for e in errs {
                for e1 in [e, -e] {
                    let y = snap_min(e1, d0);
                    assert!(y.is_finite(), "snap_min({e1}, {d0}) = {y}");
                    // x ≥ 10 for all of these: at most 1 % attenuation.
                    assert!((y - e1).abs() <= 0.01 * e1.abs() + 1e-6, "snap_min({e1}, {d0}) = {y}");
                }
            }
            assert_eq!(snap_min(0.0, d0), 0.0);
        }
    }

    #[test]
    fn flex_weight_curve() {
        // F = 0: always 1.
        for d in [0.0, 30.0, 49.0, 80.0] {
            assert_eq!(flex_weight(d, 100.0, 0.0), 1.0);
        }
        // F = 50, chromatic: R = 25 c, flat to 15 c, raised cosine to 25 c.
        assert_eq!(flex_weight(0.0, 100.0, 50.0), 1.0);
        assert_eq!(flex_weight(15.0, 100.0, 50.0), 1.0);
        assert!((flex_weight(20.0, 100.0, 50.0) - 0.5).abs() < 1e-5);
        assert!(flex_weight(24.9, 100.0, 50.0) < 1e-3);
        assert_eq!(flex_weight(25.0, 100.0, 50.0), 0.0);
        assert_eq!(flex_weight(40.0, 100.0, 50.0), 0.0);
        // Monotone non-increasing in d.
        let mut prev = 1.0;
        for i in 0..300 {
            let w = flex_weight(i as f32 * 0.1, 100.0, 50.0);
            assert!(w <= prev + 1e-6);
            prev = w;
        }
        // A whole-tone gap doubles the radius.
        assert!((flex_weight(40.0, 200.0, 50.0) - 0.5).abs() < 1e-5);
        // F = 100 corrects nothing.
        assert_eq!(flex_weight(0.0, 100.0, 100.0), 0.0);
    }

    #[test]
    fn snap_max_gate() {
        assert!(!beyond_snap_max(500.0, 0.0));
        assert!(beyond_snap_max(-301.0, 300.0));
        assert!(!beyond_snap_max(299.0, 300.0));
    }
}
