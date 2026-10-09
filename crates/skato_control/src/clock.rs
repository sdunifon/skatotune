//! Hop-unit time keeping.
//!
//! The control law runs once per control hop Δ (64 samples at 44.1/48 kHz, so
//! about 1.33 ms). Every time constant in R§7.6 is specified in milliseconds, so
//! each one is converted to hops here, from the actual sample rate and hop, and
//! re-derived whenever the settings change (R§8.0 "hop-dependent constants").
//! Nothing in the control law may assume a particular hop length.

/// Converts milliseconds to control hops and builds one-pole coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HopClock {
    /// Hop length in seconds.
    pub hop_s: f32,
    /// Hop length in milliseconds.
    pub hop_ms: f32,
}

impl HopClock {
    pub fn new(sample_rate: f32, hop: usize) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 1000.0 { sample_rate } else { 48_000.0 };
        let hop = hop.max(1) as f32;
        let hop_s = hop / sr;
        Self { hop_s, hop_ms: hop_s * 1000.0 }
    }

    /// Duration in (fractional) hops.
    #[inline]
    pub fn hops(&self, ms: f32) -> f32 {
        ms / self.hop_ms
    }

    /// Duration rounded to whole hops (at least `min`).
    #[inline]
    pub fn whole_hops(&self, ms: f32, min: u32) -> u32 {
        let h = self.hops(ms).round();
        if h.is_finite() && h > min as f32 { h.min(u32::MAX as f32) as u32 } else { min }
    }

    /// One-pole coefficient for a time constant in ms, R§7.6.1:
    /// `α(τ) = 1 − exp(−Δ/τ)`, with `α = 1` when τ = 0.
    #[inline]
    pub fn alpha(&self, tau_ms: f32) -> f32 {
        onepole_alpha(tau_ms, self.hop_ms)
    }
}

/// `α = 1 − exp(−step/τ)` (both in the same unit); 1.0 when `τ <= 0` or NaN.
#[inline]
pub(crate) fn onepole_alpha(tau: f32, step: f32) -> f32 {
    if tau > 0.0 && step > 0.0 { (1.0 - (-step / tau).exp()).clamp(0.0, 1.0) } else { 1.0 }
}

/// Replaces NaN/Inf with `default`; leaves finite values alone.
#[inline]
pub(crate) fn finite_or(x: f32, default: f32) -> f32 {
    if x.is_finite() { x } else { default }
}

/// Finite value clamped to `[lo, hi]` (NaN → `default`).
#[inline]
pub(crate) fn sane(x: f32, lo: f32, hi: f32, default: f32) -> f32 {
    finite_or(x, default).clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_units_follow_rate() {
        let c48 = HopClock::new(48_000.0, 64);
        assert!((c48.hop_ms - 1.333_333).abs() < 1e-4);
        assert!((c48.hops(20.0) - 15.0).abs() < 1e-4);
        let c96 = HopClock::new(96_000.0, 128);
        assert!((c96.hop_ms - c48.hop_ms).abs() < 1e-6);
        let c44 = HopClock::new(44_100.0, 64);
        assert_eq!(c44.whole_hops(20.0, 0), 14);
    }

    #[test]
    fn alpha_definition() {
        assert_eq!(onepole_alpha(0.0, 1.0), 1.0);
        assert_eq!(onepole_alpha(f32::NAN, 1.0), 1.0);
        let a = onepole_alpha(15.0, 1.0);
        // After τ/Δ steps the one-pole has covered 63.2 % of a step.
        let covered = 1.0 - (1.0 - a).powi(15);
        assert!((covered - (1.0 - (-1.0f32).exp())).abs() < 1e-5);
    }

    #[test]
    fn sanitizers() {
        assert_eq!(finite_or(f32::NAN, 3.0), 3.0);
        assert_eq!(sane(f32::INFINITY, 0.0, 1.0, 0.5), 0.5);
        assert_eq!(sane(7.0, 0.0, 1.0, 0.5), 1.0);
    }
}
