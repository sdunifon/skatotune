//! Mean-pitch Kalman filter with innovation gating (R§8.10).
//!
//! State: pitch p (cents) and its velocity v (cents/s), constant-velocity
//! model with white acceleration noise of spectral density `q`:
//!
//! ```text
//! F = [1 dt; 0 1],   Q = q·[dt³/3  dt²/2; dt²/2  dt],   z = p + noise,  R = (R₀ / max(clarity, 0.1))²
//! ```
//!
//! A constant-velocity filter follows ramps (glides, the slopes of vibrato)
//! without steady-state lag. Its equivalent loop bandwidth at full clarity is
//! `ω ≈ (q / (R·dt))^¼`: [`Q_LIVE`] gives ≈ 80 Hz at a 1.33 ms hop, so a 6 Hz
//! vibrato passes with < 1 % amplitude error and a pitch step settles in
//! ≈ 2 ms, while estimator noise and isolated outliers are smoothed (lag is
//! the defect for a tuner, R§10.1, and the control law pre-smooths at 4 ms
//! anyway, R§7.6.2 C). Studio uses [`Q_STUDIO`] (≈ 145 Hz): its evidence is
//! already centred and HMM-smoothed, so the filter is mainly an outlier
//! guard. Low clarity raises R and smooths harder. This is *not*
//! Christensen's per-sample AR(1) "fast" state, which R§8.10 rejects as a
//! vibrato model.
//!
//! Innovation gating: an innovation above 120 c is treated as an outlier and
//! skipped, unless it repeats (two consecutive hops agreeing within 50 c),
//! which is a real note change: the filter then jumps to the measurement
//! (R§8.10 "… and the HMM bin has moved for 2 or more hops"). Studio passes
//! `accept_jump` because its fixed-lag HMM has already validated the move.

/// Measurement noise at clarity 1, cents.
const R0_CENTS: f64 = 1.0;
/// Acceleration noise density for Live (causal) analysis, cents²/s³.
pub(crate) const Q_LIVE: f64 = 1.0e8;
/// Acceleration noise density for Studio (centred) analysis, cents²/s³.
pub(crate) const Q_STUDIO: f64 = 1.0e9;
/// Innovation gate, cents (R§8.10: 100-150 c).
const GATE_CENTS: f64 = 120.0;
/// Two gated measurements closer than this confirm a note change.
const CONFIRM_CENTS: f64 = 50.0;
/// Initial velocity uncertainty, cents/s.
const V0_SIGMA: f64 = 200.0;
/// Velocity clamp, cents/s (faster than any sung glide).
const V_MAX: f64 = 30_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeanKalman {
    q: f64,
    p: f64,
    v: f64,
    /// Covariance [[ppp, ppv], [ppv, pvv]].
    ppp: f64,
    ppv: f64,
    pvv: f64,
    init: bool,
    gated: u32,
    last_gated: f64,
}

impl MeanKalman {
    /// Filter with acceleration noise density `q` (cents²/s³).
    pub(crate) fn new(q: f64) -> Self {
        Self { q, p: 0.0, v: 0.0, ppp: 0.0, ppv: 0.0, pvv: 0.0, init: false, gated: 0, last_gated: 0.0 }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.q);
    }

    pub(crate) fn is_init(&self) -> bool {
        self.init
    }

    /// Current pitch estimate, cents.
    pub(crate) fn pitch(&self) -> f64 {
        self.p
    }

    /// Current velocity estimate, cents/s.
    pub(crate) fn velocity(&self) -> f64 {
        self.v
    }

    /// Starts the filter at measurement `z`.
    pub(crate) fn start(&mut self, z: f64, clarity: f32) {
        let r = Self::r(clarity);
        *self = Self { p: z, v: 0.0, ppp: r, ppv: 0.0, pvv: V0_SIGMA * V0_SIGMA, init: true, ..Self::new(self.q) };
    }

    fn r(clarity: f32) -> f64 {
        let c = if clarity.is_finite() { (clarity as f64).max(0.1) } else { 0.1 };
        let s = R0_CENTS / c;
        s * s
    }

    fn predict(&mut self, dt: f64) {
        let q = self.q;
        self.p += self.v * dt;
        // P ← F P Fᵀ + Q
        let ppp = self.ppp + 2.0 * dt * self.ppv + dt * dt * self.pvv + q * dt * dt * dt / 3.0;
        let ppv = self.ppv + dt * self.pvv + q * dt * dt / 2.0;
        let pvv = self.pvv + q * dt;
        self.ppp = ppp;
        self.ppv = ppv;
        self.pvv = pvv;
    }

    /// One hop of `dt` seconds with measurement `z` (cents) of the given
    /// clarity. `accept_jump`: a gated innovation is a validated note change
    /// (jump at once). Returns the filtered pitch.
    pub(crate) fn update(&mut self, z: f64, clarity: f32, dt: f64, accept_jump: bool) -> f64 {
        if !z.is_finite() {
            return self.p;
        }
        if !self.init {
            self.start(z, clarity);
            return self.p;
        }
        self.predict(dt);
        let innov = z - self.p;
        if innov.abs() > GATE_CENTS {
            let confirmed = accept_jump || (self.gated >= 1 && (z - self.last_gated).abs() < CONFIRM_CENTS);
            if confirmed {
                self.start(z, clarity);
            } else {
                self.gated += 1;
                self.last_gated = z;
            }
            return self.p;
        }
        self.gated = 0;
        let r = Self::r(clarity);
        let s = self.ppp + r;
        let (k0, k1) = (self.ppp / s, self.ppv / s);
        self.p += k0 * innov;
        self.v = (self.v + k1 * innov).clamp(-V_MAX, V_MAX);
        // P ← (I − K H) P
        let (ppp, ppv, pvv) = (self.ppp, self.ppv, self.pvv);
        self.ppp = (1.0 - k0) * ppp;
        self.ppv = (1.0 - k0) * ppv;
        self.pvv = pvv - k1 * ppv;
        self.p
    }

    /// Prediction-only hop (no usable measurement): the velocity decays by
    /// `v_decay` so a hold does not run away along a stale slope.
    pub(crate) fn coast(&mut self, dt: f64, v_decay: f64) {
        if self.init {
            self.predict(dt);
            self.v *= v_decay;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 64.0 / 48_000.0;

    #[test]
    fn tracks_vibrato_with_small_error() {
        let mut k = MeanKalman::new(Q_LIVE);
        let mut worst = 0.0f64;
        for i in 0..3000 {
            let t = i as f64 * DT;
            let z = 5700.0 + 50.0 * (std::f64::consts::TAU * 6.0 * t).sin();
            let p = k.update(z, 0.95, DT, false);
            if i > 300 {
                worst = worst.max((p - z).abs());
            }
        }
        assert!(worst < 1.5, "vibrato tracking error {worst} c");
    }

    #[test]
    fn rejects_single_outlier_and_follows_note_change() {
        let mut k = MeanKalman::new(Q_LIVE);
        for _ in 0..100 {
            k.update(6000.0, 0.9, DT, false);
        }
        // One octave glitch is ignored.
        let p = k.update(7200.0, 0.9, DT, false);
        assert!((p - 6000.0).abs() < 1.0, "{p}");
        let p = k.update(6000.0, 0.9, DT, false);
        assert!((p - 6000.0).abs() < 1.0);
        // A real leap: second consistent measurement jumps.
        k.update(6500.0, 0.9, DT, false);
        let p = k.update(6502.0, 0.9, DT, false);
        assert!((p - 6502.0).abs() < 1e-6, "{p}");
        assert!(k.velocity().abs() < 1e-9);
        // A validated jump (Studio) is taken at once.
        let p = k.update(7000.0, 0.9, DT, true);
        assert!((p - 7000.0).abs() < 1e-6, "{p}");
    }

    #[test]
    fn step_settles_quickly() {
        // 40 c step (below the gate): 50 % within 2 hops, 90 % within 4.
        let mut k = MeanKalman::new(Q_LIVE);
        for _ in 0..200 {
            k.update(6000.0, 0.95, DT, false);
        }
        let steps: Vec<f64> = (0..6).map(|_| k.update(6040.0, 0.95, DT, false) - 6000.0).collect();
        assert!(steps[1] > 20.0 && steps[3] > 36.0, "{steps:?}");
    }

    #[test]
    fn noise_is_smoothed_and_low_clarity_smooths_more() {
        let run = |clarity: f32| {
            let mut k = MeanKalman::new(Q_LIVE);
            let mut s = 1u32;
            let mut err = 0.0f64;
            for i in 0..4000 {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let n = (s >> 8) as f64 / (1u64 << 24) as f64 - 0.5;
                let p = k.update(6000.0 + 8.0 * n, clarity, DT, false);
                if i > 500 {
                    err += (p - 6000.0) * (p - 6000.0);
                }
            }
            (err / 3500.0).sqrt()
        };
        let raw = 8.0 / 12f64.sqrt();
        let hi = run(1.0);
        let lo = run(0.3);
        assert!(hi < raw, "{hi} vs raw {raw}");
        assert!(lo < hi, "{lo} vs {hi}");
    }

    #[test]
    fn coast_and_nan() {
        let mut k = MeanKalman::new(Q_LIVE);
        k.coast(DT, 0.5);
        assert!(!k.is_init());
        assert_eq!(k.update(f64::NAN, 1.0, DT, false), 0.0);
        k.update(5000.0, f32::NAN, DT, false);
        assert!(k.is_init());
        k.coast(DT, 0.5);
        assert!((k.pitch() - 5000.0).abs() < 1e-9);
        k.reset();
        assert!(!k.is_init());
    }
}
