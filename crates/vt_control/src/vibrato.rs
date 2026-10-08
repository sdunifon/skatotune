//! Vibrato / centre tracker, R§8.11.
//!
//! Splits the pre-smoothed pitch `p_s` into a vibrato-free *centre* and an
//! engagement weight `w_v`, so the control law can correct the centre and
//! keep (or scale) the sung vibrato (DVA-like behaviour, R§7.6.2 D/J).
//!
//! **Extrema tracker** (every tier, causal):
//! - turning points of `p_s` with 5 c hysteresis;
//! - each extremum gives a half-period `hp` and half-extent `A`; it is valid
//!   when `1/(2·hp)` is 3.5-9 Hz and `A` is 8-150 c (larger is a trill/note);
//! - after 3 valid, consistent half-periods (`hp_max/hp_min ≤ 1.6`,
//!   `A_max/A_min ≤ 2`) the split engages and `w_v` ramps 0 → 1 over one
//!   half-period. Engagement therefore takes about one vibrato cycle; in Live
//!   at retune 0 the first cycle is flattened (unavoidable causally, R§8.11);
//! - centre anchors `(p_{e−2} + 2·p_{e−1} + p_e)/4`, placed at the middle
//!   extremum, cancel both the vibrato and linear drift. Between anchors the
//!   centre is extrapolated along the slope of the last two anchors (clamped
//!   to ±300 c/s). When a new anchor arrives the jump between the old and new
//!   line is spread over about half a period so the centre stays continuous
//!   (design choice: a step in the centre would be a step in the correction);
//! - it disengages when no extremum arrives within `1.6·mean hp`, or on a
//!   transition (`|p_s − centre| > 1.8·A_v + 25 c`).
//!
//! **HQ/Max refinement**: a Kalman filter `[c, ċ, v, v_prev]` where the centre
//! `c` drifts with a slowly varying velocity `ċ` and `v` is an AR(2)
//! resonator at the tracked vibrato rate, fed with `p_s` and with the anchor
//! line as a second (pseudo) measurement of `c`. Once engaged it gives a
//! smooth centre with near-zero group delay (an adaptive notch). See the
//! tuning notes on the constants below for the measured deviations from
//! R§8.11's starting values.

use core::f64::consts::PI;

/// Turning-point hysteresis, cents.
const HYSTERESIS: f32 = 5.0;
const RATE_MIN_HZ: f32 = 3.5;
const RATE_MAX_HZ: f32 = 9.0;
const EXTENT_MIN: f32 = 8.0;
const EXTENT_MAX: f32 = 150.0;
const HP_RATIO_MAX: f32 = 1.6;
const EXTENT_RATIO_MAX: f32 = 2.0;
/// Valid half-periods needed before engaging.
const ENGAGE_COUNT: usize = 3;
/// Centre slope clamp, cents per second.
const SLOPE_MAX: f32 = 300.0;
/// Disengage ramp after a transition, ms.
const TRANSITION_RAMP_MS: f32 = 10.0;

// Kalman tuning at the reference hop (64 samples at 48 kHz); rescaled to the
// actual hop in `new()`. Chosen by measurement (wandering 5-6 Hz, 18-42 c
// vibrato with drifting centre and detector noise), not taken verbatim from
// R§8.11:
// - R§8.11's ρ = 0.98 damps the resonator so fast that it under-predicts the
//   vibrato every hop and 2-4 c of it leaks into the centre; ρ = 0.995
//   (≈ 0.27 s time constant) still adapts to rate/extent changes.
// - R§8.11's pure random-walk centre with q = (2 c)² lags drift (a 30 c/s
//   drift gave a 10 c error) and absorbs vibrato; a constant-velocity centre
//   with small noise tracks drift without lag.
// - The extrema anchor line is fused in as a pseudo-measurement of the centre
//   with R = (4 c)². Alone the Kalman was worse than the anchor line; fused it
//   matches the line's accuracy (≈ 1 c RMS) with about 4x less jerk, which is
//   the smooth centre between anchors that R§8.11 wants from it.
const RHO_REF: f32 = 0.995;
const Q_CENTRE_REF: f32 = 0.05;
const Q_SLOPE_REF: f32 = 1e-6;
const LINE_R: f64 = 16.0;
const REF_HOP_MS: f32 = 64.0 / 48.0;

/// Output of one tracker update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VibratoOut {
    /// Vibrato-free centre, cents (equals the input while not engaged).
    pub centre: f32,
    /// Engagement weight `w_v`, 0-1.
    pub w: f32,
    /// Vibrato rate `f_v`, Hz (0 when not engaged).
    pub rate_hz: f32,
    /// Vibrato half-extent `A_v`, cents (0 when not engaged).
    pub extent: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Point {
    t: f64,
    p: f32,
}

/// AR(2)-resonator Kalman filter on `[c, ċ, v, v_prev]` (R§8.11 HQ/Max).
///
/// R§8.11 models the centre `c` as a pure random walk. That lags any steady
/// drift (a random walk has no velocity, so a 30 c/s drift left a 10 c
/// error in the tests), so the centre here is an integrated random walk
/// (constant velocity `ċ` per hop). `v` is the AR(2) resonator
/// `v_t = 2ρ·cos(ω_v·Δ)·v_{t−1} − ρ²·v_{t−2}`. f64 because `ċ` and its
/// variance are tiny next to the centre's.
#[derive(Clone, Copy, Debug, Default)]
struct Resonator {
    x: [f64; 4],
    p: [[f64; 4]; 4],
    a1: f64,
    a2: f64,
    q: [f64; 4],
    r: f64,
    active: bool,
}

impl Resonator {
    /// Set the resonator for vibrato rate `f_v` and half-extent `a_v`.
    /// `rho`, `q_c` and `q_d` are already rescaled to the hop.
    fn configure(&mut self, f_v: f32, a_v: f32, hop_s: f32, rho: f32, q_c: f32, q_d: f32) {
        let theta = 2.0 * core::f64::consts::PI * f64::from(f_v) * f64::from(hop_s);
        let rho = f64::from(rho);
        self.a1 = 2.0 * rho * theta.cos();
        self.a2 = -rho * rho;
        // Process noise that makes the AR(2) stationary variance equal to the
        // variance of a sinusoid of half-extent a_v (A²/2):
        // γ0 = σ²(1 − a2) / ((1 + a2)((1 − a2)² − a1²)).
        let gamma0 = 0.5 * f64::from(a_v).powi(2);
        let num = (1.0 + self.a2) * ((1.0 - self.a2).powi(2) - self.a1 * self.a1);
        let q_v = (gamma0 * num / (1.0 - self.a2)).max(1e-4);
        self.q = [f64::from(q_c), f64::from(q_d), q_v, 0.0];
        // Measurement noise (3 c)², R§8.11.
        self.r = 9.0;
    }

    fn start(&mut self, c: f32, slope_per_hop: f32, v: f32, v_prev: f32, a_v: f32) {
        self.x = [f64::from(c), f64::from(slope_per_hop), f64::from(v), f64::from(v_prev)];
        let pv = f64::from(a_v * a_v).max(25.0);
        self.p = [[0.0; 4]; 4];
        self.p[0][0] = 25.0;
        self.p[1][1] = 0.01;
        self.p[2][2] = pv;
        self.p[3][3] = pv;
        self.active = true;
    }

    /// One predict/update step with measurement `z = c + v + noise`.
    fn step(&mut self, z: f32) -> f32 {
        let f = [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, self.a1, self.a2],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut xp = [0.0f64; 4];
        for i in 0..4 {
            xp[i] = (0..4).map(|k| f[i][k] * self.x[k]).sum();
        }
        // P = F P Fᵀ + Q
        let mut fp = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                fp[i][j] = (0..4).map(|k| f[i][k] * self.p[k][j]).sum();
            }
        }
        let mut pp = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                pp[i][j] = (0..4).map(|k| fp[i][k] * f[j][k]).sum();
            }
            pp[i][i] += self.q[i];
        }
        // Update with H = [1, 0, 1, 0].
        let hp: [f64; 4] = core::array::from_fn(|j| pp[0][j] + pp[2][j]);
        let s = hp[0] + hp[2] + self.r;
        let innov = f64::from(z) - (xp[0] + xp[2]);
        if !(s > 0.0) || !innov.is_finite() {
            self.active = false;
            return self.x[0] as f32;
        }
        for i in 0..4 {
            let k = hp[i] / s;
            self.x[i] = xp[i] + k * innov;
            for j in 0..4 {
                self.p[i][j] = pp[i][j] - k * hp[j];
            }
        }
        // Keep P symmetric against round-off.
        for i in 0..4 {
            for j in (i + 1)..4 {
                let m = 0.5 * (self.p[i][j] + self.p[j][i]);
                self.p[i][j] = m;
                self.p[j][i] = m;
            }
        }
        self.x[0] as f32
    }

    /// Scalar pseudo-measurement of the centre alone (`H = [1, 0, 0, 0]`).
    fn observe_centre(&mut self, z: f32, r: f64) -> f32 {
        let s = self.p[0][0] + r;
        let innov = f64::from(z) - self.x[0];
        if !(s > 0.0) || !innov.is_finite() {
            return self.x[0] as f32;
        }
        let hp = self.p[0];
        for i in 0..4 {
            let k = hp[i] / s;
            self.x[i] += k * innov;
            for j in 0..4 {
                self.p[i][j] -= k * hp[j];
            }
        }
        self.x[0] as f32
    }
}

/// Causal vibrato/centre tracker (see module docs).
pub(crate) struct VibratoTracker {
    hop_s: f32,
    use_kalman: bool,
    /// AR(2) pole radius per hop.
    rho: f32,
    /// Centre random-walk variance per hop.
    q_c: f32,
    /// Centre-velocity random-walk variance per hop.
    q_d: f32,
    t: f64,
    // Turning-point detection.
    dir: i8,
    run: Point,
    run_min: Point,
    run_max: Point,
    started: bool,
    ext: [Point; 3],
    n_ext: usize,
    // Valid (hp seconds, half-extent) history, newest last.
    hist: [(f32, f32); ENGAGE_COUNT],
    n_hist: usize,
    mean_hp: f32,
    mean_a: f32,
    engaged: bool,
    w: f32,
    w_step: f32,
    anchors: [Point; 2],
    n_anchor: usize,
    slope: f32,
    offset: f32,
    last_ext_t: f64,
    prev_input: f32,
    kalman: Resonator,
}

impl VibratoTracker {
    pub fn new(hop_s: f32, use_kalman: bool) -> Self {
        let scale = hop_s * 1000.0 / REF_HOP_MS;
        Self {
            hop_s,
            use_kalman,
            rho: RHO_REF.powf(scale),
            q_c: Q_CENTRE_REF * scale,
            q_d: Q_SLOPE_REF * scale,
            t: 0.0,
            dir: 0,
            run: Point::default(),
            run_min: Point::default(),
            run_max: Point::default(),
            started: false,
            ext: [Point::default(); 3],
            n_ext: 0,
            hist: [(0.0, 0.0); ENGAGE_COUNT],
            n_hist: 0,
            mean_hp: 0.0,
            mean_a: 0.0,
            engaged: false,
            w: 0.0,
            w_step: 0.0,
            anchors: [Point::default(); 2],
            n_anchor: 0,
            slope: 0.0,
            offset: 0.0,
            last_ext_t: 0.0,
            prev_input: 0.0,
            kalman: Resonator::default(),
        }
    }

    /// Forget everything (note forgotten, or unvoiced past the hold time).
    pub fn reset(&mut self) {
        *self = Self::new(self.hop_s, self.use_kalman);
    }

    /// Enable/disable the HQ/Max Kalman refinement (tier change).
    pub fn set_kalman(&mut self, on: bool) {
        if on != self.use_kalman {
            self.use_kalman = on;
            self.kalman.active = false;
            if on && self.engaged {
                let c = self.line_centre();
                self.kalman.start(c, self.slope * self.hop_s, self.prev_input - c, self.prev_input - c, self.mean_a);
                self.configure_kalman();
            }
        }
    }

    fn hops(&self, seconds: f32) -> f32 {
        (seconds / self.hop_s).max(1.0)
    }

    fn line_centre(&self) -> f32 {
        match self.n_anchor {
            0 => self.prev_input,
            _ => {
                let a = self.anchors[1];
                a.p + self.slope * (self.t - a.t) as f32 + self.offset
            }
        }
    }

    fn configure_kalman(&mut self) {
        if self.mean_hp > 0.0 {
            let f_v = 0.5 / self.mean_hp;
            self.kalman.configure(f_v, self.mean_a, self.hop_s, self.rho, self.q_c, self.q_d);
        }
    }

    /// Feed one hop of pre-smoothed pitch (cents). Returns the split.
    pub fn update(&mut self, y: f32) -> VibratoOut {
        self.t += f64::from(self.hop_s);
        if let Some(e) = self.detect(y) {
            self.on_extremum(e);
        }

        // Timeout: no extremum within 1.6 mean half-periods.
        if self.engaged && (self.t - self.last_ext_t) as f32 > HP_RATIO_MAX * self.mean_hp {
            self.ramp_out(self.mean_hp);
        }

        let decay = (-self.hop_s / (0.5 * self.mean_hp).max(self.hop_s)).exp();
        self.offset *= decay;
        let mut centre = self.line_centre();
        if self.kalman.active && self.w > 0.0 {
            centre = self.kalman.step(y);
            if self.n_anchor > 0 {
                centre = self.kalman.observe_centre(self.line_centre(), LINE_R);
            }
        }

        // Transition: the pitch left the vibrato band, so this is a new note
        // or a glide. Reset the centre to the input and drop out quickly.
        if self.w > 0.0 && (y - centre).abs() > 1.8 * self.mean_a + 25.0 {
            self.clear_history();
            self.n_anchor = 0;
            self.offset = 0.0;
            self.ramp_out(TRANSITION_RAMP_MS * 1e-3);
            centre = y;
        }

        self.w = (self.w + self.w_step).clamp(0.0, 1.0);
        if self.w <= 0.0 {
            self.w_step = 0.0;
            self.kalman.active = false;
            if !self.engaged {
                centre = y;
            }
        }
        self.prev_input = y;

        let on = self.w > 0.0;
        VibratoOut {
            centre: if centre.is_finite() { centre } else { y },
            w: self.w,
            rate_hz: if on && self.mean_hp > 0.0 { 0.5 / self.mean_hp } else { 0.0 },
            extent: if on { self.mean_a } else { 0.0 },
        }
    }

    fn ramp_out(&mut self, seconds: f32) {
        self.engaged = false;
        self.w_step = -1.0 / self.hops(seconds);
    }

    fn clear_history(&mut self) {
        self.n_hist = 0;
    }

    /// Turning-point detector with hysteresis.
    fn detect(&mut self, y: f32) -> Option<Point> {
        let now = Point { t: self.t, p: y };
        if !self.started {
            self.started = true;
            self.run_min = now;
            self.run_max = now;
            return None;
        }
        match self.dir {
            0 => {
                if y > self.run_max.p {
                    self.run_max = now;
                }
                if y < self.run_min.p {
                    self.run_min = now;
                }
                if y < self.run_max.p - HYSTERESIS {
                    let e = self.run_max;
                    self.dir = -1;
                    self.run = now;
                    Some(e)
                } else if y > self.run_min.p + HYSTERESIS {
                    let e = self.run_min;
                    self.dir = 1;
                    self.run = now;
                    Some(e)
                } else {
                    None
                }
            }
            1 => {
                if y > self.run.p {
                    self.run = now;
                    None
                } else if y < self.run.p - HYSTERESIS {
                    let e = self.run;
                    self.dir = -1;
                    self.run = now;
                    Some(e)
                } else {
                    None
                }
            }
            _ => {
                if y < self.run.p {
                    self.run = now;
                    None
                } else if y > self.run.p + HYSTERESIS {
                    let e = self.run;
                    self.dir = 1;
                    self.run = now;
                    Some(e)
                } else {
                    None
                }
            }
        }
    }

    fn on_extremum(&mut self, e: Point) {
        self.ext = [self.ext[1], self.ext[2], e];
        self.n_ext = (self.n_ext + 1).min(3);
        self.last_ext_t = e.t;
        if self.n_ext < 2 {
            return;
        }
        let prev = self.ext[1];
        let hp = (e.t - prev.t) as f32;
        let a = 0.5 * (e.p - prev.p).abs();
        let rate = if hp > 0.0 { 0.5 / hp } else { 0.0 };
        let valid = (RATE_MIN_HZ..=RATE_MAX_HZ).contains(&rate) && (EXTENT_MIN..=EXTENT_MAX).contains(&a);
        if !valid {
            self.clear_history();
            if self.engaged {
                self.ramp_out(self.mean_hp);
            }
            return;
        }
        self.hist = [self.hist[1], self.hist[2], (hp, a)];
        self.n_hist = (self.n_hist + 1).min(ENGAGE_COUNT);
        let recent = &self.hist[ENGAGE_COUNT - self.n_hist..];
        let n = recent.len() as f32;
        self.mean_hp = recent.iter().map(|h| h.0).sum::<f32>() / n;
        self.mean_a = recent.iter().map(|h| h.1).sum::<f32>() / n;

        // Anchor from the last three extrema (two valid half-periods).
        if self.n_hist >= 2 && self.n_ext == 3 {
            let before = self.line_centre();
            let anchor = Point {
                t: self.ext[1].t,
                p: 0.25 * (self.ext[0].p + 2.0 * self.ext[1].p + self.ext[2].p),
            };
            self.anchors = [self.anchors[1], anchor];
            self.n_anchor = (self.n_anchor + 1).min(2);
            self.slope = if self.n_anchor == 2 {
                let dt = (self.anchors[1].t - self.anchors[0].t) as f32;
                if dt > 0.0 { ((self.anchors[1].p - self.anchors[0].p) / dt).clamp(-SLOPE_MAX, SLOPE_MAX) } else { 0.0 }
            } else {
                0.0
            };
            // Continuity: carry the old/new line difference as a decaying offset.
            self.offset = 0.0;
            if self.w > 0.0 {
                let after = self.line_centre();
                self.offset = before - after;
            }
        }

        if self.use_kalman && self.w > 0.0 {
            self.configure_kalman();
        }

        if !self.engaged && self.n_hist >= ENGAGE_COUNT && self.consistent() {
            self.engaged = true;
            self.w_step = 1.0 / self.hops(self.mean_hp);
            if self.use_kalman {
                self.configure_kalman();
                if !self.kalman.active {
                    let c = self.line_centre();
                    self.kalman.start(c, self.slope * self.hop_s, self.prev_input - c, self.prev_input - c, self.mean_a);
                }
            }
        }
    }

    fn consistent(&self) -> bool {
        let (mut hp_lo, mut hp_hi, mut a_lo, mut a_hi) = (f32::MAX, 0.0f32, f32::MAX, 0.0f32);
        for &(hp, a) in &self.hist {
            hp_lo = hp_lo.min(hp);
            hp_hi = hp_hi.max(hp);
            a_lo = a_lo.min(a);
            a_hi = a_hi.max(a);
        }
        hp_lo > 0.0 && a_lo > 0.0 && hp_hi / hp_lo <= HP_RATIO_MAX && a_hi / a_lo <= EXTENT_RATIO_MAX
    }
}

/// Half-extent (cents) and rate (Hz) of a sinusoid sampled every hop: a helper
/// for tests that need to know what the tracker should report.
#[cfg(test)]
pub(crate) fn sine_contour(t: f64, centre: f32, extent: f32, rate: f32) -> f32 {
    centre + extent * (2.0 * PI * f64::from(rate) * t).sin() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOP_S: f32 = 64.0 / 48_000.0;

    fn run(tracker: &mut VibratoTracker, secs: f32, f: impl Fn(f64) -> f32) -> Vec<(f32, VibratoOut)> {
        let n = (secs / HOP_S) as usize;
        (0..n)
            .map(|i| {
                let y = f(f64::from(i as f32 * HOP_S));
                (y, tracker.update(y))
            })
            .collect()
    }

    fn check_steady_vibrato(kalman: bool) {
        let mut tr = VibratoTracker::new(HOP_S, kalman);
        let out = run(&mut tr, 2.0, |t| sine_contour(t, 6030.0, 40.0, 6.0));
        // Engages within ~1.5 cycles (≈ 250 ms) and reports rate/extent.
        let engaged_at = out.iter().position(|(_, o)| o.w >= 1.0).unwrap_or(usize::MAX);
        assert!((engaged_at as f32) * HOP_S < 0.45, "engaged at {engaged_at}");
        let (_, last) = out[out.len() - 1];
        assert!((last.rate_hz - 6.0).abs() < 0.3, "rate {}", last.rate_hz);
        assert!((last.extent - 40.0).abs() < 4.0, "extent {}", last.extent);
        // Centre ripple after engagement is small compared with the vibrato.
        let tail = &out[out.len() / 2..];
        let max_err = tail.iter().map(|(_, o)| (o.centre - 6030.0).abs()).fold(0.0, f32::max);
        assert!(max_err < 6.0, "centre error {max_err} (kalman {kalman})");
    }

    #[test]
    fn steady_vibrato_extrema() {
        check_steady_vibrato(false);
    }

    #[test]
    fn steady_vibrato_kalman() {
        check_steady_vibrato(true);
    }

    #[test]
    fn drifting_centre_is_followed() {
        for kalman in [false, true] {
            let mut tr = VibratoTracker::new(HOP_S, kalman);
            let out = run(&mut tr, 2.0, |t| sine_contour(t, 6000.0 + 30.0 * t as f32, 35.0, 5.5));
            let tail = &out[out.len() / 2..];
            let n = out.len() / 2;
            let max_err = tail
                .iter()
                .enumerate()
                .map(|(i, (_, o))| {
                    let t = (n + i) as f32 * HOP_S;
                    (o.centre - (6000.0 + 30.0 * t)).abs()
                })
                .fold(0.0, f32::max);
            assert!(max_err < 8.0, "drift error {max_err} (kalman {kalman})");
        }
    }

    #[test]
    fn steady_and_slow_pitch_never_engage() {
        let mut tr = VibratoTracker::new(HOP_S, true);
        let out = run(&mut tr, 1.0, |_| 6000.0);
        assert!(out.iter().all(|(y, o)| o.w == 0.0 && o.centre == *y));
        // ±5 c hover at 2 Hz: too small and too slow.
        let mut tr = VibratoTracker::new(HOP_S, false);
        let out = run(&mut tr, 2.0, |t| sine_contour(t, 6000.0, 5.0, 2.0));
        assert!(out.iter().all(|(_, o)| o.w == 0.0));
        // A 2 Hz ±60 c wobble is too slow to be vibrato.
        let mut tr = VibratoTracker::new(HOP_S, false);
        let out = run(&mut tr, 2.0, |t| sine_contour(t, 6000.0, 60.0, 2.0));
        assert!(out.iter().all(|(_, o)| o.w == 0.0));
    }

    #[test]
    fn transition_resets_centre() {
        let mut tr = VibratoTracker::new(HOP_S, false);
        let _ = run(&mut tr, 1.0, |t| sine_contour(t, 6000.0, 30.0, 6.0));
        // Jump up a whole tone: the centre must follow within a few hops.
        let out = run(&mut tr, 0.05, |_| 6200.0);
        let (_, last) = out[out.len() - 1];
        assert!((last.centre - 6200.0).abs() < 1.0);
        assert!(last.w < 1e-6);
    }

    #[test]
    fn vibrato_stop_disengages() {
        let mut tr = VibratoTracker::new(HOP_S, true);
        let _ = run(&mut tr, 1.0, |t| sine_contour(t, 6000.0, 30.0, 6.0));
        let out = run(&mut tr, 0.5, |_| 6000.0);
        let (_, last) = out[out.len() - 1];
        assert_eq!(last.w, 0.0);
        assert_eq!(last.centre, 6000.0);
    }
}
