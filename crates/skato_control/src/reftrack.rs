//! Singer tuning tracker for the Labs "Follow singer's tuning"
//! ([`skato_core::ReferenceMode::FollowSinger`],
//! `docs/research/note-intent.md` §5.3): an online estimate of how far the
//! singer's notes sit from the 12-TET grid, which then places the note grid
//! (and, latched per note, the target) on the singer's own reference.
//!
//! **Evidence.** The vibrato-free centre of each voiced hop is split into
//! segments (held notes: stretches whose centre stays within 40 c of the
//! segment's running mean). Hops from 50 ms into a segment are evidence; the
//! sample is their deviation from the nearest legal note (hops more than
//! 50 c from every legal note are skipped).
//!
//! **Estimate.** A circular mean on the 100-cent circle of every evidence
//! hop, confidence-weighted and forgotten with a time constant of 4 s of
//! evidence (silence forgets nothing). It is shrunk toward 0 (A440) by its
//! resultant length R (0 at R ≤ 0.25, full at R ≥ 0.45), clamped to ±45 c
//! and smoothed over 100 ms. A seam guard keeps singers who alternate on both
//! sides of the grid (say ±40 c, which a circular mean puts at the ±50 c
//! seam) near A440: when the circular estimate disagrees by more than 15 c
//! with a per-segment linear estimate anchored at A440 (Mauch et al., JASA
//! 2014: `r ← r + (1 − μ)·wrap(segment deviation − r)`, μ = 0.85), the linear
//! one is used. The applied offset never sweeps across the seam: it jumps.
//!
//! This is the note-intent prototype A's tracker ("fg45"). On real vocadito
//! singers it lowered the evaluation objective in every configuration, so it
//! is an opt-in for consistently detuned material, never a default; on a
//! detuned-singer benchmark it recovered most of the lost note intent.
//!
//! Realtime: fixed-size state, O(1) per hop (at most one sin/cos, exp and
//! atan2 per evidence hop), no allocation.

use core::f32::consts::TAU;

/// Forgetting time constant, seconds of evidence.
const TAU_S: f32 = 4.0;
/// Per-segment forgetting of the linear (seam-guard) estimate.
const MU: f32 = 0.85;
/// Resultant-length gate: 0 → 1 as R goes from `R0` to `R1`.
const R0: f32 = 0.25;
const R1: f32 = 0.45;
/// Clamp of the applied offset, cents.
const R_MAX: f32 = 45.0;
/// Segment band, cents.
const BAND_C: f32 = 40.0;
/// Hops count as evidence from this long into a segment, ms.
const MIN_RUN_MS: f32 = 50.0;
/// Shortest segment (ms of evidence) that updates the linear estimate.
const MIN_SEG_MS: f32 = 80.0;
/// Smoothing of the applied offset, ms.
const SMOOTH_MS: f32 = 100.0;
/// Seam guard, cents.
const GUARD_C: f32 = 15.0;
/// How far past ±50 c the unwrapped circular offset may go before it wraps.
const SEAM_H: f32 = 10.0;
/// The segment's running mean averages over at most this long, ms.
const MEAN_MS: f32 = 100.0;

#[inline]
fn wrap50(x: f32) -> f32 {
    x - 100.0 * (x / 100.0).round()
}

#[inline]
fn ramp(x: f32, a: f32, b: f32) -> f32 {
    ((x - a) / (b - a)).clamp(0.0, 1.0)
}

/// The tracker (see the module docs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefTracker {
    hop_s: f32,
    /// Hop counts of `MIN_RUN_MS`, `MIN_SEG_MS` and `MEAN_MS`.
    run_hops: u32,
    seg_min_hops: u32,
    mean_hops: u32,
    /// Smoothing coefficient per hop.
    alpha: f32,
    // Evidence-weighted phasor.
    gx: f32,
    gy: f32,
    gw: f32,
    /// Linear (A440-anchored) estimate.
    lr: f32,
    // Current segment: evidence phasor, evidence hops, run length and mean.
    sx: f32,
    sy: f32,
    sw: f32,
    seg_hops: u32,
    run: u32,
    seg_mean: f32,
    /// Applied offset (smoothed, clamped), cents.
    r: f32,
    /// Unwrapped circular estimate.
    u: f32,
    r_goal: f32,
}

impl RefTracker {
    pub fn new(hop_s: f32) -> Self {
        let hops = |ms: f32| (ms * 1e-3 / hop_s).round().max(1.0) as u32;
        Self {
            hop_s,
            run_hops: hops(MIN_RUN_MS),
            seg_min_hops: hops(MIN_SEG_MS),
            mean_hops: hops(MEAN_MS),
            alpha: 1.0 - (-hop_s * 1e3 / SMOOTH_MS).exp(),
            gx: 0.0,
            gy: 0.0,
            gw: 0.0,
            lr: 0.0,
            sx: 0.0,
            sy: 0.0,
            sw: 0.0,
            seg_hops: 0,
            run: 0,
            seg_mean: 0.0,
            r: 0.0,
            u: 0.0,
            r_goal: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.hop_s);
    }

    /// The offset of the singer's grid from A440, cents (0 until there is
    /// evidence).
    #[inline]
    pub fn offset(&self) -> f32 {
        self.r
    }

    /// An unvoiced (or octave-guarded) hop.
    pub fn unvoiced(&mut self) {
        self.end_segment();
        self.run = 0;
        self.smooth();
    }

    /// A voiced hop. `d`: the analysis centre relative to the base grid
    /// (absolute cents, for the segmenter); `dev`: its deviation from the
    /// nearest legal note (None when more than 50 c from every legal note);
    /// `restart`: a fresh note or phrase.
    pub fn voiced(&mut self, d: f32, dev: Option<f32>, conf: f32, restart: bool) {
        if !d.is_finite() {
            return;
        }
        if restart || self.run == 0 || (d - self.seg_mean).abs() > BAND_C {
            self.end_segment();
            self.run = 1;
            self.seg_mean = d;
        } else {
            self.run = self.run.saturating_add(1);
            let n = (self.run as f32).min(self.mean_hops as f32);
            self.seg_mean += (d - self.seg_mean) / n;
        }
        let Some(dev) = dev.filter(|x| x.is_finite() && x.abs() <= 50.0) else {
            self.smooth();
            return;
        };
        if self.run >= self.run_hops && conf >= 0.0 {
            let th = TAU * dev / 100.0;
            let (s, c) = th.sin_cos();
            let w = conf.clamp(0.05, 1.0) * self.hop_s;
            self.sx += w * c;
            self.sy += w * s;
            self.sw += w;
            self.seg_hops = self.seg_hops.saturating_add(1);
            let lam = (-w / TAU_S).exp();
            self.gx = lam * self.gx + w * c;
            self.gy = lam * self.gy + w * s;
            self.gw = lam * self.gw + w;
            self.update_goal();
        }
        self.smooth();
    }

    /// Close the current segment: a long enough one updates the linear
    /// estimate of the seam guard.
    fn end_segment(&mut self) {
        if self.seg_hops >= self.seg_min_hops && self.sw > 0.0 {
            let m = (self.sx * self.sx + self.sy * self.sy).sqrt();
            if m > 0.0 {
                let seg_dev = self.sy.atan2(self.sx) * 100.0 / TAU;
                let e = wrap50(seg_dev - self.lr);
                self.lr += (1.0 - MU) * e;
                self.update_goal();
            }
        }
        self.sx = 0.0;
        self.sy = 0.0;
        self.sw = 0.0;
        self.seg_hops = 0;
    }

    fn update_goal(&mut self) {
        let (x, y, w) = (self.gx, self.gy, self.gw);
        let (est, g_r) = if w <= 0.0 {
            (0.0, 0.0)
        } else {
            let m = (x * x + y * y).sqrt();
            let res = (m / w).clamp(0.0, 1.0);
            let raw = if m > 0.0 { y.atan2(x) * 100.0 / TAU } else { 0.0 };
            // Unwrap against the previous estimate; wrap only SEAM_H past the seam.
            let mut u = raw;
            while u - self.u > 50.0 {
                u -= 100.0;
            }
            while u - self.u < -50.0 {
                u += 100.0;
            }
            if u > 50.0 + SEAM_H {
                u -= 100.0;
            } else if u < -50.0 - SEAM_H {
                u += 100.0;
            }
            self.u = u;
            let g_r = ramp(res, R0, R1);
            // Seam guard: a circular mean far from the A440-anchored linear
            // estimate yields to it.
            if wrap50(u - self.lr).abs() > GUARD_C { (self.lr, g_r) } else { (u, g_r) }
        };
        self.r_goal = (g_r * est).clamp(-R_MAX, R_MAX);
    }

    fn smooth(&mut self) {
        if (self.r_goal - self.r).abs() > 50.0 {
            // Crossed the seam: jump (a sweep would re-decide every note on the way).
            self.r = self.r_goal;
            return;
        }
        self.r += self.alpha * (self.r_goal - self.r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOP_S: f32 = 64.0 / 44_100.0;

    /// Held notes of `(deviation c, duration s)` on consecutive semitones
    /// with 60 ms gaps; returns the applied offset at the end.
    fn sing(t: &mut RefTracker, notes: &[(f32, f32)]) -> f32 {
        for (i, &(dev, dur)) in notes.iter().enumerate() {
            let base = 6000.0 + 100.0 * (i % 5) as f32;
            for _ in 0..(dur / HOP_S) as usize {
                let d = base + dev;
                t.voiced(d, Some(wrap50(d)), 0.9, false);
            }
            for _ in 0..(0.06 / HOP_S) as usize {
                t.unvoiced();
            }
        }
        t.offset()
    }

    #[test]
    fn a_consistently_detuned_singer_is_followed() {
        let mut t = RefTracker::new(HOP_S);
        // The guard holds A440 until the linear estimate has moved toward
        // +30 c (a few segments), then the circular mean takes over.
        let r = sing(&mut t, &[(30.0, 0.5); 24]);
        assert!((r - 30.0).abs() < 3.0, "{r}");
        let mut flat = RefTracker::new(HOP_S);
        let r = sing(&mut flat, &[(-12.0, 0.5); 12]);
        assert!((r + 12.0).abs() < 2.0, "{r}");
    }

    #[test]
    fn alternating_plus_minus_40_stays_near_a440() {
        let notes: Vec<(f32, f32)> = (0..16).map(|i| (if i % 2 == 0 { 40.0 } else { -40.0 }, 0.5)).collect();
        let mut t = RefTracker::new(HOP_S);
        assert!(sing(&mut t, &notes).abs() < 10.0);
    }

    #[test]
    fn an_inconsistent_singer_is_gated_to_a440() {
        let devs = [-45.0, 10.0, 35.0, -20.0, 0.0, 45.0, -30.0, 20.0, -10.0, 30.0, -40.0, 5.0];
        let notes: Vec<(f32, f32)> = devs.iter().map(|&d| (d, 0.4)).collect();
        let mut t = RefTracker::new(HOP_S);
        assert!(sing(&mut t, &notes).abs() < 5.0);
    }

    #[test]
    fn bounded_finite_and_reset() {
        let mut t = RefTracker::new(HOP_S);
        let mut notes = vec![(48.0, 0.6); 6];
        notes.extend(vec![(-48.0, 0.6); 6]);
        let r = sing(&mut t, &notes);
        assert!(r.is_finite() && r.abs() <= R_MAX);
        t.voiced(f32::NAN, None, 0.9, false);
        t.voiced(6000.0, Some(f32::INFINITY), f32::NAN, false);
        t.voiced(6000.0, Some(10.0), f32::NAN, true);
        assert!(t.offset().is_finite());
        t.reset();
        assert_eq!(t.offset(), 0.0);
    }
}
