//! Eco note decision by rules, R§7.6.2 F: fat1-style bias toward the current
//! note plus Waves' OR-tolerance. HQ/Max reuse the dwell timer
//! ([`RuleDecider::observe`]) for the same rule, applied there as a gate on
//! the note HMM's decisions.
//!
//! With k the candidate nearest the decision pitch and n* the current note:
//! `mid = (pitch(n*) + pitch(k))/2`, `x = (p_dec − mid)·sign(pitch(k) − pitch(n*))`
//! (how far past the midpoint the pitch is, toward k). Switch to k when
//! `(Cents on ∧ x > T_c) ∨ (Time on ∧ x > 0 continuously for ≥ T_t) ∨
//! (both off ∧ x > 0)`: whichever enabled threshold is crossed first.
//! (Up/Down redirection is not available: `NoteState` has no Up/Down in v1.)

use crate::targets::TargetSet;

/// Rule-based decider state (the dwell timer of the Time tolerance).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RuleDecider {
    dwell: u32,
    dwell_note: usize,
}

impl RuleDecider {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns the new note when the decision switches away from `current`.
    ///
    /// `rel` is the decision pitch relative to the note grid (note n at
    /// 100·n), `tol_cents` the cents tolerance (0 = off) and `tol_hops` the
    /// time tolerance in hops (0 = off).
    pub fn decide(&mut self, current: usize, rel: f32, targets: &TargetSet, tol_cents: f32, tol_hops: u32) -> Option<usize> {
        let k = targets.nearest(rel)?;
        if !targets.is_allowed(current) {
            // The current note stopped being legal (scale/MIDI change).
            self.reset();
            return Some(k);
        }
        if k == current {
            self.dwell = 0;
            return None;
        }
        let x = past_midpoint(current, k, rel);
        if self.confirmed(k, x, tol_cents, tol_hops) {
            self.reset();
            Some(k)
        } else {
            None
        }
    }

    /// Studio variant: the rule is judged on `rel_ahead` (the median of the
    /// lookahead window), so the tolerance costs no dwell, but the switch is
    /// applied only once the frame itself (`rel_now`) has crossed the
    /// midpoint, so the delayed audio is never retuned before the singer
    /// actually moves.
    pub fn decide_ahead(
        &mut self,
        current: usize,
        rel_now: f32,
        rel_ahead: f32,
        targets: &TargetSet,
        tol_cents: f32,
        tol_hops: u32,
    ) -> Option<usize> {
        if !targets.is_allowed(current) {
            self.reset();
            return targets.nearest(rel_now);
        }
        let k = targets.nearest(rel_ahead)?;
        if k == current {
            self.dwell = 0;
            return None;
        }
        let x_ahead = past_midpoint(current, k, rel_ahead);
        let confirmed = self.confirmed(k, x_ahead, tol_cents, tol_hops);
        if confirmed && past_midpoint(current, k, rel_now) > 0.0 {
            self.reset();
            Some(k)
        } else {
            None
        }
    }

    /// HQ/Max (CL-2): advance the dwell timer of the time tolerance with
    /// `rel` (toward the nearest candidate, 0 while on `current`'s side).
    /// [`Self::dwell_toward`] then reads it for the OR-rule gate.
    pub fn observe(&mut self, current: usize, rel: f32, targets: &TargetSet) {
        match targets.nearest(rel) {
            Some(k) if k != current => self.track_dwell(k, past_midpoint(current, k, rel)),
            _ => self.dwell = 0,
        }
    }

    /// Hops the pitch has continuously been past the midpoint toward `k`.
    pub fn dwell_toward(&self, k: usize) -> u32 {
        if self.dwell_note == k { self.dwell } else { 0 }
    }

    /// Updates the dwell timer and applies the OR-tolerance to a candidate
    /// `x` cents past the midpoint toward `k`.
    fn confirmed(&mut self, k: usize, x: f32, tol_cents: f32, tol_hops: u32) -> bool {
        self.track_dwell(k, x);
        let cents_on = tol_cents > 0.0;
        let time_on = tol_hops > 0;
        (cents_on && x > tol_cents) || (time_on && self.dwell >= tol_hops) || (!cents_on && !time_on && x > 0.0)
    }

    /// Dwell timer: consecutive hops with `x > 0` toward the same candidate.
    fn track_dwell(&mut self, k: usize, x: f32) {
        if x > 0.0 {
            if self.dwell > 0 && self.dwell_note == k {
                self.dwell = self.dwell.saturating_add(1);
            } else {
                self.dwell = 1;
                self.dwell_note = k;
            }
        } else {
            self.dwell = 0;
        }
    }
}

/// How far `rel` is past the midpoint between notes `current` and `k`,
/// measured toward `k` (cents; negative when on `current`'s side).
pub(crate) fn past_midpoint(current: usize, k: usize, rel: f32) -> f32 {
    let mid = 50.0 * (current + k) as f32;
    let dir = if k > current { 1.0 } else { -1.0 };
    (rel - mid) * dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cents_tolerance() {
        let t = TargetSet::default();
        let mut r = RuleDecider::default();
        assert_eq!(r.decide(60, 6057.0, &t, 8.0, 0), None);
        assert_eq!(r.decide(60, 6059.0, &t, 8.0, 0), Some(61));
        // Downward too.
        assert_eq!(r.decide(60, 5943.0, &t, 8.0, 0), None);
        assert_eq!(r.decide(60, 5941.0, &t, 8.0, 0), Some(59));
    }

    #[test]
    fn time_tolerance_or_cents() {
        let t = TargetSet::default();
        let mut r = RuleDecider::default();
        // 5 c past the midpoint: only the dwell timer can trigger.
        for _ in 0..9 {
            assert_eq!(r.decide(60, 6055.0, &t, 20.0, 10), None);
        }
        assert_eq!(r.decide(60, 6055.0, &t, 20.0, 10), Some(61));
        // Dropping back below the midpoint restarts the timer.
        let mut r = RuleDecider::default();
        for _ in 0..5 {
            r.decide(60, 6055.0, &t, 20.0, 10);
        }
        r.decide(60, 6045.0, &t, 20.0, 10);
        for _ in 0..9 {
            assert_eq!(r.decide(60, 6055.0, &t, 20.0, 10), None);
        }
        // The cents threshold still fires immediately (OR).
        assert_eq!(r.decide(60, 6075.0, &t, 20.0, 10), Some(61));
    }

    #[test]
    fn both_off_switches_at_midpoint() {
        let t = TargetSet::default();
        let mut r = RuleDecider::default();
        assert_eq!(r.decide(60, 6049.0, &t, 0.0, 0), None);
        assert_eq!(r.decide(60, 6051.0, &t, 0.0, 0), Some(61));
    }

    #[test]
    fn lookahead_confirms_but_waits_for_the_crossing() {
        let t = TargetSet::default();
        let mut r = RuleDecider::default();
        // The future is clearly on C#, but this frame is still on C.
        assert_eq!(r.decide_ahead(60, 6020.0, 6100.0, &t, 8.0, 0), None);
        // Crossed the midpoint (inside the tolerance) with the future confirming.
        assert_eq!(r.decide_ahead(60, 6052.0, 6100.0, &t, 8.0, 0), Some(61));
        // Crossing without confirmation (the future comes back) holds.
        let mut r = RuleDecider::default();
        assert_eq!(r.decide_ahead(60, 6060.0, 6040.0, &t, 8.0, 0), None);
    }

    #[test]
    fn illegal_current_note_moves() {
        let mut t = TargetSet::default();
        let mut states = [skato_core::NoteState::On; 12];
        states[0] = skato_core::NoteState::Remove;
        t.set_scale(&states);
        let mut r = RuleDecider::default();
        assert_eq!(r.decide(60, 6010.0, &t, 8.0, 0), Some(61));
        assert_eq!(r.decide(60, 5990.0, &t, 8.0, 0), Some(59));
    }
}
