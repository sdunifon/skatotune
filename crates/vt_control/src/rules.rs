//! Eco note decision by rules, R§7.6.2 F: fat1-style bias toward the current
//! note plus Waves' OR-tolerance.
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
        let mid = 50.0 * (current + k) as f32;
        let dir = if k > current { 1.0 } else { -1.0 };
        let x = (rel - mid) * dir;
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
        let cents_on = tol_cents > 0.0;
        let time_on = tol_hops > 0;
        let switch = (cents_on && x > tol_cents) || (time_on && self.dwell >= tol_hops) || (!cents_on && !time_on && x > 0.0);
        if switch {
            self.reset();
            Some(k)
        } else {
            None
        }
    }
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
    fn illegal_current_note_moves() {
        let mut t = TargetSet::default();
        let mut states = [vt_core::NoteState::On; 12];
        states[0] = vt_core::NoteState::Remove;
        t.set_scale(&states);
        let mut r = RuleDecider::default();
        assert_eq!(r.decide(60, 6010.0, &t, 8.0, 0), Some(59));
    }
}
