//! Candidate target notes, R§7.6.2 E.
//!
//! Notes are 12-TET MIDI numbers 0-127; the pitch of note `n` is
//! `100·n + offset` cents, where the offset is Detune plus the smoothed MIDI
//! bend (R§7.6.1). The candidate set is either
//!
//! - the scale: every note whose pitch class is `On` or `Bypass` (`Remove`
//!   notes are excluded, so pitch near them is pulled to the nearest *other*
//!   legal note), with a chromatic fallback when nothing is legal; or
//! - held MIDI notes (Target, or Auto with notes held): the held notes
//!   themselves, or every octave of their pitch classes with All Octaves. MIDI
//!   targets override `Bypass` (Waves/Antares semantics); or
//! - nothing (MIDI Target with no note held): audio passes through untuned.
//!
//! The set is a fixed 128-entry mask so building and searching it never
//! allocates; it is rebuilt only when the settings or the held notes change.

use vt_core::NoteState;

/// Number of note slots (MIDI 0-127).
pub(crate) const N_NOTES: usize = 128;

/// Where the current candidates come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TargetSource {
    Scale,
    Midi,
    /// MIDI Target mode with no note held: correction releases to 0.
    PassThrough,
}

/// The legal target notes for the current hop.
#[derive(Clone, Debug)]
pub(crate) struct TargetSet {
    allowed: [bool; N_NOTES],
    bypass: [bool; N_NOTES],
    count: usize,
    source: TargetSource,
}

impl Default for TargetSet {
    fn default() -> Self {
        let mut t = Self {
            allowed: [false; N_NOTES],
            bypass: [false; N_NOTES],
            count: 0,
            source: TargetSource::Scale,
        };
        t.set_scale(&[NoteState::On; 12]);
        t
    }
}

impl TargetSet {
    /// Candidates from per-pitch-class states (On/Bypass legal, Remove not).
    /// Falls back to chromatic when no pitch class is legal (R§7.6.2 E).
    pub fn set_scale(&mut self, states: &[NoteState; 12]) {
        let legal = states.iter().any(|s| *s != NoteState::Remove);
        self.count = 0;
        for n in 0..N_NOTES {
            let st = if legal { states[n % 12] } else { NoteState::On };
            self.allowed[n] = st != NoteState::Remove;
            self.bypass[n] = st == NoteState::Bypass;
            self.count += usize::from(self.allowed[n]);
        }
        self.source = TargetSource::Scale;
    }

    /// Candidates from held MIDI notes. `held` is indexed by MIDI note.
    /// With nothing held the set becomes [`TargetSource::PassThrough`].
    pub fn set_midi(&mut self, held: &[bool; N_NOTES], all_octaves: bool) {
        let mut pc = [false; 12];
        for (n, h) in held.iter().enumerate() {
            if *h {
                pc[n % 12] = true;
            }
        }
        self.count = 0;
        for n in 0..N_NOTES {
            self.allowed[n] = if all_octaves { pc[n % 12] } else { held[n] };
            self.bypass[n] = false;
            self.count += usize::from(self.allowed[n]);
        }
        self.source = if self.count == 0 { TargetSource::PassThrough } else { TargetSource::Midi };
    }

    #[inline]
    pub fn source(&self) -> TargetSource {
        self.source
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn is_allowed(&self, n: usize) -> bool {
        self.allowed.get(n).copied().unwrap_or(false)
    }

    /// The note is legal but must not be corrected (scale Bypass, never MIDI).
    #[inline]
    pub fn is_bypass(&self, n: usize) -> bool {
        self.bypass.get(n).copied().unwrap_or(false)
    }

    /// Allowed-note mask (for the note HMM).
    #[inline]
    pub fn allowed_mask(&self) -> &[bool; N_NOTES] {
        &self.allowed
    }

    /// Nearest legal note to `rel` cents, where `rel = pitch − offset` (so
    /// note `n` sits at exactly `100·n`). `None` when the set is empty.
    pub fn nearest(&self, rel: f32) -> Option<usize> {
        if self.count == 0 || !rel.is_finite() {
            return None;
        }
        let n0 = (rel / 100.0).round().clamp(0.0, (N_NOTES - 1) as f32) as i32;
        for k in 0..N_NOTES as i32 {
            let lo = n0 - k;
            let hi = n0 + k;
            let lo_ok = lo >= 0 && self.is_allowed(lo as usize);
            let hi_ok = hi < N_NOTES as i32 && self.is_allowed(hi as usize);
            match (lo_ok, hi_ok) {
                (true, true) => {
                    let dl = (rel - 100.0 * lo as f32).abs();
                    let dh = (rel - 100.0 * hi as f32).abs();
                    return Some(if dl <= dh { lo as usize } else { hi as usize });
                }
                (true, false) => return Some(lo as usize),
                (false, true) => return Some(hi as usize),
                (false, false) => {}
            }
        }
        None
    }

    /// Next legal note strictly above (`dir > 0`) or below (`dir < 0`) `n`.
    pub fn neighbour(&self, n: usize, dir: i32) -> Option<usize> {
        let step = if dir >= 0 { 1 } else { -1 };
        let mut m = n as i32 + step;
        while (0..N_NOTES as i32).contains(&m) {
            if self.is_allowed(m as usize) {
                return Some(m as usize);
            }
            m += step;
        }
        None
    }

    /// Gap in cents from `n` to the adjacent legal note on side `dir` (used by
    /// Flex-Tune and the HMM capture widths). Falls back to the other side,
    /// then to 100 c, at the ends of the range.
    pub fn gap(&self, n: usize, dir: i32) -> f32 {
        let dist = |m: usize| 100.0 * (m as f32 - n as f32).abs();
        match (self.neighbour(n, dir), self.neighbour(n, -dir)) {
            (Some(m), _) | (None, Some(m)) => dist(m),
            (None, None) => 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vt_core::ScaleKind;

    #[test]
    fn chromatic_nearest() {
        let t = TargetSet::default();
        assert_eq!(t.nearest(6020.0), Some(60));
        assert_eq!(t.nearest(6049.0), Some(60));
        assert_eq!(t.nearest(6051.0), Some(61));
        assert_eq!(t.nearest(-500.0), Some(0));
        assert_eq!(t.nearest(99_999.0), Some(127));
        assert_eq!(t.nearest(f32::NAN), None);
        assert_eq!(t.gap(60, 1), 100.0);
    }

    #[test]
    fn scale_remove_and_bypass() {
        // C major: C# (61) is Remove, so 6060 goes to the nearest legal note.
        let mut states = ScaleKind::Major.note_states(0);
        states[4] = NoteState::Bypass; // E
        let mut t = TargetSet::default();
        t.set_scale(&states);
        assert_eq!(t.nearest(6110.0), Some(62)); // C#+10 → D
        assert_eq!(t.nearest(6090.0), Some(60)); // C#-10 → C
        assert!(t.is_allowed(64) && t.is_bypass(64));
        assert!(!t.is_allowed(61));
        assert_eq!(t.gap(60, 1), 200.0);
        assert_eq!(t.gap(64, 1), 100.0);
        assert_eq!(t.neighbour(62, -1), Some(60));
    }

    #[test]
    fn empty_scale_falls_back_to_chromatic() {
        let mut t = TargetSet::default();
        t.set_scale(&[NoteState::Remove; 12]);
        assert!(!t.is_empty());
        assert_eq!(t.nearest(6130.0), Some(61));
    }

    #[test]
    fn midi_sets() {
        let mut held = [false; N_NOTES];
        held[64] = true;
        let mut t = TargetSet::default();
        t.set_midi(&held, false);
        assert_eq!(t.source(), TargetSource::Midi);
        assert_eq!(t.nearest(5200.0), Some(64));
        t.set_midi(&held, true);
        assert_eq!(t.nearest(5200.0), Some(52));
        t.set_midi(&[false; N_NOTES], false);
        assert_eq!(t.source(), TargetSource::PassThrough);
        assert_eq!(t.nearest(6000.0), None);
    }
}
