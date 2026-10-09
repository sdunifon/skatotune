//! MIDI target state, R§7.6.3.
//!
//! Tracks held keys, the sustain pedal (CC64 holds released notes), pitch bend
//! and the Learn-Scale pitch classes. Events are applied either immediately
//! (Live modes: they take effect at the next hop, ≤ 1.33 ms later) or through
//! a small fixed-capacity queue (Studio: delayed so they line up with the
//! delayed audio, minus half the transition time so a glide is centred on the
//! note-on, R§7.6.3 "Timing"). The queue preserves event order and never
//! allocates; if it overflows, the oldest event is applied early.

use skato_core::{MidiEvent, NoteState};

use crate::targets::N_NOTES;

/// Queue capacity (events pending for Studio alignment).
const QUEUE_CAP: usize = 256;

pub(crate) struct MidiState {
    /// Keys physically held.
    held: [bool; N_NOTES],
    /// Keys released while the pedal was down (still sounding).
    sustained: [bool; N_NOTES],
    pedal: bool,
    /// Normalised bend, -1..=1.
    bend: f32,
    learned: [bool; 12],
    /// At least one note has been learned since Learn mode was entered.
    learn_has: bool,
    /// The next note-on clears the learned scale first.
    learn_armed: bool,
    queue: [(u64, MidiEvent); QUEUE_CAP],
    q_head: usize,
    q_len: usize,
    last_due: u64,
    /// The effective held set changed (targets must be rebuilt).
    changed: bool,
    /// A note-on was applied (used to arm an onset snap during unvoiced gaps).
    note_on: bool,
}

impl MidiState {
    pub fn new() -> Self {
        Self {
            held: [false; N_NOTES],
            sustained: [false; N_NOTES],
            pedal: false,
            bend: 0.0,
            learned: [false; 12],
            learn_has: false,
            learn_armed: true,
            queue: [(0, MidiEvent::AllNotesOff); QUEUE_CAP],
            q_head: 0,
            q_len: 0,
            last_due: 0,
            changed: true,
            note_on: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Called when the MIDI mode switches into Learn: the next note-on
    /// clears the learned scale (R§7.6.3).
    pub fn arm_learn(&mut self) {
        self.learn_armed = true;
        self.learn_has = false;
        self.changed = true;
    }

    /// Queue an event to take effect at hop `due` (applied immediately when
    /// `due <= now`). Due times are forced monotone so order is preserved.
    pub fn push(&mut self, ev: MidiEvent, due: u64, now: u64, learn_mode: bool) {
        let due = due.max(self.last_due);
        self.last_due = due;
        if due <= now && self.q_len == 0 {
            self.apply(ev, learn_mode);
            return;
        }
        if self.q_len == QUEUE_CAP {
            // Overflow: apply the oldest event early rather than drop it.
            let (_, old) = self.queue[self.q_head];
            self.q_head = (self.q_head + 1) % QUEUE_CAP;
            self.q_len -= 1;
            self.apply(old, learn_mode);
        }
        let idx = (self.q_head + self.q_len) % QUEUE_CAP;
        self.queue[idx] = (due, ev);
        self.q_len += 1;
    }

    /// Apply every queued event whose due hop has arrived.
    pub fn apply_due(&mut self, now: u64, learn_mode: bool) {
        while self.q_len > 0 {
            let (due, ev) = self.queue[self.q_head];
            if due > now {
                break;
            }
            self.q_head = (self.q_head + 1) % QUEUE_CAP;
            self.q_len -= 1;
            self.apply(ev, learn_mode);
        }
    }

    fn apply(&mut self, ev: MidiEvent, learn_mode: bool) {
        match ev {
            MidiEvent::NoteOn { note, velocity } if velocity > 0.0 => {
                let n = usize::from(note);
                if n >= N_NOTES {
                    return;
                }
                self.held[n] = true;
                self.note_on = true;
                if learn_mode {
                    if self.learn_armed {
                        self.learned = [false; 12];
                        self.learn_armed = false;
                    }
                    self.learn_has = true;
                    self.learned[n % 12] = true;
                }
                self.changed = true;
            }
            // Velocity-0 note-on is a note-off by MIDI convention.
            MidiEvent::NoteOn { note, .. } | MidiEvent::NoteOff { note } => {
                let n = usize::from(note);
                if n >= N_NOTES {
                    return;
                }
                if self.held[n] {
                    self.held[n] = false;
                    if self.pedal {
                        self.sustained[n] = true;
                    }
                    self.changed = true;
                }
            }
            MidiEvent::PitchBend { value } => {
                self.bend = if value.is_finite() { value.clamp(-1.0, 1.0) } else { 0.0 };
            }
            MidiEvent::Sustain { down } => {
                self.pedal = down;
                if !down && self.sustained.iter().any(|s| *s) {
                    self.sustained = [false; N_NOTES];
                    self.changed = true;
                }
            }
            MidiEvent::AllNotesOff => {
                self.held = [false; N_NOTES];
                self.sustained = [false; N_NOTES];
                self.pedal = false;
                self.changed = true;
            }
        }
    }

    /// Effective held notes (keys down or sustained by the pedal).
    pub fn held_effective(&self) -> [bool; N_NOTES] {
        let mut out = [false; N_NOTES];
        for (o, (h, s)) in out.iter_mut().zip(self.held.iter().zip(self.sustained.iter())) {
            *o = *h || *s;
        }
        out
    }

    pub fn any_held(&self) -> bool {
        self.held.iter().zip(self.sustained.iter()).any(|(h, s)| *h || *s)
    }

    /// Normalised pitch bend, -1..=1.
    #[inline]
    pub fn bend(&self) -> f32 {
        self.bend
    }

    /// Learned scale (learned classes On, others Remove), once anything has
    /// been learned since Learn mode was entered.
    pub fn learned_states(&self) -> Option<[NoteState; 12]> {
        if !self.learn_has {
            return None;
        }
        let mut s = [NoteState::Remove; 12];
        for (st, l) in s.iter_mut().zip(self.learned.iter()) {
            if *l {
                *st = NoteState::On;
            }
        }
        Some(s)
    }

    /// Returns and clears the "held set changed" flag.
    pub fn take_changed(&mut self) -> bool {
        core::mem::take(&mut self.changed)
    }

    /// Returns and clears the "a note-on was applied" flag.
    pub fn take_note_on(&mut self) -> bool {
        core::mem::take(&mut self.note_on)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(note: u8) -> MidiEvent {
        MidiEvent::NoteOn { note, velocity: 0.8 }
    }

    #[test]
    fn sustain_holds_released_notes() {
        let mut m = MidiState::new();
        m.push(on(60), 0, 0, false);
        m.push(MidiEvent::Sustain { down: true }, 0, 0, false);
        m.push(MidiEvent::NoteOff { note: 60 }, 0, 0, false);
        assert!(m.held_effective()[60]);
        m.push(MidiEvent::Sustain { down: false }, 0, 0, false);
        assert!(!m.any_held());
    }

    #[test]
    fn velocity_zero_is_note_off_and_panic_clears() {
        let mut m = MidiState::new();
        m.push(on(60), 0, 0, false);
        m.push(MidiEvent::NoteOn { note: 60, velocity: 0.0 }, 0, 0, false);
        assert!(!m.any_held());
        m.push(on(62), 0, 0, false);
        m.push(MidiEvent::Sustain { down: true }, 0, 0, false);
        m.push(MidiEvent::AllNotesOff, 0, 0, false);
        assert!(!m.any_held());
        m.push(on(200), 0, 0, false); // out of range: ignored, no panic
        assert!(!m.any_held());
    }

    #[test]
    fn learn_clears_on_first_note() {
        let mut m = MidiState::new();
        assert!(m.learned_states().is_none());
        m.push(on(62), 0, 0, true);
        m.push(on(66), 0, 0, true);
        let s = m.learned_states().unwrap_or([NoteState::Remove; 12]);
        assert_eq!(s[2], NoteState::On);
        assert_eq!(s[6], NoteState::On);
        assert_eq!(s[0], NoteState::Remove);
        m.arm_learn();
        assert!(m.learned_states().is_none());
        m.push(on(60), 0, 0, true);
        let s = m.learned_states().unwrap_or([NoteState::Remove; 12]);
        assert_eq!(s[2], NoteState::Remove);
        assert_eq!(s[0], NoteState::On);
    }

    #[test]
    fn queue_delays_and_keeps_order() {
        let mut m = MidiState::new();
        m.push(on(60), 10, 0, false);
        // A later event with an earlier nominal due time cannot overtake it.
        m.push(MidiEvent::NoteOff { note: 60 }, 5, 1, false);
        m.apply_due(9, false);
        assert!(!m.any_held());
        m.apply_due(10, false);
        assert!(!m.any_held()); // on then off, both at hop 10
        m.push(on(64), 20, 11, false);
        m.apply_due(19, false);
        assert!(!m.any_held());
        m.apply_due(20, false);
        assert!(m.held_effective()[64]);
    }

    #[test]
    fn queue_overflow_applies_oldest() {
        let mut m = MidiState::new();
        for i in 0..(QUEUE_CAP + 10) {
            m.push(on((i % 100) as u8), 1_000, 0, false);
        }
        // The overflowed (oldest) note-ons were applied early.
        assert!(m.any_held());
    }
}
