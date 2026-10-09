//! MIDI Learn Scale, kept by the plugin as user state (R§7.6.3).
//!
//! The control law can learn a scale itself (`MidiMode::Learn`), but it keeps
//! the result in signal state: `Engine::reset()` (every VST3
//! `setProcessing(true)`, every CLAP `start_processing`), any restart and any
//! project reload would forget it while the mode still reads Learn. So the
//! plugin learns the scale here instead:
//!
//! - The learned pitch classes live in one word,
//!   [`VocalTuneParams::learned_scale`](crate::params::VocalTuneParams), a
//!   `#[persist]` field. The host saves it with the project (always, even
//!   when empty, so loading a project without one clears it), and engine
//!   resets or rebuilds never touch it.
//! - Learn *edits the scale* (R§7.6.3: "clears the scale, then each note-on
//!   sets that pitch class On"). So a learned scale replaces the Key/Scale
//!   note states in every MIDI mode (Off, Auto without held notes, Learn)
//!   until the user clears it ([`clear`], the editor's CLEAR) or turns it
//!   into note overrides (the editor's KEEP). Leaving Learn keeps it.
//! - While MIDI Mode is Learn, the engine runs with `MidiMode::Off` and the
//!   learned scale as its note states. That is exactly what the corrector's
//!   own Learn mode does with its learned states, so correction behaves the
//!   same.
//! - Entering Learn arms it: the first note-on after that clears the scale,
//!   and every note-on adds its pitch class. Until that first note the
//!   previous learned scale (if any) stays in use, so leaving Learn and
//!   coming back does not lose it.
//!
//! Word layout: bit 31 = something has been learned, bit `pc` = pitch class
//! `pc` learned. All operations are single atomic loads, stores or
//! read-modify-writes (realtime safe), so the audio thread (note-ons) and the
//! GUI thread (clicks on the keyboard, CLEAR) can both edit it.

use std::sync::atomic::{AtomicU32, Ordering};

use vt_core::{MidiEvent, MidiMode, NoteState, Settings};

use crate::params::MidiModeChoice;

/// Bit 31 of the learned-scale word: a scale has been learned.
pub const LEARNED_VALID: u32 = 1 << 31;

/// Learned scale (learned classes On, the rest Remove), if any.
pub fn unpack(word: u32) -> Option<[NoteState; 12]> {
    if word & LEARNED_VALID == 0 {
        return None;
    }
    Some(std::array::from_fn(|pc| {
        if word & (1 << pc) != 0 {
            NoteState::On
        } else {
            NoteState::Remove
        }
    }))
}

/// Word for a set of learned pitch classes (empty = nothing learned).
pub fn pack(classes: &[u8]) -> u32 {
    classes
        .iter()
        .fold(LEARNED_VALID, |w, &pc| w | (1 << (pc % 12)))
}

/// Apply Learn to the engine settings: a learned scale replaces the
/// parameter scale in every MIDI mode, and in Learn mode the engine gets
/// `MidiMode::Off` (the plugin does the learning).
pub fn apply(settings: &mut Settings, mode: MidiModeChoice, word: u32) {
    if mode == MidiModeChoice::Learn {
        settings.midi_mode = MidiMode::Off;
    }
    if let Some(states) = unpack(word) {
        settings.note_states = states;
    }
}

/// Forget the learned scale (back to Key/Scale and the note overrides).
pub fn clear(word: &AtomicU32) {
    word.store(0, Ordering::Relaxed);
}

/// Toggle pitch class `pc` in the learned scale (a click on the keyboard
/// while a learned scale is in use). Never removes the last learned class:
/// an empty scale would silently stop all correction, and CLEAR is the way
/// back to Key/Scale. Does nothing while nothing has been learned.
pub fn toggle(word: &AtomicU32, pc: usize) {
    let bit = 1u32 << (pc % 12);
    // A compare-exchange loop so a note-on landing at the same moment is
    // never lost (the audio thread uses a single fetch_or).
    let _ = word.try_update(Ordering::Relaxed, Ordering::Relaxed, |w| {
        let classes = w & !LEARNED_VALID;
        (w & LEARNED_VALID != 0 && classes != bit).then_some(w ^ bit)
    });
}

/// Per-instance Learn bookkeeping on the audio thread: whether the next
/// note-on starts a new scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct LearnTracker {
    /// The next note-on clears the scale first.
    armed: bool,
    /// Whether the previous block was in Learn (`None` before the first block,
    /// so loading a project saved in Learn continues its scale).
    was_learning: Option<bool>,
}

impl LearnTracker {
    /// Call once per block, before its events, with the block's MIDI mode.
    pub fn begin_block(&mut self, mode: MidiModeChoice) {
        let learning = mode == MidiModeChoice::Learn;
        if learning && self.was_learning == Some(false) {
            self.armed = true;
        }
        self.was_learning = Some(learning);
    }

    /// Feed one converted MIDI event; note-ons in Learn mode update `word`.
    pub fn event(&mut self, ev: &MidiEvent, word: &AtomicU32) {
        if self.was_learning != Some(true) {
            return;
        }
        let MidiEvent::NoteOn { note, velocity } = *ev else {
            return;
        };
        if velocity <= 0.0 {
            return;
        }
        let learned = LEARNED_VALID | (1 << (note % 12));
        if self.armed {
            word.store(learned, Ordering::Relaxed);
        } else {
            word.fetch_or(learned, Ordering::Relaxed);
        }
        self.armed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(note: u8) -> MidiEvent {
        MidiEvent::NoteOn {
            note,
            velocity: 0.8,
        }
    }

    #[test]
    fn note_ons_build_the_scale_in_learn_mode_only() {
        let word = AtomicU32::new(0);
        let mut t = LearnTracker::default();
        t.begin_block(MidiModeChoice::Off);
        t.event(&on(60), &word);
        assert_eq!(unpack(word.load(Ordering::Relaxed)), None);

        t.begin_block(MidiModeChoice::Learn);
        for n in [60, 64, 67, 72] {
            t.event(&on(n), &word);
        }
        t.event(&MidiEvent::NoteOff { note: 62 }, &word);
        assert_eq!(word.load(Ordering::Relaxed), pack(&[0, 4, 7]));
    }

    #[test]
    fn re_entering_learn_keeps_the_scale_until_the_next_note() {
        let word = AtomicU32::new(0);
        let mut t = LearnTracker::default();
        t.begin_block(MidiModeChoice::Off);
        t.begin_block(MidiModeChoice::Learn);
        t.event(&on(62), &word);
        t.event(&on(66), &word);
        t.begin_block(MidiModeChoice::Off);
        t.begin_block(MidiModeChoice::Learn);
        assert_eq!(word.load(Ordering::Relaxed), pack(&[2, 6]), "still there");
        t.event(&on(69), &word);
        assert_eq!(
            word.load(Ordering::Relaxed),
            pack(&[9]),
            "first note after arming starts a new scale"
        );
    }

    #[test]
    fn a_session_that_starts_in_learn_continues_the_saved_scale() {
        // Project saved in Learn and reloaded: the first block is already
        // Learn, so new notes add to the restored scale.
        let word = AtomicU32::new(pack(&[0, 4, 7]));
        let mut t = LearnTracker::default();
        t.begin_block(MidiModeChoice::Learn);
        t.event(&on(71), &word);
        assert_eq!(word.load(Ordering::Relaxed), pack(&[0, 4, 7, 11]));
    }

    #[test]
    fn keyboard_edits_and_clear() {
        let word = AtomicU32::new(0);
        toggle(&word, 4);
        assert_eq!(
            unpack(word.load(Ordering::Relaxed)),
            None,
            "nothing learned"
        );
        word.store(pack(&[0, 4]), Ordering::Relaxed);
        toggle(&word, 7);
        assert_eq!(word.load(Ordering::Relaxed), pack(&[0, 4, 7]));
        toggle(&word, 0);
        assert_eq!(word.load(Ordering::Relaxed), pack(&[4, 7]));
        toggle(&word, 4);
        toggle(&word, 7);
        assert_eq!(
            word.load(Ordering::Relaxed),
            pack(&[7]),
            "the last learned note stays"
        );
        clear(&word);
        assert_eq!(unpack(word.load(Ordering::Relaxed)), None);
    }

    #[test]
    fn apply_swaps_learn_for_the_learned_scale() {
        let mut s = Settings {
            midi_mode: MidiMode::Learn,
            ..Settings::default()
        };
        let param_states = s.note_states;
        apply(&mut s, MidiModeChoice::Learn, 0);
        assert_eq!(s.midi_mode, MidiMode::Off);
        assert_eq!(s.note_states, param_states, "nothing learned yet");
        apply(&mut s, MidiModeChoice::Learn, pack(&[2, 4, 6]));
        assert_eq!(s.note_states, unpack(pack(&[2, 4, 6])).unwrap());
        assert_eq!(s.note_states[2], NoteState::On);
        assert_eq!(s.note_states[3], NoteState::Remove);

        // Outside Learn the learned scale still is the scale (Auto falls back
        // to it), and the MIDI mode is left alone.
        let mut auto = Settings {
            midi_mode: MidiMode::Auto,
            ..Settings::default()
        };
        apply(&mut auto, MidiModeChoice::Auto, pack(&[2]));
        assert_eq!(auto.midi_mode, MidiMode::Auto);
        assert_eq!(auto.note_states, unpack(pack(&[2])).unwrap());
        let mut off = Settings::default();
        let before = off;
        apply(&mut off, MidiModeChoice::Off, 0);
        assert_eq!(off, before, "nothing learned: untouched");
    }
}
