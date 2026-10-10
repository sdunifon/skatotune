//! Key Source (#5, #6): where the base scale comes from
//! (docs/research/host-key-scale.md §3).
//!
//! - **Manual:** Key + Scale, or a MIDI-learned scale (exactly as before).
//! - **MIDI Scale:** a chord on the MIDI input sets the scale and stays
//!   latched after release. This is how a DAW's key reaches the plugin today
//!   (Bitwig: a cluster through Key Filter+ into Skatotune's MIDI input), and
//!   how a singer can set the key from a keyboard. MIDI notes don't act as
//!   targets in this source.
//! - **Detect:** Auto-Key's confirmed key (`skato_control::autokey`).
//!
//! Until MIDI Scale has latched a chord or Detect has confirmed a key, the
//! Manual scale applies. Per-note overrides apply on top of MIDI Scale and
//! Detect as absolute pitch classes ("Remove F" stays F), and MIDI Target /
//! Auto with notes held keep precedence under Detect (performance input).
//!
//! Both followed keys are plugin state saved with the project, like the
//! learned scale (`#[persist]` atomic words), so a project reloads in key
//! before anything is played or sung. All operations are single atomic
//! loads and stores (realtime safe).

use std::sync::atomic::{AtomicU32, Ordering};

use skato_engine::KeyName;
use skato_core::{MidiEvent, MidiMode, NoteState, ScaleKind, Settings};

use crate::learn;
use crate::params::{KeySourceChoice, MidiModeChoice, NoteOverride, ParamValues};

/// Bit 31 of a saved key word: a key is stored.
pub const VALID: u32 = 1 << 31;

/// A chord needs at least this many pitch classes to set the scale (one
/// note is a melody, not a key).
pub const MIN_CHORD: u32 = 2;
/// Note-offs this soon after a note-on belong to the same chord change (a
/// legato change lifts the old notes just after the new ones arrive).
pub const CHANGE_WINDOW_MS: f32 = 50.0;

/// MIDI Scale word: bit 31 valid, bits 0-11 the pitch-class set, bits 12-15
/// the pitch class of the chord's lowest note (used to name it).
pub fn pack_midi(mask: u16, bass: u8) -> u32 {
    VALID | u32::from(mask & 0x0FFF) | (u32::from(bass % 12) << 12)
}

/// The latched set and its bass note, if any.
pub fn unpack_midi(word: u32) -> Option<(u16, u8)> {
    (word & VALID != 0 && word & 0x0FFF != 0).then_some(((word & 0x0FFF) as u16, ((word >> 12) & 0xF) as u8 % 12))
}

/// Detected-key word: bit 31 valid, bits 0-3 the tonic, bits 4-8 the scale
/// (its index in `ScaleKind::ALL`).
pub fn pack_key(key: KeyName) -> u32 {
    let kind = ScaleKind::ALL.iter().position(|&k| k == key.kind).unwrap_or(0) as u32;
    VALID | u32::from(key.root % 12) | (kind << 4)
}

pub fn unpack_key(word: u32) -> Option<KeyName> {
    if word & VALID == 0 {
        return None;
    }
    let kind = *ScaleKind::ALL.get(((word >> 4) & 0x1F) as usize)?;
    Some(KeyName::new((word & 0xF) as u8 % 12, kind))
}

/// The name for a latched MIDI set: a built-in scale on the bass note if one
/// fits, else the first that fits; None for a set no scale matches.
pub fn name_midi(mask: u16, bass: u8) -> Option<KeyName> {
    ScaleKind::identify(mask, Some(bass)).map(|(r, k)| KeyName::new(r, k))
}

/// The base set `mask` with the per-note overrides on top (absolute pitch
/// classes; `Scale` defers to the set).
pub fn mask_states(mask: u16, overrides: &[NoteOverride; 12]) -> [NoteState; 12] {
    std::array::from_fn(|pc| match overrides[pc] {
        NoteOverride::Scale if mask & (1 << pc) != 0 => NoteState::On,
        NoteOverride::Scale => NoteState::Remove,
        NoteOverride::On => NoteState::On,
        NoteOverride::Bypass => NoteState::Bypass,
        NoteOverride::Remove => NoteState::Remove,
    })
}

/// Bit 30 of the confirmed-key word in `SharedState`: restored from the
/// saved project, not yet confirmed by the singing.
pub const KEY_SAVED: u32 = 1 << 30;

/// `InEffect` as one word for the editor: bits 28-30 the variant, bits 0-15
/// the MIDI set and bass, or bits 0-8 the key (as in [`pack_key`]).
pub fn pack_in_effect(e: InEffect) -> u32 {
    match e {
        InEffect::Manual => 0,
        InEffect::Midi { mask, bass } => (1 << 28) | u32::from(mask & 0x0FFF) | (u32::from(bass % 12) << 12),
        InEffect::MidiWaiting => 2 << 28,
        InEffect::Detect(k) => (3 << 28) | (pack_key(k) & 0x1FF),
        InEffect::DetectWaiting => 4 << 28,
    }
}

pub fn unpack_in_effect(w: u32) -> InEffect {
    match (w >> 28) & 0x7 {
        1 => InEffect::Midi { mask: (w & 0x0FFF) as u16, bass: ((w >> 12) & 0xF) as u8 % 12 },
        2 => InEffect::MidiWaiting,
        3 => unpack_key(VALID | (w & 0x1FF)).map_or(InEffect::DetectWaiting, InEffect::Detect),
        4 => InEffect::DetectWaiting,
        _ => InEffect::Manual,
    }
}

/// Which scale is in effect this block (for the editor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InEffect {
    /// Key/Scale or the learned scale.
    Manual,
    /// MIDI Scale with a latched chord.
    Midi { mask: u16, bass: u8 },
    /// MIDI Scale before any chord: the manual scale applies.
    MidiWaiting,
    /// Detect with a confirmed key.
    Detect(KeyName),
    /// Detect before a key is confirmed: the manual scale applies.
    DetectWaiting,
}

/// Resolve this block's engine note states (and MIDI mode) from the Key
/// Source. `learned` is the learned-scale word, `midi` the MIDI Scale word,
/// `confirmed` Auto-Key's confirmed key.
pub fn apply(settings: &mut Settings, v: &ParamValues, learned: u32, midi: u32, confirmed: Option<KeyName>) -> InEffect {
    match v.key_source {
        KeySourceChoice::Manual => {
            learn::apply(settings, v.midi_mode, learned);
            InEffect::Manual
        }
        KeySourceChoice::MidiScale => {
            // MIDI notes define the scale here, so they are not targets.
            settings.midi_mode = MidiMode::Off;
            match unpack_midi(midi) {
                Some((mask, bass)) => {
                    settings.note_states = mask_states(mask, &v.note_overrides);
                    if let Some(k) = name_midi(mask, bass) {
                        settings.key = k.root;
                        settings.scale = k.kind;
                    }
                    InEffect::Midi { mask, bass }
                }
                None => {
                    learn::apply(settings, MidiModeChoice::Off, learned);
                    InEffect::MidiWaiting
                }
            }
        }
        KeySourceChoice::Detect => {
            // Learn belongs to the manual scale; Target/Auto stay (they are
            // performance input and take precedence over any scale).
            let mode = if v.midi_mode == MidiModeChoice::Learn { MidiModeChoice::Off } else { v.midi_mode };
            match confirmed {
                Some(k) => {
                    if mode != v.midi_mode {
                        settings.midi_mode = MidiMode::Off;
                    }
                    settings.note_states = mask_states(k.mask, &v.note_overrides);
                    settings.key = k.root;
                    settings.scale = k.kind;
                    InEffect::Detect(k)
                }
                None => {
                    learn::apply(settings, mode, learned);
                    if mode != v.midi_mode {
                        settings.midi_mode = MidiMode::Off;
                    }
                    InEffect::DetectWaiting
                }
            }
        }
    }
}

/// Latches chords from the MIDI input into the MIDI Scale word (audio
/// thread). A note-on sets the scale to every note held at that moment
/// (from two pitch classes up). Note-offs never shrink it, except when a
/// note pressed *before* the latest chord is lifted within
/// `CHANGE_WINDOW_MS` of it (a legato chord change). So a chord stays
/// latched however and however soon its notes are released.
#[derive(Clone, Debug)]
pub struct MidiScaleTracker {
    held: [u8; 128],
    /// Input position of each note's latest note-on, samples.
    on_pos: [u64; 128],
    /// Input position of the latest note-on, samples.
    last_on: Option<u64>,
}

impl Default for MidiScaleTracker {
    fn default() -> Self {
        Self { held: [0; 128], on_pos: [0; 128], last_on: None }
    }
}

impl MidiScaleTracker {
    /// Forget which notes are held (the source changed or the host reset).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn held_set(&self) -> (u16, Option<u8>) {
        let mut mask = 0u16;
        let mut bass = None;
        for (n, &c) in self.held.iter().enumerate() {
            if c > 0 {
                mask |= 1 << (n % 12);
                bass.get_or_insert((n % 12) as u8);
            }
        }
        (mask, bass)
    }

    /// Feed one MIDI event at input position `pos` (samples). Returns
    /// whether it latched a chord.
    pub fn event(&mut self, ev: &MidiEvent, pos: u64, sample_rate: f32, word: &AtomicU32) -> bool {
        match *ev {
            MidiEvent::NoteOn { note, velocity } if velocity > 0.0 => {
                let n = usize::from(note.min(127));
                self.held[n] = self.held[n].saturating_add(1);
                self.on_pos[n] = pos;
                self.last_on = Some(pos);
                self.latch(word)
            }
            MidiEvent::NoteOn { note, .. } | MidiEvent::NoteOff { note } => {
                let n = usize::from(note.min(127));
                self.held[n] = self.held[n].saturating_sub(1);
                let window = (CHANGE_WINDOW_MS * 1e-3 * sample_rate.max(1.0)) as u64;
                let older = self.last_on.is_some_and(|t| self.on_pos[n].saturating_add(window) < t);
                older && self.last_on.is_some_and(|t| pos.saturating_sub(t) <= window) && self.latch(word)
            }
            _ => false,
        }
    }

    fn latch(&self, word: &AtomicU32) -> bool {
        let (mask, bass) = self.held_set();
        let ok = mask.count_ones() >= MIN_CHORD;
        if ok {
            word.store(pack_midi(mask, bass.unwrap_or(0)), Ordering::Relaxed);
        }
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn on(note: u8) -> MidiEvent {
        MidiEvent::NoteOn { note, velocity: 0.8 }
    }

    fn off(note: u8) -> MidiEvent {
        MidiEvent::NoteOff { note }
    }

    fn set(word: &AtomicU32) -> Option<u16> {
        unpack_midi(word.load(Ordering::Relaxed)).map(|(m, _)| m)
    }

    fn mask(pcs: &[u8]) -> u16 {
        pcs.iter().fold(0, |m, &p| m | 1 << p)
    }

    #[test]
    fn a_chord_latches_and_stays_through_a_staggered_release() {
        let w = AtomicU32::new(0);
        let mut t = MidiScaleTracker::default();
        for (i, n) in [62u8, 65, 69].into_iter().enumerate() {
            t.event(&on(n), i as u64 * 100, SR, &w);
        }
        assert_eq!(set(&w), Some(mask(&[2, 5, 9])));
        // Released one by one, well after the chord.
        for (i, n) in [69u8, 62, 65].into_iter().enumerate() {
            t.event(&off(n), 48_000 + i as u64 * 4_800, SR, &w);
        }
        assert_eq!(set(&w), Some(mask(&[2, 5, 9])), "a chord stays latched");
        assert_eq!(unpack_midi(w.load(Ordering::Relaxed)).map(|(_, b)| b), Some(2), "bass D");
    }

    #[test]
    fn a_short_staccato_chord_latches_whole() {
        let w = AtomicU32::new(0);
        let mut t = MidiScaleTracker::default();
        for n in [62u8, 66, 69] {
            t.event(&on(n), 0, SR, &w);
        }
        for (i, n) in [69u8, 62, 66].into_iter().enumerate() {
            t.event(&off(n), 500 + 100 * i as u64, SR, &w); // ~10 ms later
        }
        assert_eq!(set(&w), Some(mask(&[2, 6, 9])));
    }

    #[test]
    fn a_single_note_does_not_set_a_scale() {
        let w = AtomicU32::new(0);
        let mut t = MidiScaleTracker::default();
        t.event(&on(60), 0, SR, &w);
        t.event(&off(60), 1_000, SR, &w);
        assert_eq!(set(&w), None);
    }

    #[test]
    fn a_new_chord_replaces_the_old_one_also_when_played_legato() {
        let w = AtomicU32::new(0);
        let mut t = MidiScaleTracker::default();
        // C major cluster (Key Filter+ output), released, then D minor.
        let c_major = [60u8, 62, 64, 65, 67, 69, 71];
        for &n in &c_major {
            t.event(&on(n), 0, SR, &w);
        }
        for &n in &c_major {
            t.event(&off(n), 96_000, SR, &w);
        }
        let d_dorian_ish = [60u8, 62, 64, 65, 67, 69, 70]; // F major set
        for &n in &d_dorian_ish {
            t.event(&on(n), 192_000, SR, &w);
        }
        assert_eq!(set(&w), Some(mask(&[0, 2, 4, 5, 7, 9, 10])));
        // Legato: the next cluster arrives before the old notes are lifted
        // (at the same moment, or a few ms later).
        let g_major = [60u8, 62, 64, 66, 67, 69, 71];
        for &n in &g_major {
            t.event(&on(n), 288_000, SR, &w);
        }
        for &n in &d_dorian_ish {
            t.event(&off(n), 288_000 + 240, SR, &w);
        }
        assert_eq!(set(&w), Some(mask(&[0, 2, 4, 6, 7, 9, 11])), "old notes lifted with the change");
    }

    #[test]
    fn overrides_apply_on_top_of_a_followed_set_as_absolute_pitch_classes() {
        let c_major = ScaleKind::Major.mask(0);
        let mut ovr = [NoteOverride::Scale; 12];
        ovr[5] = NoteOverride::Remove; // F
        ovr[6] = NoteOverride::Bypass; // F#
        let s = mask_states(c_major, &ovr);
        assert_eq!(s[0], NoteState::On);
        assert_eq!(s[1], NoteState::Remove);
        assert_eq!(s[5], NoteState::Remove);
        assert_eq!(s[6], NoteState::Bypass);
        // A modulation to G major keeps "Remove F" on F.
        let s = mask_states(ScaleKind::Major.mask(7), &ovr);
        assert_eq!(s[5], NoteState::Remove);
        assert_eq!(s[6], NoteState::Bypass);
    }

    #[test]
    fn saved_words_round_trip() {
        for kind in ScaleKind::ALL {
            for r in 0..12 {
                let k = KeyName::new(r, kind);
                assert_eq!(unpack_key(pack_key(k)), Some(k));
            }
        }
        assert_eq!(unpack_key(0), None);
        assert_eq!(unpack_midi(pack_midi(0b1010_1011_0101, 9)), Some((0b1010_1011_0101, 9)));
        assert_eq!(unpack_midi(0), None);
        let k = KeyName::new(4, ScaleKind::HarmonicMinor);
        for e in [
            InEffect::Manual,
            InEffect::Midi { mask: 0b1010_1101_0101, bass: 7 },
            InEffect::MidiWaiting,
            InEffect::Detect(k),
            InEffect::DetectWaiting,
        ] {
            assert_eq!(unpack_in_effect(pack_in_effect(e)), e);
        }
        assert_eq!(name_midi(ScaleKind::Major.mask(0), 9), Some(KeyName::new(9, ScaleKind::NaturalMinor)));
    }

    fn values(source: KeySourceChoice) -> ParamValues {
        ParamValues { key_source: source, ..ParamValues::default() }
    }

    #[test]
    fn each_source_resolves_its_scale_and_falls_back_to_manual() {
        let manual = values(KeySourceChoice::Manual).to_settings();
        // MIDI Scale before a chord: manual; with one: the chord, MIDI off.
        let mut v = values(KeySourceChoice::MidiScale);
        v.midi_mode = MidiModeChoice::Target;
        let mut s = v.to_settings();
        assert_eq!(apply(&mut s, &v, 0, 0, None), InEffect::MidiWaiting);
        assert_eq!(s.note_states, manual.note_states);
        assert_eq!(s.midi_mode, MidiMode::Off);
        let mut s = v.to_settings();
        let w = pack_midi(ScaleKind::Major.mask(2), 2);
        assert_eq!(apply(&mut s, &v, 0, w, None), InEffect::Midi { mask: ScaleKind::Major.mask(2), bass: 2 });
        assert_eq!(s.note_states, ScaleKind::Major.note_states(2));
        assert_eq!((s.key, s.scale), (2, ScaleKind::Major));
        // Detect before confirmation: manual; after: the key. MIDI Target
        // stays (performance input), Learn does not.
        let mut v = values(KeySourceChoice::Detect);
        v.midi_mode = MidiModeChoice::Target;
        let mut s = v.to_settings();
        assert_eq!(apply(&mut s, &v, 0, 0, None), InEffect::DetectWaiting);
        assert_eq!(s.note_states, manual.note_states);
        assert_eq!(s.midi_mode, MidiMode::Target);
        let a_minor = KeyName::new(9, ScaleKind::NaturalMinor);
        let mut s = v.to_settings();
        assert_eq!(apply(&mut s, &v, 0, 0, Some(a_minor)), InEffect::Detect(a_minor));
        assert_eq!(s.note_states, ScaleKind::NaturalMinor.note_states(9));
        assert_eq!(s.midi_mode, MidiMode::Target);
        v.midi_mode = MidiModeChoice::Learn;
        let mut s = v.to_settings();
        apply(&mut s, &v, learn::pack(&[0, 4, 7]), 0, Some(a_minor));
        assert_eq!(s.midi_mode, MidiMode::Off);
        assert_eq!(s.note_states, ScaleKind::NaturalMinor.note_states(9), "the learned scale is manual");
    }
}
