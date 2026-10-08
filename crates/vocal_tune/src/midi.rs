//! Host note events → engine MIDI events.
//!
//! nice-plug decodes host MIDI into [`NoteEvent`]s; the control law wants the
//! smaller [`MidiEvent`] vocabulary of R§7.6.3 (notes, bend, sustain, panic).
//! Conversion is a pure function, and the converted events of one host block
//! live in a fixed-capacity array so the audio thread never allocates.

use nice_plug::midi::Key;
use nice_plug::prelude::*;
use vt_core::MidiEvent;
use vt_engine::TimedMidi;

/// Most MIDI events kept per host block (DESIGN §3.6). Extra events are
/// dropped; a sung line never needs anywhere near this many.
pub const MIDI_CAPACITY: usize = 512;

/// CC64: sustain (damper) pedal.
const CC_SUSTAIN: u8 = 64;
/// CC120: all sound off.
const CC_ALL_SOUND_OFF: u8 = 120;
/// CC123: all notes off.
const CC_ALL_NOTES_OFF: u8 = 123;

/// Convert one host event. `block_len` clamps the timing into the block.
/// Returns `None` for events the corrector has no use for.
pub fn convert_event<S>(ev: &NoteEvent<S>, block_len: u32) -> Option<TimedMidi> {
    let (timing, event) = match *ev {
        NoteEvent::NoteOn {
            timing,
            key: Key::Number(note),
            velocity,
            ..
        } => {
            let note = note.min(127);
            if velocity > 0.0 {
                (
                    timing,
                    MidiEvent::NoteOn {
                        note,
                        velocity: velocity.min(1.0),
                    },
                )
            } else {
                // A note-on with zero velocity is a note-off by MIDI convention.
                (timing, MidiEvent::NoteOff { note })
            }
        }
        // A host choke ends the note immediately; for a target note that is
        // the same as releasing it.
        NoteEvent::NoteOff {
            timing,
            key: Key::Number(note),
            ..
        }
        | NoteEvent::Choke {
            timing,
            key: Key::Number(note),
            ..
        } => (
            timing,
            MidiEvent::NoteOff {
                note: note.min(127),
            },
        ),
        // CLAP wildcard note-off/choke (key -1) releases every note.
        NoteEvent::NoteOff {
            timing,
            key: Key::Wildcard,
            ..
        }
        | NoteEvent::Choke {
            timing,
            key: Key::Wildcard,
            ..
        } => (timing, MidiEvent::AllNotesOff),
        NoteEvent::MidiPitchBend { timing, value, .. } => {
            // nice-plug normalises bend to [0, 1] with 0.5 = centre.
            let v = if value.is_finite() {
                (value * 2.0 - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            (timing, MidiEvent::PitchBend { value: v })
        }
        NoteEvent::MidiCC {
            timing,
            cc: CC_SUSTAIN,
            value,
            ..
        } => (timing, MidiEvent::Sustain { down: value >= 0.5 }),
        NoteEvent::MidiCC {
            timing,
            cc: CC_ALL_SOUND_OFF | CC_ALL_NOTES_OFF,
            ..
        } => (timing, MidiEvent::AllNotesOff),
        _ => return None,
    };
    Some(TimedMidi {
        timing: timing.min(block_len.saturating_sub(1)),
        event,
    })
}

/// Fixed-capacity, allocation-free list of one block's MIDI events, kept in
/// non-decreasing timing order as the engine requires.
pub struct MidiBuffer {
    events: [TimedMidi; MIDI_CAPACITY],
    len: usize,
    /// Events dropped because the buffer was full (diagnostics).
    dropped: u64,
}

impl Default for MidiBuffer {
    fn default() -> Self {
        Self {
            events: [TimedMidi {
                timing: 0,
                event: MidiEvent::AllNotesOff,
            }; MIDI_CAPACITY],
            len: 0,
            dropped: 0,
        }
    }
}

impl MidiBuffer {
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append an event. Hosts deliver events sorted, but a timing that goes
    /// backwards is clamped up to the previous one rather than trusted, so the
    /// engine's sorted-input contract always holds.
    pub fn push(&mut self, mut ev: TimedMidi) {
        if let Some(prev) = self.as_slice().last() {
            ev.timing = ev.timing.max(prev.timing);
        }
        let Some(slot) = self.events.get_mut(self.len) else {
            self.dropped = self.dropped.saturating_add(1);
            return;
        };
        *slot = ev;
        self.len += 1;
    }

    /// The events collected so far.
    pub fn as_slice(&self) -> &[TimedMidi] {
        self.events.get(..self.len).unwrap_or(&[])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use nice_plug::midi::{Channel, VoiceID};

    type Ev = NoteEvent<()>;

    fn on(timing: u32, key: u8, velocity: f32) -> Ev {
        NoteEvent::NoteOn {
            timing,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key: Key::Number(key),
            velocity,
        }
    }

    fn off(timing: u32, key: Key) -> Ev {
        NoteEvent::NoteOff {
            timing,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key,
            velocity: 0.0,
        }
    }

    #[test]
    fn notes_convert() {
        let t = convert_event(&on(10, 60, 0.8), 64).unwrap();
        assert_eq!(
            t,
            TimedMidi {
                timing: 10,
                event: MidiEvent::NoteOn {
                    note: 60,
                    velocity: 0.8
                }
            }
        );
        assert_eq!(
            convert_event(&off(3, Key::Number(60)), 64).unwrap().event,
            MidiEvent::NoteOff { note: 60 }
        );
        assert_eq!(
            convert_event(&off(3, Key::Wildcard), 64).unwrap().event,
            MidiEvent::AllNotesOff
        );
        // Velocity 0 note-on is a note-off.
        assert_eq!(
            convert_event(&on(0, 62, 0.0), 64).unwrap().event,
            MidiEvent::NoteOff { note: 62 }
        );
        let choke: Ev = NoteEvent::Choke {
            timing: 0,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Wildcard,
            key: Key::Number(64),
        };
        assert_eq!(
            convert_event(&choke, 64).unwrap().event,
            MidiEvent::NoteOff { note: 64 }
        );
    }

    #[test]
    fn bend_is_recentred() {
        let ev = |value| -> Ev {
            NoteEvent::MidiPitchBend {
                timing: 0,
                channel: 0,
                value,
            }
        };
        let get = |e: &Ev| match convert_event(e, 1).unwrap().event {
            MidiEvent::PitchBend { value } => value,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(get(&ev(0.5)), 0.0);
        assert_eq!(get(&ev(1.0)), 1.0);
        assert_eq!(get(&ev(0.0)), -1.0);
        assert_eq!(get(&ev(f32::NAN)), 0.0);
    }

    #[test]
    fn controllers() {
        let cc = |cc, value| -> Ev {
            NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc,
                value,
            }
        };
        assert_eq!(
            convert_event(&cc(64, 1.0), 1).unwrap().event,
            MidiEvent::Sustain { down: true }
        );
        assert_eq!(
            convert_event(&cc(64, 0.2), 1).unwrap().event,
            MidiEvent::Sustain { down: false }
        );
        assert_eq!(
            convert_event(&cc(123, 0.0), 1).unwrap().event,
            MidiEvent::AllNotesOff
        );
        assert_eq!(
            convert_event(&cc(120, 0.0), 1).unwrap().event,
            MidiEvent::AllNotesOff
        );
        assert!(convert_event(&cc(1, 0.5), 1).is_none());
        let pressure: Ev = NoteEvent::MidiChannelPressure {
            timing: 0,
            channel: 0,
            pressure: 0.3,
        };
        assert!(convert_event(&pressure, 1).is_none());
    }

    #[test]
    fn timing_is_clamped_into_the_block() {
        assert_eq!(convert_event(&on(500, 60, 1.0), 128).unwrap().timing, 127);
        assert_eq!(convert_event(&on(5, 60, 1.0), 0).unwrap().timing, 0);
    }

    #[test]
    fn buffer_caps_and_keeps_order() {
        let mut b = MidiBuffer::default();
        b.push(TimedMidi {
            timing: 10,
            event: MidiEvent::AllNotesOff,
        });
        b.push(TimedMidi {
            timing: 4,
            event: MidiEvent::AllNotesOff,
        });
        assert_eq!(b.as_slice()[1].timing, 10, "backwards timing is clamped up");
        for _ in 0..MIDI_CAPACITY + 10 {
            b.push(TimedMidi {
                timing: 20,
                event: MidiEvent::AllNotesOff,
            });
        }
        assert_eq!(b.len(), MIDI_CAPACITY);
        assert_eq!(b.dropped(), 12);
        b.clear();
        assert!(b.is_empty());
    }
}
