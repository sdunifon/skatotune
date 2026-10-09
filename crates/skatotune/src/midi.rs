//! Host note events → engine MIDI events.
//!
//! nice-plug decodes host MIDI into [`NoteEvent`]s; the control law wants the
//! smaller [`MidiEvent`] vocabulary of R§7.6.3 (notes, bend, sustain, panic).
//! Conversion is a pure function, and the converted events of one host block
//! live in a fixed-capacity array so the audio thread never allocates.

use nice_plug::midi::Key;
use nice_plug::prelude::*;
use skato_core::MidiEvent;
use skato_engine::TimedMidi;

/// Most MIDI events kept per host block (DESIGN §3.6). The buffer never
/// drops state: pitch bends are coalesced (see [`MidiBuffer::push`]) and the
/// last [`RESERVED_SLOTS`] slots are kept for note and pedal events.
pub const MIDI_CAPACITY: usize = 512;

/// Slots only note, sustain and all-notes-off events may use, so a flood of
/// pitch bends can never push out a note-off.
pub const RESERVED_SLOTS: usize = 32;

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
///
/// The engine hands MIDI to the control law at hop boundaries, and the pitch
/// bend is plain state: only the last bend value delivered before each
/// boundary is ever used (R§7.6.3). So bends that fall in the same control hop
/// collapse into one slot without changing the result, which bounds them to
/// one per hop. Should bends still fill the buffer (huge offline blocks), the
/// newest value overwrites the latest queued bend, so the bend state at the
/// end of the block is always right; and the last [`RESERVED_SLOTS`] slots
/// stay free for notes and the pedal. If even those run out, a release
/// (note-off, sustain up, all-notes-off) turns the final slot into
/// all-notes-off: no note can be left stuck by an overflow.
pub struct MidiBuffer {
    events: [TimedMidi; MIDI_CAPACITY],
    len: usize,
    /// Index of the newest queued pitch bend in this block.
    last_bend: Option<usize>,
    /// Control hop and the engine's position inside it at the block start,
    /// so `(phase + timing) / hop` names the hop an event is delivered in.
    hop: u32,
    phase: u32,
    /// Events dropped or merged because the buffer was full (diagnostics).
    /// Same-hop bend coalescing is lossless and not counted.
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
            last_bend: None,
            hop: 1,
            phase: 0,
            dropped: 0,
        }
    }
}

/// Whether dropping this event could leave a note or the pedal held.
fn is_release(ev: &MidiEvent) -> bool {
    matches!(
        ev,
        MidiEvent::NoteOff { .. } | MidiEvent::AllNotesOff | MidiEvent::Sustain { down: false }
    )
}

impl MidiBuffer {
    /// Empty the buffer for a new block without hop information: only bends
    /// at the same sample are coalesced.
    pub fn clear(&mut self) {
        self.begin_block(1, 0);
    }

    /// Empty the buffer for a new block. `hop` is the engine's control hop
    /// and `phase` how many samples of the current hop the engine has
    /// already consumed (its absolute position modulo `hop`).
    pub fn begin_block(&mut self, hop: u32, phase: u32) {
        self.len = 0;
        self.last_bend = None;
        self.hop = hop.max(1);
        self.phase = phase % self.hop;
    }

    /// Control hop an event at `timing` is delivered in (relative to the block).
    fn hop_of(&self, timing: u32) -> u64 {
        (u64::from(self.phase) + u64::from(timing)) / u64::from(self.hop)
    }

    /// Append an event. Hosts deliver events sorted, but a timing that goes
    /// backwards is clamped up to the previous one rather than trusted, so the
    /// engine's sorted-input contract always holds. See the type docs for
    /// what happens when the buffer fills up.
    pub fn push(&mut self, mut ev: TimedMidi) {
        if let Some(prev) = self.as_slice().last() {
            ev.timing = ev.timing.max(prev.timing);
        }
        if matches!(ev.event, MidiEvent::PitchBend { .. }) {
            self.push_bend(ev);
        } else {
            self.push_other(ev);
        }
    }

    fn push_bend(&mut self, ev: TimedMidi) {
        let MidiEvent::PitchBend { value } = ev.event else {
            return;
        };
        let last_index = self.len.checked_sub(1);
        if let Some(i) = self.last_bend {
            let same_hop = self
                .events
                .get(i)
                .is_some_and(|b| self.hop_of(b.timing) == self.hop_of(ev.timing));
            // Every event queued after slot `i` is in the same hop as both
            // bends (timings are sorted), so updating the value in place is
            // exactly what the control law would have seen.
            let full = self.len >= MIDI_CAPACITY - RESERVED_SLOTS;
            if same_hop || full {
                if !same_hop {
                    self.dropped = self.dropped.saturating_add(1);
                }
                if let Some(slot) = self.events.get_mut(i) {
                    if Some(i) == last_index {
                        // Nothing after it: move it to the new time as well.
                        *slot = ev;
                    } else {
                        slot.event = MidiEvent::PitchBend { value };
                    }
                }
                return;
            }
        } else if self.len >= MIDI_CAPACITY - RESERVED_SLOTS {
            // Only reachable with more than 480 note/pedal events in one
            // block. Bends never enter the reserved slots, so the release
            // fallback below never has to overwrite one.
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        if let Some(slot) = self.events.get_mut(self.len) {
            *slot = ev;
            self.last_bend = Some(self.len);
            self.len += 1;
        }
    }

    fn push_other(&mut self, ev: TimedMidi) {
        if let Some(slot) = self.events.get_mut(self.len) {
            *slot = ev;
            self.len += 1;
            return;
        }
        self.dropped = self.dropped.saturating_add(1);
        if is_release(&ev.event)
            && let Some(i) = self.len.checked_sub(1)
            && let Some(slot) = self.events.get_mut(i)
        {
            // Full even for notes. Releasing everything is the only way to
            // be sure the dropped release cannot leave a note stuck. The
            // final slot never holds a bend here (bends stay out of the
            // reserved slots).
            *slot = TimedMidi {
                timing: ev.timing,
                event: MidiEvent::AllNotesOff,
            };
            if self.last_bend == Some(i) {
                self.last_bend = None;
            }
        }
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

    fn bend(timing: u32, value: f32) -> TimedMidi {
        TimedMidi {
            timing,
            event: MidiEvent::PitchBend { value },
        }
    }

    fn last_bend(b: &MidiBuffer) -> Option<f32> {
        b.as_slice().iter().rev().find_map(|e| match e.event {
            MidiEvent::PitchBend { value } => Some(value),
            _ => None,
        })
    }

    #[test]
    fn bend_flood_never_drops_the_note_off() {
        // RT-3: 1000 bends in a 2048-sample block followed by a note-off.
        let mut b = MidiBuffer::default();
        b.clear();
        for i in 0..1000u32 {
            b.push(bend(i * 2, (i as f32 / 999.0) * 2.0 - 1.0));
        }
        b.push(TimedMidi {
            timing: 2040,
            event: MidiEvent::NoteOff { note: 60 },
        });
        assert!(b.len() <= MIDI_CAPACITY);
        assert_eq!(
            b.as_slice().last().map(|e| e.event),
            Some(MidiEvent::NoteOff { note: 60 }),
            "the note-off must survive the flood"
        );
        assert_eq!(last_bend(&b), Some(1.0), "the newest bend value is kept");
    }

    #[test]
    fn bends_coalesce_per_control_hop_only() {
        // Hop 64, engine already 60 samples into the current hop: samples 0-3
        // close that hop, 4-67 are the next one.
        let mut b = MidiBuffer::default();
        b.begin_block(64, 60);
        b.push(bend(0, 0.1));
        b.push(bend(3, 0.2)); // same hop: replaces 0.1
        b.push(bend(4, 0.3)); // next hop
        b.push(TimedMidi {
            timing: 10,
            event: MidiEvent::NoteOn {
                note: 60,
                velocity: 1.0,
            },
        });
        b.push(bend(67, 0.4)); // same hop as 0.3, a note in between
        b.push(bend(68, 0.5)); // third hop
        let got: Vec<_> = b.as_slice().iter().map(|e| (e.timing, e.event)).collect();
        assert_eq!(
            got,
            vec![
                (3, MidiEvent::PitchBend { value: 0.2 }),
                (4, MidiEvent::PitchBend { value: 0.4 }),
                (
                    10,
                    MidiEvent::NoteOn {
                        note: 60,
                        velocity: 1.0
                    }
                ),
                (68, MidiEvent::PitchBend { value: 0.5 }),
            ]
        );
        assert_eq!(b.dropped(), 0, "same-hop coalescing is lossless");
    }

    #[test]
    fn notes_keep_reserved_slots_after_a_bend_flood() {
        // Hop 1: every bend is its own hop, so none coalesce.
        let mut b = MidiBuffer::default();
        b.begin_block(1, 0);
        for i in 0..2000u32 {
            b.push(bend(i, if i % 2 == 0 { 0.5 } else { -0.5 }));
        }
        assert_eq!(b.len(), MIDI_CAPACITY - RESERVED_SLOTS);
        assert_eq!(last_bend(&b), Some(-0.5), "newest bend value kept");
        for n in 0..RESERVED_SLOTS as u8 {
            b.push(TimedMidi {
                timing: 2000,
                event: MidiEvent::NoteOff { note: n },
            });
        }
        assert_eq!(b.len(), MIDI_CAPACITY);
        let offs = b
            .as_slice()
            .iter()
            .filter(|e| matches!(e.event, MidiEvent::NoteOff { .. }))
            .count();
        assert_eq!(offs, RESERVED_SLOTS, "every note-off has a slot");
        let sorted = b.as_slice().windows(2).all(|w| w[0].timing <= w[1].timing);
        assert!(sorted);
    }

    #[test]
    fn a_release_that_cannot_fit_becomes_all_notes_off() {
        let mut b = MidiBuffer::default();
        for i in 0..MIDI_CAPACITY as u32 {
            b.push(TimedMidi {
                timing: i,
                event: MidiEvent::NoteOn {
                    note: (i % 128) as u8,
                    velocity: 1.0,
                },
            });
        }
        // A note-on that does not fit is simply dropped (nothing gets stuck).
        b.push(TimedMidi {
            timing: 600,
            event: MidiEvent::NoteOn {
                note: 1,
                velocity: 1.0,
            },
        });
        assert_eq!(b.dropped(), 1);
        for release in [
            MidiEvent::NoteOff { note: 3 },
            MidiEvent::Sustain { down: false },
        ] {
            b.push(TimedMidi {
                timing: 700,
                event: release,
            });
            assert_eq!(b.len(), MIDI_CAPACITY);
            assert_eq!(
                b.as_slice().last().copied(),
                Some(TimedMidi {
                    timing: 700,
                    event: MidiEvent::AllNotesOff
                })
            );
        }
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
