//! MIDI targets (R§7.6.3): Target, Learn, Auto, All Octaves, bend, sustain,
//! all-notes-off, note-on during an unvoiced gap.

use super::*;
use skato_core::{MidiEvent, MidiMode, NoteState, QualityTier, ScaleKind, frame_flags};

fn on(note: u8) -> MidiEvent {
    MidiEvent::NoteOn { note, velocity: 0.9 }
}

fn last(outs: &[ControlOutput]) -> ControlOutput {
    outs[outs.len() - 1]
}

fn midi_live(mode: MidiMode, edit: impl FnOnce(&mut Settings)) -> Corrector {
    live(|s| {
        s.midi_mode = mode;
        s.retune_ms = 0.0;
        edit(s);
    })
}

#[test]
fn target_mode_passes_through_until_a_note_is_held() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = midi_live(MidiMode::Target, |s| {
            s.tier = tier;
            s.transition_ms = Some(30.0);
        });
        let outs = hold(&mut c, 6030.0, hops(200.0));
        let o = last(&outs);
        assert!(o.shift_cents.abs() < 1e-3, "pass-through corrected: {}", o.shift_cents);
        assert_eq!(o.frame.target, 0.0);
        assert_ne!(o.frame.flags & frame_flags::BYPASSED, 0);

        // Hold E4 while singing C4+30: the only target is E4 (400 c away;
        // MIDI targets are allowed up to 1200 c).
        c.midi(on(64));
        let outs = hold(&mut c, 6030.0, hops(300.0));
        let o = last(&outs);
        assert_eq!(o.frame.target, 6400.0, "{tier:?}");
        assert!((o.frame.output - 6400.0).abs() < 1e-2, "{tier:?}: {}", o.frame.output);
        assert_ne!(o.frame.flags & frame_flags::MIDI, 0);
        // It was a target change mid-phrase: transition flag on the first hop.
        assert_ne!(outs[0].frame.flags & frame_flags::TRANSITION, 0);

        // Releasing it passes audio through again, with a 50 ms release.
        c.midi(MidiEvent::NoteOff { note: 64 });
        let outs = hold(&mut c, 6030.0, hops(400.0));
        let k = hops(50.0) - 1;
        let frac = outs[k].shift_cents / 370.0;
        assert!((frac - (-1.0f32).exp()).abs() < 0.03, "release after 50 ms: {frac}");
        assert!(last(&outs).shift_cents.abs() < 0.5);
        assert_eq!(last(&outs).frame.target, 0.0);
    }
}

#[test]
fn nearest_held_note_wins_and_all_octaves_matches_pitch_classes() {
    let mut c = midi_live(MidiMode::Target, |_| {});
    c.midi(on(60));
    c.midi(on(67));
    let outs = hold(&mut c, 6550.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 6700.0);

    // Only E5 held; singing E4: absolute match is 12 st away (beyond Snap
    // Max 1200 → no correction), All Octaves finds E4.
    let mut c = midi_live(MidiMode::Target, |_| {});
    c.midi(on(76));
    let outs = hold(&mut c, 6420.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 7600.0);
    let mut c = midi_live(MidiMode::Target, |s| s.midi_all_octaves = true);
    c.midi(on(76));
    let outs = hold(&mut c, 6420.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 6400.0);
    assert!((last(&outs).frame.output - 6400.0).abs() < 1e-2);
}

#[test]
fn pitch_bend_moves_the_target_with_5ms_smoothing() {
    let mut c = midi_live(MidiMode::Target, |s| s.bend_range_st = 2.0);
    c.midi(on(60));
    hold(&mut c, 6000.0, hops(100.0));
    c.midi(MidiEvent::PitchBend { value: 0.5 });
    let outs = hold(&mut c, 6000.0, hops(100.0));
    // +1 st after the 5 ms one-pole settles; 63 % after 5 ms.
    let k = hops(5.0) - 1;
    let frac = (outs[k].frame.target - 6000.0) / 100.0;
    assert!((frac - 0.63).abs() < 0.05, "bend after 5 ms: {frac}");
    let o = last(&outs);
    assert!((o.frame.target - 6100.0).abs() < 0.05);
    assert!((o.frame.output - 6100.0).abs() < 0.05);
    // Bend is ignored with MIDI off.
    let mut c = live(|s| s.retune_ms = 0.0);
    c.midi(MidiEvent::PitchBend { value: 1.0 });
    let outs = hold(&mut c, 6000.0, hops(100.0));
    assert_eq!(last(&outs).frame.target, 6000.0);
}

#[test]
fn sustain_holds_and_all_notes_off_releases() {
    let mut c = midi_live(MidiMode::Target, |_| {});
    c.midi(on(64));
    c.midi(MidiEvent::Sustain { down: true });
    c.midi(MidiEvent::NoteOff { note: 64 });
    let outs = hold(&mut c, 6380.0, hops(100.0));
    assert_eq!(last(&outs).frame.target, 6400.0, "pedal should hold E4");
    c.midi(MidiEvent::Sustain { down: false });
    let outs = hold(&mut c, 6380.0, hops(100.0));
    assert_eq!(last(&outs).frame.target, 0.0);

    c.midi(on(62));
    c.midi(on(64));
    hold(&mut c, 6380.0, hops(50.0));
    c.midi(MidiEvent::AllNotesOff);
    let outs = hold(&mut c, 6380.0, hops(50.0));
    assert_eq!(last(&outs).frame.target, 0.0);
}

#[test]
fn learn_mode_builds_a_scale_from_note_ons() {
    let mut c = midi_live(MidiMode::Learn, |_| {});
    assert!(c.learned_states().is_none());
    // Before anything is learned the configured scale (chromatic) applies.
    let outs = hold(&mut c, 6130.0, hops(100.0));
    assert_eq!(last(&outs).frame.target, 6100.0);
    // Learn D and F#: the first note-on clears the scale.
    c.midi(on(62));
    c.midi(MidiEvent::NoteOff { note: 62 });
    c.midi(on(78));
    let learned = c.learned_states().unwrap_or([NoteState::Remove; 12]);
    let on_pcs: Vec<usize> = (0..12).filter(|&i| learned[i] == NoteState::On).collect();
    assert_eq!(on_pcs, vec![2, 6]);
    // Learned classes apply in every octave and are not MIDI targets (the
    // keys need not stay held).
    c.midi(MidiEvent::NoteOff { note: 78 });
    let outs = hold(&mut c, 6130.0, hops(300.0));
    let o = last(&outs);
    assert_eq!(o.frame.target, 6200.0);
    assert_eq!(o.frame.flags & frame_flags::MIDI, 0);
    let outs = hold(&mut c, 6560.0, hops(300.0));
    assert_eq!(last(&outs).frame.target, 6600.0);
    // Leaving and re-entering Learn re-arms the clearing.
    let mut s = Settings { midi_mode: MidiMode::Off, retune_ms: 0.0, ..Settings::default() };
    c.set_settings(&s);
    assert!(c.learned_states().is_none());
    s.midi_mode = MidiMode::Learn;
    c.set_settings(&s);
    c.midi(on(64));
    let learned = c.learned_states().unwrap_or([NoteState::Remove; 12]);
    assert_eq!((0..12).filter(|&i| learned[i] == NoteState::On).count(), 1);
}

#[test]
fn auto_mode_uses_held_notes_else_the_scale() {
    let mut c = midi_live(MidiMode::Auto, |s| s.note_states = ScaleKind::Major.note_states(0));
    let outs = hold(&mut c, 6130.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 6200.0); // scale: C# removed → D
    c.midi(on(61));
    let outs = hold(&mut c, 6130.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 6100.0); // MIDI overrides the scale
    assert_ne!(last(&outs).frame.flags & frame_flags::MIDI, 0);
    c.midi(MidiEvent::NoteOff { note: 61 });
    let outs = hold(&mut c, 6130.0, hops(200.0));
    assert_eq!(last(&outs).frame.target, 6200.0);
}

#[test]
fn midi_overrides_bypass() {
    let mut c = midi_live(MidiMode::Auto, |s| s.note_states = [NoteState::Bypass; 12]);
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert!(last(&outs).shift_cents.abs() < 1e-3);
    c.midi(on(60));
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert!((last(&outs).frame.output - 6000.0).abs() < 1e-2);
}

#[test]
fn note_on_during_a_gap_snaps_at_the_voiced_onset() {
    // Slow retune: without the armed snap the correction would take ~200 ms.
    let mut c = midi_live(MidiMode::Target, |s| s.retune_ms = 200.0);
    run(&mut c, 10, |_| unvoiced());
    c.midi(on(62));
    run(&mut c, 5, |_| unvoiced());
    let outs = hold(&mut c, 6150.0, hops(50.0));
    // Onset confirmation (2 hops) + the 5 ms voiced ramp, then exactly D.
    let k = 1 + hops(5.0) + 1;
    assert!((outs[k].frame.output - 6200.0).abs() < 0.5, "output {}", outs[k].frame.output);
}

#[test]
fn hq_and_max_follow_midi_targets_too() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        let mut c = midi_live(MidiMode::Target, |s| s.tier = tier);
        c.midi(on(55));
        let outs = hold(&mut c, 6020.0, hops(300.0));
        assert_eq!(last(&outs).frame.target, 5500.0, "{tier:?}");
        c.midi(MidiEvent::NoteOff { note: 55 });
        c.midi(on(67));
        let outs = hold(&mut c, 6020.0, hops(300.0));
        assert_eq!(last(&outs).frame.target, 6700.0, "{tier:?}");
        assert!((last(&outs).frame.output - 6700.0).abs() < 1e-2, "{tier:?}");
    }
}
