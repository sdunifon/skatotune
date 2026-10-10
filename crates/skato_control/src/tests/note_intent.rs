//! The Labs note-intent options (docs/research/note-intent.md §5-6): the
//! learned note hold only ever removes switches the control law proposed,
//! never touches an onset, steps aside for Eco and MIDI targets and without a
//! model; the release freeze waits out a falling level; follow singer's
//! tuning finds a consistently detuned singer, latches the offset per note
//! and ignores MIDI targets.

use super::*;
use crate::corrector::Change;
use skato_core::{MidiEvent, MidiMode, QualityTier, ReferenceMode};

/// Deterministic pseudo-random numbers in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A wandering line like the real singing the veto was trained on: notes
/// on random semitones sitting anywhere within ±45 c of the grid, wobble and
/// drift across midpoints, scoops, level decays at the ends of notes, gaps.
fn wandering(seed: u64, n: usize, level: bool) -> Vec<PitchEstimate> {
    let mut r = Lcg(seed);
    let mut out = Vec::with_capacity(n);
    let mut note = 6000.0f32;
    while out.len() < n {
        note += 100.0 * ((r.next() * 5.0).floor() - 2.0);
        note = note.clamp(5400.0, 6600.0);
        let centre = note + 90.0 * (r.next() - 0.5);
        let len = hops(120.0 + 500.0 * r.next());
        let (rate, depth) = (3.0 + 4.0 * r.next(), 5.0 + 35.0 * r.next());
        let scoop = if r.next() < 0.4 { -80.0 * r.next() } else { 0.0 };
        for i in 0..len {
            let t = i as f32 * HOP_S;
            let wob = depth * (std::f32::consts::TAU * rate * t).sin();
            let s = scoop * (-t / 0.05).exp();
            let tail = (i as f32 - len as f32 + hops(60.0) as f32).max(0.0);
            let db = if level { -18.0 - 0.25 * tail } else { -18.0 };
            out.push(PitchEstimate { level_db: db, ..est(centre + wob + s + 4.0 * (r.next() - 0.5)) });
        }
        if r.next() < 0.3 {
            for _ in 0..hops(30.0 + 150.0 * r.next()) {
                out.push(unvoiced());
            }
        }
    }
    out.truncate(n);
    out
}

fn outputs(c: &mut Corrector, stream: &[PitchEstimate]) -> Vec<ControlOutput> {
    stream.iter().map(|e| c.process_hop(e, None)).collect()
}

fn hard(tier: QualityTier, note_hold: bool) -> impl FnOnce(&mut Settings) {
    move |s: &mut Settings| {
        s.tier = tier;
        s.retune_ms = 0.0;
        s.vibrato_depth = 0.0;
        s.tolerance_cents = 5.0;
        s.note_hold = note_hold;
    }
}

/// "Rule proposes, model vetoes": every decision the corrector makes is the
/// control law's own (with the release freeze), except that a proposed
/// switch may become a Keep. Onsets are never altered and no switch is
/// added or redirected.
#[test]
fn note_hold_only_removes_switches_the_rule_proposed() {
    for (cfg, tier) in [
        (config(LatencyMode::Live, 0), QualityTier::Hq),
        (config(LatencyMode::Live, 0), QualityTier::Max),
        (config(LatencyMode::LivePlus, 0), QualityTier::Hq),
        (config(LatencyMode::Studio, 30), QualityTier::Hq),
    ] {
        let mut c = with_config(cfg, hard(tier, true));
        c.decisions = Some(Vec::new());
        let lag = if cfg.latency_mode == LatencyMode::Studio { 10 * HOP } else { 0 };
        let stream: Vec<PitchEstimate> =
            wandering(7, hops(30_000.0), true).into_iter().map(|e| PitchEstimate { info_lag: lag as f32, ..e }).collect();
        let outs = outputs(&mut c, &stream);
        let log = c.decisions.take().unwrap_or_default();
        let (mut vetoed, mut switches, mut onsets) = (0, 0, 0);
        for &(rule, out) in &log {
            match (rule, out) {
                (r, o) if r == o => {
                    switches += usize::from(matches!(r, Change::Switch(_)));
                    onsets += usize::from(matches!(r, Change::Onset(_)));
                }
                (Change::Switch(_), Change::Keep) => vetoed += 1,
                other => panic!("{:?} {tier:?}: the veto changed {other:?}", cfg.latency_mode),
            }
        }
        let stats = c.note_hold_stats();
        assert_eq!(stats.vetoed, vetoed as u64, "{:?}", cfg.latency_mode);
        assert!(vetoed > 10 && switches > 10 && onsets > 5, "{:?} {tier:?}: vetoed {vetoed}, switches {switches}, onsets {onsets}", cfg.latency_mode);
        assert!(stats.proposals >= stats.vetoed && stats.frozen > 0, "{stats:?}");
        outs.iter().for_each(|o| assert!(o.shift_cents.is_finite()));
    }
}

/// With the same input the learned note hold changes nothing in Eco and with
/// MIDI targets (bit for bit), and a missing model leaves only the release
/// freeze (nothing at all with a steady level).
#[test]
fn eco_midi_targets_and_a_missing_model_are_unaffected() {
    let stream = wandering(11, hops(20_000.0), true);
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        let cfg = config(mode, if mode == LatencyMode::Studio { 30 } else { 0 });
        let off = outputs(&mut with_config(cfg, hard(QualityTier::Eco, false)), &stream);
        let on = outputs(&mut with_config(cfg, hard(QualityTier::Eco, true)), &stream);
        assert_eq!(off, on, "{mode:?} Eco");

        let midi = |hold: bool| {
            let mut c = with_config(cfg, |s| {
                hard(QualityTier::Hq, hold)(s);
                s.midi_mode = MidiMode::Target;
            });
            for note in [55, 57, 60, 62, 64, 65] {
                c.midi(MidiEvent::NoteOn { note, velocity: 0.8 });
            }
            outputs(&mut c, &stream)
        };
        assert_eq!(midi(false), midi(true), "{mode:?} MIDI Target");
    }
    let steady = wandering(11, hops(20_000.0), false);
    let off = outputs(&mut live(hard(QualityTier::Hq, false)), &steady);
    let mut c = live(hard(QualityTier::Hq, true));
    c.drop_model_for_test();
    assert_eq!(off, outputs(&mut c, &steady));
    assert_eq!(c.note_hold_stats().proposals, 0);
}

/// The release freeze: a step down while the level falls more than 4 dB in
/// 40 ms waits until the level stops falling; a step up does not wait.
#[test]
fn the_release_freeze_waits_out_a_falling_level() {
    // C4 held; the level starts to fall 30 ms before the pitch steps to
    // B3 + 5 c (55 c past the midpoint) and keeps falling (0.2 dB/ms) until
    // 60 ms after it. The window then holds less than 4 dB of fall 80 ms
    // after the step.
    let n0 = hops(300.0);
    let fall_from = n0 - hops(30.0);
    let fall = |i: usize| -20.0 - 0.2 * ((i.saturating_sub(fall_from)) as f32 * HOP_MS).min(90.0);
    let switch_ms = |note_hold: bool, to: f32| {
        let mut c = live(hard(QualityTier::Hq, note_hold));
        c.drop_model_for_test(); // the freeze alone
        let outs = run(&mut c, n0 + hops(300.0), |i| {
            let p = if i < n0 { 6000.0 } else { to };
            PitchEstimate { level_db: fall(i), ..est(p) }
        });
        let want = 100.0 * (to / 100.0).round();
        outs[n0..].iter().position(|o| o.frame.target == want).map(|k| k as f32 * HOP_MS)
    };
    let off = switch_ms(false, 5905.0).expect("switches without the freeze");
    let on = switch_ms(true, 5905.0).expect("switches once the level settles");
    assert!(off < 15.0, "{off}");
    assert!((60.0..=110.0).contains(&on), "frozen until the fall ends: {on} ms");
    let up = switch_ms(true, 6105.0).expect("a step up is not frozen");
    assert!(up < 15.0, "{up}");
}

/// A singer sitting high in a note who drifts just past the midpoint now
/// and then: the hard-tune target flips each time without the learned note
/// hold and stays with it; a real move to the next note still switches.
#[test]
fn note_hold_keeps_a_drifting_note_and_follows_a_real_move() {
    let drift = |i: usize| {
        let cycle = hops(400.0);
        let k = i % cycle;
        est(if k > cycle - hops(30.0) { 6062.0 } else { 6040.0 })
    };
    let n = hops(4000.0);
    let flips = |note_hold: bool| {
        let mut c = live(hard(QualityTier::Hq, note_hold));
        target_changes(&run(&mut c, n, drift))
    };
    let (off, on) = (flips(false), flips(true));
    assert!(off >= 10, "the drift flips the plain control law: {off}");
    assert!(on <= off / 5, "note hold: {on} flips against {off}");
    let mut c = live(hard(QualityTier::Hq, true));
    run(&mut c, n, drift);
    let outs = hold(&mut c, 6140.0, hops(300.0));
    let k = outs.iter().position(|o| o.frame.target == 6100.0).map(|k| k as f32 * HOP_MS);
    assert!(k.is_some_and(|ms| ms < 60.0), "a real note change: {k:?}");
}

/// Held notes on random semitones, every one `dev` cents off A440, 400 ms
/// each with 60 ms gaps.
fn detuned_singer(c: &mut Corrector, dev: f32, notes: usize, seed: u64) -> Vec<ControlOutput> {
    let mut r = Lcg(seed);
    let mut outs = Vec::new();
    for _ in 0..notes {
        let n = 55.0 + (r.next() * 12.0).floor();
        outs.extend(hold(c, 100.0 * n + dev, hops(400.0)));
        outs.extend(run(c, hops(60.0), |_| unvoiced()));
    }
    outs
}

fn follow(on: bool) -> impl FnOnce(&mut Settings) {
    move |s: &mut Settings| {
        s.retune_ms = 0.0;
        s.reference = if on { ReferenceMode::FollowSinger } else { ReferenceMode::A440 };
    }
}

/// Grid offset of a voiced frame's target from A440, cents.
fn target_offset(o: &ControlOutput) -> f32 {
    o.frame.target - 100.0 * (o.frame.target / 100.0).round()
}

#[test]
fn follow_singer_finds_a_consistently_sharp_singer() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = live(|s| {
            follow(true)(s);
            s.tier = tier;
        });
        let outs = detuned_singer(&mut c, 30.0, 24, 1);
        assert!((c.singer_offset() - 30.0).abs() < 3.0, "{tier:?}: {}", c.singer_offset());
        // The last notes: targets on the singer's grid, almost no correction.
        let last = &outs[outs.len() - hops(300.0)..];
        for o in last.iter().filter(|o| o.shift.voiced && o.frame.target > 0.0) {
            assert!((target_offset(o) - 30.0).abs() < 3.0, "{tier:?}: target {}", o.frame.target);
            assert!(o.shift_cents.abs() < 3.0, "{tier:?}: shift {}", o.shift_cents);
        }
        // A440: the same singer is pulled down 30 c.
        let mut a = live(|s| {
            follow(false)(s);
            s.tier = tier;
        });
        let outs = detuned_singer(&mut a, 30.0, 24, 1);
        let o = outs[outs.len() - hops(100.0)];
        assert_eq!(target_offset(&o), 0.0);
        assert!((o.shift_cents + 30.0).abs() < 1.0, "{}", o.shift_cents);
        assert_eq!(a.singer_offset(), 0.0);
    }
}

/// The offset a note starts with is its target's for the whole note, even
/// while the estimate moves.
#[test]
fn the_target_offset_is_latched_per_note() {
    let mut c = live(follow(true));
    detuned_singer(&mut c, 30.0, 24, 2);
    let before = c.singer_offset();
    // A long note 20 c lower pulls the estimate down while it is sung.
    let outs = hold(&mut c, 6010.0, hops(3000.0));
    let after = c.singer_offset();
    assert!(before - after > 2.0, "the estimate moved: {before} → {after}");
    let voiced: Vec<f32> = outs.iter().filter(|o| o.shift.voiced && o.frame.target > 0.0).map(|o| o.frame.target).collect();
    assert!(voiced.len() > hops(2900.0));
    assert!(voiced.windows(2).all(|w| w[0] == w[1]), "target changed within the note");
    // The next note starts on the new estimate.
    run(&mut c, hops(60.0), |_| unvoiced());
    let next = hold(&mut c, 6210.0, hops(200.0));
    let o = next[next.len() - 1];
    assert!((target_offset(&o) - c.singer_offset()).abs() < 1.0, "{} vs {}", target_offset(&o), c.singer_offset());
}

#[test]
fn an_in_tune_or_inconsistent_singer_stays_on_a440() {
    let mut c = live(follow(true));
    let mut r = Lcg(5);
    for _ in 0..30 {
        let n = 55.0 + (r.next() * 12.0).floor();
        let dev = 90.0 * (r.next() - 0.5);
        hold(&mut c, 100.0 * n + dev, hops(400.0));
        run(&mut c, hops(60.0), |_| unvoiced());
    }
    assert!(c.singer_offset().abs() < 5.0, "inconsistent: {}", c.singer_offset());
    let mut c = live(follow(true));
    detuned_singer(&mut c, 0.0, 20, 3);
    assert!(c.singer_offset().abs() < 1.0, "in tune: {}", c.singer_offset());
}

/// MIDI targets are the intent: follow singer's tuning leaves them alone.
#[test]
fn follow_singer_ignores_midi_targets() {
    let midi = |on: bool| {
        let mut c = live(|s| {
            follow(on)(s);
            s.midi_mode = MidiMode::Target;
        });
        for note in 55..67 {
            c.midi(MidiEvent::NoteOn { note, velocity: 0.8 });
        }
        let outs = detuned_singer(&mut c, 30.0, 16, 4);
        (outs, c.singer_offset())
    };
    let (off, _) = midi(false);
    let (on, offset) = midi(true);
    assert_eq!(off, on);
    assert_eq!(offset, 0.0);
}

/// Switching the option on mid-note leaves that note on A440 (no stale
/// offset from an earlier session); the next note starts on the estimate.
#[test]
fn switching_follow_on_mid_note_waits_for_the_next_note() {
    let mut c = live(follow(true));
    detuned_singer(&mut c, 30.0, 24, 7);
    c.set_settings(&Settings { retune_ms: 0.0, ..Settings::default() });
    run(&mut c, hops(60.0), |_| unvoiced());
    hold(&mut c, 6030.0, hops(200.0));
    c.set_settings(&Settings { retune_ms: 0.0, reference: ReferenceMode::FollowSinger, ..Settings::default() });
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert!(outs.iter().all(|o| target_offset(o) == 0.0), "the sounding note keeps A440");
}

#[test]
fn switching_follow_off_returns_to_a440_and_reset_forgets() {
    let mut c = live(follow(true));
    detuned_singer(&mut c, 30.0, 24, 6);
    assert!(c.singer_offset() > 20.0);
    c.set_settings(&Settings { retune_ms: 0.0, ..Settings::default() });
    assert_eq!(c.singer_offset(), 0.0);
    run(&mut c, hops(60.0), |_| unvoiced());
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert_eq!(target_offset(&outs[outs.len() - 1]), 0.0);
    let mut c = live(|s| {
        follow(true)(s);
        s.note_hold = true;
    });
    detuned_singer(&mut c, 30.0, 24, 6);
    c.reset();
    assert_eq!(c.singer_offset(), 0.0);
    assert_eq!(c.note_hold_stats(), crate::NoteHoldStats::default());
}
