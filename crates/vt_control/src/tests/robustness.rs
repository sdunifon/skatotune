//! Realtime robustness: garbage input, fuzzing, determinism, no allocation.

use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use vt_core::{LatencyMode, MidiEvent, MidiMode, NoteState, QualityTier, SynthVibrato};

// Counts allocations made by the current thread while armed, so parallel
// tests do not disturb each other.
struct CountingAlloc;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.with(Cell::get) {
            COUNT.with(|c| c.set(c.get() + 1));
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.with(Cell::get) {
            COUNT.with(|c| c.set(c.get() + 1));
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn count_allocs(f: impl FnOnce()) -> usize {
    COUNT.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    COUNT.with(Cell::get)
}

/// Deterministic pseudo-random numbers in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn random_settings(r: &mut Lcg) -> Settings {
    let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
    let modes = [MidiMode::Off, MidiMode::Target, MidiMode::Learn, MidiMode::Auto];
    let states = [NoteState::On, NoteState::Bypass, NoteState::Remove];
    let mut note_states = [NoteState::On; 12];
    for s in note_states.iter_mut() {
        *s = states[(r.next() * 3.0) as usize % 3];
    }
    Settings {
        tier: tiers[(r.next() * 3.0) as usize % 3],
        note_states,
        detune_cents: r.next() * 200.0 - 100.0,
        retune_ms: if r.next() < 0.3 { 0.0 } else { r.next() * 400.0 },
        transition_ms: if r.next() < 0.5 { None } else { Some(r.next() * 800.0) },
        humanize: r.next() * 100.0,
        flex_tune: r.next() * 100.0,
        amount: r.next(),
        tolerance_cents: r.next() * 40.0,
        tolerance_ms: if r.next() < 0.5 { 0.0 } else { r.next() * 300.0 },
        snap_min_cents: r.next() * 50.0,
        snap_max_st: r.next() * 12.0,
        preserve_vibrato: r.next() < 0.5,
        ignore_vibrato: r.next() < 0.5,
        vibrato_depth: r.next() * 3.0 - 1.0,
        synth_vibrato: SynthVibrato { enabled: r.next() < 0.5, ..SynthVibrato::default() },
        transpose_st: (r.next() * 24.0 - 12.0).round(),
        formant_st: r.next() * 24.0 - 12.0,
        formant_follows_transpose: r.next() < 0.5,
        midi_mode: modes[(r.next() * 4.0) as usize % 4],
        midi_all_octaves: r.next() < 0.5,
        bend_range_st: r.next() * 12.0,
        ..Settings::default()
    }
}

fn random_estimate(r: &mut Lcg, base: f32) -> PitchEstimate {
    match (r.next() * 12.0) as u32 {
        0 => unvoiced(),
        1 => PitchEstimate { cents: f32::NAN, f0_hz: f32::NAN, voiced: true, ..est(6000.0) },
        2 => PitchEstimate { cents: f32::INFINITY, confidence: f32::NAN, ..est(6000.0) },
        3 => PitchEstimate { cents: -500.0, f0_hz: -1.0, ..est(6000.0) },
        4 => est(base + 1200.0 * (r.next() - 0.5) * 4.0),
        5 => PitchEstimate { info_lag: f32::NAN, period: f32::INFINITY, ..est(base) },
        _ => est(base + 60.0 * (r.next() - 0.5)),
    }
}

fn random_midi(r: &mut Lcg) -> MidiEvent {
    match (r.next() * 6.0) as u32 {
        0 => MidiEvent::NoteOn { note: (r.next() * 140.0) as u8, velocity: r.next() },
        1 => MidiEvent::NoteOff { note: (r.next() * 140.0) as u8 },
        2 => MidiEvent::PitchBend { value: if r.next() < 0.1 { f32::NAN } else { r.next() * 2.4 - 1.2 } },
        3 => MidiEvent::Sustain { down: r.next() < 0.5 },
        4 => MidiEvent::AllNotesOff,
        _ => MidiEvent::NoteOn { note: 60 + (r.next() * 12.0) as u8, velocity: 0.8 },
    }
}

fn check_finite(o: &ControlOutput) {
    assert!(o.shift_cents.is_finite() && o.shift_cents.abs() <= 2400.0, "{o:?}");
    assert!(o.shift.ratio.is_finite() && o.shift.ratio > 0.24 && o.shift.ratio < 4.01, "{o:?}");
    assert!(o.shift.formant_ratio.is_finite() && o.shift.formant_ratio > 0.0, "{o:?}");
    let f = o.frame;
    for v in [f.detected, f.centre, f.target, f.output, f.confidence, f.vib_rate, f.vib_extent] {
        assert!(v.is_finite(), "{o:?}");
    }
}

#[test]
fn fuzzed_inputs_settings_and_midi_stay_finite_without_allocating() {
    for (mode, look) in [(LatencyMode::Live, 0), (LatencyMode::LiveMin, 0), (LatencyMode::Studio, 30)] {
        let mut r = Lcg(0xC0FFEE ^ look as u64);
        let mut c = Corrector::new(config(mode, look));
        // Pre-generate everything so the measured loop itself is alloc-free.
        let settings: Vec<Settings> = (0..40).map(|_| random_settings(&mut r)).collect();
        let mut nan_settings = random_settings(&mut r);
        nan_settings.retune_ms = f32::NAN;
        nan_settings.amount = f32::INFINITY;
        nan_settings.vibrato_depth = f32::NAN;
        nan_settings.transpose_st = f32::NEG_INFINITY;
        let estimates: Vec<PitchEstimate> = (0..20_000).map(|i| random_estimate(&mut r, 4800.0 + (i % 3000) as f32)).collect();
        let midi: Vec<(bool, MidiEvent)> = (0..20_000).map(|_| (r.next() < 0.05, random_midi(&mut r))).collect();
        let mut outs = Vec::with_capacity(estimates.len());
        let allocs = count_allocs(|| {
            for (i, e) in estimates.iter().enumerate() {
                if i % 500 == 0 {
                    c.set_settings(&settings[(i / 500) % settings.len()]);
                }
                if i % 7777 == 0 {
                    c.set_settings(&nan_settings);
                }
                if let Some((true, ev)) = midi.get(i) {
                    c.midi(*ev);
                }
                let mark = if i % 3 == 0 { Some(e.cents + 10.0) } else if i % 11 == 0 { Some(f32::NAN) } else { None };
                outs.push(c.process_hop(e, mark));
                if i == 15_000 {
                    c.reset();
                }
            }
        });
        assert_eq!(allocs, 0, "{mode:?}: process path allocated");
        outs.iter().for_each(check_finite);
    }
}

/// Settings with float knob `k` (an index into every float field) set to `v`.
fn with_knob(base: Settings, k: usize, v: f32) -> Settings {
    let mut s = base;
    match k {
        0 => s.detune_cents = v,
        1 => s.retune_ms = v,
        2 => s.transition_ms = Some(v),
        3 => s.humanize = v,
        4 => s.flex_tune = v,
        5 => s.amount = v,
        6 => s.tolerance_cents = v,
        7 => s.tolerance_ms = v,
        8 => s.snap_min_cents = v,
        9 => s.snap_max_st = v,
        10 => s.vibrato_depth = v,
        11 => s.transpose_st = v,
        12 => s.formant_st = v,
        13 => s.bend_range_st = v,
        14 => s.synth_vibrato.rate_hz = v,
        15 => s.synth_vibrato.depth_cents = v,
        16 => s.synth_vibrato.delay_ms = v,
        17 => s.synth_vibrato.ramp_ms = v,
        18 => s.tracking = v,
        _ => s.gate_db = v,
    }
    s
}

/// RT-1: tiny and subnormal knob values (exponentially decaying automation
/// or CLAP modulation tails) must never poison the control law. A Snap Min of
/// 1e-20 c used to turn the error NaN, the retune one-pole latched it, and
/// correction stayed silently off for the rest of the session, including
/// the next note after a gap.
#[test]
fn tiny_and_subnormal_knob_values_never_poison_the_correction() {
    let tinies = [1e-45f32, 1e-40, 1e-38, 1e-30, 1e-20, 1e-17];
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let base = Settings {
            tier,
            synth_vibrato: SynthVibrato { enabled: false, ..SynthVibrato::default() },
            ..Settings::default()
        };
        for k in 0..20 {
            for tiny in tinies {
                for v in [tiny, -tiny] {
                    let mut c = live(|s| *s = base);
                    // A3 + 35 c, with the knob at the tiny value for 0.2 s.
                    hold(&mut c, 5735.0, hops(100.0));
                    c.set_settings(&with_knob(base, k, v));
                    let outs = hold(&mut c, 5735.0, hops(200.0));
                    outs.iter().for_each(check_finite);
                    // Back to the defaults: the note is corrected again...
                    c.set_settings(&base);
                    let outs = hold(&mut c, 5735.0, hops(300.0));
                    outs.iter().for_each(check_finite);
                    let last = outs[outs.len() - 1].frame;
                    assert!((last.output - 5700.0).abs() < 0.5, "{tier:?} knob {k} = {v}: output {}", last.output);
                    // ... and so is the next note after 1 s of silence.
                    run(&mut c, hops(1000.0), |_| unvoiced());
                    let outs = hold(&mut c, 6035.0, hops(300.0));
                    let last = outs[outs.len() - 1].frame;
                    assert!((last.output - 6000.0).abs() < 0.5, "{tier:?} knob {k} = {v}: next note {}", last.output);
                }
            }
        }
    }
}

/// RT-1, defence in depth: whatever corrupts the correction state, the back
/// end restarts it instead of latching NaN, both mid-note and in a gap.
#[test]
fn a_non_finite_correction_state_heals_itself() {
    for retune in [0.0f32, 20.0] {
        let mut c = live(|s| s.retune_ms = retune);
        hold(&mut c, 5735.0, hops(100.0));
        c.poison_filter_state_for_test();
        let outs = hold(&mut c, 5735.0, hops(200.0));
        outs.iter().for_each(check_finite);
        assert!((outs[outs.len() - 1].frame.output - 5700.0).abs() < 0.5, "retune {retune}");
        // Poisoned during a gap (hold, release and the unvoiced ramp).
        c.poison_filter_state_for_test();
        let outs = run(&mut c, hops(200.0), |_| unvoiced());
        outs.iter().for_each(check_finite);
        let outs = hold(&mut c, 6035.0, hops(200.0));
        outs.iter().for_each(check_finite);
        assert!((outs[outs.len() - 1].frame.output - 6000.0).abs() < 0.5, "retune {retune}");
    }
}

#[test]
fn the_allocation_counter_works() {
    // Negative control for the test above.
    let n = count_allocs(|| {
        let v: Vec<u8> = Vec::with_capacity(16);
        std::hint::black_box(v);
    });
    assert_eq!(n, 1);
}

#[test]
fn identical_input_gives_identical_output() {
    let run_once = || {
        let mut r = Lcg(42);
        let mut c = live(|s| {
            s.tier = QualityTier::Max;
            s.humanize = 40.0;
            s.synth_vibrato.enabled = true;
        });
        (0..5000).map(|i| c.process_hop(&est(6000.0 + 80.0 * (i as f32 * 0.01).sin() + 3.0 * r.next()), None)).collect::<Vec<_>>()
    };
    assert_eq!(run_once(), run_once());
}

#[test]
fn tier_switch_mid_note_keeps_the_note_and_does_not_jump() {
    let mut s = Settings { tier: QualityTier::Eco, retune_ms: 15.0, ..Settings::default() };
    let mut c = live(|x| *x = s);
    let a = hold(&mut c, 6430.0, hops(300.0));
    for tier in [QualityTier::Hq, QualityTier::Max, QualityTier::Eco, QualityTier::Hq] {
        s.tier = tier;
        c.set_settings(&s);
        let b = hold(&mut c, 6430.0, hops(100.0));
        assert_eq!(target_changes(&b), 0, "{tier:?}");
        assert_eq!(b[0].frame.target, a[a.len() - 1].frame.target);
        assert!((b[0].shift_cents - a[a.len() - 1].shift_cents).abs() < 0.5);
    }
}

#[test]
fn reset_forgets_everything() {
    let mut c = live(|s| {
        s.midi_mode = MidiMode::Target;
        s.retune_ms = 0.0;
    });
    c.midi(MidiEvent::NoteOn { note: 64, velocity: 1.0 });
    hold(&mut c, 6400.0, 100);
    c.reset();
    let outs = hold(&mut c, 6430.0, 100);
    // Held note cleared by reset → pass-through.
    assert_eq!(outs[outs.len() - 1].frame.target, 0.0);
    assert!(outs[outs.len() - 1].shift_cents.abs() < 1e-3);
    // First hop after reset: unvoiced until the onset is confirmed.
    let mut c = live(|_| {});
    hold(&mut c, 6000.0, 100);
    c.reset();
    assert!(!c.process_hop(&est(6000.0), None).shift.voiced);
}

#[test]
fn synthetic_vibrato_starts_after_its_delay() {
    let mut c = live(|s| {
        s.retune_ms = 0.0;
        s.synth_vibrato = SynthVibrato { enabled: true, rate_hz: 5.0, depth_cents: 30.0, delay_ms: 200.0, ramp_ms: 100.0 };
    });
    let outs = hold(&mut c, 6000.0, hops(1500.0));
    let early = half_range(outs[hops(10.0)..hops(190.0)].iter().map(|o| o.frame.output));
    assert!(early < 1e-3, "LFO before its delay: {early}");
    let late = half_range(outs[hops(600.0)..].iter().map(|o| o.frame.output));
    assert!(late > 0.85 * 30.0 && late < 1.15 * 30.0, "LFO depth {late}");
}
