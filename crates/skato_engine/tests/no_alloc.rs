//! Zero-allocation proof for the realtime path (DESIGN §2.1 rule 1, §3.5).
//!
//! A counting `#[global_allocator]` (its own test binary, so it cannot clash
//! with anything else) counts allocations, reallocations and frees made by
//! the current thread while it is *armed*. Everything the audio thread calls
//! per block runs armed: `set_settings`, `process` with every block size,
//! MIDI, telemetry, tier/engine/mode switches, NaN input, `reset`, `stats`
//! and `learned_states`. A negative control proves the counter sees an
//! allocation made inside the armed region.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Rng, noise_bursts, silence, vowel};
use skato_core::{
    AnalysisChannel, EngineConfig, EngineKind, LatencyMode, MidiEvent, MidiMode, QualityTier, ReferenceMode, ScaleKind,
    Settings, VoiceRange,
};
use skato_engine::{Engine, TimedMidi};

struct Counting;

static EVENTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Per-thread arm flag (const-initialised: reading it never allocates),
    /// so tests running in parallel do not count each other.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    if ARMED.with(Cell::get) {
        EVENTS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: forwards every call unchanged to the system allocator; the extra
// work is a thread-local read and an atomic increment, neither of which
// allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `f` armed and returns how many allocator calls it made.
fn armed<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = EVENTS.load(Ordering::Relaxed);
    ARMED.with(|a| a.set(true));
    let r = f();
    ARMED.with(|a| a.set(false));
    (EVENTS.load(Ordering::Relaxed) - before, r)
}

#[test]
fn negative_control_counts_an_allocation() {
    let (n, v) = armed(|| vec![1u8; 64]);
    assert!(n >= 1, "the counting allocator did not see a Vec allocation");
    drop(v);
    let (n, _) = armed(|| 3 + 4);
    assert_eq!(n, 0);
}

/// Material that exercises every path: voiced notes (seeds, onsets, grain
/// repeats under correction), unvoiced noise, silence, NaN/Inf and garbage.
fn material(sr: f32) -> Vec<f32> {
    let mut x = silence(sr, 0.03);
    x.extend(vowel(sr, 0.35, 1.0, |t| {
        180.0 * 2f64.powf((40.0 * (std::f64::consts::TAU * 6.0 * t).sin() + 30.0) / 1200.0)
    }));
    x.extend(noise_bursts(sr, 0.08, 0.04, 1.0, 0.1, 3));
    let mut bad = vec![0.0f32; 600];
    for (i, v) in bad.iter_mut().enumerate() {
        *v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0e30, -1.0e30, 0.0][i % 6];
    }
    x.extend(bad);
    x.extend(silence(sr, 0.05));
    x.extend(vowel(sr, 0.3, 0.8, |_| 400.0));
    x
}

fn settings_for(i: usize, mode: LatencyMode) -> Settings {
    let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
    Settings {
        tier: tiers[i % 3],
        engine: if (i / 3).is_multiple_of(2) { EngineKind::Natural } else { EngineKind::Classic },
        retune_ms: [0.0, 20.0, 80.0][i % 3],
        scale: ScaleKind::ALL[i % ScaleKind::ALL.len()],
        note_states: ScaleKind::ALL[i % ScaleKind::ALL.len()].note_states((i % 12) as u8),
        midi_mode: [MidiMode::Off, MidiMode::Target, MidiMode::Learn, MidiMode::Auto][i % 4],
        analysis_channel: [AnalysisChannel::Mix, AnalysisChannel::Left, AnalysisChannel::Right][i % 3],
        transpose_st: if mode == LatencyMode::Studio { 3.0 } else { 0.0 },
        formant_st: if i.is_multiple_of(2) { 0.0 } else { -2.0 },
        synth_vibrato: skato_core::SynthVibrato { enabled: i % 2 == 1, ..Default::default() },
        ..Settings::default()
    }
}

#[test]
fn process_never_allocates_after_construction() {
    let sr = 48_000.0;
    let x = material(sr);
    let modes = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
    for (mi, mode) in modes.into_iter().enumerate() {
        let cfg = EngineConfig {
            sample_rate: sr,
            max_block: 4096,
            channels: 2,
            latency_mode: mode,
            voice_range: [VoiceRange::AltoTenor, VoiceRange::LowMale, VoiceRange::Wide, VoiceRange::Soprano][mi],
            ..EngineConfig::default()
        };
        // Construction may allocate; nothing after it may.
        let mut e = Engine::new(cfg);
        let mut rx = e.take_telemetry().expect("telemetry consumer");
        let mut l = x.clone();
        let mut r = x.clone();
        let mut midi: Vec<TimedMidi> = Vec::with_capacity(8);
        let mut rng = Rng::new(mi as u64 + 1);
        let sizes = [1usize, 7, 32, 64, 128, 480, 4096, 6000];
        let mut pos = 0usize;
        let mut block = 0usize;
        let mut total_events = 0usize;
        while pos < l.len() {
            let n = sizes[block % sizes.len()].min(l.len() - pos);
            midi.clear();
            if block.is_multiple_of(5) {
                midi.push(TimedMidi {
                    timing: 0,
                    event: MidiEvent::NoteOn { note: 50 + (block % 20) as u8, velocity: 0.7 },
                });
                midi.push(TimedMidi { timing: (n / 2) as u32, event: MidiEvent::PitchBend { value: rng.bipolar() } });
            }
            if block % 7 == 3 {
                midi.push(TimedMidi {
                    timing: (n - 1) as u32,
                    event: MidiEvent::NoteOff { note: 50 + ((block - 3) % 20) as u8 },
                });
                midi.push(TimedMidi {
                    timing: (n - 1) as u32,
                    event: MidiEvent::Sustain { down: block.is_multiple_of(2) },
                });
            }
            if block % 11 == 10 {
                midi.push(TimedMidi { timing: 0, event: MidiEvent::AllNotesOff });
            }
            let s = settings_for(block, mode);
            let (lb, rb) = (&mut l[pos..pos + n], &mut r[pos..pos + n]);
            let (count, _) = armed(|| {
                e.set_settings(&s);
                e.set_telemetry_enabled(!block.is_multiple_of(3));
                e.process(&mut [lb, rb], &midi);
                let _ = e.stats();
                let _ = e.learned_states();
                let _ = e.latency_samples();
                if block == 9 {
                    e.reset();
                }
            });
            total_events += count;
            // Drain like the GUI would (outside the armed region: the
            // consumer runs on another thread in the plugin).
            while rx.pop().is_ok() {}
            pos += n;
            block += 1;
        }
        assert_eq!(total_events, 0, "{mode:?}: {total_events} allocator calls in the realtime path");
        assert!(e.stats().hops > 0);
        assert!(l.iter().chain(&r).all(|v| v.is_finite()), "{mode:?}: non-finite output");
    }
}

/// Mono and other sample rates use differently sized internals; still no
/// allocation, and a telemetry ring left full only drops frames.
#[test]
fn other_rates_and_mono_never_allocate() {
    for sr in [44_100.0f32, 96_000.0, 192_000.0] {
        let cfg =
            EngineConfig { sample_rate: sr, channels: 1, latency_mode: LatencyMode::Live, ..EngineConfig::default() };
        let mut e = Engine::new(cfg);
        let _rx = e.take_telemetry();
        let mut x = material(sr);
        let s = Settings { tier: QualityTier::Max, ..Settings::default() };
        let (count, _) = armed(|| {
            e.set_settings(&s);
            e.set_telemetry_enabled(true);
            for chunk in x.chunks_mut(333) {
                e.process(&mut [chunk], &[]);
            }
        });
        assert_eq!(count, 0, "{sr} Hz: {count} allocator calls");
    }
}

/// HI-3: voice-range switches (and refused ones) never allocate: every
/// range sharing the engine's latency was built in `Engine::new`.
#[test]
fn voice_range_switches_never_allocate() {
    let sr = 48_000.0;
    for mode in [LatencyMode::Studio, LatencyMode::LivePlus, LatencyMode::LiveMin, LatencyMode::Live] {
        let cfg = EngineConfig { sample_rate: sr, channels: 2, latency_mode: mode, ..EngineConfig::default() };
        let mut e = Engine::new(cfg);
        let _rx = e.take_telemetry();
        let mut l = material(sr);
        let mut r = l.clone();
        let ranges = [VoiceRange::LowMale, VoiceRange::Soprano, VoiceRange::Wide, VoiceRange::AltoTenor];
        let (count, _) = armed(|| {
            e.set_telemetry_enabled(true);
            for (i, (lc, rc)) in l.chunks_mut(97).zip(r.chunks_mut(97)).enumerate() {
                if i % 40 == 7 {
                    let _ = e.set_voice_range(ranges[(i / 40) % ranges.len()]);
                }
                if i == 333 {
                    e.reset();
                }
                e.process(&mut [lc, rc], &[]);
            }
        });
        assert_eq!(count, 0, "{mode:?}: {count} allocator calls");
        assert!(l.iter().chain(&r).all(|v| v.is_finite()));
    }
}

/// A sung line that wobbles across semitone midpoints and steps between
/// notes (so the note decision proposes switches for the learned veto to
/// judge), with level decays, gaps and garbage.
fn wandering_line(sr: f32) -> Vec<f32> {
    let mut x = Vec::new();
    for (k, base) in [196.0f64, 207.65, 185.0, 220.0, 196.0, 174.6].into_iter().enumerate() {
        let off = [45.0, -40.0, 30.0, -48.0, 10.0, 42.0][k];
        let mut v = vowel(sr, 0.4, 0.8, move |t| base * 2f64.powf((off + 25.0 * (std::f64::consts::TAU * 5.0 * t).sin()) / 1200.0));
        let n = v.len();
        for (i, s) in v.iter_mut().enumerate() {
            // The last 80 ms decay by 20 dB (a phrase-end release).
            let tail = (i as f64 - (n as f64 - 0.08 * f64::from(sr))).max(0.0) / (0.08 * f64::from(sr));
            *s *= 10f32.powf(-tail as f32);
        }
        x.extend(v);
        if k % 2 == 1 {
            x.extend(silence(sr, 0.05));
        }
    }
    x.extend(material(sr));
    x
}

/// The Labs note-intent options (learned note hold, follow singer's tuning)
/// in every mode and tier, switched on and off mid-stream: no allocation in
/// the realtime path (the learned veto runs its tree ensemble on the audio
/// thread) and finite output.
#[test]
fn note_intent_options_never_allocate() {
    let sr = 48_000.0;
    let x = wandering_line(sr);
    for mode in [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            let cfg = EngineConfig { sample_rate: sr, channels: 1, latency_mode: mode, ..EngineConfig::default() };
            let mut e = Engine::new(cfg);
            let _rx = e.take_telemetry();
            let on = Settings {
                tier,
                retune_ms: 0.0,
                tolerance_cents: 5.0,
                note_hold: true,
                reference: ReferenceMode::FollowSinger,
                ..Settings::default()
            };
            let off = Settings { note_hold: false, reference: ReferenceMode::A440, ..on };
            let mut buf = x.clone();
            let sizes = [32usize, 7, 128, 64, 480, 1];
            let (count, _) = armed(|| {
                e.set_telemetry_enabled(true);
                let mut pos = 0;
                let mut i = 0;
                while pos < buf.len() {
                    let n = sizes[i % sizes.len()].min(buf.len() - pos);
                    e.set_settings(if i % 400 < 300 { &on } else { &off });
                    e.process(&mut [&mut buf[pos..pos + n]], &[]);
                    if i == 2000 {
                        e.reset();
                    }
                    pos += n;
                    i += 1;
                }
            });
            assert_eq!(count, 0, "{mode:?} {tier:?}: {count} allocator calls");
            assert!(buf.iter().all(|v| v.is_finite()), "{mode:?} {tier:?}: non-finite output");
            // The veto ran (HQ/Max; Eco keeps the plain control law).
            let judged = e.note_hold_stats().proposals;
            assert_eq!(judged > 0, tier != QualityTier::Eco, "{mode:?} {tier:?}: {judged} switches judged");
        }
    }
}
