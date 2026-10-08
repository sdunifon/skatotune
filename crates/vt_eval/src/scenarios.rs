//! Named evaluation scenarios: a signal plus the settings under which its
//! `target_cents` is the correct answer.
//!
//! Detune ranges are chosen per scale so the intended note stays the nearest
//! legal one (in C major the E-F and B-C neighbours are a semitone apart, so
//! |detune + drift| is soft-limited to 45 c), mirroring the R§10.2 detuning
//! model (per-note offsets, OU drift σ ≈ 10-15 c / τ = 1 s, onset scoops of
//! −50…−150 c decaying with τ = 60 ms, +30 c overshoot) within those limits.

use vt_core::{ScaleKind, Settings, VoiceRange};

use crate::rng::Rng;
use crate::synth::{Consonant, Melody, Note, Signal, SignalBuilder, Vibrato, Voice, Vowel};

/// Which metrics are meaningful for a scenario.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measures {
    pub tuning: bool,
    pub settling: bool,
    /// Settling also after legato boundaries (step tests).
    pub settling_legato: bool,
    pub vibrato: bool,
    pub jitter: bool,
    pub voicing: bool,
    pub latency: bool,
    /// Score sweeps against the nearest-semitone staircase.
    pub sweep: bool,
}

pub struct Scenario {
    pub name: &'static str,
    pub description: &'static str,
    pub signal: Signal,
    /// Key/scale (and preferences) that make `target_cents` correct.
    pub settings: Settings,
    pub voice_range: VoiceRange,
    /// Retune speeds the suite renders, ms.
    pub retunes: &'static [f32],
    pub measures: Measures,
}

/// Every scenario, in suite order.
pub const NAMES: &[&str] = &[
    "melody_male",
    "melody_female",
    "vibrato",
    "steps",
    "sweep",
    "steady",
    "consonants",
    "low_male",
    "high_soprano",
    "noise_bursts",
    "edge",
];

/// Settings with `key`/`scale` resolved into note states.
pub fn settings_for(key: u8, scale: ScaleKind) -> Settings {
    Settings { key, scale, note_states: scale.note_states(key), ..Settings::default() }
}

/// Build scenario `name` at `sample_rate` (None for unknown names).
pub fn build(name: &str, sample_rate: f32) -> Option<Scenario> {
    let sr = sample_rate;
    let c_major = settings_for(0, ScaleKind::Major);
    let chromatic = settings_for(0, ScaleKind::Chromatic);
    let tuning = Measures { tuning: true, settling: true, voicing: true, ..Measures::default() };
    Some(match name {
        "melody_male" => Scenario {
            name: "melody_male",
            description: "Detuned C-major melody, male voice (C3-A3): legato and gaps, scoops, overshoot, drift, consonants",
            signal: SignalBuilder::new(sr).seed(11).silence(0.25).melody(sung_line(Voice::male(), 48, 101)).silence(0.3).build(),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "melody_female" => Scenario {
            name: "melody_female",
            description: "The same line an octave and a third higher, female voice (E4-C5)",
            signal: SignalBuilder::new(sr).seed(12).silence(0.25).melody(sung_line(Voice::female(), 64, 202)).silence(0.3).build(),
            settings: settings_for(4, ScaleKind::Major),
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "vibrato" => Scenario {
            name: "vibrato",
            description: "Long detuned notes with natural vibrato 5.5-6.5 Hz, ±40-60 c (male A2-A3)",
            signal: SignalBuilder::new(sr).seed(13).silence(0.25).melody(vibrato_line()).silence(0.3).build(),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures { tuning: true, vibrato: true, voicing: true, ..Measures::default() },
        },
        "steps" => Scenario {
            name: "steps",
            description: "Sustained A3 alternating +40/−40 c every 500 ms (R§10.5 correction-lag test)",
            signal: SignalBuilder::new(sr)
                .seed(14)
                .silence(0.25)
                .melody(Melody::steps(Voice::male(), 5700.0, 40.0, 0.5, 8, Vowel::A))
                .silence(0.3)
                .build(),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures { tuning: true, settling: true, settling_legato: true, ..Measures::default() },
        },
        "sweep" => Scenario {
            name: "sweep",
            description: "A2-A3-A2 glide, 2 s each way (Flex-Tune / staircase calibration)",
            signal: SignalBuilder::new(sr)
                .seed(15)
                .silence(0.25)
                .melody(
                    Melody::new(Voice::male())
                        .note(Note::new(4500.0, 2.0).sweep_to(5700.0).detune(0.0).vowel(Vowel::O))
                        .note(Note::new(5700.0, 2.0).sweep_to(4500.0).detune(0.0).vowel(Vowel::O).glide(0.0)),
                )
                .silence(0.3)
                .build(),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures { sweep: true, ..Measures::default() },
        },
        "steady" => Scenario {
            name: "steady",
            description: "Steady sustained notes, small fixed detune, no vibrato (added-jitter test)",
            signal: steady_signal(sr),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: Measures { tuning: true, jitter: true, ..Measures::default() },
        },
        "consonants" => Scenario {
            name: "consonants",
            description: "Short syllables separated by s, sh, f, t, k and breaths (voicing / sibilance)",
            signal: consonant_signal(sr),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[20.0],
            measures: Measures { voicing: true, tuning: true, ..Measures::default() },
        },
        "low_male" => Scenario {
            name: "low_male",
            description: "Bass line E2-D3 (82-147 Hz) with drift and a vibrato note",
            signal: SignalBuilder::new(sr).seed(17).silence(0.25).melody(low_line()).silence(0.3).build(),
            settings: settings_for(4, ScaleKind::NaturalMinor),
            voice_range: VoiceRange::LowMale,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "high_soprano" => Scenario {
            name: "high_soprano",
            description: "Soprano line C5-C6 (523-1047 Hz), female voice",
            signal: SignalBuilder::new(sr).seed(18).silence(0.25).melody(high_line()).silence(0.3).build(),
            settings: c_major,
            voice_range: VoiceRange::Soprano,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "noise_bursts" => Scenario {
            name: "noise_bursts",
            description: "White-noise bursts and an aperiodic click train (latency measurement)",
            signal: noise_burst_signal(sr),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[20.0],
            measures: Measures { latency: true, ..Measures::default() },
        },
        "edge" => Scenario {
            name: "edge",
            description: "Silence, DC, denormals, a clipped full-scale tone, a near-Nyquist tone, then a note",
            signal: edge_signal(sr),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[20.0],
            measures: Measures::default(),
        },
        _ => return None,
    })
}

/// Every scenario at `sample_rate`.
pub fn all(sample_rate: f32) -> Vec<Scenario> {
    NAMES.iter().filter_map(|n| build(n, sample_rate)).collect()
}

/// The main test melody: (semitones above `root`, dur, gap before, onset consonant).
fn sung_line(voice: Voice, root: i32, seed: u64) -> Melody {
    // C D E | G F E | D C | A G E | C  (in C major, transposed by `root`).
    let line: [(i32, f64, f64, Option<(Consonant, f64)>); 14] = [
        (0, 0.50, 0.0, None),
        (2, 0.40, 0.0, None),
        (4, 0.60, 0.0, None),
        (7, 0.50, 0.18, Some((Consonant::S, 0.12))),
        (5, 0.40, 0.0, None),
        (4, 0.45, 0.0, None),
        (2, 0.60, 0.20, Some((Consonant::T, 0.03))),
        (0, 0.50, 0.0, None),
        (9, 0.70, 0.35, Some((Consonant::Breath, 0.25))),
        (7, 0.50, 0.0, None),
        (4, 0.45, 0.0, None),
        (5, 0.40, 0.18, Some((Consonant::Sh, 0.12))),
        (4, 0.40, 0.0, None),
        (0, 0.80, 0.20, Some((Consonant::K, 0.04))),
    ];
    let vowels = [Vowel::A, Vowel::E, Vowel::O, Vowel::A, Vowel::I, Vowel::U, Vowel::A];
    let mut rng = Rng::new(seed);
    let notes = line.iter().enumerate().map(|(i, &(st, dur, gap, cons))| {
        let mut n = Note::midi((root + st) as f64, dur).vowel(vowels[i % vowels.len()]).glide(0.08);
        if gap > 0.0 || i == 0 {
            n = n.gap(gap).scoop(-rng.range(50.0, 120.0));
        } else {
            n = n.overshoot(30.0);
        }
        if let Some((c, d)) = cons {
            let level = if c == Consonant::Breath { -42.0 } else { -32.0 };
            n = n.consonant(c, d, level);
        }
        n
    });
    Melody::new(voice).notes(notes).detune_range(35.0).drift(10.0, 1.0).deviation_limit(45.0).seed(seed)
}

fn vibrato_line() -> Melody {
    let spec: [(f64, f64, f64, f64, f64, f64); 5] = [
        // (midi, dur, gap, rate, extent, detune)
        (45.0, 1.6, 0.0, 5.5, 50.0, 25.0),
        (48.0, 1.4, 0.0, 6.0, 40.0, -20.0),
        (52.0, 1.6, 0.25, 6.5, 60.0, 30.0),
        (55.0, 1.4, 0.0, 5.8, 45.0, -30.0),
        (57.0, 1.8, 0.25, 6.2, 55.0, 15.0),
    ];
    let vowels = [Vowel::A, Vowel::O, Vowel::E, Vowel::A, Vowel::U];
    let notes = spec.iter().enumerate().map(|(i, &(m, d, gap, rate, ext, det))| {
        let mut n = Note::midi(m, d).vowel(vowels[i]).detune(det).vibrato(Vibrato::new(rate, ext)).glide(0.1);
        n = if gap > 0.0 || i == 0 { n.gap(gap).scoop(-60.0) } else { n.overshoot(20.0) };
        n
    });
    Melody::new(Voice::male()).notes(notes).drift(6.0, 1.0).deviation_limit(40.0).seed(31)
}

fn low_line() -> Melody {
    // E2 G2 A2 B2 | D3 B2 A2 (vibrato) | E2 in E natural minor.
    let line: [(f64, f64, f64, bool); 8] = [
        (40.0, 0.6, 0.0, false),
        (43.0, 0.5, 0.0, false),
        (45.0, 0.6, 0.0, false),
        (47.0, 0.5, 0.2, false),
        (50.0, 0.6, 0.0, false),
        (47.0, 0.5, 0.0, false),
        (45.0, 1.4, 0.25, true),
        (40.0, 0.9, 0.25, false),
    ];
    let notes = line.iter().enumerate().map(|(i, &(m, d, gap, vib))| {
        let mut n = Note::midi(m, d).vowel(if i % 2 == 0 { Vowel::O } else { Vowel::A }).glide(0.1);
        n = if gap > 0.0 || i == 0 { n.gap(gap).scoop(-80.0) } else { n.overshoot(25.0) };
        if vib {
            n = n.vibrato(Vibrato::new(5.5, 45.0));
        }
        n
    });
    Melody::new(Voice::male()).notes(notes).detune_range(35.0).drift(10.0, 1.0).deviation_limit(45.0).seed(41)
}

fn high_line() -> Melody {
    // C5 E5 G5 A5 | G5 E5 | C6 | A5 G5 (C major, 523-1047 Hz).
    let line: [(f64, f64, f64); 9] = [
        (72.0, 0.5, 0.0),
        (76.0, 0.4, 0.0),
        (79.0, 0.5, 0.0),
        (81.0, 0.6, 0.2),
        (79.0, 0.4, 0.0),
        (76.0, 0.5, 0.0),
        (84.0, 0.7, 0.25),
        (81.0, 0.5, 0.0),
        (79.0, 0.8, 0.0),
    ];
    let notes = line.iter().enumerate().map(|(i, &(m, d, gap))| {
        let mut n = Note::midi(m, d).vowel(if i % 3 == 0 { Vowel::A } else { Vowel::O }).glide(0.08);
        n = if gap > 0.0 || i == 0 { n.gap(gap).scoop(-60.0) } else { n.overshoot(20.0) };
        n
    });
    Melody::new(Voice::female()).notes(notes).detune_range(35.0).drift(8.0, 1.0).deviation_limit(45.0).seed(51)
}

fn steady_signal(sr: f32) -> Signal {
    let mut male = Voice::male();
    male.jitter = 0.001;
    male.shimmer = 0.01;
    let mut female = Voice::female();
    female.jitter = 0.001;
    female.shimmer = 0.01;
    SignalBuilder::new(sr)
        .seed(16)
        .silence(0.25)
        .melody(
            Melody::new(male)
                .note(Note::midi(45.0, 2.0).detune(18.0).vowel(Vowel::A))
                .note(Note::midi(50.0, 2.0).detune(-22.0).vowel(Vowel::O).gap(0.3)),
        )
        .silence(0.3)
        .melody(Melody::new(female).note(Note::midi(69.0, 2.0).detune(15.0).vowel(Vowel::A)))
        .silence(0.3)
        .build()
}

fn consonant_signal(sr: f32) -> Signal {
    let syll: [(Consonant, f64, f64, Vowel, f64); 8] = [
        (Consonant::S, 0.15, -30.0, Vowel::A, 52.0),
        (Consonant::Sh, 0.15, -30.0, Vowel::I, 55.0),
        (Consonant::F, 0.12, -40.0, Vowel::O, 53.0),
        (Consonant::T, 0.04, -28.0, Vowel::A, 52.0),
        (Consonant::K, 0.05, -28.0, Vowel::E, 50.0),
        (Consonant::Breath, 0.30, -42.0, Vowel::U, 48.0),
        (Consonant::S, 0.20, -26.0, Vowel::E, 57.0),
        (Consonant::Sh, 0.10, -32.0, Vowel::O, 55.0),
    ];
    let notes = syll.iter().map(|&(c, d, lvl, v, m)| Note::midi(m, 0.35).vowel(v).consonant(c, d, lvl).gap(d + 0.06).scoop(-50.0));
    SignalBuilder::new(sr)
        .seed(19)
        .silence(0.2)
        .melody(Melody::new(Voice::male()).notes(notes).detune_range(30.0).deviation_limit(45.0).seed(61))
        .consonant(Consonant::Breath, 0.4, -40.0)
        .silence(0.3)
        .build()
}

fn noise_burst_signal(sr: f32) -> Signal {
    SignalBuilder::new(sr)
        .seed(20)
        .silence(0.15)
        .noise_burst(0.06, -20.0)
        .silence(0.2)
        .noise_burst(0.03, -14.0)
        .silence(0.2)
        .pulse_train(0.2, 0.013, 0.6, 0.4)
        .silence(0.2)
        .noise_burst(0.08, -26.0)
        .silence(0.25)
        .build()
}

fn edge_signal(sr: f32) -> Signal {
    let n = |s: f64| (s * sr as f64) as usize;
    let dc = vec![0.3f32; n(0.1)];
    let denormal: Vec<f32> = (0..n(0.1)).map(|i| if i % 2 == 0 { 1.0e-39 } else { -1.0e-39 }).collect();
    let clipped: Vec<f32> =
        (0..n(0.2)).map(|i| (1.5 * (2.0 * std::f64::consts::PI * 150.0 * i as f64 / sr as f64).sin()).clamp(-0.99, 0.99) as f32).collect();
    let nyq: Vec<f32> = (0..n(0.1)).map(|i| (0.5 * (std::f64::consts::PI * 0.98 * i as f64).sin()) as f32).collect();
    SignalBuilder::new(sr)
        .seed(21)
        .silence(0.1)
        .raw(dc)
        .raw(denormal)
        .raw(clipped)
        .silence(0.05)
        .raw(nyq)
        .silence(0.1)
        .melody(Melody::new(Voice::male()).note(Note::midi(57.0, 0.5).detune(20.0)))
        .silence(0.2)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_builds_with_consistent_truth() {
        for name in NAMES {
            let s = build(name, 48_000.0).unwrap();
            assert_eq!(s.name, *name);
            let sig = &s.signal;
            assert!(sig.len() > 4800, "{name}");
            assert!(sig.audio.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{name}");
            assert_eq!(sig.f0_truth.len(), sig.len());
            // Every intended note is a legal (On) note of the scenario's scale.
            for n in sig.notes.iter().filter(|n| n.target_cents.is_finite()) {
                let pc = ((n.target_cents / 100.0).round() as i64).rem_euclid(12) as usize;
                assert_eq!(s.settings.note_states[pc], vt_core::NoteState::On, "{name}: note {}", n.target_cents);
            }
        }
        assert!(build("nope", 48_000.0).is_none());
    }

    #[test]
    fn detuned_pitch_stays_nearest_to_its_target() {
        // Centre pitch (no vibrato) after the scoop has decayed must stay
        // within ±50 c of the target so the note decision is unambiguous.
        for name in ["melody_male", "melody_female", "low_male", "high_soprano", "vibrato", "steps"] {
            let s = build(name, 48_000.0).unwrap();
            let sig = &s.signal;
            for n in &sig.notes {
                let a = n.start + (0.35 * 48_000.0) as usize;
                for i in a.min(n.end)..n.end {
                    let d = (sig.centre_cents[i] as f64 - n.target_cents).abs();
                    assert!(d < 50.0, "{name}: deviation {d} at {i}");
                }
            }
        }
    }
}
