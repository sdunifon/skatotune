//! Named evaluation scenarios: a signal plus the settings under which its
//! `target_cents` is the correct answer.
//!
//! Detune ranges are chosen per scale so the intended note stays the nearest
//! legal one (in C major the E-F and B-C neighbours are a semitone apart, so
//! |detune + drift| is soft-limited to 45 c), mirroring the R§10.2 detuning
//! model (per-note offsets, OU drift σ ≈ 10-15 c / τ = 1 s, onset scoops of
//! −50…−150 c decaying with τ = 60 ms, +30 c overshoot) within those limits.

use skato_core::{ScaleKind, Settings, VoiceRange};

use crate::rng::Rng;
use crate::synth::{
    Consonant, Melody, Note, Signal, SignalBuilder, Vibrato, Voice, VoicedConsonant, Vowel,
};

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
    "vibrato_onset",
    "steps",
    "sweep",
    "steady",
    "consonants",
    "low_male",
    "high_soprano",
    "noise_bursts",
    "edge",
    "heldout_pentatonic",
    "heldout_room",
];

/// Held-out scenarios (EV-4): built after the engine's HMM and vibrato
/// constants were tuned, from material the tuning never saw (other voices,
/// seeds and keys, R§10.2's ±80 c detune, voiced-consonant onsets, hard and
/// soft attacks, room reverb and a noise floor). The suite applies the same
/// DESIGN §4 targets to them in separate gate rows; nobody may tune engine
/// constants on them, or they stop being held out.
pub const HELDOUT: [&str; 2] = ["heldout_pentatonic", "heldout_room"];

/// Settings with `key`/`scale` resolved into note states.
pub fn settings_for(key: u8, scale: ScaleKind) -> Settings {
    Settings {
        key,
        scale,
        note_states: scale.note_states(key),
        ..Settings::default()
    }
}

/// Legal (On) pitch classes of a settings value, C = 0.
pub fn legal_classes(s: &Settings) -> [bool; 12] {
    s.note_states.map(|st| st == skato_core::NoteState::On)
}

/// Build scenario `name` at `sample_rate` (None for unknown names).
pub fn build(name: &str, sample_rate: f32) -> Option<Scenario> {
    let sr = sample_rate;
    let c_major = settings_for(0, ScaleKind::Major);
    let chromatic = settings_for(0, ScaleKind::Chromatic);
    let tuning = Measures {
        tuning: true,
        settling: true,
        voicing: true,
        ..Measures::default()
    };
    Some(match name {
        "melody_male" => Scenario {
            name: "melody_male",
            description: "Detuned C-major melody, male voice (C3-A3): legato and gaps, scoops, overshoot, drift, consonants",
            signal: SignalBuilder::new(sr)
                .seed(11)
                .silence(0.25)
                .melody(sung_line(Voice::male(), 48, 101))
                .silence(0.3)
                .build(),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "melody_female" => Scenario {
            name: "melody_female",
            description: "The same line an octave and a third higher, female voice (E4-C#5, E major)",
            signal: SignalBuilder::new(sr)
                .seed(12)
                .silence(0.25)
                .melody(sung_line(Voice::female(), 64, 202))
                .silence(0.3)
                .build(),
            settings: settings_for(4, ScaleKind::Major),
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "vibrato" => Scenario {
            name: "vibrato",
            description: "Long detuned notes with natural vibrato 5.5-6.5 Hz, ±40-60 c (male A2-A3)",
            signal: SignalBuilder::new(sr)
                .seed(13)
                .silence(0.25)
                .melody(vibrato_line())
                .silence(0.3)
                .build(),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures {
                tuning: true,
                vibrato: true,
                voicing: true,
                ..Measures::default()
            },
        },
        "vibrato_onset" => Scenario {
            name: "vibrato_onset",
            description: "Vibrato from the very note onset (no delay or fade-in), 5.5-6.5 Hz, ±40-60 c, held male notes A2-D4 after gaps (R§8.11 engagement, EV-3)",
            signal: SignalBuilder::new(sr)
                .seed(22)
                .silence(0.25)
                .melody(vibrato_onset_line())
                .silence(0.3)
                .build(),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures {
                tuning: true,
                vibrato: true,
                voicing: true,
                ..Measures::default()
            },
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
            measures: Measures {
                tuning: true,
                settling: true,
                settling_legato: true,
                ..Measures::default()
            },
        },
        "sweep" => Scenario {
            name: "sweep",
            description: "A2-A3-A2 glide, 2 s each way (Flex-Tune / staircase calibration)",
            signal: SignalBuilder::new(sr)
                .seed(15)
                .silence(0.25)
                .melody(
                    Melody::new(Voice::male())
                        .note(
                            Note::new(4500.0, 2.0)
                                .sweep_to(5700.0)
                                .detune(0.0)
                                .vowel(Vowel::O),
                        )
                        .note(
                            Note::new(5700.0, 2.0)
                                .sweep_to(4500.0)
                                .detune(0.0)
                                .vowel(Vowel::O)
                                .glide(0.0),
                        ),
                )
                .silence(0.3)
                .build(),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0],
            measures: Measures {
                sweep: true,
                ..Measures::default()
            },
        },
        "steady" => Scenario {
            name: "steady",
            description: "Steady sustained notes, small fixed detune, no vibrato (added-jitter test)",
            signal: steady_signal(sr),
            settings: chromatic,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: Measures {
                tuning: true,
                jitter: true,
                ..Measures::default()
            },
        },
        "consonants" => Scenario {
            name: "consonants",
            description: "Short syllables separated by s, sh, f, t, k and breaths (voicing / sibilance)",
            signal: consonant_signal(sr),
            settings: c_major,
            voice_range: VoiceRange::AltoTenor,
            retunes: &[20.0],
            measures: Measures {
                voicing: true,
                tuning: true,
                ..Measures::default()
            },
        },
        "low_male" => Scenario {
            name: "low_male",
            description: "Bass line E2-D3 (82-147 Hz) with drift and a vibrato note",
            signal: SignalBuilder::new(sr)
                .seed(17)
                .silence(0.25)
                .melody(low_line())
                .silence(0.3)
                .build(),
            settings: settings_for(4, ScaleKind::NaturalMinor),
            voice_range: VoiceRange::LowMale,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "high_soprano" => Scenario {
            name: "high_soprano",
            description: "Soprano line C5-C6 (523-1047 Hz), female voice",
            signal: SignalBuilder::new(sr)
                .seed(18)
                .silence(0.25)
                .melody(high_line())
                .silence(0.3)
                .build(),
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
            measures: Measures {
                latency: true,
                ..Measures::default()
            },
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
        "heldout_pentatonic" => Scenario {
            name: "heldout_pentatonic",
            description: "HELD OUT. Male line in C major pentatonic with R§10.2 detuning: per-note ±80 c, OU drift σ 15 c, scoops −50…−150 c, +30 c overshoot; voiced-consonant onsets, hard and soft attacks",
            signal: SignalBuilder::new(sr)
                .seed(901)
                .silence(0.25)
                .melody(pentatonic_line())
                .silence(0.3)
                .build(),
            settings: settings_for(0, ScaleKind::MajorPentatonic),
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        "heldout_room" => Scenario {
            name: "heldout_room",
            description: "HELD OUT. Breathy female line in D major (D4-B4) with voiced-consonant onsets and breathy attacks, in a room (RT60 0.6 s, wet −12 dB) over a −60 dBFS noise floor",
            signal: SignalBuilder::new(sr)
                .seed(902)
                .silence(0.25)
                .melody(room_line())
                .silence(0.3)
                .room(0.6, -12.0)
                .noise_floor(-60.0)
                .build(),
            settings: settings_for(2, ScaleKind::Major),
            voice_range: VoiceRange::AltoTenor,
            retunes: &[0.0, 20.0],
            measures: tuning,
        },
        _ => return None,
    })
}

/// Every scenario at `sample_rate`.
pub fn all(sample_rate: f32) -> Vec<Scenario> {
    NAMES.iter().filter_map(|n| build(n, sample_rate)).collect()
}

/// One step of [`sung_line`]: semitones above the root, duration, gap
/// before, and an optional onset consonant with its duration.
type LineNote = (i32, f64, f64, Option<(Consonant, f64)>);

/// The main test melody.
fn sung_line(voice: Voice, root: i32, seed: u64) -> Melody {
    // C D E | G F E | D C | A G E | C  (in C major, transposed by `root`).
    let line: [LineNote; 14] = [
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
    let vowels = [
        Vowel::A,
        Vowel::E,
        Vowel::O,
        Vowel::A,
        Vowel::I,
        Vowel::U,
        Vowel::A,
    ];
    let mut rng = Rng::new(seed);
    let notes = line.iter().enumerate().map(|(i, &(st, dur, gap, cons))| {
        let mut n = Note::midi((root + st) as f64, dur)
            .vowel(vowels[i % vowels.len()])
            .glide(0.08);
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
    Melody::new(voice)
        .notes(notes)
        .detune_range(35.0)
        .drift(10.0, 1.0)
        .deviation_limit(45.0)
        .seed(seed)
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
    let notes = spec
        .iter()
        .enumerate()
        .map(|(i, &(m, d, gap, rate, ext, det))| {
            let mut n = Note::midi(m, d)
                .vowel(vowels[i])
                .detune(det)
                .vibrato(Vibrato::new(rate, ext))
                .glide(0.1);
            n = if gap > 0.0 || i == 0 {
                n.gap(gap).scoop(-60.0)
            } else {
                n.overshoot(20.0)
            };
            n
        });
    Melody::new(Voice::male())
        .notes(notes)
        .drift(6.0, 1.0)
        .deviation_limit(40.0)
        .seed(31)
}

/// Held notes whose vibrato starts with the voice (as many singers do):
/// nothing before the vibrato for a tracker to settle on. Every note (A2,
/// D3, G3, A3, D4) has whole-tone neighbours in C major and |detune| +
/// extent ≤ 80 c, inside the 100 c − tolerance capture region, so before
/// the vibrato split engages the raw pitch never makes a neighbour the
/// right answer: what is measured is the vibrato tracker, not a note flip.
fn vibrato_onset_line() -> Melody {
    let spec: [(f64, f64, f64, f64); 5] = [
        // (midi, rate, extent, detune)
        (45.0, 5.5, 50.0, 20.0),
        (50.0, 6.0, 40.0, -15.0),
        (55.0, 6.5, 60.0, 20.0),
        (57.0, 5.8, 45.0, -25.0),
        (62.0, 6.2, 55.0, 10.0),
    ];
    let vowels = [Vowel::O, Vowel::A, Vowel::E, Vowel::A, Vowel::O];
    let notes = spec.iter().enumerate().map(|(i, &(m, rate, ext, det))| {
        Note::midi(m, 1.2)
            .vowel(vowels[i])
            .detune(det)
            .vibrato(Vibrato::immediate(rate, ext))
            .gap(if i == 0 { 0.0 } else { 0.25 })
    });
    Melody::new(Voice::male())
        .notes(notes)
        .drift(5.0, 1.0)
        .deviation_limit(40.0)
        .seed(71)
}

/// An onset style: optional voiced consonant (and its length, s) and
/// optional attack time, s.
type OnsetStyle = (Option<(VoicedConsonant, f64)>, Option<f64>);

/// Onset styles cycled through by the held-out pentatonic line.
const ONSETS: [OnsetStyle; 6] = [
    (None, Some(0.004)),
    (Some((VoicedConsonant::M, 0.07)), None),
    (None, Some(0.08)),
    (Some((VoicedConsonant::L, 0.05)), Some(0.03)),
    (Some((VoicedConsonant::Z, 0.06)), None),
    (Some((VoicedConsonant::N, 0.06)), Some(0.004)),
];

/// The ±80 c held-out line: C major pentatonic (C D E G A), whose smallest
/// gap is a whole tone, so the intended note stays the nearest legal one
/// for |detune + drift| < 100 c − tolerance; capped at 85 c.
fn pentatonic_line() -> Melody {
    // C3 D3 E3 G3 | A3 G3 E3 | D3 C3 | E3 G3 A3 C4 | A3 G3
    let line: [(i32, f64, f64); 15] = [
        (48, 0.45, 0.0),
        (50, 0.40, 0.0),
        (52, 0.50, 0.0),
        (55, 0.55, 0.20),
        (57, 0.45, 0.0),
        (55, 0.40, 0.0),
        (52, 0.60, 0.22),
        (50, 0.45, 0.0),
        (48, 0.60, 0.25),
        (52, 0.40, 0.0),
        (55, 0.40, 0.0),
        (57, 0.45, 0.0),
        (60, 0.70, 0.30),
        (57, 0.45, 0.0),
        (55, 0.80, 0.20),
    ];
    let vowels = [Vowel::A, Vowel::O, Vowel::E, Vowel::U, Vowel::A, Vowel::I];
    let mut rng = Rng::new(9010);
    let mut onset = 0usize;
    let notes = line.iter().enumerate().map(|(i, &(m, dur, gap))| {
        let mut n = Note::midi(f64::from(m), dur)
            .vowel(vowels[i % vowels.len()])
            .glide(0.07);
        if gap > 0.0 || i == 0 {
            n = n.gap(gap).scoop(-rng.range(50.0, 150.0));
            let (voiced, attack) = ONSETS[onset % ONSETS.len()];
            onset += 1;
            if let Some((v, d)) = voiced {
                n = n.voiced_onset(v, d);
            }
            if let Some(a) = attack {
                n = n.attack(a);
            }
        } else {
            n = n.overshoot(30.0);
            if i % 3 == 1 {
                n = n.voiced_onset(VoicedConsonant::L, 0.04);
            }
        }
        n
    });
    let mut voice = Voice::male();
    voice.rd = 0.9;
    voice.jitter = 0.003;
    voice.shimmer = 0.03;
    Melody::new(voice)
        .notes(notes)
        .detune_range(80.0)
        .drift(15.0, 1.0)
        .deviation_limit(85.0)
        .seed(9011)
}

/// The roomy held-out line: breathy female voice in D major (semitone
/// neighbours F#-G and C#-D, so |detune + drift| ≤ 40 c), breath before
/// soft attacks, voiced consonants.
fn room_line() -> Melody {
    // D4 E4 F#4 A4 | B4 A4 F#4 | E4 D4 | F#4 G4 A4 | D4
    let line: [(i32, f64, f64); 13] = [
        (62, 0.45, 0.0),
        (64, 0.40, 0.0),
        (66, 0.55, 0.0),
        (69, 0.50, 0.22),
        (71, 0.45, 0.0),
        (69, 0.40, 0.0),
        (66, 0.60, 0.25),
        (64, 0.45, 0.0),
        (62, 0.55, 0.28),
        (66, 0.40, 0.0),
        (67, 0.40, 0.0),
        (69, 0.55, 0.0),
        (62, 0.80, 0.30),
    ];
    let vowels = [Vowel::O, Vowel::A, Vowel::E, Vowel::A, Vowel::U];
    let mut rng = Rng::new(9020);
    let mut onset = 0usize;
    let notes = line.iter().enumerate().map(|(i, &(m, dur, gap))| {
        let mut n = Note::midi(f64::from(m), dur)
            .vowel(vowels[i % vowels.len()])
            .glide(0.08);
        if gap > 0.0 || i == 0 {
            n = n.gap(gap).scoop(-rng.range(50.0, 120.0));
            match onset % 3 {
                // Breathy attack: audible breath straight into a slow onset.
                0 => n = n.consonant(Consonant::Breath, 0.12, -40.0).attack(0.09),
                1 => n = n.voiced_onset(VoicedConsonant::M, 0.07),
                _ => n = n.voiced_onset(VoicedConsonant::V, 0.05).attack(0.05),
            }
            onset += 1;
        } else {
            n = n.overshoot(30.0);
        }
        n
    });
    let mut voice = Voice::female();
    voice.rd = 1.7;
    voice.aspiration_db = -20.0;
    Melody::new(voice)
        .notes(notes)
        .detune_range(35.0)
        .drift(10.0, 1.0)
        .deviation_limit(40.0)
        .seed(9021)
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
        let mut n = Note::midi(m, d)
            .vowel(if i % 2 == 0 { Vowel::O } else { Vowel::A })
            .glide(0.1);
        n = if gap > 0.0 || i == 0 {
            n.gap(gap).scoop(-80.0)
        } else {
            n.overshoot(25.0)
        };
        if vib {
            n = n.vibrato(Vibrato::new(5.5, 45.0));
        }
        n
    });
    Melody::new(Voice::male())
        .notes(notes)
        .detune_range(35.0)
        .drift(10.0, 1.0)
        .deviation_limit(45.0)
        .seed(41)
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
        let mut n = Note::midi(m, d)
            .vowel(if i % 3 == 0 { Vowel::A } else { Vowel::O })
            .glide(0.08);
        n = if gap > 0.0 || i == 0 {
            n.gap(gap).scoop(-60.0)
        } else {
            n.overshoot(20.0)
        };
        n
    });
    Melody::new(Voice::female())
        .notes(notes)
        .detune_range(35.0)
        .drift(8.0, 1.0)
        .deviation_limit(45.0)
        .seed(51)
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
    let notes = syll.iter().map(|&(c, d, lvl, v, m)| {
        Note::midi(m, 0.35)
            .vowel(v)
            .consonant(c, d, lvl)
            .gap(d + 0.06)
            .scoop(-50.0)
    });
    SignalBuilder::new(sr)
        .seed(19)
        .silence(0.2)
        .melody(
            Melody::new(Voice::male())
                .notes(notes)
                .detune_range(30.0)
                .deviation_limit(45.0)
                .seed(61),
        )
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
    let denormal: Vec<f32> = (0..n(0.1))
        .map(|i| if i % 2 == 0 { 1.0e-39 } else { -1.0e-39 })
        .collect();
    let clipped: Vec<f32> = (0..n(0.2))
        .map(|i| {
            (1.5 * (2.0 * std::f64::consts::PI * 150.0 * i as f64 / sr as f64).sin())
                .clamp(-0.99, 0.99) as f32
        })
        .collect();
    let nyq: Vec<f32> = (0..n(0.1))
        .map(|i| (0.5 * (std::f64::consts::PI * 0.98 * i as f64).sin()) as f32)
        .collect();
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
            assert!(
                sig.audio.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "{name}"
            );
            assert_eq!(sig.f0_truth.len(), sig.len());
            // Every intended note is a legal (On) note of the scenario's scale.
            for n in sig.notes.iter().filter(|n| n.target_cents.is_finite()) {
                let pc = ((n.target_cents / 100.0).round() as i64).rem_euclid(12) as usize;
                assert_eq!(
                    s.settings.note_states[pc],
                    skato_core::NoteState::On,
                    "{name}: note {}",
                    n.target_cents
                );
            }
        }
        assert!(build("nope", 48_000.0).is_none());
    }

    /// The melody an ideal corrector would output: every note exactly on its
    /// target (no detune, drift, scoop or overshoot), vibrato kept, same
    /// voice and noise seeds.
    fn oracle(m: &Melody) -> Melody {
        let mut o = m.clone();
        o.drift = None;
        for n in &mut o.notes {
            n.detune_cents = Some(0.0);
            n.scoop_cents = 0.0;
            n.overshoot_cents = 0.0;
            n.glide_s = 0.0; // a hard-tuner jumps at the note boundary
        }
        o
    }

    /// The measurement floor: an oracle output passes every pitch gate with a
    /// wide margin, so the §4 thresholds measure the engine, not the harness.
    #[test]
    fn oracle_output_passes_pitch_gates_with_margin() {
        use crate::metrics::{TuningOptions, jitter, settling, tuning, vibrato};
        use crate::reference::track;
        let sr = 48_000.0;
        for (name, m) in [
            ("male", sung_line(Voice::male(), 48, 101)),
            ("female", sung_line(Voice::female(), 64, 202)),
            ("vibrato", vibrato_line()),
        ] {
            let build = |m: Melody| {
                SignalBuilder::new(sr)
                    .seed(11)
                    .silence(0.25)
                    .melody(m)
                    .silence(0.3)
                    .build()
            };
            let input = build(m.clone());
            let ideal = build(oracle(&m));
            let (ti, to) = (track(&input.audio, sr), track(&ideal.audio, sr));
            let t = tuning(&input, &to, &TuningOptions::default());
            let st = settling(&input, &to, 10.0, 0.02, false);
            eprintln!("oracle {name}: tuning {t:?}\n  settling {:?}", st.per_note);
            // Floor ≈ the voice's own 0.2 % period jitter (~2 c RMS over a
            // 20 ms window), which a pitch-preserving engine reproduces.
            assert!(t.median_abs < 1.6 && t.p95_abs < 5.0, "{name}: {t:?}");
            assert!(
                t.unvoiced * 50 < t.frames,
                "{name}: tracker lost {} frames",
                t.unvoiced
            );
            if name == "vibrato" {
                let v = vibrato(&input, &ti, &to, 0.03);
                eprintln!("  vibrato {v:?}");
                assert!(
                    v.worst_extent_ratio_error < 0.05 && v.centre_error_median < 2.0,
                    "{v:?}"
                );
            } else {
                assert!(
                    st.settled == st.per_note.len() && st.max_s < 0.02,
                    "{name}: settling {:?}",
                    st.per_note
                );
            }
        }
        // Jitter floor on the steady scenario: the oracle is the same notes undetuned.
        let steady = build("steady", sr).unwrap().signal;
        let t = track(&steady.audio, sr);
        let j = jitter(&steady, &t, &t, 12.0);
        eprintln!("steady jitter floor {j:?}");
        // ≈ 1 c: the voice's own 0.1 % jitter as resolved per cycle.
        assert!(j.out_rms < 1.5, "tracker jitter floor {j:?}");
    }

    /// Independent check of the retune-20 tuning metric (the integrator's
    /// `TuningOptions::stable_window_s = 2·τ`, which skips frames within
    /// 2·τ of an input movement): an *ideal* R§7.6 corrector (perfect
    /// pitch, instant note decisions, snap at voiced onsets, then the τ one-
    /// pole on the correction) cannot meet DESIGN §4's p95 < 15 c at
    /// τ = 20 ms on `melody_male` when every frame counts, because scoop
    /// tails, glides and overshoot keep its output converging for tens of
    /// ms. With the window it is nearly perfect, so the window removes only
    /// by-design convergence. No engine involved.
    #[test]
    fn ideal_retune_corrector_needs_the_stable_window() {
        use crate::metrics::{TuningOptions, tuning};
        use crate::reference::{PitchTrack, RefFrame};
        let sr = 48_000.0f32;
        let s = build("melody_male", sr).unwrap();
        let sig = &s.signal;
        let tau = 0.02;
        let a = (-1.0 / (tau * f64::from(sr))).exp();
        let (mut corr, mut voiced_before) = (0.0f64, false);
        let mut ideal = vec![f64::NAN; sig.len()];
        for (i, slot) in ideal.iter_mut().enumerate() {
            let (c, t) = (
                f64::from(sig.centre_cents[i]),
                f64::from(sig.target_cents[i]),
            );
            if c.is_finite() && t.is_finite() {
                let want = t - c;
                corr = if voiced_before {
                    a * corr + (1.0 - a) * want
                } else {
                    want
                };
                *slot = c + corr;
                voiced_before = true;
            } else {
                voiced_before = false;
            }
        }
        let hop = 48;
        let frames = (0..sig.len() / hop)
            .map(|k| {
                let c = ideal[k * hop];
                RefFrame {
                    pos: k * hop,
                    voiced: c.is_finite(),
                    f0_hz: if c.is_finite() {
                        crate::synth::cents_to_hz64(c)
                    } else {
                        0.0
                    },
                    cents: c,
                    cents_raw: c,
                    clarity: 1.0,
                    level_dbfs: -20.0,
                }
            })
            .collect();
        let track = PitchTrack {
            sample_rate: f64::from(sr),
            hop,
            frames,
        };
        let every = tuning(sig, &track, &TuningOptions::default());
        let stable = tuning(
            sig,
            &track,
            &TuningOptions {
                stable_window_s: 2.0 * tau,
                ..TuningOptions::default()
            },
        );
        eprintln!("ideal τ = 20 ms: every frame {every:?}\n  stable window {stable:?}");
        assert!(every.p95_abs > 15.0, "{every:?}");
        assert!(stable.p95_abs < 5.0, "{stable:?}");
        assert!(
            stable.frames * 10 > every.frames * 7,
            "window keeps most frames"
        );
    }

    #[test]
    fn heldout_pentatonic_stresses_the_note_decision() {
        // R§10.2's ±80 c per-note detune: several notes sit more than 50 c
        // off (a chromatic corrector would pick the neighbour), yet every
        // note stays inside its pentatonic capture region (whole-tone or
        // wider gaps: < 100 c − 8 c tolerance) once its scoop has decayed.
        let s = build("heldout_pentatonic", 48_000.0).unwrap();
        let sig = &s.signal;
        let mut far = 0;
        for n in &sig.notes {
            let a = n.start + (0.35 * 48_000.0) as usize;
            let mut worst = 0.0f64;
            for i in a.min(n.end)..n.end {
                let d = (f64::from(sig.centre_cents[i]) - n.target_cents).abs();
                assert!(d < 92.0, "deviation {d} at {i}");
                worst = worst.max(d);
            }
            if worst > 50.0 {
                far += 1;
            }
        }
        assert!(far >= 3, "only {far} notes beyond 50 c");
        // The /z/ onset's frication is in the take.
        assert!(!sig.events.is_empty());
    }

    #[test]
    fn detuned_pitch_stays_nearest_to_its_target() {
        // Centre pitch (no vibrato) after the scoop has decayed must stay
        // within ±50 c of the target so the note decision is unambiguous.
        for name in [
            "melody_male",
            "melody_female",
            "low_male",
            "high_soprano",
            "vibrato",
            "steps",
        ] {
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
