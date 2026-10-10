//! With the note-intent Labs options off (learned note hold off, reference
//! A440: the `Settings` defaults), the engine is bit-identical to the code
//! before those options existed.
//!
//! The golden hashes below were recorded from the engine at commit 0cab284,
//! before the options were added, on aarch64 macOS (Apple M4 Max). Output
//! bits depend on the SIMD backend and the platform's libm, so the
//! comparison only runs there; elsewhere the test prints the hashes and
//! passes. Re-record (`SKATO_PRINT_HASHES=1 cargo test -p skato_eval
//! --release --test note_intent_off -- --nocapture`) only for a change that
//! is meant to alter the output, and say so in the commit.

use skato_core::{
    EngineConfig, LatencyMode, MidiEvent, MidiMode, PitchFrame, QualityTier, Settings,
};
use skato_eval::render::{self, BlockPattern, RenderOptions, default_engine};
use skato_eval::scenarios;

const SR: f32 = 48_000.0;

/// One render: scenario, mode, tier, and what is special about it.
#[derive(Clone, Copy, Debug)]
struct Case {
    scenario: &'static str,
    mode: LatencyMode,
    tier: QualityTier,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// The scenario's own key and scale, hard tune (retune 0, Tolerance 5 c).
    Hard,
    /// The Natural preset (retune 25 ms, Humanize 30, Flex 20, 15 c / 30 ms).
    Natural,
    /// MIDI Target with a few held notes, random host blocks.
    Midi,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let modes = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
    let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
    for scenario in ["melody_male", "melody_female", "vibrato_onset", "heldout_pentatonic", "heldout_room"] {
        for mode in modes {
            for tier in tiers {
                v.push(Case { scenario, mode, tier, kind: Kind::Hard });
            }
        }
    }
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        for tier in tiers {
            v.push(Case { scenario: "melody_male", mode, tier, kind: Kind::Natural });
            v.push(Case { scenario: "melody_female", mode, tier, kind: Kind::Midi });
        }
    }
    v
}

fn options(c: &Case, scn: &scenarios::Scenario) -> RenderOptions {
    let mut s = Settings { tier: c.tier, engine: default_engine(c.mode), ..scn.settings };
    match c.kind {
        Kind::Hard | Kind::Midi => {
            s.retune_ms = 0.0;
            s.vibrato_depth = 0.0;
            s.tolerance_cents = 5.0;
            s.tolerance_ms = 0.0;
        }
        Kind::Natural => {
            s.retune_ms = 25.0;
            s.humanize = 30.0;
            s.flex_tune = 20.0;
            s.tolerance_cents = 15.0;
            s.tolerance_ms = 30.0;
        }
    }
    let config = EngineConfig {
        sample_rate: SR,
        channels: 1,
        latency_mode: c.mode,
        voice_range: scn.voice_range,
        ..EngineConfig::default()
    };
    let mut opts = RenderOptions::new(SR, c.mode, c.tier).with_config(config).with_settings(s);
    if c.kind == Kind::Midi {
        opts.settings.midi_mode = MidiMode::Target;
        opts.blocks = BlockPattern::Random { seed: 0x5eed, min: 1, max: 700 };
        let sec = |t: f64| (t * f64::from(SR)) as u64;
        opts.midi = vec![
            (sec(0.0), MidiEvent::NoteOn { note: 64, velocity: 0.8 }),
            (sec(0.0), MidiEvent::NoteOn { note: 68, velocity: 0.8 }),
            (sec(1.2), MidiEvent::NoteOff { note: 64 }),
            (sec(1.3), MidiEvent::NoteOn { note: 71, velocity: 0.8 }),
            (sec(2.0), MidiEvent::PitchBend { value: 0.25 }),
            (sec(2.6), MidiEvent::AllNotesOff),
            (sec(3.1), MidiEvent::NoteOn { note: 73, velocity: 0.8 }),
        ];
    } else {
        opts.blocks = BlockPattern::Fixed(128);
    }
    opts
}

/// FNV-1a over the output samples and every telemetry field.
fn hash(out: &[f32], tel: &[PitchFrame]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for x in out {
        eat(&x.to_bits().to_le_bytes());
    }
    for f in tel {
        eat(&f.sample_pos.to_le_bytes());
        for v in [f.detected, f.centre, f.target, f.output, f.confidence, f.vib_rate, f.vib_extent] {
            eat(&v.to_bits().to_le_bytes());
        }
        eat(&[f.flags]);
    }
    h
}

fn render_hash(c: &Case) -> u64 {
    let scn = scenarios::build(c.scenario, SR).expect("scenario");
    let opts = options(c, &scn);
    // The defaults are the "off" state of the note-intent options.
    let r = render::render_signal(&scn.signal, &opts);
    hash(r.raw(0), &r.telemetry)
}

/// Recorded at 0cab284 (before the note-intent options), aarch64 macOS, in
/// `cases()` order.
const GOLDEN: &[u64] = &[
    0x6feedd2cc460046a, // melody_male LiveMin Eco Hard
    0x3b8cd575282f890f, // melody_male LiveMin Hq Hard
    0x9c369e132a8becdb, // melody_male LiveMin Max Hard
    0x0b5e0686ed16c8ef, // melody_male Live Eco Hard
    0x7f3bff9ac262bb3d, // melody_male Live Hq Hard
    0xcb82428cba6eb715, // melody_male Live Max Hard
    0xac6154701e5618ca, // melody_male LivePlus Eco Hard
    0x0fe0a5b4cdaea7f6, // melody_male LivePlus Hq Hard
    0x1ad00aa234f6e870, // melody_male LivePlus Max Hard
    0x8026399c3cfec7d1, // melody_male Studio Eco Hard
    0x803f325b99d5a6ca, // melody_male Studio Hq Hard
    0x2342860caeab47a2, // melody_male Studio Max Hard
    0xe462f149741ce765, // melody_female LiveMin Eco Hard
    0x515899527edfefec, // melody_female LiveMin Hq Hard
    0xff869f1b221c8e2e, // melody_female LiveMin Max Hard
    0x08e3aa381ce3f6da, // melody_female Live Eco Hard
    0x9b7e4089b82453e6, // melody_female Live Hq Hard
    0x90b7fcf0326436c5, // melody_female Live Max Hard
    0x1e78da50c2c25e3e, // melody_female LivePlus Eco Hard
    0x5d8e3ae84e2b1124, // melody_female LivePlus Hq Hard
    0x1ffad8bf0afd9d68, // melody_female LivePlus Max Hard
    0x60c569e16693fdc4, // melody_female Studio Eco Hard
    0xae766862a1659ed8, // melody_female Studio Hq Hard
    0x2c0c41ef9e002ba0, // melody_female Studio Max Hard
    0x632f0e7f507b3935, // vibrato_onset LiveMin Eco Hard
    0xe0b103b53797199d, // vibrato_onset LiveMin Hq Hard
    0x7b68e637f99dd412, // vibrato_onset LiveMin Max Hard
    0x3acea6f97b4cb396, // vibrato_onset Live Eco Hard
    0x4c0fe28917bf9c67, // vibrato_onset Live Hq Hard
    0x8cc077ececab36da, // vibrato_onset Live Max Hard
    0xc5f7750f45d7315c, // vibrato_onset LivePlus Eco Hard
    0xf9b615cfcf8fbc70, // vibrato_onset LivePlus Hq Hard
    0x2ff965210875284b, // vibrato_onset LivePlus Max Hard
    0xcd419e045419272c, // vibrato_onset Studio Eco Hard
    0x1018adeb350c4f89, // vibrato_onset Studio Hq Hard
    0x641359583b76c6cd, // vibrato_onset Studio Max Hard
    0x8febb98e53f00976, // heldout_pentatonic LiveMin Eco Hard
    0x6a9c1e530112a1a8, // heldout_pentatonic LiveMin Hq Hard
    0xa65ba3c59839c752, // heldout_pentatonic LiveMin Max Hard
    0xf258e21d2a481792, // heldout_pentatonic Live Eco Hard
    0xeaf343fbc32a226f, // heldout_pentatonic Live Hq Hard
    0x2033aff2c241b06a, // heldout_pentatonic Live Max Hard
    0x96dff74381016b9f, // heldout_pentatonic LivePlus Eco Hard
    0x8c3d8f1d8a31cab0, // heldout_pentatonic LivePlus Hq Hard
    0x7461a8ff10e2df0c, // heldout_pentatonic LivePlus Max Hard
    0x1b0060425bfe3b24, // heldout_pentatonic Studio Eco Hard
    0xbc87e5f348119fcb, // heldout_pentatonic Studio Hq Hard
    0x78bbf31bd238edfd, // heldout_pentatonic Studio Max Hard
    0x68a6fc54c64a0a01, // heldout_room LiveMin Eco Hard
    0x83e57ee33816ece4, // heldout_room LiveMin Hq Hard
    0x185569ce3d07cf0e, // heldout_room LiveMin Max Hard
    0x659deaf8ca51709b, // heldout_room Live Eco Hard
    0x960b30ccab5c74bf, // heldout_room Live Hq Hard
    0x0437f978c7feb329, // heldout_room Live Max Hard
    0x8be4de3b00e963e8, // heldout_room LivePlus Eco Hard
    0x6e84f257a81619b0, // heldout_room LivePlus Hq Hard
    0xdcbee50f501c429d, // heldout_room LivePlus Max Hard
    0xc2f92e4474dcf9a4, // heldout_room Studio Eco Hard
    0x65eb830677c60772, // heldout_room Studio Hq Hard
    0x3462ebfa35ecc8dd, // heldout_room Studio Max Hard
    0x6271c14921706e5f, // melody_male Live Eco Natural
    0xec1d4a5e3718140a, // melody_female Live Eco Midi
    0x9a86fbeb7a03f8e6, // melody_male Live Hq Natural
    0x8207d92a5c58bfa1, // melody_female Live Hq Midi
    0x1c4f69c276a34aa4, // melody_male Live Max Natural
    0x5614f8f02a9027d9, // melody_female Live Max Midi
    0x0d8d32a8e54b1c73, // melody_male Studio Eco Natural
    0xecfab120dfdb69e2, // melody_female Studio Eco Midi
    0xd392eb65eb20a974, // melody_male Studio Hq Natural
    0xa0ac9a0cc51e8c0b, // melody_female Studio Hq Midi
    0x80d3482a827f937b, // melody_male Studio Max Natural
    0x623b8b716bd62340, // melody_female Studio Max Midi
];

#[test]
fn note_intent_off_is_bit_identical_to_the_engine_before_it() {
    let cases = cases();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let hashes = std::sync::Mutex::new(vec![0u64; cases.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(c) = cases.get(i) else { break };
                    let h = render_hash(c);
                    hashes.lock().unwrap()[i] = h;
                }
            });
        }
    });
    let hashes = hashes.into_inner().unwrap();
    if std::env::var_os("SKATO_PRINT_HASHES").is_some() {
        for (c, h) in cases.iter().zip(&hashes) {
            println!("    0x{h:016x}, // {} {:?} {:?} {:?}", c.scenario, c.mode, c.tier, c.kind);
        }
    }
    if !cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        eprintln!("skipping the golden comparison: the hashes were recorded on aarch64 macOS");
        return;
    }
    assert_eq!(GOLDEN.len(), cases.len(), "golden table out of date");
    let mut bad = Vec::new();
    for ((c, h), g) in cases.iter().zip(&hashes).zip(GOLDEN) {
        if h != g {
            bad.push(format!("{} {:?} {:?} {:?}: {h:#018x} != {g:#018x}", c.scenario, c.mode, c.tier, c.kind));
        }
    }
    assert!(bad.is_empty(), "{} of {} renders changed:\n{}", bad.len(), cases.len(), bad.join("\n"));
}
