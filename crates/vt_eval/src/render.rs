//! Run the realtime [`Engine`] over a signal exactly as a host would.
//!
//! The harness feeds blocks through the public `process()` API with a chosen
//! block-size pattern, drains telemetry after every block (so the ring never
//! overflows), and pads the input with silence so the latency-delayed tail
//! comes out. Everything the engine sees is deterministic, so two renders with
//! the same options are bit-identical.

use vt_core::{
    EngineConfig, EngineKind, LatencyMode, MidiEvent, PitchFrame, QualityTier, Settings,
};
use vt_engine::{Engine, EngineStats, TimedMidi};

use crate::rng::Rng;

/// Host block sizes to use, in order (patterns repeat until the input ends).
#[derive(Clone, Debug, PartialEq)]
pub enum BlockPattern {
    Fixed(usize),
    /// Cycle through these sizes.
    Sequence(Vec<usize>),
    /// Uniform random sizes in [min, max] from a fixed seed.
    Random {
        seed: u64,
        min: usize,
        max: usize,
    },
}

impl BlockPattern {
    /// The standard invariance set of R§8.0: {1, 7, 32, 64, 128, 480, 4096, random}.
    pub fn invariance_set() -> Vec<BlockPattern> {
        let mut v: Vec<BlockPattern> = [1, 7, 32, 64, 128, 480, 4096]
            .into_iter()
            .map(BlockPattern::Fixed)
            .collect();
        v.push(BlockPattern::Random {
            seed: 0xb10c,
            min: 1,
            max: 1500,
        });
        v
    }

    /// Short label for reports.
    pub fn label(&self) -> String {
        match self {
            BlockPattern::Fixed(n) => n.to_string(),
            BlockPattern::Sequence(s) => format!("seq{}", s.len()),
            BlockPattern::Random { min, max, .. } => format!("rand{min}-{max}"),
        }
    }

    /// Block sizes covering exactly `total` samples.
    pub fn blocks(&self, total: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut done = 0usize;
        let mut rng = match self {
            BlockPattern::Random { seed, .. } => Some(Rng::new(*seed)),
            _ => None,
        };
        let mut i = 0usize;
        while done < total {
            let want = match self {
                BlockPattern::Fixed(n) => (*n).max(1),
                BlockPattern::Sequence(s) => s.get(i % s.len().max(1)).copied().unwrap_or(1).max(1),
                BlockPattern::Random { min, max, .. } => {
                    let (lo, hi) = ((*min).max(1), (*max).max((*min).max(1)));
                    let r = rng.as_mut().map_or(0.0, |r| r.uniform());
                    lo + ((hi - lo + 1) as f64 * r) as usize
                }
            };
            let n = want.min(total - done);
            out.push(n);
            done += n;
            i += 1;
        }
        out
    }
}

/// Everything that defines one render.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderOptions {
    pub config: EngineConfig,
    pub settings: Settings,
    pub blocks: BlockPattern,
    /// Collect telemetry frames (the GUI path).
    pub telemetry: bool,
    /// MIDI events at absolute input sample positions (sorted or not).
    pub midi: Vec<(u64, MidiEvent)>,
    /// Silence appended after the input, in addition to the reported latency.
    pub extra_tail: usize,
}

impl RenderOptions {
    /// Mono engine at `sample_rate` with default settings, 64-sample blocks.
    pub fn new(sample_rate: f32, mode: LatencyMode, tier: QualityTier) -> Self {
        let config = EngineConfig {
            sample_rate,
            channels: 1,
            latency_mode: mode,
            ..EngineConfig::default()
        };
        let settings = Settings {
            tier,
            engine: default_engine(mode),
            ..Settings::default()
        };
        Self {
            config,
            settings,
            blocks: BlockPattern::Fixed(64),
            telemetry: true,
            midi: Vec::new(),
            extra_tail: 0,
        }
    }

    pub fn with_settings(mut self, s: Settings) -> Self {
        self.settings = s;
        self
    }

    pub fn with_blocks(mut self, b: BlockPattern) -> Self {
        self.blocks = b;
        self
    }

    pub fn with_config(mut self, c: EngineConfig) -> Self {
        self.config = c;
        self
    }
}

/// Live-Min only allows the Classic engine (DESIGN §3.4); every other mode
/// defaults to PSOLA.
pub fn default_engine(mode: LatencyMode) -> EngineKind {
    if mode == LatencyMode::LiveMin {
        EngineKind::Classic
    } else {
        EngineKind::Natural
    }
}

/// Hops between the end of a hop and the boundary at which the engine applies
/// that hop's control and emits its telemetry frame: 1 for the Studio
/// pipeline (RT-2; the engine pipelines Studio whenever
/// [`vt_engine::studio_split`] leaves the control law a lead of at least one
/// hop), 0 otherwise. The first `pipeline_hops` hops after a reset therefore
/// apply no control and emit no frame, so a render has
/// `stats.hops − pipeline_hops` frames, and frame `j` is produced at the
/// boundary that closes hop `j + pipeline_hops`.
pub fn pipeline_hops(config: &EngineConfig) -> u64 {
    if config.latency_mode != LatencyMode::Studio {
        return 0;
    }
    let hop = vt_core::hop_for_rate(config.sample_rate);
    let d = vt_engine::latency_samples(config) as usize;
    let (_, lead) = vt_engine::studio_split(config.sample_rate, hop, config.voice_range, d);
    u64::from(lead >= 1)
}

/// The result of a render.
#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    /// Raw output per channel: `input_len + latency + extra_tail` samples.
    pub channels: Vec<Vec<f32>>,
    pub latency: u32,
    pub input_len: usize,
    pub telemetry: Vec<PitchFrame>,
    pub stats: EngineStats,
}

impl Rendered {
    /// Raw output of channel `ch` (empty if absent).
    pub fn raw(&self, ch: usize) -> &[f32] {
        self.channels.get(ch).map_or(&[], |c| c.as_slice())
    }

    /// Latency-compensated output of channel `ch`: `aligned[n]` is the output
    /// that carries input sample `n` (what a DAW with PDC plays).
    pub fn aligned(&self, ch: usize) -> &[f32] {
        let raw = self.raw(ch);
        let d = self.latency as usize;
        let end = (d + self.input_len).min(raw.len());
        raw.get(d.min(end)..end).unwrap_or(&[])
    }
}

/// Collect the events of the block `[pos, pos + n)` from the sorted absolute
/// list (continuing at `*idx`) with block-relative timings. Events before
/// `pos` that were never delivered land at offset 0.
fn block_events(
    midi: &[(u64, MidiEvent)],
    idx: &mut usize,
    pos: usize,
    n: usize,
    out: &mut Vec<TimedMidi>,
) {
    out.clear();
    while let Some((t, ev)) = midi.get(*idx) {
        let t = *t as usize;
        if t >= pos + n {
            break;
        }
        out.push(TimedMidi {
            timing: t.saturating_sub(pos) as u32,
            event: *ev,
        });
        *idx += 1;
    }
}

/// Render a synthetic [`Signal`](crate::synth::Signal) (mono, duplicated
/// onto every engine channel).
pub fn render_signal(sig: &crate::synth::Signal, opts: &RenderOptions) -> Rendered {
    render(&sig.audio, opts)
}

/// Render mono `input`, duplicated onto every engine channel.
pub fn render(input: &[f32], opts: &RenderOptions) -> Rendered {
    let chans = opts.config.channels.max(1);
    let inputs: Vec<&[f32]> = (0..chans).map(|_| input).collect();
    render_multi(&inputs, opts)
}

/// Render one input slice per engine channel (missing channels are silent,
/// extra ones ignored).
pub fn render_multi(inputs: &[&[f32]], opts: &RenderOptions) -> Rendered {
    let mut engine = Engine::new(opts.config);
    render_with(&mut engine, inputs, opts)
}

/// Render through an existing engine (it is reset first), e.g. to check that
/// `reset()` restores the initial state.
pub fn render_with(engine: &mut Engine, inputs: &[&[f32]], opts: &RenderOptions) -> Rendered {
    engine.reset();
    engine.set_settings(&opts.settings);
    engine.set_telemetry_enabled(opts.telemetry);
    // Telemetry can be taken only once per engine; a reused engine keeps
    // pushing into its ring, so drain whatever is pending first.
    let mut consumer = engine.take_telemetry();

    let chans = engine.config().channels.max(1);
    let input_len = inputs.iter().map(|c| c.len()).max().unwrap_or(0);
    let latency = engine.latency_samples();
    let total = input_len + latency as usize + opts.extra_tail;
    let mut channels: Vec<Vec<f32>> = (0..chans)
        .map(|ch| {
            let mut v = vec![0.0f32; total];
            if let Some(src) = inputs.get(ch) {
                v[..src.len()].copy_from_slice(src);
            }
            v
        })
        .collect();

    let mut midi: Vec<(u64, MidiEvent)> = opts.midi.clone();
    midi.sort_by_key(|(t, _)| *t);
    let mut midi_idx = 0usize;
    let mut block_midi: Vec<TimedMidi> = Vec::with_capacity(midi.len());
    let mut telemetry = Vec::with_capacity(total / 32 + 16);
    if let Some(c) = consumer.as_mut() {
        while c.pop().is_ok() {}
    }

    let mut pos = 0usize;
    for n in opts.blocks.blocks(total) {
        block_events(&midi, &mut midi_idx, pos, n, &mut block_midi);
        let mut io: Vec<&mut [f32]> = channels.iter_mut().map(|c| &mut c[pos..pos + n]).collect();
        engine.process(&mut io, &block_midi);
        if let Some(c) = consumer.as_mut() {
            while let Ok(f) = c.pop() {
                telemetry.push(f);
            }
        }
        pos += n;
    }
    Rendered {
        channels,
        latency,
        input_len,
        telemetry,
        stats: engine.stats(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_patterns_cover_exactly() {
        for p in BlockPattern::invariance_set() {
            let b = p.blocks(10_007);
            assert_eq!(b.iter().sum::<usize>(), 10_007, "{p:?}");
            assert!(b.iter().all(|n| *n >= 1));
        }
        let b = BlockPattern::Sequence(vec![3, 5]).blocks(12);
        assert_eq!(b, vec![3, 5, 3, 1]);
        // Random patterns are reproducible.
        let r = BlockPattern::Random {
            seed: 1,
            min: 1,
            max: 100,
        };
        assert_eq!(r.blocks(5000), r.blocks(5000));
    }

    #[test]
    fn aligned_output_strips_latency() {
        let opts = RenderOptions::new(48_000.0, LatencyMode::Live, QualityTier::Hq);
        let input: Vec<f32> = (0..4800)
            .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let r = render(&input, &opts);
        assert_eq!(r.raw(0).len(), input.len() + r.latency as usize);
        assert_eq!(r.aligned(0).len(), input.len());
        assert!(r.aligned(5).is_empty());
    }

    #[test]
    fn midi_is_split_per_block() {
        let on = MidiEvent::NoteOn {
            note: 60,
            velocity: 1.0,
        };
        let off = MidiEvent::NoteOff { note: 60 };
        let midi = vec![(5, MidiEvent::AllNotesOff), (64, on), (100, off), (130, on)];
        let (mut idx, mut out) = (0, Vec::new());
        let mut got = Vec::new();
        for (pos, n) in [(0, 64), (64, 64), (128, 64)] {
            block_events(&midi, &mut idx, pos, n, &mut out);
            got.push(out.clone());
        }
        assert_eq!(
            got[0],
            vec![TimedMidi {
                timing: 5,
                event: MidiEvent::AllNotesOff
            }]
        );
        assert_eq!(
            got[1],
            vec![
                TimedMidi {
                    timing: 0,
                    event: on
                },
                TimedMidi {
                    timing: 36,
                    event: off
                }
            ]
        );
        assert_eq!(
            got[2],
            vec![TimedMidi {
                timing: 2,
                event: on
            }]
        );
        // A full render with MIDI runs through the engine API.
        let mut opts = RenderOptions::new(48_000.0, LatencyMode::Live, QualityTier::Hq);
        opts.midi = midi;
        let r = render(&vec![0.0; 1000], &opts);
        assert_eq!(r.input_len, 1000);
    }
}
