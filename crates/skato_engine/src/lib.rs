//! Realtime engine: wires detection, control and shifting together behind one
//! block-size-agnostic `process()` call (contract: `docs/DESIGN.md` §3.5,
//! timing: §2.2).
//!
//! ## Per-sample and per-hop pipeline
//!
//! ```text
//! host block ─► sanitise (NaN/Inf → 0, clamp) ─► analysis tap (L | R | (L+R)/2)
//!   split at hop boundaries (and, in Studio, at the pipeline points below);
//!   per segment (≤ one hop):
//!     detector.push(analysis)          cheap per-sample conditioning / E-H recursion
//!     shifter.process(analysis, io)    audio path, output delayed by D
//!   Live, at every hop boundary (after k·hop input samples):
//!     MIDI events of the hop just ended ─► corrector.midi()
//!     est  = detector.analyze()
//!     ctl  = corrector.process_hop(est, shifter.mark_cents())
//!     telemetry frame ─► rtrb ring
//!     before the next sample: shifter.set_engine(..) (pending tier/engine
//!     change), shifter.set_control(ctl)
//!   Studio, hop h's analysis begins at the boundary that closes it
//!   (detector.begin_analysis), its four detector stages and then its
//!   control law run at fixed points inside hop h+1 (see below), and its
//!   control is applied at the boundary that closes hop h+1.
//! ```
//!
//! Analysis never delays audio: the reported latency D is exactly the
//! shifter's latency, `skato_shift::latency_samples` (the single source of truth,
//! R§7.2.1). The control values only update at hop boundaries and the shifter
//! ramps the ratio linearly across the next hop, so the control path adds no
//! latency either (DESIGN §2.2).
//!
//! ## Alignment (Studio vs Live)
//!
//! In the Live modes the detector is causal. The corrector is also handed the
//! PSOLA mark tracker's pitch (R§7.6.2 A), which it no longer uses (skato_control
//! SN-2).
//!
//! In Studio D is split between the detector and the control law
//! ([`studio_split`], CL-3): the detector runs [`Alignment::Centered`] on
//! `newest − (D − k·hop)`, so each estimate describes the input `k` hops
//! *ahead* of the audio being output, and the corrector holds its decisions
//! back by that lead, deciding every frame with `k` hops of future estimates
//! (fixed-lag notes, onset and switch medians, R§7.6.2 F "Studio"). With the
//! detector centred on `newest − D` (DESIGN §3.5's original wording) the
//! corrector would have no lookahead and Studio decisions would be causal.
//!
//! ## Studio pipeline (RT-2, PERF-7)
//!
//! Run at the hop boundary, a hop's detector analysis lands in whichever
//! host block crosses the boundary together with the shifter's control
//! update, which at 96/192 kHz with 32-sample blocks used most of that
//! block's deadline. Studio therefore spreads hop h's work over hop h+1
//! with the detector's split analysis (`skato_detect` "Split analysis"):
//! `begin_analysis()` at the boundary that closes hop h (a snapshot of the
//! stream positions, ≈ 1 µs), then the four stages Frames, Checks, Track and
//! Output before samples `hop/4`, `3·hop/8`, `hop/2` and `3·hop/4` of hop
//! h+1, and the control law (with the MIDI of hop h) before sample
//! `7·hop/8` (`PIPELINE_EIGHTHS`). The control is applied at the boundary
//! that closes hop h+1. Host blocks smaller than a hop therefore share the
//! work out: at 192 kHz with 32-sample blocks (eight per hop) every stage
//! has a callback of its own and the shifter's update (which re-seeds the
//! mark tracker at a voiced onset) has the first; with a quarter hop per
//! block (192 kHz / 64, 96 kHz / 32) the shifter update still has the first
//! callback to itself; at 48 kHz with 32-sample blocks Frames and Checks run
//! in the first callback and Track (the heaviest Studio stage), Output and
//! the control law in the second.
//!
//! The detector is pushed every sample as it arrives: each stage reads the
//! histories at the boundary's positions, so the estimate is bit-identical
//! to a one-shot `analyze()` at the boundary (the engine test
//! `studio_split_analysis_matches_a_one_shot_analysis` checks the output).
//! The pipeline points sit at fixed sample offsets, so the output stays
//! block-size invariant; only which callback does the work depends on the
//! host's block size. The corrector is given one hop less lead, so the
//! control still lines up with the audio it was measured on; the price is
//! one hop of the Studio lookahead. Telemetry and stats are produced at the
//! boundary that applies a control.
//!
//! Live keeps the analysis at the boundary (it has no lookahead to spare).
//! In both pipelines the shifter gets its update (engine/tier change,
//! control) just before the first sample after the boundary: the same
//! sample position, so the output is unchanged. When the boundary closes a
//! host block that is the start of the next block, so the mark tracker's
//! re-seed at a voiced onset (inside `Shifter::set_control`, up to ~55 µs at
//! 192 kHz) no longer lands in the block that ran the analysis (PERF-5).
//!
//! ## Determinism
//!
//! Every stage is sample-accurate and the hop boundaries (and the Studio
//! analysis points) sit at fixed absolute sample counts, so the output depends
//! only on the input, not on how the host splits it into blocks (DESIGN §2.1
//! rule 6). MIDI events are applied with the analysis of the hop containing
//! them; engine/tier changes requested through [`Engine::set_settings`] start
//! at the next hop boundary (or at once, without a crossfade, before the first
//! sample after a reset).
//!
//! ## Realtime
//!
//! Everything is allocated in [`Engine::new`]. `process`, `set_settings`,
//! `reset` and the accessors never allocate, lock, log or panic, and every
//! loop is bounded by the block length (DESIGN §2.1). A counting-allocator test
//! binary (`tests/no_alloc.rs`) proves it.

pub use skato_control::{KeyConfig, KeyDetector, KeyName, KeyReading};
use rtrb::{Consumer, Producer, RingBuffer};
use skato_control::{ControlOutput, Corrector, CorrectorConfig};
use skato_core::{
    AnalysisChannel, EngineConfig, EngineKind, LatencyMode, MidiEvent, NoteState, PitchEstimate, PitchFrame,
    QualityTier, Settings, ShiftControl, VoiceRange,
};
use skato_detect::{ANALYSIS_STEPS, Alignment, Detector, DetectorConfig};
use skato_shift::{Shifter, ShifterConfig};

/// Telemetry ring capacity in frames (DESIGN asks for ≥ 4096). At 48 kHz with a
/// 64-sample hop that is 11 s of frames; the GUI drains it every repaint.
const TELEMETRY_CAPACITY: usize = 8192;

/// Largest input magnitude passed on (+80 dBFS). Nothing legitimate comes
/// close; garbage such as ±1e30 would overflow the f32 energy sums of the mark
/// tracker and correlators (x² = inf → NaN), so it is clamped at the boundary
/// together with the NaN/Inf sanitising (DESIGN §2.1 rule 3).
const INPUT_LIMIT: f32 = 1.0e4;

/// Studio: delay the centred detector needs before any fixed-lag smoothing,
/// in periods of the range's lowest note. Mirrors `skato_detect`'s own minimum
/// (`Alignment::Centered` places its long frame, 2.2 periods of `f_min`
/// searched over lags up to 1.06 periods, centred on the observation
/// instant: (2.2 + 1.06)/2 periods). One hop is added on top for rounding and
/// the decimators' group delay; the engine test
/// `studio_estimates_line_up_with_the_audio` checks the alignment end to end.
const DETECTOR_CENTRE_PERIODS: f64 = 1.63;

/// Studio: fixed-lag pitch-HMM smoothing left to the detector, in hops, before
/// the rest of the delay goes to the control law's lookahead (see
/// [`studio_split`]). Measured on vocadito (Studio HQ, 40 ms): from 8 hops
/// (all of the "free" split) down to 0, the hard-tune p95 error goes 22.8 →
/// 20.2 c and warble episodes 665 → 497, but the detector's voicing onset
/// delay grows 4.2 → 9.3 ms. 4 hops keeps the onset within 1 ms of the
/// unsplit detector (5.0 ms) with most of the correction gain (20.9 c, 594).
const DETECTOR_FIXED_LAG_HOPS: usize = 4;

/// Studio: most control-law lookahead worth having, ms. The lookahead rules
/// use at most 40 ms (onset median, R§7.6.2 F) and 30 ms (switch median).
const MAX_CONTROL_LOOKAHEAD_MS: f64 = 40.0;

/// Studio: MIDI events wait in the engine until the analysis of the hop that
/// contains them (pipeline, see the crate docs). If more than this many are
/// pending, the oldest is handed to the corrector early (≤ 2 hops).
const MIDI_QUEUE: usize = 1024;

/// Studio pipeline: the steps of hop h's analysis, run during hop h+1 before
/// the sample at these offsets, in eighths of a hop: the detector stages
/// (`Detector::step_analysis`: Frames, Checks, Track, Output) and then the
/// control law. The shifter update (which re-seeds the mark tracker at a
/// voiced onset, the costliest single item at 192 kHz) runs before sample 0.
/// Chosen so that host blocks of an eighth of a hop (192 kHz / 32 samples)
/// give every step its own callback, blocks of a quarter hop (192 kHz / 64,
/// 96 kHz / 32) leave the shifter update alone in the first callback, and
/// blocks of half a hop (48 kHz / 32) split Frames + Checks from Track (the
/// heaviest Studio stage: 9 µs in HQ, 18 µs in Max) + Output + the control
/// law. The price is at two blocks per hop at 192 kHz (128 samples), where
/// the first callback carries the re-seed with Frames and Checks: its worst
/// case is ~15 µs above the one-chunk analysis (10.9 % against 8.7 % of the
/// period); moving Frames later helps there but costs at 48 kHz / 32 and at
/// four blocks per hop. Measured in `docs/PERFORMANCE.md` §11.
const PIPELINE_EIGHTHS: [usize; ANALYSIS_STEPS + 1] = [2, 3, 4, 6, 7];

/// Reported latency for a config (delegates to [`skato_shift::latency_samples`]).
/// Depends only on sample rate, latency mode, voice range and the Live+/Studio
/// milliseconds, never on the tier or the runtime settings.
pub fn latency_samples(cfg: &EngineConfig) -> u32 {
    let d =
        skato_shift::latency_samples(cfg.sample_rate, cfg.latency_mode, cfg.voice_range, cfg.live_plus_ms, cfg.studio_ms);
    u32::try_from(d).unwrap_or(u32::MAX)
}

/// How Studio spends its reported latency D (CL-3, R§7.6.2 F "Studio").
///
/// Returns `(detector_delay, lead_hops)`. The detector runs
/// `Alignment::Centered { delay: detector_delay }`, i.e. each estimate
/// describes the input `lead_hops` hops *ahead* of the audio being output,
/// and the control law decides each output frame with that many hops of
/// future estimates (one fewer once the engine's Studio pipeline has taken
/// its hop, see the crate docs). Running the detector at
/// `Centered { delay: D }` leaves the control law no lookahead at all.
///
/// The detector keeps what it needs to centre its long frame on its output
/// instant plus `DETECTOR_FIXED_LAG_HOPS` of fixed-lag smoothing; the rest,
/// up to `MAX_CONTROL_LOOKAHEAD_MS`, becomes control lookahead. Low voices
/// need longer frames and get less: at the default Studio 40 ms and 48 kHz,
/// Soprano 16 hops (21 ms), Alto/Tenor 9 (12 ms), Low Male and Wide 2. When D
/// is too short the lead is 0 and the detector gets all of D.
pub fn studio_split(sample_rate: f32, hop: usize, range: VoiceRange, latency: usize) -> (usize, usize) {
    let hop = hop.max(1);
    let sr = if sample_rate.is_finite() && sample_rate > 0.0 { f64::from(sample_rate) } else { 48_000.0 };
    let centre = (DETECTOR_CENTRE_PERIODS * sr / f64::from(range.f_min())).ceil() as usize + hop;
    let need = centre + DETECTOR_FIXED_LAG_HOPS * hop;
    let max_lead = (MAX_CONTROL_LOOKAHEAD_MS * 1e-3 * sr / hop as f64).floor() as usize;
    let lead = (latency.saturating_sub(need) / hop).min(max_lead);
    (latency - lead * hop, lead)
}

/// Output fade around a voice-range change, ms (each way).
const RANGE_FADE_MS: f64 = 5.0;

/// Detector and shifter built for one voice range (HI-3).
struct RangeKit {
    detector: Detector,
    shifter: Shifter,
}

impl RangeKit {
    fn new(cfg: &EngineConfig, hop: usize, channels: usize, d: usize, alignment: Alignment) -> Self {
        let sr = cfg.sample_rate;
        Self {
            detector: Detector::new(DetectorConfig { sample_rate: sr, voice_range: cfg.voice_range, hop, alignment }),
            shifter: Shifter::new(ShifterConfig {
                sample_rate: sr,
                channels,
                hop,
                latency_mode: cfg.latency_mode,
                voice_range: cfg.voice_range,
                latency: d,
            }),
        }
    }
}

/// Detector alignment for a configuration and whether its Studio analysis
/// is pipelined (needs one hop of control lead; lead 0 only happens when D
/// barely covers the detector, e.g. Low Male below ~33 ms).
fn alignment_for(cfg: &EngineConfig, hop: usize, d: usize) -> (Alignment, bool) {
    if cfg.latency_mode == LatencyMode::Studio {
        let (delay, lead) = studio_split(cfg.sample_rate, hop, cfg.voice_range, d);
        (Alignment::Centered { delay }, lead >= 1)
    } else {
        (Alignment::Causal, false)
    }
}

/// Every voice range, in `kits` order.
const RANGES: [VoiceRange; 4] = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];

/// Index of a voice range in [`RANGES`].
fn range_index(r: VoiceRange) -> usize {
    match r {
        VoiceRange::LowMale => 0,
        VoiceRange::AltoTenor => 1,
        VoiceRange::Soprano => 2,
        VoiceRange::Wide => 3,
    }
}

/// Output gain state around a voice-range change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fade {
    Off,
    /// Fading out; samples left.
    Out(usize),
    /// Faded out, waiting for the hop boundary that switches.
    Muted,
    /// After the switch: silent for `wait` samples (the new shifter's delay
    /// line fills), then fading in over `left`.
    In { wait: usize, left: usize },
}

/// A shifter update decided at a hop boundary (Studio's first boundary
/// after a reset has an engine change at most, no control yet).
#[derive(Clone, Copy, Debug)]
struct ShifterUpdate {
    engine: Option<(EngineKind, QualityTier)>,
    control: Option<ShiftControl>,
}

/// A MIDI event with its sample offset inside the current host block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedMidi {
    pub timing: u32,
    pub event: MidiEvent,
}

/// Cumulative counters since construction or the last [`Engine::reset`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineStats {
    /// Control hops processed (one per `hop` input samples).
    pub hops: u64,
    /// Hops the control law treated as voiced (pitch being shifted).
    pub voiced_hops: u64,
    /// Telemetry frames dropped because the ring was full (GUI not draining).
    pub telemetry_dropped: u64,
}

/// The realtime pitch-correction engine (see the crate docs).
pub struct Engine {
    cfg: EngineConfig,
    /// Reported latency D in samples.
    latency: u32,
    hop: usize,
    /// Channels processed (1 or 2).
    channels: usize,
    studio: bool,
    detector: Detector,
    corrector: Corrector,
    shifter: Shifter,
    settings: Settings,
    /// The engine/tier change is applied to the shifter at the next hop boundary.
    shifter_dirty: bool,
    /// Mono analysis signal of the current segment (a segment never exceeds one hop).
    analysis: Box<[f32]>,
    /// Input samples of the current, incomplete hop.
    hop_fill: usize,
    /// Input samples processed since the last reset.
    pos: u64,
    /// Studio pipeline (RT-2, PERF-7): the analysis of each hop begins at
    /// the boundary that closes it; its steps (detector stages, then the
    /// control law) run before the samples `step_at` of the next hop, and
    /// the control is applied at the boundary after that.
    pipelined: bool,
    step_at: [usize; ANALYSIS_STEPS + 1],
    /// Index into `step_at` of the next pipeline step due (`None`: none).
    step: Option<usize>,
    /// Estimate of the analysis in flight once its last stage has run.
    pending_est: PitchEstimate,
    /// Control and telemetry frame of the latest analysis, applied at the
    /// next hop boundary (pipeline only).
    pending: Option<(ControlOutput, PitchFrame)>,
    /// The shifter update of the last hop boundary (engine/tier change,
    /// control), applied before the first sample after it (see
    /// [`Engine::process`]).
    shifter_due: Option<ShifterUpdate>,
    /// HI-3: detector and shifter for each other voice range that shares
    /// this engine's D and pipeline shape (indexed like [`RANGES`]),
    /// and the parked components of the ranges switched away from.
    kits: [Option<Box<RangeKit>>; 4],
    /// A voice-range change requested by `set_voice_range`, applied at the
    /// first hop boundary after the output has faded out.
    range_change: Option<VoiceRange>,
    /// Output gain envelope around a voice-range change.
    fade: Fade,
    fade_len: usize,
    /// Input samples processed before the last voice-range change (keeps
    /// telemetry positions monotonic across it).
    pos_base: u64,
    /// MIDI events (absolute input position) waiting for the analysis of the
    /// hop that contains them (pipeline only). Ring buffer.
    midi_q: Box<[(u64, MidiEvent)]>,
    midi_head: usize,
    midi_len: usize,
    telemetry_tx: Producer<PitchFrame>,
    telemetry_rx: Option<Consumer<PitchFrame>>,
    telemetry_enabled: bool,
    stats: EngineStats,
    /// Analyses, control-law runs and shifter updates (tests check where in
    /// the block stream they happen), and a bit per kind of work done since
    /// the test last cleared it: `1 << i` detector stage `i` (Studio
    /// pipeline), [`WORK_CONTROL`], [`WORK_SHIFTER`], [`WORK_BEGIN`],
    /// [`WORK_ANALYZE`].
    #[cfg(test)]
    analyses: u64,
    #[cfg(test)]
    controls: u64,
    #[cfg(test)]
    shifter_updates: u64,
    #[cfg(test)]
    work: u32,
    /// Test reference: run each Studio analysis in one go at the boundary
    /// (`begin_analysis` + `finish_analysis`), as the pipeline did before
    /// the split.
    #[cfg(test)]
    one_shot: bool,
}

/// Test bits of [`Engine::work`].
#[cfg(test)]
const WORK_CONTROL: u32 = 1 << 8;
#[cfg(test)]
const WORK_SHIFTER: u32 = 1 << 9;
#[cfg(test)]
const WORK_BEGIN: u32 = 1 << 10;
#[cfg(test)]
const WORK_ANALYZE: u32 = 1 << 11;

impl Engine {
    /// Builds the engine for a structural configuration. Allocates everything
    /// (detector and shifter for the Max tier, rings, sinc tables, telemetry);
    /// not for the audio thread.
    pub fn new(cfg: EngineConfig) -> Self {
        let sr = cfg.sample_rate;
        let hop = skato_core::hop_for_rate(sr).max(1);
        let latency = latency_samples(&cfg);
        let d = latency as usize;
        let channels = cfg.channels.clamp(1, 2);
        let studio = cfg.latency_mode == LatencyMode::Studio;
        let (alignment, pipelined) = alignment_for(&cfg, hop, d);
        // The corrector derives its lead as `lookahead_hops − ceil(info_lag /
        // hop)` with info_lag = delay = D − lead·hop, so ⌈D/hop⌉ gives exactly
        // `lead` at every rate (also when hop does not divide D, e.g.
        // 44.1 kHz), and one less with the pipeline, whose control is applied
        // one hop after its analysis.
        let lookahead_hops = if studio { d.div_ceil(hop) - usize::from(pipelined) } else { 0 };
        let RangeKit { detector, shifter } = RangeKit::new(&cfg, hop, channels, d, alignment);
        let corrector =
            Corrector::new(CorrectorConfig { sample_rate: sr, hop, latency_mode: cfg.latency_mode, lookahead_hops });
        // HI-3: every other voice range with the same D (and, in Studio, the
        // same pipeline shape) gets its detector and shifter now, so
        // `set_voice_range` can switch to it without allocating.
        let kits = RANGES.map(|r| {
            let rc = EngineConfig { voice_range: r, ..cfg };
            let (al, pl) = alignment_for(&rc, hop, d);
            (r != cfg.voice_range && latency_samples(&rc) == latency && pl == pipelined)
                .then(|| Box::new(RangeKit::new(&rc, hop, channels, d, al)))
        });
        let (telemetry_tx, telemetry_rx) = RingBuffer::new(TELEMETRY_CAPACITY);
        let mut engine = Self {
            cfg,
            latency,
            hop,
            channels,
            studio,
            detector,
            corrector,
            shifter,
            settings: Settings::default(),
            shifter_dirty: false,
            analysis: vec![0.0; hop].into_boxed_slice(),
            hop_fill: 0,
            pos: 0,
            pipelined,
            // Inside the hop (the control law before its last sample) and
            // non-decreasing, whatever the hop.
            step_at: PIPELINE_EIGHTHS.map(|e| (e * hop / 8).clamp(1, hop.saturating_sub(1).max(1))),
            step: None,
            pending_est: PitchEstimate::default(),
            pending: None,
            shifter_due: None,
            kits,
            range_change: None,
            fade: Fade::Off,
            fade_len: ((RANGE_FADE_MS * 1e-3 * f64::from(sr)).round() as usize).max(hop),
            pos_base: 0,
            midi_q: vec![(0, MidiEvent::AllNotesOff); if pipelined { MIDI_QUEUE } else { 0 }].into_boxed_slice(),
            midi_head: 0,
            midi_len: 0,
            telemetry_tx,
            telemetry_rx: Some(telemetry_rx),
            telemetry_enabled: false,
            stats: EngineStats::default(),
            #[cfg(test)]
            analyses: 0,
            #[cfg(test)]
            controls: 0,
            #[cfg(test)]
            shifter_updates: 0,
            #[cfg(test)]
            work: 0,
            #[cfg(test)]
            one_shot: false,
        };
        let defaults = engine.settings;
        engine.apply_settings(&defaults);
        engine
    }

    /// The configuration this engine was built with.
    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// Reported latency D in samples (constant for the engine's lifetime).
    pub fn latency_samples(&self) -> u32 {
        self.latency
    }

    /// Clears all signal state (histories, trackers, notes, grains, MIDI) and
    /// restarts the sample clock at 0. Settings, the selected engine/tier and
    /// the telemetry flag are kept.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.corrector.reset();
        self.shifter.reset();
        self.hop_fill = 0;
        self.pos = 0;
        self.step = None;
        self.pending = None;
        // An engine change decided at the last boundary but not yet handed
        // to the shifter is still pending.
        if self.shifter_due.take().is_some_and(|u| u.engine.is_some()) {
            self.shifter_dirty = true;
        }
        self.midi_head = 0;
        self.midi_len = 0;
        self.stats = EngineStats::default();
        // A pending voice-range change takes effect now (no fades needed).
        if let Some(r) = self.range_change.take() {
            self.switch_range(r);
        }
        self.fade = Fade::Off;
        self.pos_base = 0;
        // A pending engine change starts cleanly from the reset state.
        self.sync_shifter_now();
    }

    /// Applies the runtime settings (call once per host block; RT-safe and
    /// cheap when nothing changed). Detector and control-law settings take
    /// effect at the next hop (a tier change that lands while a Studio
    /// analysis is in flight lets that analysis finish with the tier it
    /// began with); an engine or tier change starts its shifter crossfade
    /// at the next hop boundary, or immediately without a fade if no sample
    /// has been processed since construction/reset.
    pub fn set_settings(&mut self, s: &Settings) {
        self.apply_settings(s);
    }

    /// Switches the voice range at runtime when that keeps the reported
    /// latency (HI-3). RT-safe: the detector and shifter of every range that
    /// shares this engine's D (and, in Studio, its pipeline shape) were built
    /// in [`Engine::new`], so nothing is allocated. Returns `false` (and
    /// changes nothing) when the range would change D, i.e. when the host has
    /// to rebuild the engine with the new [`EngineConfig`]; `true` otherwise,
    /// also when the range is already active.
    ///
    /// The switch is applied in `process()`: the output fades out over
    /// 5 ms, the new range's detector and shifter take over (reset) at the
    /// next hop boundary, and the output fades back in once their delay line
    /// has refilled (D samples later), so a change during playback costs a
    /// dropout of about D + 10 ms instead of a host restart. Settings, the
    /// control law (notes, MIDI, learned scale) and the telemetry stream are
    /// kept; [`Engine::config`] reports the new range from the switch on.
    pub fn set_voice_range(&mut self, range: VoiceRange) -> bool {
        if range == self.cfg.voice_range {
            if self.range_change.take().is_some() {
                // Cancelled while fading out: fade back in from the gain
                // reached so far.
                self.fade = match self.fade {
                    Fade::Out(k) => Fade::In { wait: 0, left: self.fade_len.saturating_sub(k) },
                    _ => Fade::In { wait: 0, left: self.fade_len },
                };
            }
            return true;
        }
        if self.kits.get(range_index(range)).is_none_or(Option::is_none) {
            return false;
        }
        self.range_change = Some(range);
        if self.pos == 0 {
            // Nothing processed since construction or reset: switch at once.
            self.switch_range(range);
            self.range_change = None;
            self.fade = Fade::Off;
        } else {
            // Fade out from the gain the output has now.
            self.fade = match self.fade {
                Fade::Off => Fade::Out(self.fade_len),
                Fade::In { wait: 0, left } => Fade::Out(self.fade_len.saturating_sub(left)),
                Fade::In { .. } => Fade::Muted,
                f @ (Fade::Out(_) | Fade::Muted) => f,
            };
        }
        true
    }

    /// Process a host block in place. Any length (internally split at hop
    /// boundaries). MIDI events must be sorted by `timing` (sample offset within
    /// this block); each is applied with the analysis of the hop containing
    /// it. Channels beyond the configured count are left untouched; only the
    /// common length of the processed channels is processed.
    pub fn process(&mut self, io: &mut [&mut [f32]], midi: &[TimedMidi]) {
        #[cfg(target_arch = "x86_64")]
        let _ftz = denormals::FlushGuard::new();

        let nch = io.len().min(self.channels);
        let n = io.iter().take(nch).map(|c| c.len()).min().unwrap_or(0);
        let mut next_midi = 0usize;
        if self.pipelined {
            // Events wait for their hop's (deferred) analysis.
            let start = self.pos;
            for ev in midi {
                self.enqueue_midi(start.saturating_add(u64::from(ev.timing)), ev.event);
            }
            next_midi = midi.len();
        }
        let mut off = 0usize;
        while off < n {
            // The last boundary's shifter update lands before the first
            // sample after it, i.e. at the same sample position as when it
            // was applied at the boundary, but in the next host block when
            // the boundary closed the previous one: the mark tracker's
            // re-seed at a voiced onset (`Shifter::set_control`) then no
            // longer shares a callback with the hop's analysis (PERF-5).
            if let Some(u) = self.shifter_due.take() {
                if let Some((kind, tier)) = u.engine {
                    self.shifter.set_engine(kind, tier);
                }
                if let Some(control) = u.control {
                    self.shifter.set_control(&control);
                }
                #[cfg(test)]
                {
                    self.shifter_updates += 1;
                    self.work |= WORK_SHIFTER;
                }
            }
            // Studio pipeline: the steps due before this sample.
            let mut stop = self.hop;
            while let Some(i) = self.step {
                match self.step_at.get(i) {
                    Some(&at) if at > self.hop_fill => {
                        stop = at;
                        break;
                    }
                    _ => self.pipeline_step(i),
                }
            }
            let len = (n - off).min(stop - self.hop_fill);
            let end = off + len;
            self.process_segment(io, nch, off, end);
            if self.fade != Fade::Off {
                self.apply_fade(io, nch, off, end);
            }
            off = end;
            self.hop_fill += len;
            self.pos += len as u64;
            if self.hop_fill >= self.hop {
                self.hop_fill = 0;
                if self.fade == Fade::Muted
                    && let Some(r) = self.range_change.take()
                {
                    self.switch_range(r);
                    self.fade = Fade::In { wait: self.latency as usize, left: self.fade_len };
                }
                if self.pipelined {
                    self.pipeline_boundary();
                } else {
                    next_midi = self.deliver_midi(midi, next_midi, off);
                    self.hop_boundary();
                }
            }
        }
        // Live: events in the hop still open (and any with out-of-block
        // timings) are queued in the corrector now; it applies them at the
        // next boundary. (Studio has queued every event above.)
        self.deliver_midi(midi, next_midi, usize::MAX);
    }

    /// The GUI side of the telemetry ring (capacity 8192 frames). Returns
    /// `Some` once.
    pub fn take_telemetry(&mut self) -> Option<Consumer<PitchFrame>> {
        self.telemetry_rx.take()
    }

    /// Telemetry is only produced while enabled (editor open). A plain flag set
    /// by the plugin each block.
    pub fn set_telemetry_enabled(&mut self, on: bool) {
        self.telemetry_enabled = on;
    }

    /// Counters since construction or the last reset.
    pub fn stats(&self) -> EngineStats {
        self.stats
    }

    /// Effective scale after MIDI Learn (forwarded from the corrector): `Some`
    /// only in `MidiMode::Learn` after the first learned note.
    pub fn learned_states(&self) -> Option<[NoteState; 12]> {
        self.corrector.learned_states()
    }

    /// What the Labs learned note hold did since the last reset (forwarded
    /// from the corrector): switches the model judged and held back, and
    /// release freezes.
    pub fn note_hold_stats(&self) -> skato_control::NoteHoldStats {
        self.corrector.note_hold_stats()
    }

    /// Auto-Key (forwarded from the corrector): the proposed key, its
    /// confidence and the confirmed key the Detect key source follows.
    pub fn key_reading(&self) -> skato_control::KeyReading {
        self.corrector.key_reading()
    }

    /// The Auto-Key detector's state, to carry into a rebuilt engine
    /// ([`Self::set_key_detector`]). It survives [`Self::reset`].
    pub fn key_detector(&self) -> &skato_control::KeyDetector {
        self.corrector.key_detector()
    }

    /// Continue Auto-Key from another engine's detector (same control hop,
    /// i.e. the same sample rate). Copies fixed-size state; no allocation.
    pub fn set_key_detector(&mut self, detector: &skato_control::KeyDetector) {
        self.corrector.set_key_detector(detector);
    }

    /// Restore an Auto-Key confirmed key saved with the project.
    pub fn seed_key(&mut self, key: skato_control::KeyName) {
        self.corrector.seed_key(key);
    }

    /// Forget everything Auto-Key has heard.
    pub fn clear_key(&mut self) {
        self.corrector.clear_key();
    }

    /// Evaluation: tune the Auto-Key detector.
    #[doc(hidden)]
    pub fn set_key_config(&mut self, cfg: skato_control::KeyConfig) {
        self.corrector.set_key_config(cfg);
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn apply_settings(&mut self, s: &Settings) {
        self.corrector.set_settings(s);
        if s.tier != self.settings.tier && self.detector.analysis_steps_left() > 0 {
            // Studio pipeline: the analysis in flight finishes with the tier
            // it began with (a tier change re-lays out the pitch HMM); its
            // remaining stages become no-ops.
            self.pending_est = self.detector.finish_analysis();
            self.count_analysis();
        }
        self.detector.set_tier(s.tier);
        self.detector.set_tracking(s.tracking, s.gate_db);
        if s.engine != self.settings.engine || s.tier != self.settings.tier {
            self.shifter_dirty = true;
        }
        self.settings = *s;
        if self.pos == 0 {
            self.sync_shifter_now();
        }
    }

    /// Selects the configured engine/tier without a crossfade. Only valid while
    /// the shifter holds no signal (fresh or just reset).
    fn sync_shifter_now(&mut self) {
        if self.shifter_dirty {
            self.shifter.set_engine(self.settings.engine, self.settings.tier);
            // The shifter is empty, so resetting it only cancels the fade.
            self.shifter.reset();
            self.shifter_dirty = false;
        }
    }

    /// Sanitises input samples `off..end` of every processed channel in place,
    /// builds the analysis tap, and runs the detector and shifter on them.
    fn process_segment(&mut self, io: &mut [&mut [f32]], nch: usize, off: usize, end: usize) {
        let len = end - off;
        for ch in io.iter_mut().take(nch) {
            if let Some(xs) = ch.get_mut(off..end) {
                for x in xs {
                    *x = sanitize(*x);
                }
            }
        }
        let Some(analysis) = self.analysis.get_mut(..len) else { return };
        let left = io.first().and_then(|c| c.get(off..end));
        let right = if nch > 1 { io.get(1).and_then(|c| c.get(off..end)) } else { None };
        match (self.settings.analysis_channel, left, right) {
            (AnalysisChannel::Mix, Some(l), Some(r)) => {
                for ((a, &x), &y) in analysis.iter_mut().zip(l).zip(r) {
                    *a = 0.5 * (x + y);
                }
            }
            (AnalysisChannel::Right, _, Some(r)) => analysis.copy_from_slice(r),
            // Left, or a mono configuration (any choice analyses the only channel).
            (_, Some(l), _) => analysis.copy_from_slice(l),
            _ => analysis.fill(0.0),
        }
        // In the Studio pipeline the previous hop's analysis may still be in
        // flight: its stages read the histories at the boundary's positions
        // (one hop of slack), so these samples can go in now.
        self.detector.push(analysis);
        match io {
            [l] => {
                if let Some(l) = l.get_mut(off..end) {
                    self.shifter.process(analysis, &mut [l]);
                }
            }
            [l, r, ..] if nch > 1 => {
                if let (Some(l), Some(r)) = (l.get_mut(off..end), r.get_mut(off..end)) {
                    self.shifter.process(analysis, &mut [l, r]);
                }
            }
            [l, ..] => {
                if let Some(l) = l.get_mut(off..end) {
                    self.shifter.process(analysis, &mut [l]);
                }
            }
            [] => {}
        }
    }

    /// Hands the corrector every event (from `from`) with `timing < until`;
    /// returns the index of the first event not delivered.
    fn deliver_midi(&mut self, midi: &[TimedMidi], from: usize, until: usize) -> usize {
        let mut i = from;
        while let Some(ev) = midi.get(i) {
            if ev.timing as usize >= until {
                break;
            }
            self.corrector.midi(ev.event);
            i += 1;
        }
        i
    }

    /// Studio pipeline: queue an event at absolute input position `at`.
    fn enqueue_midi(&mut self, at: u64, event: MidiEvent) {
        let cap = self.midi_q.len();
        if cap == 0 {
            self.corrector.midi(event);
            return;
        }
        if self.midi_len == cap {
            // Full: hand the oldest event over early rather than drop it.
            if let Some(&(_, old)) = self.midi_q.get(self.midi_head) {
                self.corrector.midi(old);
            }
            self.midi_head = (self.midi_head + 1) % cap;
            self.midi_len -= 1;
        }
        if let Some(slot) = self.midi_q.get_mut((self.midi_head + self.midi_len) % cap) {
            *slot = (at, event);
            self.midi_len += 1;
        }
    }

    /// Studio pipeline: hand the corrector every queued event before `until`.
    fn deliver_queued_midi(&mut self, until: u64) {
        let cap = self.midi_q.len();
        while self.midi_len > 0 {
            let Some(&(at, ev)) = self.midi_q.get(self.midi_head) else { break };
            if at >= until {
                break;
            }
            self.corrector.midi(ev);
            self.midi_head = (self.midi_head + 1) % cap;
            self.midi_len -= 1;
        }
    }

    /// The per-hop pipeline of DESIGN §2.2 (Live, and Studio without lead).
    fn hop_boundary(&mut self) {
        let est = self.detector.analyze();
        self.count_analysis();
        #[cfg(test)]
        {
            self.work |= WORK_ANALYZE;
        }
        // R§7.6.2 A: the mark tracker's pitch is a Live-mode source; Studio's
        // centred detector is already aligned with the output.
        let mark = if self.studio { None } else { self.shifter.mark_cents() };
        let out = self.corrector.process_hop(&est, mark);
        let frame = self.telemetry_frame(&est, &out);
        // The shifter update is deferred to the next sample (see `process`);
        // stats and telemetry belong to this boundary.
        self.defer_shifter_update(Some(out.shift));
        self.record(&out, frame);
    }

    /// Hands the shifter update decided at this boundary (a control and/or
    /// a pending engine/tier change) to the next sample (see `process`).
    fn defer_shifter_update(&mut self, control: Option<ShiftControl>) {
        let engine = self.shifter_dirty.then_some((self.settings.engine, self.settings.tier));
        self.shifter_dirty = false;
        if control.is_some() || engine.is_some() {
            self.shifter_due = Some(ShifterUpdate { engine, control });
        }
    }

    /// Studio pipeline, at the boundary that closes a hop: the control of
    /// the previous hop's analysis is recorded now and reaches the shifter
    /// with the next sample (nothing yet at the first boundary after a
    /// reset), and the analysis of the hop just closed begins; its steps run
    /// during the next hop (see `process`).
    fn pipeline_boundary(&mut self) {
        let control = match self.pending.take() {
            Some((out, frame)) => {
                self.record(&out, frame);
                Some(out.shift)
            }
            None => {
                self.stats.hops += 1;
                None
            }
        };
        self.defer_shifter_update(control);
        // Every step runs before `hop` (`step_at` < hop), so the previous
        // analysis is complete here.
        self.detector.begin_analysis();
        #[cfg(test)]
        {
            self.work |= WORK_BEGIN;
            if self.one_shot {
                self.pending_est = self.detector.finish_analysis();
                self.count_analysis();
            }
        }
        self.step = Some(0);
    }

    /// Studio pipeline step `i` of the analysis in flight: a detector stage,
    /// or the control law after the last one.
    fn pipeline_step(&mut self, i: usize) {
        if i < ANALYSIS_STEPS {
            // A no-op once the analysis has finished early (tier change).
            if let Some(est) = self.detector.step_analysis() {
                self.pending_est = est;
                self.count_analysis();
            }
            #[cfg(test)]
            {
                self.work |= 1 << i;
            }
            self.step = Some(i + 1);
        } else {
            self.pipeline_control();
            self.step = None;
        }
    }

    /// Studio pipeline: the control law for the finished analysis. Its MIDI
    /// is every event before the boundary that closed the analysed hop.
    fn pipeline_control(&mut self) {
        let boundary = self.pos.saturating_sub(self.hop_fill as u64);
        self.deliver_queued_midi(boundary);
        let est = self.pending_est;
        let out = self.corrector.process_hop(&est, None);
        #[cfg(test)]
        {
            self.controls += 1;
            self.work |= WORK_CONTROL;
        }
        let frame = self.telemetry_frame(&est, &out);
        self.pending = Some((out, frame));
    }

    /// Test counter of finished detector analyses (no-op otherwise).
    #[inline]
    fn count_analysis(&mut self) {
        #[cfg(test)]
        {
            self.analyses += 1;
        }
    }

    /// Stats and telemetry for a control applied at this hop boundary.
    fn record(&mut self, out: &ControlOutput, frame: PitchFrame) {
        self.stats.hops += 1;
        if out.shift.voiced {
            self.stats.voiced_hops += 1;
        }
        if self.telemetry_enabled && self.telemetry_tx.push(frame).is_err() {
            self.stats.telemetry_dropped += 1;
        }
    }

    /// HI-3: swaps in the detector and shifter built for `range` (the old
    /// ones are parked in its slot), resets them and applies the current
    /// settings. Signal state of the old range's analysis (pending Studio
    /// control, deferred shifter update) is dropped; the control law keeps
    /// running and the sample clock goes on (telemetry stays monotonic).
    fn switch_range(&mut self, range: VoiceRange) {
        let old = range_index(self.cfg.voice_range);
        let Some(mut kit) = self.kits.get_mut(range_index(range)).and_then(Option::take) else { return };
        core::mem::swap(&mut self.detector, &mut kit.detector);
        core::mem::swap(&mut self.shifter, &mut kit.shifter);
        if let Some(slot) = self.kits.get_mut(old) {
            *slot = Some(kit);
        }
        self.cfg.voice_range = range;
        self.detector.reset();
        self.detector.set_tier(self.settings.tier);
        self.detector.set_tracking(self.settings.tracking, self.settings.gate_db);
        self.shifter.reset();
        self.shifter.set_engine(self.settings.engine, self.settings.tier);
        // The shifter is empty, so resetting it only cancels the fade.
        self.shifter.reset();
        self.shifter_dirty = false;
        self.shifter_due = None;
        self.pending = None;
        self.step = None;
        self.pos_base = self.pos;
    }

    /// Output gain around a voice-range change, on samples `off..end`.
    fn apply_fade(&mut self, io: &mut [&mut [f32]], nch: usize, off: usize, end: usize) {
        let len = self.fade_len.max(1) as f32;
        let mut fade = self.fade;
        for i in off..end {
            let g = match fade {
                Fade::Off => 1.0,
                Fade::Out(left) => {
                    fade = if left <= 1 { Fade::Muted } else { Fade::Out(left - 1) };
                    raised_cosine(left as f32 / len)
                }
                Fade::Muted => 0.0,
                Fade::In { wait, left } => {
                    if wait > 0 {
                        fade = Fade::In { wait: wait - 1, left };
                        0.0
                    } else {
                        fade = if left <= 1 { Fade::Off } else { Fade::In { wait: 0, left: left - 1 } };
                        raised_cosine(1.0 - left as f32 / len)
                    }
                }
            };
            for ch in io.iter_mut().take(nch) {
                if let Some(x) = ch.get_mut(i) {
                    *x *= g;
                }
            }
        }
        self.fade = fade;
    }

    /// Fills in the output-time position of the corrector's frame. The frame
    /// describes the estimate passed `lookahead_lead()` hops ago, whose
    /// content is heard D later; in the Studio pipeline that is exactly the
    /// boundary at which the control is applied.
    fn telemetry_frame(&self, est: &PitchEstimate, out: &ControlOutput) -> PitchFrame {
        let lead = (self.corrector.lookahead_lead() as u64).saturating_mul(self.hop as u64);
        let sample_pos =
            est.sample_pos.saturating_sub(lead).saturating_add(u64::from(self.latency)).saturating_add(self.pos_base);
        PitchFrame { sample_pos, ..out.frame }
    }
}

/// Equal-gain raised-cosine fade weight for `t` in [0, 1].
#[inline]
fn raised_cosine(t: f32) -> f32 {
    0.5 - 0.5 * (core::f32::consts::PI * t.clamp(0.0, 1.0)).cos()
}

/// NaN/±Inf → silence; clamp absurd magnitudes (see [`INPUT_LIMIT`]).
#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-INPUT_LIMIT, INPUT_LIMIT) } else { 0.0 }
}

#[cfg(target_arch = "x86_64")]
mod denormals {
    //! Flush-to-zero / denormals-are-zero for the duration of `process()` on
    //! x86 (R§9.4 rule 5). The IIR filters in the detector decay into
    //! subnormals after long silence, which is very slow on x86 SSE; ARM
    //! flushes in hardware-friendly time and needs nothing.

    /// MXCSR bits: FTZ (bit 15) and DAZ (bit 6).
    const FTZ_DAZ: u32 = 0x8040;

    /// Sets FTZ/DAZ on creation and restores the caller's MXCSR on drop, so
    /// the host thread's floating-point state is left as it was.
    pub(crate) struct FlushGuard {
        saved: u32,
    }

    impl FlushGuard {
        #[inline]
        pub(crate) fn new() -> Self {
            let mut saved: u32 = 0;
            // SAFETY: `stmxcsr` stores the 32-bit MXCSR into the valid,
            // aligned `saved`; `ldmxcsr` loads a value that only adds the
            // FTZ/DAZ mode bits to the current one (no reserved bits set).
            unsafe {
                core::arch::asm!("stmxcsr dword ptr [{}]", in(reg) &mut saved, options(nostack, preserves_flags));
                let flushed = saved | FTZ_DAZ;
                core::arch::asm!("ldmxcsr dword ptr [{}]", in(reg) &flushed, options(nostack, preserves_flags, readonly));
            }
            Self { saved }
        }
    }

    impl Drop for FlushGuard {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: restores the MXCSR value read in `new`.
            unsafe {
                core::arch::asm!("ldmxcsr dword ptr [{}]", in(reg) &self.saved, options(nostack, preserves_flags, readonly));
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn mxcsr() -> u32 {
            let mut v: u32 = 0;
            // SAFETY: stores MXCSR into a valid, aligned u32.
            unsafe { core::arch::asm!("stmxcsr dword ptr [{}]", in(reg) &mut v, options(nostack, preserves_flags)) };
            v
        }

        #[test]
        fn guard_sets_ftz_daz_and_restores_the_caller_state() {
            let before = mxcsr();
            {
                let _g = FlushGuard::new();
                assert_eq!(mxcsr() & FTZ_DAZ, FTZ_DAZ);
                // A subnormal result is flushed to zero.
                let tiny = std::hint::black_box(f32::MIN_POSITIVE);
                assert_eq!(std::hint::black_box(tiny * 0.5), 0.0);
            }
            assert_eq!(mxcsr(), before);
        }
    }
}

#[cfg(test)]
mod tests;
