//! Real-singing evaluation on the vocadito dataset (CC BY 4.0; 40 solo
//! a-cappella excerpts, 29 singers, 44.1 kHz, F0 + two note annotations).
//!
//! Per file and per latency mode × quality tier ([`RunConfig`]) it runs:
//!
//! - **Detector** ([`Condition::Detect`]): the engine at amount 0 (so the
//!   detector sees the raw take and the output is a pass-through); the
//!   telemetry `detected` pitch, mapped to input time (`sample_pos − D`), is
//!   scored against the F0 annotation (RPA/RCA, fine error, octave errors,
//!   voicing P/R/F1, onset delay, timestamp offset). The offline reference
//!   tracker is scored the same way once per file as the offline-ideal
//!   baseline. The amount-0 output also gives the measured audio delay.
//! - **Correction** (hard-tune and Natural presets, chromatic and in the key
//!   estimated from annotator 1's notes): the output is measured with the
//!   reference tracker; in-tune distance to the nearest legal note on stable
//!   frames (input baseline alongside), note-level agreement with the
//!   annotation, onset settling, clicks, formant LSD, NaN/blow-ups and
//!   failure spans.
//!
//! [`report`] writes the CSVs, `report.md`, the failure segments and the
//! listening WAVs; [`gates`] holds the real-singing gates (EV-4). The
//! `skato_eval vocadito` subcommand drives [`run`].

pub mod data;
pub mod gates;
pub mod measure;
pub mod report;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use skato_core::{
    EngineConfig, EngineKind, LatencyMode, QualityTier, ScaleKind, Settings, VoiceRange,
};

use crate::metrics::{self, LatencyMeasurement};
use crate::reference::{PitchTrack, ReferenceConfig, ReferenceTracker};
use crate::render::{self, BlockPattern, RenderOptions, default_engine};
use crate::suite::{engine_name, mode_name, tier_name};
use crate::synth::{NoteKind, NoteSpan, Signal};
use crate::{scenarios, wav};

use data::{KeyEstimate, Track, TrackMeta};
use measure::{CorrCounts, DetCounts, InputSide, RawSegment, TimedTrack};

/// One engine configuration: latency mode × tier × engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunConfig {
    pub mode: LatencyMode,
    pub tier: QualityTier,
    pub engine: EngineKind,
}

impl RunConfig {
    pub fn new(mode: LatencyMode, tier: QualityTier) -> Self {
        Self {
            mode,
            tier,
            engine: default_engine(mode),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{}_{}_{}",
            mode_name(self.mode),
            tier_name(self.tier),
            engine_name(self.engine)
        )
    }
}

/// Live and Studio × Eco/HQ/Max (Natural engine) plus Live-Min HQ (Classic).
pub fn default_configs() -> Vec<RunConfig> {
    let mut v = Vec::new();
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            v.push(RunConfig::new(mode, tier));
        }
    }
    v.push(RunConfig::new(LatencyMode::LiveMin, QualityTier::Hq));
    v
}

/// What is rendered for each file × config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Condition {
    /// Amount 0: detector telemetry vs annotation, pass-through output.
    Detect,
    HardChromatic,
    NaturalChromatic,
    HardKey,
    NaturalKey,
}

impl Condition {
    pub const ALL: [Condition; 5] = [
        Condition::Detect,
        Condition::HardChromatic,
        Condition::NaturalChromatic,
        Condition::HardKey,
        Condition::NaturalKey,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Condition::Detect => "detect",
            Condition::HardChromatic => "hard_chromatic",
            Condition::NaturalChromatic => "natural_chromatic",
            Condition::HardKey => "hard_key",
            Condition::NaturalKey => "natural_key",
        }
    }

    /// Label of the correction measures of this condition (`passthrough` for
    /// the amount-0 render).
    pub fn corr_label(self) -> &'static str {
        match self {
            Condition::Detect => "passthrough",
            c => c.label(),
        }
    }

    pub fn parse(s: &str) -> Option<Condition> {
        Condition::ALL.into_iter().find(|c| c.label() == s)
    }

    fn uses_key(self) -> bool {
        matches!(self, Condition::HardKey | Condition::NaturalKey)
    }
}

/// Engine settings for a condition (None when it needs a key and the file
/// has none). Presets follow R§7.6.4 as the plugin applies them
/// (`skatotune::params::Preset`): Hard Tune = τ 0, depth 0 %, tolerance
/// 5 c / 0 ms; Natural = τ 25 ms, Humanize 30, Flex 20, tolerance 15 c / 30 ms.
/// Live-Min keeps the Classic engine.
pub fn condition_settings(
    cond: Condition,
    cfg: RunConfig,
    key: Option<&KeyEstimate>,
) -> Option<Settings> {
    let (k, scale) = if cond.uses_key() {
        let k = key?;
        (k.key, k.scale)
    } else {
        (0, ScaleKind::Chromatic)
    };
    let mut s = scenarios::settings_for(k, scale);
    s.tier = cfg.tier;
    s.engine = cfg.engine;
    s.preserve_vibrato = true;
    s.ignore_vibrato = true;
    match cond {
        Condition::Detect => s.amount = 0.0,
        Condition::HardChromatic | Condition::HardKey => {
            s.retune_ms = 0.0;
            s.transition_ms = None;
            s.humanize = 0.0;
            s.flex_tune = 0.0;
            s.vibrato_depth = 0.0;
            s.tolerance_cents = 5.0;
            s.tolerance_ms = 0.0;
        }
        Condition::NaturalChromatic | Condition::NaturalKey => {
            s.retune_ms = 25.0;
            s.transition_ms = None;
            s.humanize = 30.0;
            s.flex_tune = 20.0;
            s.vibrato_depth = 1.0;
            s.tolerance_cents = 15.0;
            s.tolerance_ms = 30.0;
        }
    }
    Some(s)
}

/// Legal pitch classes of a settings value (On notes).
pub fn legal_classes(s: &Settings) -> [bool; 12] {
    scenarios::legal_classes(s)
}

#[derive(Clone, Debug)]
pub struct VocaditoOptions {
    pub data: PathBuf,
    pub out: PathBuf,
    pub configs: Vec<RunConfig>,
    pub conditions: Vec<Condition>,
    /// Track ids (empty = all found).
    pub files: Vec<u32>,
    /// Force one voice range for every file (None = per file from the
    /// annotation, see [`data::choose_range`]).
    pub range: Option<VoiceRange>,
    pub blocks: BlockPattern,
    /// Worker threads (0 = available parallelism).
    pub threads: usize,
    /// Tracks to write listening WAVs for.
    pub listen: Vec<u32>,
    pub wavs: bool,
    /// Tracks to write per-frame dumps for (`<out>/dump/*.csv`, 1 ms rows).
    pub dump: Vec<u32>,
}

impl Default for VocaditoOptions {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/vocadito"),
            out: PathBuf::from("out/vocadito"),
            configs: default_configs(),
            conditions: Condition::ALL.to_vec(),
            files: Vec::new(),
            range: None,
            blocks: BlockPattern::Fixed(256),
            threads: 0,
            listen: vec![26, 1, 9, 2, 34, 21],
            wavs: true,
            dump: Vec::new(),
        }
    }
}

/// A failure span with context for the failure analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub file: u32,
    pub singer: String,
    /// Config label, or "reference" for the offline tracker baseline.
    pub config: String,
    pub condition: &'static str,
    pub raw: RawSegment,
    /// Median annotated f0 around the span, Hz (NaN if none).
    pub ann_f0: f64,
    /// Median input level (reference-tracker window), dBFS.
    pub level_db: f64,
    /// Median input periodicity (reference-tracker clarity).
    pub clarity: f64,
    pub tags: Vec<&'static str>,
}

/// Per-file facts and the offline-reference baseline.
#[derive(Clone, Debug)]
pub struct FileSummary {
    pub id: u32,
    pub meta: TrackMeta,
    pub sample_rate: f32,
    pub dur_s: f64,
    pub range: VoiceRange,
    pub min_f0: f64,
    pub max_f0: f64,
    pub voiced_fraction: f64,
    pub key: Option<KeyEstimate>,
    pub notes_a1: usize,
    pub notes_a2: usize,
    /// Median |A1 − A2| note pitch, cents, and fraction of A1 notes matched.
    pub annot_diff_c: f64,
    pub annot_matched: f64,
    pub reference: DetCounts,
    pub reference_lag_ms: f64,
    /// Stable frames / moving frames on the input side.
    pub stable_frames: u64,
    pub moving_frames: u64,
}

/// Detector result of one file × config.
#[derive(Clone, Debug)]
pub struct DetRow {
    pub file: u32,
    pub config: RunConfig,
    pub range: VoiceRange,
    pub latency: u32,
    pub counts: DetCounts,
    /// Timestamp offset (ms; positive = telemetry late) and RPA50 there.
    pub best_lag_ms: f64,
    pub rpa50_at_best: f64,
    /// Output-vs-input cross-correlation of the amount-0 render.
    pub measured: Option<LatencyMeasurement>,
    pub transparency_snr_db: f64,
    /// Median age of the pitch information when its correction is applied:
    /// hop-boundary time − telemetry output time, ms (≤ 0 = in time).
    pub info_age_ms: f64,
    /// Content delay of voiced audio in the amount-0 render (D + PSOLA
    /// wander, [`metrics::content_delay`]) on annotator-1 notes.
    pub content: metrics::ContentDelayStats,
    pub finite: bool,
    pub telemetry_ok: bool,
    pub render_s: f64,
}

/// Correction result of one file × config × condition.
#[derive(Clone, Debug)]
pub struct CorrRow {
    pub file: u32,
    pub config: RunConfig,
    pub condition: Condition,
    pub range: VoiceRange,
    pub latency: u32,
    pub counts: CorrCounts,
    pub lsd_median: f64,
    pub finite: bool,
    pub render_s: f64,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub files: Vec<FileSummary>,
    pub det: Vec<DetRow>,
    pub corr: Vec<CorrRow>,
    pub segments: Vec<Segment>,
    pub elapsed_s: f64,
    pub threads: usize,
}

/// Everything precomputed for a file (shared read-only by its jobs).
struct FileCtx {
    track: Track,
    /// Input audio + annotator-1 notes (for the formant LSD).
    sig: Signal,
    range: VoiceRange,
    key: Option<KeyEstimate>,
    ann: TimedTrack,
    /// Annotated voicing onsets (time, gap before) and offsets.
    onsets: Vec<(f64, f64)>,
    offsets: Vec<f64>,
    /// All annotated voicing transitions, sorted.
    transitions: Vec<f64>,
    in_track: PitchTrack,
    input: InputSide,
    /// Median input level over annotated-voiced frames, dBFS.
    voiced_level_db: f64,
}

fn reference_config() -> ReferenceConfig {
    ReferenceConfig::default()
}

fn build_ctx(track: Track, range_override: Option<VoiceRange>) -> FileCtx {
    let sr = track.sample_rate;
    let range = range_override.unwrap_or_else(|| data::choose_range(track.min_f0()));
    let key = data::estimate_key(&track.notes_a1);
    let ann = TimedTrack::from_annotation(&track.f0);
    let (onsets, offsets) = ann.transitions();
    let mut transitions: Vec<f64> = onsets
        .iter()
        .map(|o| o.0)
        .chain(offsets.iter().copied())
        .collect();
    transitions.sort_by(f64::total_cmp);
    let in_track = ReferenceTracker::new(sr, reference_config()).track(&track.audio);
    let input = InputSide::new(&in_track, &ann, &track.notes_a1, &onsets);
    let levels: Vec<f64> = in_track
        .frames
        .iter()
        .enumerate()
        .filter(|(k, _)| input.ann_v.get(*k).copied().unwrap_or(false))
        .map(|(_, f)| f.level_dbfs)
        .filter(|v| v.is_finite())
        .collect();
    let voiced_level_db = crate::dsp::median(&levels);
    let mut sig = Signal::from_audio(Vec::new(), sr);
    sig.notes = track
        .notes_a1
        .iter()
        .map(|n| NoteSpan {
            start: (n.start_s * sr as f64).round().max(0.0) as usize,
            end: (n.end_s() * sr as f64).round() as usize,
            target_cents: n.cents(),
            detune_cents: 0.0,
            vibrato: None,
            legato_in: false,
            kind: NoteKind::Note,
        })
        .collect();
    let mut track = track;
    sig.audio = std::mem::take(&mut track.audio);
    let n = sig.audio.len();
    // Annotated f0 per sample (nearest annotation frame; 0 = unvoiced), so
    // sample-domain metrics (the voiced content delay) see the voicing.
    sig.f0_truth = (0..n)
        .map(|i| {
            ann.nearest(i as f64 / f64::from(sr))
                .filter(|k| ann.voiced[*k])
                .map_or(0.0, |k| crate::synth::cents_to_hz64(ann.cents[k]) as f32)
        })
        .collect();
    sig.target_cents = vec![f32::NAN; n];
    sig.centre_cents = vec![f32::NAN; n];
    FileCtx {
        track,
        sig,
        range,
        key,
        ann,
        onsets,
        offsets,
        transitions,
        in_track,
        input,
        voiced_level_db,
    }
}

impl FileCtx {
    fn audio(&self) -> &[f32] {
        &self.sig.audio
    }

    fn sample_rate(&self) -> f32 {
        self.track.sample_rate
    }

    fn engine_config(&self, cfg: RunConfig) -> EngineConfig {
        EngineConfig {
            sample_rate: self.sample_rate(),
            max_block: 4096,
            channels: 1,
            latency_mode: cfg.mode,
            voice_range: self.range,
            ..EngineConfig::default()
        }
    }

    /// Context and cause tags for a failure span.
    fn segment(&self, config: String, condition: &'static str, raw: RawSegment) -> Segment {
        let (t0, t1) = (raw.start_s, raw.end_s.max(raw.start_s));
        let around: Vec<f64> = (0..self.ann.len())
            .filter(|i| {
                self.ann.t[*i] >= t0 - 0.05 && self.ann.t[*i] <= t1 + 0.05 && self.ann.voiced[*i]
            })
            .map(|i| crate::synth::cents_to_hz64(self.ann.cents[i]))
            .collect();
        let ann_f0 = crate::dsp::median(&around);
        let k0 = self.input.frame_at(t0);
        let k1 = self.input.frame_at(t1).max(k0);
        let frames = self
            .in_track
            .frames
            .get(k0..=k1.min(self.in_track.frames.len().saturating_sub(1)));
        let (level_db, clarity) = frames.map_or((f64::NAN, f64::NAN), |fs| {
            (
                crate::dsp::median(&fs.iter().map(|f| f.level_dbfs).collect::<Vec<_>>()),
                crate::dsp::median(
                    &fs.iter()
                        .map(|f| if f.voiced { f.clarity } else { 0.0 })
                        .collect::<Vec<_>>(),
                ),
            )
        });
        let mut tags = Vec::new();
        if ann_f0 < 110.0 {
            tags.push("low_f0");
        }
        if ann_f0 < 1.15 * self.range.f_min() as f64 {
            tags.push("range_edge");
        }
        if clarity < 0.75 {
            tags.push("aperiodic");
        }
        if level_db < self.voiced_level_db - 15.0 {
            tags.push("quiet");
        }
        if self
            .onsets
            .iter()
            .any(|(o, _)| t0 >= o - 0.03 && t0 <= o + 0.08)
        {
            tags.push("onset");
        }
        if self.offsets.iter().any(|o| t1 >= o - 0.08 && t0 <= o + 0.1) {
            tags.push("release");
        }
        // Fast annotated pitch movement (glide, scoop, melisma).
        let mut max_slope = 0.0f64;
        for i in 1..self.ann.len() {
            let (ta, tb) = (self.ann.t[i - 1], self.ann.t[i]);
            if tb < t0 - 0.01 || ta > t1 + 0.01 {
                continue;
            }
            let d = self.ann.cents[i] - self.ann.cents[i - 1];
            if d.is_finite() && tb > ta {
                max_slope = max_slope.max(d.abs() / ((tb - ta) * 1000.0));
            }
        }
        if max_slope > 1.5 {
            tags.push("fast_pitch");
        }
        let near_notes: Vec<f64> = self
            .track
            .notes_a1
            .iter()
            .filter(|n| n.start_s >= t0 - 0.2 && n.start_s <= t1 + 0.2)
            .map(|n| n.dur_s)
            .collect();
        if near_notes.len() >= 3 && crate::dsp::median(&near_notes) < 0.2 {
            tags.push("melisma");
        }
        if raw.kind == "false_voicing" && clarity >= 0.8 {
            tags.push("pitch_present");
        }
        if raw.kind == "false_voicing"
            && self
                .offsets
                .iter()
                .any(|o| t0 >= o - 0.01 && t0 <= o + 0.05)
        {
            tags.push("tail_after_note");
        }
        Segment {
            file: self.track.id,
            singer: self.track.meta.singer.clone(),
            config,
            condition,
            raw,
            ann_f0,
            level_db,
            clarity,
            tags,
        }
    }

    fn summary(&self) -> FileSummary {
        let est = TimedTrack::from_reference(&self.in_track);
        let cmp = measure::compare_frames(&self.ann, &est, 0.0);
        let mut reference = measure::det_counts(&cmp, &self.transitions, 0.02);
        let (d, missed, bridged) = measure::onset_delays(&self.onsets, &est, 0.1);
        reference.onset_delay_ms = d;
        reference.onsets_missed = missed;
        reference.onsets_bridged = bridged;
        let (checked, spikes) =
            measure::onset_spikes(&self.onsets, &self.ann, &est, 0.1, reference_config().f_max);
        reference.onsets_checked = checked;
        reference.onset_spikes = spikes.len() as u64;
        reference.onset_spikes_fmax = spikes
            .iter()
            .filter(|s| s.kind == "onset_spike_fmax")
            .count() as u64;
        let (lag, _, _) = measure::best_lag(&self.ann, &est, 30);
        let (annot_diff_c, annot_matched) =
            data::annotator_agreement(&self.track.notes_a1, &self.track.notes_a2);
        FileSummary {
            id: self.track.id,
            meta: self.track.meta.clone(),
            sample_rate: self.sample_rate(),
            dur_s: self.sig.duration_s(),
            range: self.range,
            min_f0: self.track.min_f0(),
            max_f0: self.track.max_f0(),
            voiced_fraction: self.track.voiced_fraction(),
            key: self.key,
            notes_a1: self.track.notes_a1.len(),
            notes_a2: self.track.notes_a2.len(),
            annot_diff_c,
            annot_matched,
            reference,
            reference_lag_ms: lag,
            stable_frames: self.input.stable.iter().filter(|s| **s).count() as u64,
            moving_frames: self.input.moving,
        }
    }

    fn reference_segments(&self) -> Vec<Segment> {
        let est = TimedTrack::from_reference(&self.in_track);
        let cmp = measure::compare_frames(&self.ann, &est, 0.0);
        let (_, spikes) =
            measure::onset_spikes(&self.onsets, &self.ann, &est, 0.1, reference_config().f_max);
        measure::detector_segments(&cmp)
            .into_iter()
            .chain(spikes)
            .map(|r| self.segment("reference".to_string(), "detect", r))
            .collect()
    }
}

enum JobOut {
    Det(DetRow),
    Corr(CorrRow),
}

/// Listening WAVs: (config, condition) pairs written for `listen` tracks.
fn listen_pair(cfg: RunConfig, cond: Condition) -> bool {
    matches!(
        (cfg.mode, cfg.tier, cfg.engine, cond),
        (
            LatencyMode::Live,
            QualityTier::Hq,
            EngineKind::Natural,
            Condition::HardChromatic
        ) | (
            LatencyMode::Live,
            QualityTier::Hq,
            EngineKind::Natural,
            Condition::NaturalKey
        ) | (
            LatencyMode::Studio,
            QualityTier::Hq,
            EngineKind::Natural,
            Condition::NaturalKey
        ) | (
            LatencyMode::LiveMin,
            QualityTier::Hq,
            EngineKind::Classic,
            Condition::HardChromatic
        )
    )
}

fn run_job(
    ctx: &FileCtx,
    cfg: RunConfig,
    cond: Condition,
    opts: &VocaditoOptions,
) -> wav::Result<Option<(Vec<JobOut>, Vec<Segment>)>> {
    let Some(settings) = condition_settings(cond, cfg, ctx.key.as_ref()) else {
        return Ok(None);
    };
    let sr = ctx.sample_rate();
    let ropts = RenderOptions {
        config: ctx.engine_config(cfg),
        settings,
        blocks: opts.blocks.clone(),
        telemetry: true,
        midi: Vec::new(),
        extra_tail: 0,
    };
    let t0 = Instant::now();
    let r = render::render(ctx.audio(), &ropts);
    let render_s = t0.elapsed().as_secs_f64();
    let out = r.aligned(0);
    let finite = r.raw(0).iter().all(|v| v.is_finite());
    let label = cfg.label();
    if opts.wavs && opts.listen.contains(&ctx.track.id) && listen_pair(cfg, cond) {
        let path = opts.out.join("listen").join(format!(
            "vocadito_{}__{}__{}.wav",
            ctx.track.id,
            label,
            cond.label()
        ));
        wav::write_wav(&path, &[out], sr)?;
    }
    // One frame per applied control (the Studio pipeline applies none on the
    // first hop after a reset, see `render::pipeline_hops`).
    let skipped_hops = render::pipeline_hops(&ropts.config);
    let telemetry_ok =
        r.stats.telemetry_dropped == 0 && r.telemetry.len() as u64 + skipped_hops == r.stats.hops;
    let out_track = ReferenceTracker::new(sr, reference_config()).track(out);
    if opts.dump.contains(&ctx.track.id) {
        write_dump(
            &opts.out.join("dump").join(format!(
                "vocadito_{}__{}__{}.csv",
                ctx.track.id,
                label,
                cond.label()
            )),
            ctx,
            Some(&out_track),
            &r.telemetry,
            r.latency,
        )?;
    }
    let mut outs = Vec::new();
    let mut all_segs: Vec<Segment> = Vec::new();
    if cond == Condition::Detect {
        let est = TimedTrack::from_telemetry(&r.telemetry, r.latency, f64::from(sr));
        let cmp = measure::compare_frames(&ctx.ann, &est, 0.0);
        let mut counts = measure::det_counts(&cmp, &ctx.transitions, 0.02);
        let (d, missed, bridged) = measure::onset_delays(&ctx.onsets, &est, 0.1);
        counts.onset_delay_ms = d;
        counts.onsets_missed = missed;
        counts.onsets_bridged = bridged;
        let (checked, spikes) = measure::onset_spikes(
            &ctx.onsets,
            &ctx.ann,
            &est,
            0.1,
            f64::from(ctx.range.f_max()),
        );
        counts.onsets_checked = checked;
        counts.onset_spikes = spikes.len() as u64;
        counts.onset_spikes_fmax = spikes
            .iter()
            .filter(|s| s.kind == "onset_spike_fmax")
            .count() as u64;
        let (best_lag_ms, _, _) = measure::best_lag(&ctx.ann, &est, 30);
        let rpa50_at_best = measure::det_counts(
            &measure::compare_frames(&ctx.ann, &est, best_lag_ms * 1e-3),
            &ctx.transitions,
            0.02,
        )
        .rpa50();
        // Measured delay and transparency on (at most) the first 10 s.
        let n10 = ctx.audio().len().min((10.0 * sr) as usize);
        let max_lag = r.latency as usize + (0.06 * sr as f64) as usize;
        let raw = r.raw(0);
        let measured = metrics::measure_latency(
            &ctx.audio()[..n10],
            &raw[..(n10 + max_lag).min(raw.len())],
            max_lag,
        );
        let max_wander = sr as f64 / ctx.range.f_min() as f64;
        let (snr, _) = metrics::aligned_segmental_snr(
            &ctx.audio()[..n10],
            &out[..n10.min(out.len())],
            (0.02 * sr as f64) as usize,
            -2.0,
            max_wander,
        );
        // Information age: frame j is produced (its control applied) at the
        // boundary that closes hop j + skipped_hops, i.e. sample
        // (j + skipped_hops + 1)·hop.
        let hop = skato_core::hop_for_rate(sr) as f64;
        let ages: Vec<f64> = r
            .telemetry
            .iter()
            .enumerate()
            .filter(|(_, f)| f.flags & skato_core::frame_flags::VOICED != 0)
            .map(|(j, f)| {
                ((j as f64 + skipped_hops as f64 + 1.0) * hop - f.sample_pos as f64) / f64::from(sr)
                    * 1000.0
            })
            .collect();
        all_segs.extend(
            measure::detector_segments(&cmp)
                .into_iter()
                .chain(spikes)
                .map(|s| ctx.segment(label.clone(), cond.label(), s)),
        );
        let content =
            metrics::content_delay(&ctx.sig, raw, r.latency, 1.0 / f64::from(ctx.range.f_min()));
        outs.push(JobOut::Det(DetRow {
            file: ctx.track.id,
            config: cfg,
            range: ctx.range,
            latency: r.latency,
            counts,
            best_lag_ms,
            rpa50_at_best,
            measured,
            transparency_snr_db: snr,
            info_age_ms: crate::dsp::median(&ages),
            content,
            finite,
            telemetry_ok,
            render_s,
        }));
    }

    // Correction measures (for Detect: the amount-0 pass-through, a control
    // for the measurement chain, scored against the chromatic grid).
    let legal = legal_classes(&settings);
    let targets = TimedTrack::targets_from_telemetry(&r.telemetry, r.latency, f64::from(sr));
    let (mut counts, mut raw_segs) =
        measure::correction_pitch(&ctx.input, &out_track, &targets, &legal);
    let (cs, cf) = metrics::click_frames(ctx.audio(), out, sr, 0.02);
    let click_segs = measure::click_segments(&cf, 0.01);
    counts.click_frames = cs.clicks as u64;
    counts.click_events = click_segs.len() as u64;
    counts.click_max_db = cs.max_excess_db;
    raw_segs.extend(click_segs);
    counts.dur_s = ctx.sig.duration_s();
    let lsd = metrics::formant_lsd(&ctx.sig, out);
    if lsd.frames > 0 && lsd.mean_db.is_finite() {
        counts.lsd_sum = lsd.mean_db * lsd.frames as f64;
        counts.lsd_frames = lsd.frames as u64;
    }
    let peak = |x: &[f32]| {
        x.iter().fold(
            0.0f32,
            |m, v| if v.is_finite() { m.max(v.abs()) } else { m },
        )
    };
    let (pin, pout) = (peak(ctx.audio()), peak(out));
    counts.peak_ratio_db = if pin > 0.0 {
        20.0 * f64::from(pout.max(1e-12) / pin).log10()
    } else {
        f64::NAN
    };
    counts.nonfinite = u64::from(!finite);
    counts.blowups = u64::from(counts.peak_ratio_db > 6.0);
    all_segs.extend(
        raw_segs
            .into_iter()
            .map(|s| ctx.segment(label.clone(), cond.corr_label(), s)),
    );
    outs.push(JobOut::Corr(CorrRow {
        file: ctx.track.id,
        config: cfg,
        condition: cond,
        range: ctx.range,
        latency: r.latency,
        counts,
        lsd_median: lsd.median_db,
        finite,
        render_s,
    }));
    Ok(Some((outs, all_segs)))
}

/// Per-frame dump on the reference-track grid (1 ms): annotation, input and
/// output reference pitch, and the nearest telemetry frame (input time).
fn write_dump(
    path: &Path,
    ctx: &FileCtx,
    out_track: Option<&PitchTrack>,
    telemetry: &[skato_core::PitchFrame],
    latency: u32,
) -> wav::Result<()> {
    use std::fmt::Write as _;
    let sr = f64::from(ctx.sample_rate());
    let tel = TimedTrack::from_telemetry(telemetry, latency, sr);
    // Telemetry frames that map to input time, in the same order as `tel`.
    let frames: Vec<&skato_core::PitchFrame> = telemetry
        .iter()
        .filter(|f| f.sample_pos >= u64::from(latency))
        .collect();
    let opt = |v: f64, d: usize| {
        if v.is_finite() {
            format!("{v:.d$}")
        } else {
            String::new()
        }
    };
    let mut s = String::from(
        "time_s,ann_cents,in_cents,in_clarity,in_level_db,stable,out_cents,tel_voiced,tel_detected,tel_centre,tel_target,tel_output,tel_flags\n",
    );
    for k in 0..ctx.input.len() {
        let t = ctx.input.time(k);
        let ann = ctx.ann.nearest(t).map_or(f64::NAN, |i| ctx.ann.cents[i]);
        let inf = &ctx.in_track.frames[k];
        let out_c = out_track
            .and_then(|o| o.frames.get(k))
            .map_or(f64::NAN, |f| if f.voiced { f.cents } else { f64::NAN });
        let tf = tel.nearest(t).and_then(|i| frames.get(i).copied());
        let _ = writeln!(
            s,
            "{t:.4},{},{},{:.3},{:.1},{},{},{},{},{},{},{},{}",
            opt(ann, 1),
            opt(ctx.input.in_c[k], 1),
            inf.clarity,
            inf.level_dbfs,
            u8::from(ctx.input.stable[k]),
            opt(out_c, 1),
            tf.map_or(String::new(), |f| (f.flags & skato_core::frame_flags::VOICED)
                .to_string()),
            tf.map_or(String::new(), |f| opt(f64::from(f.detected), 1)),
            tf.map_or(String::new(), |f| opt(f64::from(f.centre), 1)),
            tf.map_or(String::new(), |f| opt(f64::from(f.target), 1)),
            tf.map_or(String::new(), |f| opt(f64::from(f.output), 1)),
            tf.map_or(String::new(), |f| f.flags.to_string()),
        );
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, s)?;
    Ok(())
}

/// Run `f(i)` for `i in 0..n` on `threads` workers; results in index order.
fn parallel<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let v = f(i);
                    if let Ok(mut s) = slots.lock() {
                        s[i] = Some(v);
                    }
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect()
}

/// Load, render, measure. `progress` gets one line per finished job.
pub fn run(opts: &VocaditoOptions, progress: &(dyn Fn(&str) + Sync)) -> wav::Result<Outcome> {
    let t_start = Instant::now();
    let root: &Path = &opts.data;
    let mut ids = data::list_tracks(root);
    if ids.is_empty() {
        return Err(format!("no vocadito tracks under {}", root.display()).into());
    }
    if !opts.files.is_empty() {
        ids.retain(|i| opts.files.contains(i));
        if ids.is_empty() {
            return Err("none of the requested --files exist".into());
        }
    }
    let threads = if opts.threads == 0 {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    } else {
        opts.threads
    };
    let meta = data::load_metadata(root);
    let loaded: Vec<Result<FileCtx, String>> = parallel(ids.len(), threads, |i| {
        let t = data::load_track(root, ids[i], &meta).map_err(|e| e.to_string())?;
        Ok(build_ctx(t, opts.range))
    });
    let mut ctxs = Vec::new();
    for (id, l) in ids.iter().zip(loaded) {
        match l {
            Ok(c) => ctxs.push(c),
            Err(e) => progress(&format!("skipping vocadito_{id}: {e}")),
        }
    }
    if ctxs.is_empty() {
        return Err("no track could be loaded".into());
    }
    progress(&format!(
        "loaded {} tracks ({:.1} min) in {:.1} s",
        ctxs.len(),
        ctxs.iter().map(|c| c.sig.duration_s()).sum::<f64>() / 60.0,
        t_start.elapsed().as_secs_f64()
    ));
    if opts.wavs {
        for c in ctxs.iter().filter(|c| opts.listen.contains(&c.track.id)) {
            wav::write_wav(
                &opts
                    .out
                    .join("listen")
                    .join(format!("vocadito_{}__input.wav", c.track.id)),
                &[c.audio()],
                c.sample_rate(),
            )?;
        }
    }
    let summaries: Vec<FileSummary> = parallel(ctxs.len(), threads, |i| ctxs[i].summary());
    let mut segments: Vec<Segment> =
        parallel(ctxs.len(), threads, |i| ctxs[i].reference_segments())
            .into_iter()
            .flatten()
            .collect();

    // Longest files first so the tail of the job queue is short.
    let mut jobs: Vec<(usize, RunConfig, Condition)> = Vec::new();
    for (fi, _) in ctxs.iter().enumerate() {
        for &cfg in &opts.configs {
            for &cond in &opts.conditions {
                jobs.push((fi, cfg, cond));
            }
        }
    }
    jobs.sort_by(|a, b| ctxs[b.0].sig.len().cmp(&ctxs[a.0].sig.len()));
    let done = AtomicUsize::new(0);
    let total = jobs.len();
    let results = parallel(total, threads, |i| {
        let (fi, cfg, cond) = jobs[i];
        let r = run_job(&ctxs[fi], cfg, cond, opts).map_err(|e| e.to_string());
        let k = done.fetch_add(1, Ordering::Relaxed) + 1;
        let line = match &r {
            Ok(Some((outs, _))) => match outs.first() {
                Some(JobOut::Det(d)) => format!(
                    "rpa50 {:.3} oct {:.2}% vF1 {:.3} lag {:+.0} ms",
                    d.counts.rpa50(),
                    100.0 * d.counts.octave_rate(),
                    d.counts.f1(),
                    d.best_lag_ms
                ),
                Some(JobOut::Corr(c)) => format!(
                    "in-tune med {:.1} c p95 {:.1} c clicks {}",
                    c.counts.out.percentile(50.0),
                    c.counts.out.percentile(95.0),
                    c.counts.click_events
                ),
                None => String::new(),
            },
            Ok(None) => "skipped (no key)".to_string(),
            Err(e) => format!("ERROR {e}"),
        };
        progress(&format!(
            "[{k:>4}/{total}] vocadito_{:<2} {:<22} {:<17} {line}",
            ctxs[fi].track.id,
            cfg.label(),
            cond.label()
        ));
        r
    });
    let mut det = Vec::new();
    let mut corr = Vec::new();
    for r in results {
        if let Some((outs, s)) = r? {
            for o in outs {
                match o {
                    JobOut::Det(d) => det.push(d),
                    JobOut::Corr(c) => corr.push(c),
                }
            }
            segments.extend(s);
        }
    }
    let order = |c: &RunConfig| {
        opts.configs
            .iter()
            .position(|x| x == c)
            .unwrap_or(usize::MAX)
    };
    det.sort_by_key(|d| (order(&d.config), d.file));
    corr.sort_by_key(|c| (order(&c.config), c.condition as usize, c.file));
    Ok(Outcome {
        files: summaries,
        det,
        corr,
        segments,
        elapsed_s: t_start.elapsed().as_secs_f64(),
        threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_settings_follow_the_presets() {
        let cfg = RunConfig::new(LatencyMode::Live, QualityTier::Hq);
        let hard = condition_settings(Condition::HardChromatic, cfg, None).unwrap();
        assert_eq!(hard.retune_ms, 0.0);
        assert_eq!(hard.vibrato_depth, 0.0);
        assert_eq!(hard.scale, ScaleKind::Chromatic);
        assert_eq!(legal_classes(&hard), [true; 12]);
        let nat = condition_settings(Condition::NaturalChromatic, cfg, None).unwrap();
        assert_eq!(
            (nat.retune_ms, nat.humanize, nat.flex_tune),
            (25.0, 30.0, 20.0)
        );
        assert!(condition_settings(Condition::HardKey, cfg, None).is_none());
        let k = KeyEstimate {
            key: 2,
            scale: ScaleKind::Major,
            correlation: 0.9,
            coverage: 1.0,
        };
        let hk = condition_settings(Condition::HardKey, cfg, Some(&k)).unwrap();
        let on: Vec<usize> = (0..12).filter(|i| legal_classes(&hk)[*i]).collect();
        assert_eq!(on, vec![1, 2, 4, 6, 7, 9, 11]);
        let det = condition_settings(Condition::Detect, cfg, None).unwrap();
        assert_eq!(det.amount, 0.0);
        // Live-Min keeps the Classic engine even for the Natural preset.
        let lm = RunConfig::new(LatencyMode::LiveMin, QualityTier::Hq);
        let s = condition_settings(Condition::NaturalChromatic, lm, None).unwrap();
        assert_eq!(s.engine, EngineKind::Classic);
    }

    #[test]
    fn default_matrix() {
        let c = default_configs();
        assert_eq!(c.len(), 7);
        assert_eq!(c[6].label(), "livemin_hq_classic");
        assert_eq!(c[1].label(), "live_hq_natural");
    }
}
