//! The evaluation suite: every scenario × latency mode × quality tier (×
//! retune speed) rendered through the engine, measured, and summarised as a
//! CSV plus a Markdown report with the DESIGN §4 gates.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use skato_core::{EngineConfig, EngineKind, LatencyMode, QualityTier, Settings, VoiceRange};

use crate::metrics::{
    self, ClickStats, ContentDelayStats, JitterStats, LatencyMeasurement, LsdStats,
    SettlingOptions, SettlingStats, TuningOptions, TuningStats, VibratoStats, VoicingStats,
};
use crate::reference::{PitchTrack, ReferenceConfig, ReferenceTracker};
use crate::render::{self, BlockPattern, RenderOptions, Rendered, default_engine};
use crate::scenarios::{self, Scenario};
use crate::wav::{self, Result};

pub const MODES: [LatencyMode; 4] = [
    LatencyMode::LiveMin,
    LatencyMode::Live,
    LatencyMode::LivePlus,
    LatencyMode::Studio,
];
pub const TIERS: [QualityTier; 3] = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
pub const RANGES: [VoiceRange; 4] = [
    VoiceRange::LowMale,
    VoiceRange::AltoTenor,
    VoiceRange::Soprano,
    VoiceRange::Wide,
];

pub fn mode_name(m: LatencyMode) -> &'static str {
    match m {
        LatencyMode::LiveMin => "livemin",
        LatencyMode::Live => "live",
        LatencyMode::LivePlus => "liveplus",
        LatencyMode::Studio => "studio",
    }
}

pub fn tier_name(t: QualityTier) -> &'static str {
    match t {
        QualityTier::Eco => "eco",
        QualityTier::Hq => "hq",
        QualityTier::Max => "max",
    }
}

pub fn range_name(r: VoiceRange) -> &'static str {
    match r {
        VoiceRange::LowMale => "lowmale",
        VoiceRange::AltoTenor => "altotenor",
        VoiceRange::Soprano => "soprano",
        VoiceRange::Wide => "wide",
    }
}

pub fn engine_name(e: EngineKind) -> &'static str {
    match e {
        EngineKind::Classic => "classic",
        EngineKind::Natural => "natural",
    }
}

/// Engine configuration for a scenario run (mono: metrics only need one
/// channel, and both channels get identical processing by contract).
pub fn engine_config(sample_rate: f32, mode: LatencyMode, range: VoiceRange) -> EngineConfig {
    EngineConfig {
        sample_rate,
        max_block: 4096,
        channels: 1,
        latency_mode: mode,
        voice_range: range,
        ..EngineConfig::default()
    }
}

/// Settings for a scenario run.
pub fn run_settings(
    scn: &Scenario,
    mode: LatencyMode,
    tier: QualityTier,
    retune_ms: f32,
) -> Settings {
    Settings {
        tier,
        engine: default_engine(mode),
        retune_ms,
        ..scn.settings
    }
}

/// Settling options for a run: R§10.3's 10 c / 20 ms, with the clock started
/// at the note-decision point of the run's scale and tolerance
/// ([`metrics::SettlingOptions`]).
pub fn settling_options(settings: &Settings, include_legato: bool) -> SettlingOptions {
    SettlingOptions {
        include_legato,
        legal: Some(scenarios::legal_classes(settings)),
        decision_tolerance_cents: f64::from(settings.tolerance_cents),
        ..SettlingOptions::default()
    }
}

/// Measurements of one render.
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub scenario: &'static str,
    pub mode: LatencyMode,
    pub tier: QualityTier,
    pub engine: EngineKind,
    pub retune_ms: f32,
    pub latency: u32,
    pub measured_latency: Option<LatencyMeasurement>,
    pub tuning: Option<TuningStats>,
    pub settling: Option<SettlingStats>,
    pub vibrato: Option<VibratoStats>,
    pub jitter: Option<JitterStats>,
    pub voicing: Option<VoicingStats>,
    pub clicks: ClickStats,
    pub lsd: Option<LsdStats>,
    /// Every output sample finite.
    pub finite: bool,
    /// Wall time of the render alone, s.
    pub render_s: f64,
}

/// Render one scenario configuration and measure everything that applies.
/// `in_track` is the reference track of the input (computed once per scenario).
pub fn evaluate(
    scn: &Scenario,
    in_track: Option<&PitchTrack>,
    mode: LatencyMode,
    tier: QualityTier,
    retune_ms: f32,
    blocks: &BlockPattern,
) -> (RunResult, Rendered) {
    let sig = &scn.signal;
    let opts = RenderOptions {
        config: engine_config(sig.sample_rate, mode, scn.voice_range),
        settings: run_settings(scn, mode, tier, retune_ms),
        blocks: blocks.clone(),
        telemetry: true,
        midi: Vec::new(),
        extra_tail: 0,
    };
    let t0 = Instant::now();
    let r = render::render(&sig.audio, &opts);
    let render_s = t0.elapsed().as_secs_f64();
    let out = r.aligned(0);
    let finite = r.raw(0).iter().all(|v| v.is_finite());
    let m = scn.measures;
    let needs_pitch = m.tuning || m.settling || m.vibrato || m.jitter || m.sweep;
    let out_track = needs_pitch
        .then(|| ReferenceTracker::new(sig.sample_rate, ReferenceConfig::default()).track(out));
    let tuning = out_track.as_ref().filter(|_| m.tuning || m.sweep).map(|t| {
        metrics::tuning(
            sig,
            t,
            &TuningOptions {
                include_sweeps: m.sweep,
                // Frames within 2·τ of an input movement are still converging
                // by design (see `TuningOptions::stable_window_s`).
                stable_window_s: 2.0 * f64::from(retune_ms.max(0.0)) * 1e-3,
                // Ambiguity from the scenario's own scale (see
                // `TuningOptions::legal`).
                legal: Some(scenarios::legal_classes(&scn.settings)),
                legal_tolerance_cents: f64::from(opts.settings.tolerance_cents),
                ..TuningOptions::default()
            },
        )
    });
    let settling = out_track.as_ref().filter(|_| m.settling).map(|t| {
        metrics::settling_with(
            sig,
            t,
            &settling_options(&opts.settings, m.settling_legato),
            Some((&r.telemetry, r.latency)),
        )
    });
    let vibrato = match (&out_track, in_track) {
        (Some(o), Some(i)) if m.vibrato => Some(metrics::vibrato(sig, i, o, 0.03)),
        _ => None,
    };
    let jitter = match (&out_track, in_track) {
        (Some(o), Some(i)) if m.jitter => Some(metrics::jitter(sig, i, o, 12.0)),
        _ => None,
    };
    let voicing = m
        .voicing
        .then(|| metrics::voicing_f1(sig, &r.telemetry, r.latency, 0.02));
    let measured_latency = m
        .latency
        .then(|| {
            metrics::measure_latency(
                &sig.audio,
                r.raw(0),
                r.latency as usize + (0.06 * sig.sample_rate as f64) as usize,
            )
        })
        .flatten();
    let clicks = metrics::clicks(&sig.audio, out, sig.sample_rate, 0.02);
    let lsd = (m.tuning && !sig.notes.is_empty()).then(|| metrics::formant_lsd(sig, out));
    let result = RunResult {
        scenario: scn.name,
        mode,
        tier,
        engine: opts.settings.engine,
        retune_ms,
        latency: r.latency,
        measured_latency,
        tuning,
        settling,
        vibrato,
        jitter,
        voicing,
        clicks,
        lsd,
        finite,
        render_s,
    };
    (result, r)
}

/// Which outputs the suite saves as WAV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavPolicy {
    None,
    /// Inputs plus every HQ-tier output (default).
    Hq,
    All,
}

#[derive(Clone, Debug)]
pub struct SuiteOptions {
    pub out_dir: PathBuf,
    pub sample_rate: f32,
    /// Scenario names (empty = all).
    pub scenarios: Vec<String>,
    pub modes: Vec<LatencyMode>,
    pub tiers: Vec<QualityTier>,
    pub wavs: WavPolicy,
    pub blocks: BlockPattern,
    /// Worker threads (0 = available parallelism).
    pub threads: usize,
}

impl Default for SuiteOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("out"),
            sample_rate: 48_000.0,
            scenarios: Vec::new(),
            modes: MODES.to_vec(),
            tiers: TIERS.to_vec(),
            wavs: WavPolicy::Hq,
            blocks: BlockPattern::Fixed(128),
            threads: 0,
        }
    }
}

/// Extra checks run once per mode (HQ): amount-0 transparency and
/// block-size invariance.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeCheck {
    pub mode: LatencyMode,
    pub engine: EngineKind,
    /// max |output − input delayed by D| with amount 0.
    pub transparency_max_diff: f64,
    /// Segmental SNR of the amount-0 output vs the input after aligning each
    /// segment by its content wander, dB.
    pub transparency_snr_db: f64,
    /// Largest difference between block patterns {1, 7, 32, 64, 128, 480,
    /// 4096, random} and 64-sample blocks.
    pub block_invariance_max_diff: f64,
    /// Content delay of the voiced amount-0 output (D + PSOLA wander).
    pub content_delay: ContentDelayStats,
}

/// Transparency at amount 0 and block-size invariance for one mode/engine.
pub fn mode_check(sample_rate: f32, mode: LatencyMode, engine: EngineKind) -> Option<ModeCheck> {
    let scn = scenarios::build("melody_male", sample_rate)?;
    let sig = &scn.signal;
    let mut settings = run_settings(&scn, mode, QualityTier::Hq, 20.0);
    settings.engine = engine;
    let config = engine_config(sample_rate, mode, scn.voice_range);
    let amount0 = RenderOptions::new(sample_rate, mode, QualityTier::Hq)
        .with_config(config)
        .with_settings(Settings {
            amount: 0.0,
            ..settings
        });
    let r = render::render(&sig.audio, &amount0);
    let out = r.aligned(0);
    let skip = (0.1 * sample_rate as f64) as usize;
    let max_diff = out
        .iter()
        .zip(sig.audio.iter())
        .skip(skip)
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0, f64::max);
    // PSOLA content wander is within one input period: allow up to 1/f_min.
    let max_wander = sample_rate as f64 / scn.voice_range.f_min() as f64;
    let (snr, _) = metrics::aligned_segmental_snr(
        &sig.audio,
        out,
        (0.02 * sample_rate as f64) as usize,
        -2.0,
        max_wander,
    );

    // Block-size invariance on a 2 s excerpt (block size 1 is slow on a real engine).
    let excerpt = &sig.audio[..sig.audio.len().min((2.0 * sample_rate) as usize)];
    let base = RenderOptions::new(sample_rate, mode, QualityTier::Hq)
        .with_config(config)
        .with_settings(settings);
    let reference = render::render(excerpt, &base.clone().with_blocks(BlockPattern::Fixed(64)));
    let mut inv = 0.0f64;
    for p in BlockPattern::invariance_set() {
        let o = render::render(excerpt, &base.clone().with_blocks(p));
        let d = o
            .raw(0)
            .iter()
            .zip(reference.raw(0))
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        inv = inv.max(d);
    }
    Some(ModeCheck {
        mode,
        engine,
        transparency_max_diff: max_diff,
        transparency_snr_db: snr,
        block_invariance_max_diff: inv,
        content_delay: metrics::content_delay(
            sig,
            r.raw(0),
            r.latency,
            1.0 / f64::from(scn.voice_range.f_min()),
        ),
    })
}

/// One row of the latency gauge: what a performer monitoring through the
/// plugin hears, per mode × engine × voice range.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeRow {
    pub mode: LatencyMode,
    pub engine: EngineKind,
    pub range: VoiceRange,
    /// Scenario whose voiced notes were measured.
    pub scenario: &'static str,
    /// Reported D plus the measured voiced content delay (amount 0, HQ).
    pub content: ContentDelayStats,
}

/// Scenarios the gauge measures, one per voice range preset.
pub const GAUGE_SCENARIOS: [&str; 4] = ["low_male", "melody_male", "melody_female", "high_soprano"];

/// Latency gauge (R§10.3 / R§10.6): reported D and the measured content
/// delay of voiced audio (D + PSOLA wander, [`metrics::content_delay`]) for
/// every mode × engine on [`GAUGE_SCENARIOS`], at amount 0 so the output is
/// a delayed copy of the input. Runs on up to `threads` workers (0 = all).
pub fn latency_gauge(sample_rate: f32, modes: &[LatencyMode], threads: usize) -> Vec<GaugeRow> {
    let built: Vec<Scenario> = GAUGE_SCENARIOS
        .iter()
        .filter_map(|n| scenarios::build(n, sample_rate))
        .collect();
    let mut jobs = Vec::new();
    for (si, _) in built.iter().enumerate() {
        for &mode in modes {
            let engines: &[EngineKind] = if mode == LatencyMode::LiveMin {
                &[EngineKind::Classic]
            } else {
                &[EngineKind::Classic, EngineKind::Natural]
            };
            for &e in engines {
                jobs.push((si, mode, e));
            }
        }
    }
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    } else {
        threads
    };
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Option<GaugeRow>>> = Mutex::new(vec![None; jobs.len()]);
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, jobs.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(si, mode, engine)) = jobs.get(i) else {
                        break;
                    };
                    let scn = &built[si];
                    let mut settings = run_settings(scn, mode, QualityTier::Hq, 20.0);
                    settings.engine = engine;
                    settings.amount = 0.0;
                    let opts = RenderOptions::new(sample_rate, mode, QualityTier::Hq)
                        .with_config(engine_config(sample_rate, mode, scn.voice_range))
                        .with_settings(settings);
                    let r = render::render(&scn.signal.audio, &opts);
                    let content = metrics::content_delay(
                        &scn.signal,
                        r.raw(0),
                        r.latency,
                        1.0 / f64::from(scn.voice_range.f_min()),
                    );
                    if let Ok(mut v) = rows.lock()
                        && let Some(slot) = v.get_mut(i)
                    {
                        *slot = Some(GaugeRow {
                            mode,
                            engine,
                            range: scn.voice_range,
                            scenario: scn.name,
                            content,
                        });
                    }
                }
            });
        }
    });
    rows.into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect()
}

/// One measured-vs-reported latency comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyRow {
    pub sample_rate: f32,
    pub mode: LatencyMode,
    pub range: VoiceRange,
    pub reported: u32,
    pub measured: Option<LatencyMeasurement>,
}

impl LatencyRow {
    /// Measured within ±1 sample of reported, with a clean correlation peak.
    pub fn ok(&self) -> bool {
        self.measured
            .is_some_and(|m| (m.lag as i64 - self.reported as i64).abs() <= 1 && m.peak > 0.9)
    }
}

/// Measured vs reported latency for every mode × voice range at each rate
/// (DESIGN §3.5 / §4), by cross-correlation on noise bursts (R§10.6).
pub fn latency_sweep(sample_rates: &[f32]) -> Vec<LatencyRow> {
    let mut rows = Vec::new();
    for &sr in sample_rates {
        let Some(scn) = scenarios::build("noise_bursts", sr) else {
            continue;
        };
        for mode in MODES {
            for range in RANGES {
                let opts = RenderOptions::new(sr, mode, QualityTier::Hq)
                    .with_config(engine_config(sr, mode, range));
                let r = render::render(&scn.signal.audio, &opts);
                let max_lag = r.latency as usize + (0.06 * sr as f64) as usize;
                let measured = metrics::measure_latency(&scn.signal.audio, r.raw(0), max_lag);
                rows.push(LatencyRow {
                    sample_rate: sr,
                    mode,
                    range,
                    reported: r.latency,
                    measured,
                });
            }
        }
    }
    rows
}

/// Run the whole suite, writing WAVs as configured. `progress` receives one
/// line per finished render.
pub fn run_suite(
    opts: &SuiteOptions,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<(Vec<RunResult>, Vec<ModeCheck>)> {
    let names: Vec<&str> = if opts.scenarios.is_empty() {
        scenarios::NAMES.to_vec()
    } else {
        scenarios::NAMES
            .iter()
            .copied()
            .filter(|n| opts.scenarios.iter().any(|s| s == n))
            .collect()
    };
    let wav_dir = opts.out_dir.join("wav");
    let built: Vec<Scenario> = names
        .iter()
        .filter_map(|n| scenarios::build(n, opts.sample_rate))
        .collect();
    // Input tracks once per scenario (needed for vibrato and jitter baselines).
    let in_tracks: Vec<Option<PitchTrack>> = built
        .iter()
        .map(|s| {
            (s.measures.vibrato || s.measures.jitter).then(|| {
                ReferenceTracker::new(opts.sample_rate, ReferenceConfig::default())
                    .track(&s.signal.audio)
            })
        })
        .collect();
    if opts.wavs != WavPolicy::None {
        for s in &built {
            wav::write_wav(
                &wav_dir.join(format!("{}__input.wav", s.name)),
                &[&s.signal.audio],
                opts.sample_rate,
            )?;
            wav::write_truth_csv(&wav_dir.join(format!("{}__truth.csv", s.name)), &s.signal)?;
        }
    }

    let mut jobs = Vec::new();
    for (si, s) in built.iter().enumerate() {
        for &mode in &opts.modes {
            for &tier in &opts.tiers {
                for &rt in s.retunes {
                    jobs.push((si, mode, tier, rt));
                }
            }
        }
    }
    let threads = if opts.threads == 0 {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    } else {
        opts.threads
    };
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<RunResult>>> = Mutex::new(vec![None; jobs.len()]);
    let error: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(si, mode, tier, rt)) = jobs.get(i) else {
                        break;
                    };
                    let scn = &built[si];
                    let (res, rendered) =
                        evaluate(scn, in_tracks[si].as_ref(), mode, tier, rt, &opts.blocks);
                    let save = match opts.wavs {
                        WavPolicy::All => true,
                        WavPolicy::Hq => tier == QualityTier::Hq,
                        WavPolicy::None => false,
                    };
                    if save {
                        let path = wav_dir.join(format!(
                            "{}__{}_{}_rt{}.wav",
                            scn.name,
                            mode_name(mode),
                            tier_name(tier),
                            rt
                        ));
                        if let Err(e) =
                            wav::write_wav(&path, &[rendered.aligned(0)], opts.sample_rate)
                            && let Ok(mut slot) = error.lock()
                        {
                            *slot = Some(format!("{}: {e}", path.display()));
                        }
                    }
                    progress(&format!(
                        "{:>14} {:>8} {:>3} rt{:<3} {}",
                        scn.name,
                        mode_name(mode),
                        tier_name(tier),
                        rt,
                        short_summary(&res)
                    ));
                    if let Ok(mut v) = results.lock()
                        && let Some(slot) = v.get_mut(i)
                    {
                        *slot = Some(res);
                    }
                }
            });
        }
    });
    if let Some(e) = error.into_inner().ok().flatten() {
        return Err(e.into());
    }
    let results: Vec<RunResult> = results
        .into_inner()
        .map_err(|_| "worker panicked")?
        .into_iter()
        .flatten()
        .collect();

    let mut checks = Vec::new();
    for &mode in &opts.modes {
        let engines: &[EngineKind] = if mode == LatencyMode::LiveMin {
            &[EngineKind::Classic]
        } else {
            &[EngineKind::Classic, EngineKind::Natural]
        };
        for &e in engines {
            checks.extend(mode_check(opts.sample_rate, mode, e));
            progress(&format!(
                "{:>14} {:>8} {:>3} {}",
                "mode-check",
                mode_name(mode),
                "hq",
                engine_name(e)
            ));
        }
    }
    Ok((results, checks))
}

fn short_summary(r: &RunResult) -> String {
    let mut s = String::new();
    if let Some(t) = &r.tuning {
        let _ = write!(s, "tune med {:.1} p95 {:.1} ", t.median_abs, t.p95_abs);
    }
    if let Some(v) = &r.voicing {
        let _ = write!(s, "F1 {:.3} ", v.f1);
    }
    if let Some(m) = &r.measured_latency {
        let _ = write!(s, "lat {}/{} ", m.lag, r.latency);
    }
    s
}

/// One DESIGN §4 gate evaluated over suite results.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: &'static str,
    pub target: &'static str,
    /// Measured value(s), human readable.
    pub value: String,
    /// None when the suite did not include the needed runs.
    pub pass: Option<bool>,
}

const TUNING_SCENARIOS: [&str; 4] = ["melody_male", "melody_female", "low_male", "high_soprano"];

fn worst<'a>(
    rs: impl Iterator<Item = &'a RunResult>,
    f: impl Fn(&RunResult) -> Option<f64>,
) -> Option<(f64, String)> {
    rs.filter_map(|r| {
        f(r).map(|v| {
            (
                v,
                format!(
                    "{} {}/{} rt{}",
                    r.scenario,
                    mode_name(r.mode),
                    tier_name(r.tier),
                    r.retune_ms
                ),
            )
        })
    })
    .max_by(|a, b| a.0.total_cmp(&b.0))
}

/// The DESIGN §4 Live tuning target on the held-out scenarios
/// ([`scenarios::HELDOUT`]): median < 5 c, p95 < 15 c.
pub fn heldout_tuning_gate<'a>(runs: impl Iterator<Item = &'a RunResult> + Clone) -> Gate {
    const NAME: &str = "Held-out tuning (Live HQ, retune 0-20 ms)";
    const TARGET: &str = "median < 5 c, p95 < 15 c on scenarios never used for tuning";
    let med = worst(runs.clone(), |r| r.tuning.map(|t| t.median_abs));
    let p95 = worst(runs, |r| r.tuning.map(|t| t.p95_abs));
    match (med, p95) {
        (Some((m, ms)), Some((p, ps))) => Gate {
            name: NAME,
            target: TARGET,
            value: format!("worst median {m:.2} c ({ms}); worst p95 {p:.2} c ({ps})"),
            pass: Some(m < 5.0 && p < 15.0),
        },
        _ => Gate {
            name: NAME,
            target: TARGET,
            value: "not run".into(),
            pass: None,
        },
    }
}

/// DESIGN §4 settling limit, s.
pub const SETTLING_LIMIT_S: f64 = 0.030;

fn ms(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.1}", x * 1e3),
        Some(_) => "never".into(),
        None => "-".into(),
    }
}

/// DESIGN §4 "Settling (Live, retune 0): < 30 ms to stay within 10 c after
/// onset", applied to **every** onset (no median; EV-1). An onset that never
/// settles is a failure. The value names the worst onset with its latency
/// breakdown (decision point, detection, target switch, convergence).
pub fn settling_gate<'a>(runs: impl Iterator<Item = &'a RunResult>) -> Gate {
    settling_gate_named("Settling (Live HQ, retune 0)", runs)
}

/// [`settling_gate`] under another name (the held-out row).
pub fn settling_gate_named<'a>(
    name: &'static str,
    runs: impl Iterator<Item = &'a RunResult>,
) -> Gate {
    const TARGET: &str = "every onset < 30 ms to stay within 10 c for 20 ms (clock from the note-decision point; never settled = fail)";
    let mut total = 0usize;
    let mut failing = 0usize;
    let mut unsettled = 0usize;
    let mut worst: Option<(f64, String, metrics::OnsetTiming)> = None;
    for r in runs {
        let Some(s) = &r.settling else { continue };
        for (i, o) in s.onsets.iter().enumerate() {
            total += 1;
            let v = o.settle_s.unwrap_or(f64::INFINITY);
            if o.settle_s.is_none() {
                unsettled += 1;
            }
            if v.is_nan() || v >= SETTLING_LIMIT_S {
                failing += 1;
            }
            if worst.as_ref().is_none_or(|w| v.total_cmp(&w.0).is_gt()) {
                worst = Some((v, format!("{} onset {i}", r.scenario), *o));
            }
        }
    }
    match worst {
        Some((v, at, o)) => Gate {
            name,
            target: TARGET,
            value: format!(
                "worst {} ms ({at}: decision point {} ms into the note; after it, detected at {} ms, target switch at {} ms; convergence {} ms); {failing} of {total} onsets ≥ 30 ms, {unsettled} never settled",
                ms(Some(v)),
                ms(Some(o.clock_s)),
                ms(o.detect_s),
                ms(o.switch_s),
                ms(o.convergence_s()),
            ),
            pass: Some(failing == 0),
        },
        None => Gate {
            name,
            target: TARGET,
            value: "not run".into(),
            pass: None,
        },
    }
}

/// The PSOLA content wander behind DESIGN §4's transparency tolerance,
/// against R§10.3's target "wander ≤ T" (one sung period), with the content
/// delay a performer hears on voiced audio (D + wander).
pub fn content_wander_gate(checks: &[ModeCheck]) -> Gate {
    const NAME: &str = "Content delay (amount 0, voiced, melody_male)";
    const TARGET: &str = "|wander| ≤ one sung period (R§10.3); heard delay = D + wander";
    let parts: Vec<String> = checks
        .iter()
        .filter(|c| c.engine == EngineKind::Natural)
        .map(|c| {
            let d = &c.content_delay;
            format!(
                "{} D {:.2} ms → median {:.2} ms (max |wander| {:.2} periods)",
                mode_name(c.mode),
                d.reported_ms(),
                d.delay_ms(50.0),
                d.max_abs_wander_periods()
            )
        })
        .collect();
    let ok = checks.iter().all(|c| {
        let d = &c.content_delay;
        d.segments > 0 && d.max_abs_wander_periods() <= 1.0
    });
    Gate {
        name: NAME,
        target: TARGET,
        value: if parts.is_empty() {
            "Classic only (exact delay)".into()
        } else {
            parts.join("; ")
        },
        pass: Some(ok),
    }
}

/// Scenarios of the vibrato gates: delayed, faded-in vibrato and vibrato
/// from the note onset (EV-3).
pub const VIBRATO_SCENARIOS: [&str; 2] = ["vibrato", "vibrato_onset"];

/// Scenario the vibrato engagement gate applies to (vibrato from the onset).
pub const ENGAGEMENT_SCENARIO: &str = "vibrato_onset";

/// Engagement bound, vibrato periods after the vibrato start. R§8.11's
/// extrema tracker engages after three valid extrema (about one cycle,
/// 170-250 ms) and ramps the split in over half a period, so the output
/// reaches full extent ≈ 1.75 periods after a vibrato starts; the bound adds
/// one period for detection lag and the one-period measurement window.
pub const ENGAGEMENT_LIMIT_PERIODS: f64 = 2.5;

/// Largest early-window (first two periods) output/input extent ratio: the
/// first cycle may be flattened (R§8.11, accepted in R§11 risk 5) but must
/// not be exaggerated beyond DESIGN §4's ±15 % extent tolerance.
pub const EARLY_RATIO_MAX: f64 = 1.15;

/// DESIGN §4 vibrato preservation (PreserveVibrato on, τ = 0) on every
/// vibrato scenario run: developed extent within ±15 %, centre median < 8 c.
pub fn vibrato_gate(runs: &[&RunResult]) -> Gate {
    const NAME: &str = "Vibrato preservation (Live HQ, τ = 0)";
    const TARGET: &str = "extent within ±15 %, centre median < 8 c (developed vibrato; delayed and from-onset scenarios)";
    if runs.is_empty() {
        return Gate {
            name: NAME,
            target: TARGET,
            value: "not run".into(),
            pass: None,
        };
    }
    let mut parts = Vec::new();
    let mut ok = true;
    for r in runs {
        match r.vibrato.as_ref().filter(|v| !v.notes.is_empty()) {
            Some(v) => {
                ok &= v.worst_extent_ratio_error <= 0.15 && v.centre_error_median < 8.0;
                parts.push(format!(
                    "{}: worst extent error {:.1} %, centre median {:.2} c",
                    r.scenario,
                    v.worst_extent_ratio_error * 100.0,
                    v.centre_error_median
                ));
            }
            None => {
                ok = false;
                parts.push(format!("{}: no vibrato measured in output", r.scenario));
            }
        }
    }
    Gate {
        name: NAME,
        target: TARGET,
        value: parts.join("; "),
        pass: Some(ok),
    }
}

/// Vibrato onset (EV-3, R§8.11): how soon the output vibrato reaches 80 % of
/// the input's after the vibrato starts, and the early-window extent ratio.
pub fn vibrato_engagement_gate(runs: &[&RunResult]) -> Gate {
    const NAME: &str = "Vibrato engagement (Live HQ, τ = 0)";
    const TARGET: &str = "vibrato from the onset: output ≥ 80 % of input extent ≤ 2.5 periods after it starts; first-two-period extent ratio ≤ 1.15 (a flattened first cycle is the documented causal limit, R§8.11)";
    let mut parts = Vec::new();
    let mut ok = true;
    let mut any = false;
    for r in runs {
        let Some(v) = r.vibrato.as_ref().filter(|v| !v.engagement.is_empty()) else {
            continue;
        };
        let worst = v.worst_engagement_periods();
        let (lo, hi) = v.early_ratio_range();
        let slowest = v
            .engagement
            .iter()
            .filter_map(|e| e.engagement_s)
            .fold(f64::NAN, f64::max);
        // Gated on vibrato that starts at full extent with the note; a
        // faded-in vibrato (the `vibrato` scenario) gives the tracker small
        // early extrema by construction, so it is reported only.
        let gated = r.scenario == ENGAGEMENT_SCENARIO;
        if gated {
            any = true;
            ok &= worst <= ENGAGEMENT_LIMIT_PERIODS && hi <= EARLY_RATIO_MAX;
        }
        parts.push(format!(
            "{}{}: slowest {:.2} periods ({:.0} ms), early extent ratio {:.2}-{:.2}",
            r.scenario,
            if gated {
                ""
            } else {
                " (reported only: faded-in vibrato)"
            },
            worst,
            slowest * 1e3,
            lo,
            hi
        ));
    }
    Gate {
        name: NAME,
        target: TARGET,
        value: if any {
            parts.join("; ")
        } else {
            "not run".into()
        },
        pass: any.then_some(ok),
    }
}

/// Evaluate the §4 gates that the suite covers.
pub fn gates(results: &[RunResult], checks: &[ModeCheck]) -> Vec<Gate> {
    let sel = |mode: LatencyMode, scenarios: &'static [&'static str]| {
        results.iter().filter(move |r| {
            r.mode == mode && r.tier == QualityTier::Hq && scenarios.contains(&r.scenario)
        })
    };
    let mut g = Vec::new();
    let nan_fail = |v: f64| v.is_nan();

    let live = sel(LatencyMode::Live, &TUNING_SCENARIOS).filter(|r| r.retune_ms <= 20.0);
    let med = worst(live.clone(), |r| r.tuning.map(|t| t.median_abs));
    let p95 = worst(live, |r| r.tuning.map(|t| t.p95_abs));
    g.push(match (med, p95) {
        (Some((m, ms)), Some((p, ps))) => Gate {
            name: "Tuning (Live HQ, retune 0-20 ms)",
            target: "median < 5 c, p95 < 15 c",
            value: format!("worst median {m:.2} c ({ms}); worst p95 {p:.2} c ({ps})"),
            pass: Some(m < 5.0 && p < 15.0 && !nan_fail(m) && !nan_fail(p)),
        },
        _ => Gate {
            name: "Tuning (Live HQ, retune 0-20 ms)",
            target: "median < 5 c, p95 < 15 c",
            value: "not run".into(),
            pass: None,
        },
    });

    let studio = worst(sel(LatencyMode::Studio, &TUNING_SCENARIOS), |r| {
        r.tuning.map(|t| t.median_abs)
    });
    g.push(match studio {
        Some((m, s)) => Gate {
            name: "Tuning (Studio HQ)",
            target: "median < 3 c",
            value: format!("worst median {m:.2} c ({s})"),
            pass: Some(m < 3.0 && !m.is_nan()),
        },
        None => Gate {
            name: "Tuning (Studio HQ)",
            target: "median < 3 c",
            value: "not run".into(),
            pass: None,
        },
    });

    g.push(settling_gate(
        sel(LatencyMode::Live, &TUNING_SCENARIOS).filter(|r| r.retune_ms == 0.0),
    ));
    g.push(heldout_tuning_gate(
        sel(LatencyMode::Live, &scenarios::HELDOUT).filter(|r| r.retune_ms <= 20.0),
    ));
    g.push(settling_gate_named(
        "Held-out settling (Live HQ, retune 0)",
        sel(LatencyMode::Live, &scenarios::HELDOUT).filter(|r| r.retune_ms == 0.0),
    ));
    // Voicing on the held-out line without a room: reverb tails are pitched
    // sound where the dry truth is unvoiced, so `heldout_room` is reported
    // in its table only.
    let held_f1 = worst(sel(LatencyMode::Live, &["heldout_pentatonic"]), |r| {
        r.voicing.map(|v| -v.f1)
    });
    g.push(match held_f1 {
        Some((v, s)) => Gate {
            name: "Held-out voicing F1 (Live HQ telemetry vs truth)",
            target: ">= 0.97 (voiced-consonant onsets, hard and soft attacks)",
            value: format!("worst {:.3} ({s})", -v),
            pass: Some(-v >= 0.97),
        },
        None => Gate {
            name: "Held-out voicing F1 (Live HQ telemetry vs truth)",
            target: ">= 0.97",
            value: "not run".into(),
            pass: None,
        },
    });

    let vib: Vec<&RunResult> = sel(LatencyMode::Live, &VIBRATO_SCENARIOS)
        .filter(|r| r.retune_ms == 0.0)
        .collect();
    g.push(vibrato_gate(&vib));
    g.push(vibrato_engagement_gate(&vib));

    let jit = worst(sel(LatencyMode::Live, &["steady"]), |r| {
        r.jitter.map(|j| {
            if j.frames == 0 {
                f64::INFINITY
            } else {
                j.added_rms
            }
        })
    });
    g.push(match jit {
        Some((v, s)) => Gate {
            name: "Added jitter (Live HQ, steady notes)",
            target: "< 3 c RMS (> 12 Hz)",
            value: format!("worst {v:.2} c ({s})"),
            pass: Some(v < 3.0),
        },
        None => Gate {
            name: "Added jitter (Live HQ, steady notes)",
            target: "< 3 c RMS",
            value: "not run".into(),
            pass: None,
        },
    });

    let f1 = worst(
        sel(
            LatencyMode::Live,
            &["consonants", "melody_male", "melody_female"],
        ),
        |r| r.voicing.map(|v| -v.f1),
    );
    g.push(match f1 {
        Some((v, s)) => Gate {
            name: "Voicing F1 (Live HQ telemetry vs truth)",
            target: ">= 0.97",
            value: format!("worst {:.3} ({s})", -v),
            pass: Some(-v >= 0.97),
        },
        None => Gate {
            name: "Voicing F1 (Live HQ telemetry vs truth)",
            target: ">= 0.97",
            value: "not run".into(),
            pass: None,
        },
    });

    let lat: Vec<&RunResult> = results
        .iter()
        .filter(|r| r.scenario == "noise_bursts")
        .collect();
    let lat_bad = lat
        .iter()
        .filter(|r| {
            !r.measured_latency
                .is_some_and(|m| (m.lag as i64 - r.latency as i64).abs() <= 1)
        })
        .count();
    g.push(Gate {
        name: "Latency (suite rate, scenario voice range)",
        target: "measured == reported ± 1 sample",
        value: if lat.is_empty() {
            "not run (see `skato_eval latency`)".into()
        } else {
            format!("{} of {} runs off", lat_bad, lat.len())
        },
        pass: (!lat.is_empty()).then_some(lat_bad == 0),
    });

    if !checks.is_empty() {
        let classic = checks
            .iter()
            .filter(|c| c.engine == EngineKind::Classic)
            .map(|c| c.transparency_max_diff)
            .fold(0.0, f64::max);
        let psola = checks
            .iter()
            .filter(|c| c.engine == EngineKind::Natural)
            .map(|c| c.transparency_snr_db)
            .fold(f64::INFINITY, f64::min);
        g.push(Gate {
            name: "Transparency (amount 0)",
            target: "Classic exact (≤ 1e-6); PSOLA seg. SNR ≥ 25 dB after wander alignment",
            value: format!("Classic max diff {classic:.2e}; PSOLA min SNR {psola:.1} dB"),
            pass: Some(classic <= 1e-6 && (psola >= 25.0 || psola.is_infinite())),
        });
        let inv = checks
            .iter()
            .map(|c| c.block_invariance_max_diff)
            .fold(0.0, f64::max);
        let finite = results.iter().all(|r| r.finite);
        g.push(Gate {
            name: "RT safety: block-size invariance, finite output",
            target: "≤ 1e-6 across {1,7,32,64,128,480,4096,random}; no NaN/Inf",
            value: format!("max diff {inv:.2e}; all outputs finite: {finite}"),
            pass: Some(inv <= 1e-6 && finite),
        });
        g.push(content_wander_gate(checks));
    }
    g.push(Gate {
        name: "CPU (p99.9 block time)",
        target: "Live HQ < 5 % @128, < 10 % @32; Max < 25 %",
        value: "run `skato_eval rt-bench` (release build)".into(),
        pass: None,
    });
    g
}

fn opt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.prec$}"),
        _ => String::new(),
    }
}

/// Largest per-onset value in ms (None if no onset has it).
fn worst_of(
    onsets: &[metrics::OnsetTiming],
    f: impl Fn(&metrics::OnsetTiming) -> Option<f64>,
) -> Option<f64> {
    onsets
        .iter()
        .filter_map(f)
        .reduce(f64::max)
        .map(|v| v * 1000.0)
}

/// One CSV row per render.
pub fn write_csv(path: &Path, results: &[RunResult]) -> Result<()> {
    let mut s = String::from(
        "scenario,mode,tier,engine,retune_ms,latency,measured_latency,latency_peak,tune_frames,tune_unvoiced,tune_ambiguous,tune_moving,tune_median_c,tune_p95_c,tune_mean_c,within5_pct,within10_pct,within25_pct,settle_median_ms,settle_p90_ms,settle_max_ms,settle_unsettled,vib_extent_err_pct,vib_centre_median_c,vib_rate_err_hz,jitter_added_c,jitter_out_c,jitter_in_c,voicing_f1,voicing_precision,voicing_recall,clicks,click_max_excess_db,lsd_mean_db,finite,render_ms,settle_onsets,settle_fail30,settle_detect_max_ms,settle_switch_max_ms,settle_converge_max_ms\n",
    );
    for r in results {
        let t = r.tuning;
        let st = r.settling.as_ref();
        let v = r.vibrato.as_ref();
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1},{},{},{},{},{}",
            r.scenario,
            mode_name(r.mode),
            tier_name(r.tier),
            engine_name(r.engine),
            r.retune_ms,
            r.latency,
            r.measured_latency
                .map_or(String::new(), |m| m.lag.to_string()),
            opt(r.measured_latency.map(|m| m.peak), 4),
            t.map_or(String::new(), |t| t.frames.to_string()),
            t.map_or(String::new(), |t| t.unvoiced.to_string()),
            t.map_or(String::new(), |t| t.ambiguous.to_string()),
            t.map_or(String::new(), |t| t.moving.to_string()),
            opt(t.map(|t| t.median_abs), 3),
            opt(t.map(|t| t.p95_abs), 3),
            opt(t.map(|t| t.mean), 3),
            opt(t.map(|t| t.within_5), 1),
            opt(t.map(|t| t.within_10), 1),
            opt(t.map(|t| t.within_25), 1),
            opt(st.map(|s| s.median_s * 1000.0), 1),
            opt(st.map(|s| s.p90_s * 1000.0), 1),
            opt(st.map(|s| s.max_s * 1000.0), 1),
            st.map_or(String::new(), |s| (s.per_note.len() - s.settled)
                .to_string()),
            opt(v.map(|v| v.worst_extent_ratio_error * 100.0), 1),
            opt(v.map(|v| v.centre_error_median), 2),
            opt(v.map(|v| v.worst_rate_error_hz), 3),
            opt(r.jitter.map(|j| j.added_rms), 3),
            opt(r.jitter.map(|j| j.out_rms), 3),
            opt(r.jitter.map(|j| j.in_rms), 3),
            opt(r.voicing.map(|v| v.f1), 4),
            opt(r.voicing.map(|v| v.precision), 4),
            opt(r.voicing.map(|v| v.recall), 4),
            r.clicks.clicks,
            opt(Some(r.clicks.max_excess_db), 1),
            opt(r.lsd.map(|l| l.mean_db), 2),
            r.finite,
            r.render_s * 1000.0,
            st.map_or(String::new(), |s| s.onsets.len().to_string()),
            st.map_or(String::new(), |s| s
                .onsets
                .iter()
                .filter(|o| o.settle_s.is_none_or(|v| v >= SETTLING_LIMIT_S))
                .count()
                .to_string()),
            opt(st.and_then(|s| worst_of(&s.onsets, |o| o.detect_s)), 1),
            opt(st.and_then(|s| worst_of(&s.onsets, |o| o.switch_s)), 1),
            opt(
                st.and_then(|s| worst_of(&s.onsets, |o| o.convergence_s())),
                1
            ),
        );
    }
    std::fs::write(path, s)?;
    Ok(())
}

/// "Latency gauge" report section: what a performer monitoring through the
/// plugin hears, per mode, engine and voice range.
fn gauge_section(s: &mut String, gauge: &[GaugeRow]) {
    if gauge.is_empty() {
        return;
    }
    s.push_str("\n## Latency gauge (amount 0, HQ)\n\n");
    s.push_str("*D* is the reported plugin latency (what the host compensates; exact for unvoiced audio). *Content delay* is how late the output actually carries the sung input on voiced 20 ms segments of each scenario's notes: D plus the PSOLA content wander (R§10.3 / R§10.6), measured by fractional cross-correlation of the amount-0 render. With PDC on, a DAW lines the output up by D, so the wander is the residual misalignment; when monitoring live through the plugin, the content delay is what the singer hears.\n\n");
    s.push_str("| Mode | Engine | Voice range (scenario) | D (ms) | Content delay mean / median / p95 (ms) | Max \\|wander\\| ms (periods) | Segments (rejected) |\n|---|---|---|---|---|---|---|\n");
    for g in gauge {
        let c = &g.content;
        let _ = writeln!(
            s,
            "| {} | {} | {} ({}) | {:.2} | {} / {} / {} | {} ({}) | {} ({}) |",
            mode_name(g.mode),
            engine_name(g.engine),
            range_name(g.range),
            g.scenario,
            c.reported_ms(),
            opt(Some(c.mean_delay_ms()), 2),
            opt(Some(c.delay_ms(50.0)), 2),
            opt(Some(c.delay_ms(95.0)), 2),
            opt(Some(c.max_abs_wander_ms()), 2),
            opt(Some(c.max_abs_wander_periods()), 2),
            c.segments,
            c.rejected,
        );
    }
    // One number per mode: the mean heard delay over every measured voiced
    // segment of the default engine in that mode.
    s.push_str("\nAverage heard delay on voiced audio per mode (default engine, all gauge scenarios pooled): ");
    let mut parts = Vec::new();
    for mode in MODES {
        let engine = default_engine(mode);
        let (mut sum, mut n) = (0.0, 0usize);
        for g in gauge
            .iter()
            .filter(|g| g.mode == mode && g.engine == engine)
        {
            let c = &g.content;
            for w in &c.wander {
                sum += (f64::from(c.reported) + w) / c.sample_rate * 1e3;
                n += 1;
            }
        }
        if n > 0 {
            parts.push(format!(
                "{} ({}) {:.2} ms",
                mode_name(mode),
                engine_name(engine),
                sum / n as f64
            ));
        }
    }
    let _ = writeln!(s, "{}.", parts.join(", "));
}

/// "Onset latency" report section: the per-onset settling breakdown of the
/// Live HQ retune-0 runs (the DESIGN §4 settling gate's runs).
fn onset_section(s: &mut String, results: &[RunResult]) {
    let runs: Vec<&RunResult> = results
        .iter()
        .filter(|r| {
            r.mode == LatencyMode::Live
                && r.tier == QualityTier::Hq
                && r.retune_ms == 0.0
                && r.settling.as_ref().is_some_and(|st| !st.onsets.is_empty())
        })
        .collect();
    if runs.is_empty() {
        return;
    }
    s.push_str("\n## Onset latency breakdown (Live HQ, retune 0)\n\n");
    s.push_str("All times in ms. *Decision point*: when the sung pitch passed the midpoint to the neighbouring legal note plus the note-decision tolerance (from the note start; 0 = decidable at once). Then, from the decision point: *detected* = the engine's detected pitch passes it (telemetry), *switch* = the engine's target becomes the note, *settled* = output within 10 c for 20 ms (the DESIGN §4 quantity, gate < 30 ms), *convergence* = settled − switch.\n\n");
    s.push_str("| Scenario | Onset | Decision point | Detected | Switch | Settled | Convergence |\n|---|---|---|---|---|---|---|\n");
    for r in runs {
        let Some(st) = &r.settling else { continue };
        for (i, o) in st.onsets.iter().enumerate() {
            let flag = if o.settle_s.is_none_or(|v| v >= SETTLING_LIMIT_S) {
                " **≥ 30**"
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "| {} | {i} | {} | {} | {} | {}{flag} | {} |",
                r.scenario,
                ms(Some(o.clock_s)),
                ms(o.detect_s),
                ms(o.switch_s),
                ms(Some(o.settle_s.unwrap_or(f64::INFINITY))),
                ms(o.convergence_s()),
            );
        }
    }
}

/// Markdown report: gates first, then one table per scenario.
pub fn write_report(
    path: &Path,
    opts: &SuiteOptions,
    results: &[RunResult],
    checks: &[ModeCheck],
    gates: &[Gate],
    gauge: &[GaugeRow],
) -> Result<()> {
    let mut s = String::new();
    let _ = writeln!(s, "# Skatotune evaluation report\n");
    let _ = writeln!(
        s,
        "Sample rate {} Hz, block pattern `{}`, {} renders. Generated by `skato_eval suite`; metrics per DESIGN §3.7/§4 and R§10.3. \
         Pitch is measured on the latency-compensated output with the offline reference tracker.\n",
        opts.sample_rate,
        opts.blocks.label(),
        results.len()
    );
    let _ = writeln!(
        s,
        "## Gates (DESIGN §4)\n\n| Gate | Target | Measured | Result |\n|---|---|---|---|"
    );
    for g in gates {
        let res = match g.pass {
            Some(true) => "PASS",
            Some(false) => "**FAIL**",
            None => "n/a",
        };
        let _ = writeln!(s, "| {} | {} | {} | {} |", g.name, g.target, g.value, res);
    }
    gauge_section(&mut s, gauge);
    onset_section(&mut s, results);
    if !checks.is_empty() {
        let _ = writeln!(
            s,
            "\n## Per-mode checks (HQ, melody_male)\n\n| Mode | Engine | Amount-0 max diff | Amount-0 seg. SNR (dB) | Block invariance max diff | D (ms) | Voiced content delay median / p95 (ms) | Max \\|wander\\| (periods) |\n|---|---|---|---|---|---|---|---|"
        );
        for c in checks {
            let d = &c.content_delay;
            let _ = writeln!(
                s,
                "| {} | {} | {:.2e} | {:.1} | {:.2e} | {:.2} | {} / {} | {} |",
                mode_name(c.mode),
                engine_name(c.engine),
                c.transparency_max_diff,
                c.transparency_snr_db,
                c.block_invariance_max_diff,
                d.reported_ms(),
                opt(Some(d.delay_ms(50.0)), 2),
                opt(Some(d.delay_ms(95.0)), 2),
                opt(Some(d.max_abs_wander_periods()), 2),
            );
        }
    }
    let mut names: Vec<&str> = results.iter().map(|r| r.scenario).collect();
    names.dedup();
    for name in names {
        let desc = scenarios::build(name, opts.sample_rate).map_or("", |s| s.description);
        let _ = writeln!(s, "\n## {name}\n\n{desc}\n");
        let _ = writeln!(
            s,
            "| Mode | Tier | Engine | Retune | D | Median c | p95 c | ≤5/10/25 c % | Settle med/max ms | Vib ext err % | Jitter+ c | F1 | Clicks | LSD dB |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        );
        for r in results.iter().filter(|r| r.scenario == name) {
            let t = r.tuning;
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                mode_name(r.mode),
                tier_name(r.tier),
                engine_name(r.engine),
                r.retune_ms,
                r.latency,
                r.measured_latency
                    .map_or(String::new(), |m| format!(" (meas. {})", m.lag)),
                opt(t.map(|t| t.median_abs), 2),
                opt(t.map(|t| t.p95_abs), 2),
                t.map_or(String::new(), |t| format!(
                    "{:.0}/{:.0}/{:.0}",
                    t.within_5, t.within_10, t.within_25
                )),
                r.settling.as_ref().map_or(String::new(), |st| format!(
                    "{}/{}",
                    opt(Some(st.median_s * 1000.0), 1),
                    opt(Some(st.max_s * 1000.0), 1)
                )),
                opt(
                    r.vibrato
                        .as_ref()
                        .map(|v| v.worst_extent_ratio_error * 100.0),
                    1
                ),
                opt(r.jitter.map(|j| j.added_rms), 2),
                opt(r.voicing.map(|v| v.f1), 3),
                r.clicks.clicks,
                opt(r.lsd.map(|l| l.mean_db), 2),
            );
        }
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::OnsetTiming;

    fn result_with(scenario: &'static str, settle: &[Option<f64>]) -> RunResult {
        let onsets: Vec<OnsetTiming> = settle
            .iter()
            .map(|s| OnsetTiming {
                settle_s: *s,
                ..OnsetTiming::default()
            })
            .collect();
        let mut sorted: Vec<f64> = settle.iter().map(|s| s.unwrap_or(f64::INFINITY)).collect();
        sorted.sort_by(f64::total_cmp);
        RunResult {
            scenario,
            mode: LatencyMode::Live,
            tier: QualityTier::Hq,
            engine: EngineKind::Natural,
            retune_ms: 0.0,
            latency: 213,
            measured_latency: None,
            tuning: None,
            settling: Some(SettlingStats {
                per_note: settle.to_vec(),
                settled: settle.iter().flatten().count(),
                unsettled: settle.iter().filter(|s| s.is_none()).count(),
                // What the old gate looked at: the median of the settled onsets.
                median_s: crate::dsp::median(&settle.iter().flatten().copied().collect::<Vec<_>>()),
                p90_s: f64::NAN,
                max_s: sorted.last().copied().unwrap_or(f64::NAN),
                onsets,
            }),
            vibrato: None,
            jitter: None,
            voicing: None,
            clicks: ClickStats::default(),
            lsd: None,
            finite: true,
            render_s: 0.0,
        }
    }

    fn settling_pass(results: &[RunResult]) -> Option<bool> {
        gates(results, &[])
            .into_iter()
            .find(|g| g.name.starts_with("Settling"))
            .and_then(|g| g.pass)
    }

    #[test]
    fn settling_gate_applies_to_every_onset() {
        // EV-1: two of six onsets miss 30 ms but the median is 9 ms. The old
        // gate (median < 30 ms) passed this; DESIGN §4 has no median.
        let slow = result_with(
            "melody_male",
            &[
                Some(0.067),
                Some(0.001),
                Some(0.005),
                Some(0.009),
                Some(0.027),
                Some(0.039),
            ],
        );
        assert_eq!(settling_pass(&[slow]), Some(false));
        // An onset that never settles is a failure, not a dropped sample.
        let never = result_with("melody_female", &[Some(0.010), None, Some(0.012)]);
        assert_eq!(settling_pass(&[never]), Some(false));
        let fast = result_with("high_soprano", &[Some(0.016), Some(0.015), Some(0.029)]);
        assert_eq!(settling_pass(std::slice::from_ref(&fast)), Some(true));
        let g = settling_gate(std::iter::once(&fast));
        assert!(g.value.contains("0 of 3 onsets"), "{}", g.value);
    }
}
