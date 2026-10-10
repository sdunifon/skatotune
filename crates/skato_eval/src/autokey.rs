//! Auto-Key evaluation (#6, docs/research/auto-key.md).
//!
//! Two kinds of material run through the real control law (Live HQ), whose
//! Auto-Key detector is snapshotted every 0.25 s:
//!
//! - **Synthetic melodies** in all 36 major, natural-minor and
//!   harmonic-minor keys: stepwise tunes weighted towards the tonic, fifth
//!   and third, phrases ending mostly on the tonic, occasional chromatic
//!   passing notes, scoops, vibrato, per-note jitter and a per-take singer
//!   detune of up to ±30 c. The truth is the generating key.
//! - **vocadito** (real singing, the note-intent DEV/TEST split). The
//!   truth is the *oracle set*: the candidate set that covers the most
//!   annotated note time of both annotators (singer offset removed), the
//!   smaller set on a tie. The vocadito evaluation's own Krumhansl-Kessler
//!   key of annotator 1's notes is reported alongside.
//!
//! The scoring knobs of [`KeyConfig`] are swept offline on the recorded
//! evidence (the accumulation knobs, `tau_s` and `offset_track`, by
//! replaying). Everything is chosen on DEV; TEST is reported once.

use std::path::{Path, PathBuf};

use skato_control::autokey::{Family, Profile, profile_key};
use skato_control::{Corrector, CorrectorConfig, KeyConfig, KeyDetector, KeyName, KeyReading};
use skato_core::{LatencyMode, PitchEstimate, QualityTier, ScaleKind, Settings, cents_to_hz, hop_for_rate};

use crate::note_intent::cache::{self, Config, Fingerprints};
use crate::note_intent::data::{FileCtx, Split, Splits};
use crate::note_intent::{load_contexts, parallel};
use crate::rng::Rng;
use crate::vocadito::data as vdata;

/// Snapshot interval, seconds.
const SNAP_S: f32 = 0.25;
/// Confidence at which a proposal counts as confident (the Detect source's
/// confirmation threshold).
const CONFIDENT: f32 = 0.8;

/// One snapshot of the detector.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    /// Input time, seconds.
    pub t: f32,
    pub hist: [f32; 12],
    pub finals: [f32; 12],
    /// The detector's own reading (its configured tuning, incl. confirmation).
    pub reading: KeyReading,
}

/// A take with its truth and recorded snapshots.
#[derive(Clone, Debug)]
pub struct Take {
    pub name: String,
    /// The truth set (synthetic: the generating key's; vocadito: the oracle set).
    pub truth_mask: u16,
    /// The truth key name (synthetic: generating key; vocadito: KK of A1).
    pub truth_name: Option<KeyName>,
    pub dur_s: f32,
    pub snaps: Vec<Snap>,
    /// vocadito: annotated note time per pitch class (offset removed).
    pub ann_hist: Option<[f64; 12]>,
    /// Mean confidence, steadiness and in-tune weight of the voiced hops,
    /// and the voiced time in seconds.
    pub weights: (f32, f32, f32, f32),
}

// ---------------------------------------------------------------------------
// Synthetic melodies
// ---------------------------------------------------------------------------

/// The keys of the synthetic set: every root in major, natural and harmonic minor.
pub fn synth_keys() -> Vec<KeyName> {
    let mut v = Vec::new();
    for kind in [ScaleKind::Major, ScaleKind::NaturalMinor, ScaleKind::HarmonicMinor] {
        for r in 0..12 {
            v.push(KeyName::new(r, kind));
        }
    }
    v
}

fn voiced_est(cents: f32, confidence: f32, sr: f32) -> PitchEstimate {
    let f0 = cents_to_hz(cents);
    PitchEstimate {
        sample_pos: 0,
        info_lag: 0.0,
        f0_hz: f0,
        cents,
        period: sr / f0,
        confidence,
        voiced_prob: 0.95,
        voiced: true,
        sibilant: false,
        level_db: -20.0,
    }
}

fn unvoiced_est() -> PitchEstimate {
    PitchEstimate { level_db: -80.0, ..PitchEstimate::default() }
}

/// Weight of a scale degree (semitones above the tonic) in the tunes.
fn degree_weight(iv: u8) -> f64 {
    match iv {
        0 => 3.0,
        7 => 2.5,
        3 | 4 => 2.0,
        _ => 1.0,
    }
}

/// Pick an index with probability proportional to `w`.
fn pick(rng: &mut Rng, w: &[f64]) -> usize {
    let total: f64 = w.iter().sum();
    let mut x = rng.uniform() * total;
    for (i, &wi) in w.iter().enumerate() {
        if x < wi {
            return i;
        }
        x -= wi;
    }
    w.len() - 1
}

/// A synthetic take of `dur_s` seconds in `key` (per-hop estimates at `sr`).
pub fn synth_take(key: KeyName, seed: u64, dur_s: f32, sr: f32) -> Vec<PitchEstimate> {
    let mut rng = Rng::new(seed);
    let hop_s = hop_for_rate(sr) as f32 / sr;
    let iv = key.kind.intervals();
    let centre = 52 + (rng.uniform() * 20.0) as i32;
    let detune = rng.range(-30.0, 30.0) as f32;
    let scale_notes: Vec<i32> = ((centre - 10)..=(centre + 10))
        .filter(|n| iv.contains(&(((n - key.root as i32).rem_euclid(12)) as u8)))
        .collect();
    let deg = |n: i32| ((n - key.root as i32).rem_euclid(12)) as u8;
    let mut cur = *scale_notes.iter().min_by_key(|n| (*n - centre).abs()).unwrap_or(&centre);
    let mut prev_dir = 1.0f32;
    let mut out = Vec::new();
    let push_unvoiced = |out: &mut Vec<PitchEstimate>, s: f32| {
        for _ in 0..(s / hop_s).round() as usize {
            out.push(unvoiced_est());
        }
    };
    push_unvoiced(&mut out, 0.3);
    while (out.len() as f32) * hop_s < dur_s {
        let phrase = 4 + (rng.uniform() * 6.0) as usize;
        for j in 0..phrase {
            let last = j + 1 == phrase;
            let (note, dur) = if last {
                // Phrase ends: tonic, fifth or third, near the current note.
                let ends: Vec<i32> = scale_notes.iter().copied().filter(|n| (n - cur).abs() <= 7).collect();
                let w: Vec<f64> = ends
                    .iter()
                    .map(|&n| match deg(n) {
                        0 => 5.0,
                        7 => 2.0,
                        3 | 4 => 2.0,
                        _ => 0.25,
                    })
                    .collect();
                (if ends.is_empty() { cur } else { ends[pick(&mut rng, &w)] }, rng.range(0.6, 1.2) as f32)
            } else if rng.uniform() < 0.03 {
                // Chromatic passing note.
                let up = if rng.uniform() < 0.5 { 1 } else { -1 };
                let n = cur + up;
                (if iv.contains(&deg(n)) { cur - up } else { n }, rng.range(0.12, 0.2) as f32)
            } else {
                let w: Vec<f64> = scale_notes
                    .iter()
                    .map(|&n| {
                        let d = (n - cur).abs();
                        if d > 7 {
                            0.0
                        } else {
                            degree_weight(deg(n)) * (-(d as f64) / 2.5).exp() * if d == 0 { 0.6 } else { 1.0 }
                        }
                    })
                    .collect();
                (scale_notes[pick(&mut rng, &w)], (0.15f64 * (0.7f64 / 0.15).powf(rng.uniform())) as f32)
            };
            if note != cur {
                prev_dir = (note - cur).signum() as f32;
            }
            cur = note;
            // Render the note.
            let base = 100.0 * note as f32 + detune + 10.0 * rng.gaussian() as f32;
            let scoop = if rng.uniform() < 0.35 { -prev_dir * rng.range(60.0, 150.0) as f32 } else { 0.0 };
            let scoop_s = rng.range(0.05, 0.09) as f32;
            let vib = dur >= 0.35;
            let (rate, ext) = (rng.range(5.0, 6.5) as f32, rng.range(20.0, 60.0) as f32);
            let conf = rng.range(0.85, 0.95) as f32;
            let n_hops = (dur / hop_s).round() as usize;
            for k in 0..n_hops {
                let t = k as f32 * hop_s;
                let mut c = base;
                if t < scoop_s {
                    c += scoop * (1.0 - t / scoop_s);
                }
                if vib && t > 0.15 {
                    let fade = ((t - 0.15) / 0.1).min(1.0);
                    c += fade * ext * (std::f32::consts::TAU * rate * t).sin();
                }
                out.push(voiced_est(c, conf, sr));
            }
            if !last && rng.uniform() < 0.25 {
                push_unvoiced(&mut out, rng.range(0.03, 0.06) as f32);
            }
        }
        push_unvoiced(&mut out, rng.range(0.25, 0.9) as f32);
    }
    out
}

// ---------------------------------------------------------------------------
// Running the detector
// ---------------------------------------------------------------------------

/// Run the control law over `est` and snapshot its Auto-Key detector.
pub fn record(est: &[PitchEstimate], cfg: CorrectorConfig, settings: &Settings, key: KeyConfig) -> (Vec<Snap>, (f32, f32, f32, f32)) {
    let mut c = Corrector::new(cfg);
    c.set_settings(settings);
    c.set_key_config(key);
    let hop_s = cfg.hop as f32 / cfg.sample_rate;
    let every = ((SNAP_S / hop_s).round() as usize).max(1);
    let mut snaps = Vec::with_capacity(est.len() / every + 1);
    for (i, e) in est.iter().enumerate() {
        c.process_hop(e, None);
        if (i + 1) % every == 0 {
            let (hist, finals) = c.key_detector().evidence();
            snaps.push(Snap { t: (i + 1) as f32 * hop_s, hist, finals, reading: c.key_reading() });
        }
    }
    let (a, b, w, n) = c.key_detector().weight_means();
    (snaps, (a, b, w, n * hop_s))
}

fn live_hq(sr: f32) -> (CorrectorConfig, Settings) {
    (
        CorrectorConfig { sample_rate: sr, hop: hop_for_rate(sr), latency_mode: LatencyMode::Live, lookahead_hops: 0 },
        Settings { tier: QualityTier::Hq, ..Settings::default() },
    )
}

/// Seeds: DEV takes use 0..n, TEST takes 1000..1000+n.
fn seed(key: KeyName, take: usize, test: bool) -> u64 {
    (u64::from(key.root) << 8 | key.kind as u64) << 20 | (take as u64 + if test { 1000 } else { 0 })
}

/// The synthetic takes of a split, recorded with `key`.
pub fn synth_takes(split: Split, per_key: usize, dur_s: f32, key: KeyConfig, threads: usize) -> Vec<Take> {
    let sr = 48_000.0;
    let (cfg, settings) = live_hq(sr);
    let mut jobs = Vec::new();
    for k in synth_keys() {
        for t in 0..per_key {
            match split {
                Split::Dev => jobs.push((k, t, false)),
                Split::Test => jobs.push((k, t, true)),
                Split::All => {
                    jobs.push((k, t, false));
                    jobs.push((k, t, true));
                }
            }
        }
    }
    parallel(jobs.len(), threads, |i| {
        let (k, t, test) = jobs[i];
        let est = synth_take(k, seed(k, t, test), dur_s, sr);
        let (snaps, weights) = record(&est, cfg, &settings, key);
        Take {
            name: format!("{}{} {}#{t}", NOTE_NAMES[k.root as usize], short(k.kind), if test { "test" } else { "dev" }),
            truth_mask: k.mask,
            truth_name: Some(k),
            dur_s: est.len() as f32 * cfg.hop as f32 / sr,
            snaps,
            ann_hist: None,
            weights,
        }
    })
}

/// Annotated note time per pitch class, both annotators averaged, with the
/// singer's offset removed.
fn annotation_histogram(ctx: &FileCtx) -> [f64; 12] {
    let mut h = [0.0; 12];
    for (notes, w) in [(&ctx.a1, 0.5), (&ctx.a2, 0.5)] {
        for n in notes.iter() {
            let pc = (((n.cents() - ctx.offset) / 100.0).round() as i64).rem_euclid(12) as usize;
            h[pc] += w * n.dur_s;
        }
    }
    h
}

/// The candidate sets Auto-Key can propose (every family).
fn all_candidate_sets() -> Vec<(u16, usize)> {
    let mut v = Vec::new();
    for (fi, fam) in Family::ALL.iter().enumerate() {
        let generator = match fam {
            Family::Diatonic => ScaleKind::Major,
            Family::HarmonicMinor => ScaleKind::HarmonicMinor,
            Family::MelodicMinor => ScaleKind::MelodicMinor,
            Family::Pentatonic => ScaleKind::MajorPentatonic,
        };
        for r in 0..12 {
            v.push((generator.mask(r), fi));
        }
    }
    v
}

/// Share of the histogram inside `mask`.
fn coverage(h: &[f64; 12], mask: u16) -> f64 {
    let total: f64 = h.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    (0..12).filter(|pc| mask & (1 << pc) != 0).map(|pc| h[pc]).sum::<f64>() / total
}

/// The oracle set: most annotated time covered, then the smaller set, then
/// the more common family.
pub fn oracle_set(h: &[f64; 12]) -> u16 {
    let mut best = (f64::NEG_INFINITY, 0u16, usize::MAX);
    for (mask, fi) in all_candidate_sets() {
        let c = coverage(h, mask);
        let better = c > best.0 + 1e-9
            || ((c - best.0).abs() <= 1e-9 && (mask.count_ones() < best.1.count_ones() || (mask.count_ones() == best.1.count_ones() && fi < best.2)));
        if better {
            best = (c, mask, fi);
        }
    }
    best.1
}

/// The vocadito takes of a split (Live HQ detector streams from the
/// note-intent cache, built when missing).
pub fn vocadito_takes(
    data: &Path,
    cache_dir: &Path,
    split: Split,
    key: KeyConfig,
    threads: usize,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<Vec<Take>, String> {
    let splits = Splits::committed();
    let files = splits.files(split);
    let cfgs = [Config::LIVE_HQ];
    let fp = Fingerprints::compute(&cfgs);
    cache::prepare(data, cache_dir, &fp, &files, &cfgs, threads, progress)?;
    let ctxs = load_contexts(data, cache_dir, &files, &splits, threads)?;
    let takes = parallel(ctxs.len(), threads, |i| -> Result<Take, String> {
        let ctx = &ctxs[i];
        let stream = cache::load_stream(cache_dir, &fp, Config::LIVE_HQ, ctx.id)?;
        let h = stream.h;
        let cfg = CorrectorConfig { sample_rate: h.sample_rate, hop: h.hop as usize, latency_mode: h.mode, lookahead_hops: h.lookahead_hops as usize };
        let settings = Settings { tier: QualityTier::Hq, ..Settings::default() };
        let ann = annotation_histogram(ctx);
        let (snaps, weights) = record(&stream.est, cfg, &settings, key);
        Ok(Take {
            name: format!("vocadito_{}", ctx.id),
            truth_mask: oracle_set(&ann),
            truth_name: vdata::estimate_key(&ctx.a1).map(|k| KeyName::new(k.key, k.scale)),
            dur_s: stream.est.len() as f32 * h.hop as f32 / h.sample_rate,
            snaps,
            ann_hist: Some(ann),
            weights,
        })
    });
    takes.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// How a set of takes scores under one tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct Score {
    pub takes: usize,
    /// Proposal set correct at 5 / 10 / 20 / 30 s and at the end of the take.
    pub acc: [f64; 5],
    /// Tonic and scale named right at the end (where a truth name exists).
    pub name_acc: f64,
    /// Share of snapshots with a confident proposal, and how many of those
    /// are right.
    pub confident_share: f64,
    pub confident_precision: f64,
    /// Time until the proposal is right and confident for the rest of the
    /// take: median and 90th percentile over the takes that get there, and
    /// how many never do.
    pub ttd_median: f64,
    pub ttd_p90: f64,
    pub never: usize,
    /// vocadito: annotated note time inside the final proposal's set, and
    /// inside the oracle set.
    pub ann_coverage: f64,
    pub oracle_coverage: f64,
    /// Takes with a confirmed key at the end, and how many are right (the
    /// detector's own tuning only).
    pub confirmed: usize,
    pub confirmed_right: usize,
}

pub const ACC_AT: [f32; 4] = [5.0, 10.0, 20.0, 30.0];

/// The proposal at each snapshot under `est` (a detector carrying the tuning).
fn proposals(take: &Take, det: Option<&KeyDetector>) -> Vec<Option<(KeyName, f32)>> {
    take.snaps
        .iter()
        .map(|s| match det {
            Some(d) => d.estimate(&s.hist, &s.finals),
            None => s.reading.proposal.map(|p| (p, s.reading.confidence)),
        })
        .collect()
}

/// Score `takes` with the tuning of `det` (None = the recorded readings).
pub fn score(takes: &[Take], det: Option<&KeyDetector>) -> Score {
    let mut sc = Score { takes: takes.len(), ..Score::default() };
    let mut ttds = Vec::new();
    let (mut conf_n, mut conf_right, mut snaps_n) = (0usize, 0usize, 0usize);
    let (mut named, mut named_right) = (0usize, 0usize);
    let mut cov = (0.0, 0.0, 0usize);
    for take in takes {
        let props = proposals(take, det);
        let right = |p: &Option<(KeyName, f32)>| p.is_some_and(|(k, _)| k.mask == take.truth_mask);
        for (j, &t) in ACC_AT.iter().enumerate() {
            let k = take.snaps.iter().rposition(|s| s.t <= t);
            if let Some(k) = k {
                sc.acc[j] += f64::from(u8::from(right(&props[k])));
            }
        }
        if let Some(last) = props.last() {
            sc.acc[4] += f64::from(u8::from(right(last)));
            if let (Some(truth), Some((k, _))) = (take.truth_name, last) {
                named += 1;
                named_right += usize::from(truth == *k);
            }
            if let (Some(h), Some((k, _))) = (&take.ann_hist, last) {
                cov.0 += coverage(h, k.mask);
                cov.1 += coverage(h, take.truth_mask);
                cov.2 += 1;
            }
        }
        for p in &props {
            snaps_n += 1;
            if p.is_some_and(|(_, c)| c >= CONFIDENT) {
                conf_n += 1;
                conf_right += usize::from(right(p));
            }
        }
        // Time to detect: the last snapshot that isn't right and confident.
        let ok = |p: &Option<(KeyName, f32)>| right(p) && p.is_some_and(|(_, c)| c >= CONFIDENT);
        match props.iter().rposition(|p| !ok(p)) {
            None if !props.is_empty() => ttds.push(take.snaps[0].t as f64),
            Some(k) if k + 1 < props.len() => ttds.push(take.snaps[k + 1].t as f64),
            _ => sc.never += 1,
        }
        if let Some(c) = take.snaps.last().and_then(|s| s.reading.confirmed) {
            sc.confirmed += 1;
            sc.confirmed_right += usize::from(c.mask == take.truth_mask);
        }
    }
    let n = takes.len().max(1) as f64;
    for a in sc.acc.iter_mut() {
        *a /= n;
    }
    sc.name_acc = if named > 0 { named_right as f64 / named as f64 } else { f64::NAN };
    sc.confident_share = conf_n as f64 / snaps_n.max(1) as f64;
    sc.confident_precision = if conf_n > 0 { conf_right as f64 / conf_n as f64 } else { f64::NAN };
    ttds.sort_by(f64::total_cmp);
    let q = |p: f64| if ttds.is_empty() { f64::NAN } else { ttds[((ttds.len() - 1) as f64 * p).round() as usize] };
    sc.ttd_median = q(0.5);
    sc.ttd_p90 = q(0.9);
    if cov.2 > 0 {
        sc.ann_coverage = cov.0 / cov.2 as f64;
        sc.oracle_coverage = cov.1 / cov.2 as f64;
    }
    sc
}

/// Classic Krumhansl-Schmuckler on the same evidence: the best of 24
/// major/minor keys (its set) at each snapshot, no confidence.
pub fn score_profile(takes: &[Take], profile: Profile) -> Score {
    let mut sc = Score { takes: takes.len(), ..Score::default() };
    let (mut named, mut named_right) = (0usize, 0usize);
    let mut cov = (0.0, 0.0, 0usize);
    for take in takes {
        let at = |s: &Snap| (s.hist.iter().sum::<f32>() > 1e-3).then(|| profile_key(&s.hist, profile).0);
        for (j, &t) in ACC_AT.iter().enumerate() {
            if let Some(s) = take.snaps.iter().rev().find(|s| s.t <= t) {
                sc.acc[j] += f64::from(u8::from(at(s).is_some_and(|k| k.mask == take.truth_mask)));
            }
        }
        if let Some(k) = take.snaps.last().and_then(at) {
            sc.acc[4] += f64::from(u8::from(k.mask == take.truth_mask));
            if let Some(truth) = take.truth_name {
                named += 1;
                named_right += usize::from(truth == k);
            }
            if let Some(h) = &take.ann_hist {
                cov.0 += coverage(h, k.mask);
                cov.1 += coverage(h, take.truth_mask);
                cov.2 += 1;
            }
        }
    }
    let n = takes.len().max(1) as f64;
    for a in sc.acc.iter_mut() {
        *a /= n;
    }
    sc.name_acc = if named > 0 { named_right as f64 / named as f64 } else { f64::NAN };
    sc.confident_share = f64::NAN;
    sc.confident_precision = f64::NAN;
    sc.ttd_median = f64::NAN;
    sc.ttd_p90 = f64::NAN;
    if cov.2 > 0 {
        sc.ann_coverage = cov.0 / cov.2 as f64;
        sc.oracle_coverage = cov.1 / cov.2 as f64;
    }
    sc
}

/// The DEV objective: equal parts synthetic accuracy at 10 s and 30 s,
/// precision of confident proposals, and vocadito accuracy at the end.
pub fn objective(syn: &Score, voc: &Score) -> f64 {
    let p = if syn.confident_precision.is_nan() { 0.0 } else { syn.confident_precision };
    0.25 * syn.acc[1] + 0.25 * syn.acc[3] + 0.25 * p + 0.25 * voc.acc[4]
}

/// The scoring grid swept on DEV.
pub fn scoring_grid(base: KeyConfig) -> Vec<KeyConfig> {
    let mut v = Vec::new();
    for eps in [0.03, 0.06, 0.1] {
        for rate in [1.0, 2.0, 4.0] {
            for prior in [[1.0, 0.3, 0.1, 0.0], [1.0, 0.3, 0.1, 0.3], [1.0, 0.0, 0.0, 0.0], [1.0, 0.1, 0.03, 0.0]] {
                for tonal_bias in [0.0, 1.0, 3.0] {
                    for profile in Profile::ALL {
                        for final_weight in [0.0, 0.2, 0.5] {
                            for sung_share in [0.0, 0.03, 0.08] {
                                v.push(KeyConfig { eps, rate, prior, tonal_bias, profile, final_weight, sung_share, ..base });
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

pub struct AutoKeyOptions {
    pub data: PathBuf,
    pub out: PathBuf,
    pub cache: Option<PathBuf>,
    pub per_key: usize,
    pub dur_s: f32,
    pub threads: usize,
    /// Skip the sweep and evaluate the shipped default tuning only.
    pub no_sweep: bool,
}

impl Default for AutoKeyOptions {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/vocadito"),
            out: PathBuf::from("out/autokey"),
            cache: None,
            per_key: 4,
            dur_s: 60.0,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            no_sweep: false,
        }
    }
}

pub const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

fn short(kind: ScaleKind) -> &'static str {
    match kind {
        ScaleKind::Major => "",
        ScaleKind::NaturalMinor => "m",
        ScaleKind::HarmonicMinor => "m(harm)",
        ScaleKind::MelodicMinor => "m(mel)",
        ScaleKind::MajorPentatonic => " pent",
        ScaleKind::MinorPentatonic => "m pent",
        _ => "?",
    }
}

fn pct(x: f64) -> String {
    if x.is_nan() { "–".into() } else { format!("{:.1} %", 100.0 * x) }
}

fn secs(x: f64) -> String {
    if x.is_nan() { "–".into() } else { format!("{x:.1} s") }
}

fn row(label: &str, s: &Score) -> String {
    format!(
        "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} / {} | {} |\n",
        pct(s.acc[0]),
        pct(s.acc[1]),
        pct(s.acc[2]),
        pct(s.acc[3]),
        pct(s.acc[4]),
        pct(s.name_acc),
        pct(s.confident_precision),
        pct(s.confident_share),
        secs(s.ttd_median),
        secs(s.ttd_p90),
        s.never,
    )
}

const HEADER: &str = "| System | @5 s | @10 s | @20 s | @30 s | end | name | confident: right | confident: share | time to detect (median / p90) | never |\n|---|---|---|---|---|---|---|---|---|---|---|\n";

fn describe(c: &KeyConfig) -> String {
    format!(
        "tau {} s, slope {} c/ms, offset {}, eps {}, rate {}, prior {:?}, bias {}, profile {:?}, finals {}, sung {}",
        c.tau_s, c.slope_half, c.offset_track, c.eps, c.rate, c.prior, c.tonal_bias, c.profile, c.final_weight, c.sung_share
    )
}

/// Run the evaluation and write `<out>/report.md`. Returns the report.
pub fn run(opts: &AutoKeyOptions, progress: &(dyn Fn(&str) + Sync)) -> Result<String, String> {
    let cache_dir = opts.cache.clone().unwrap_or_else(|| PathBuf::from("out/note-intent/cache"));
    let have_voc = opts.data.join("Annotations").is_dir();
    if !have_voc {
        progress("no vocadito dataset: synthetic only");
    }
    let base = KeyConfig::default();
    let mut report = String::from("# Auto-Key evaluation\n\n");
    report.push_str(&format!(
        "Synthetic: {} keys × {} takes × {:.0} s per split. vocadito: the note-intent DEV/TEST split (Live HQ detector streams). Set accuracy = the proposed pitch-class set equals the truth set; \"confident\" = confidence ≥ {CONFIDENT}.\n\n",
        synth_keys().len(),
        opts.per_key,
        opts.dur_s
    ));

    // 1. DEV, in two stages: the accumulation knobs (which need a replay)
    //    with the default scoring, then the scoring grid on the best of them.
    let mut chosen = base;
    if !opts.no_sweep {
        let mut stage1: Vec<(f64, KeyConfig, Score, Score)> = Vec::new();
        let mut kept: Option<(Vec<Take>, Vec<Take>)> = None;
        for tau_s in [30.0, 60.0, 120.0] {
            for slope_half in [0.5, 1.0, 2.0, 4.0] {
                for offset_track in [true, false] {
                    let acc = KeyConfig { tau_s, slope_half, offset_track, ..base };
                    progress(&format!("DEV replay: tau {tau_s} s, slope {slope_half} c/ms, offset tracking {offset_track}"));
                    let syn = synth_takes(Split::Dev, opts.per_key, opts.dur_s, acc, opts.threads);
                    let voc = if have_voc { vocadito_takes(&opts.data, &cache_dir, Split::Dev, acc, opts.threads, progress)? } else { Vec::new() };
                    let (s, v) = (score(&syn, None), score(&voc, None));
                    let o = objective(&s, &v);
                    if stage1.iter().all(|r| o > r.0) {
                        kept = Some((syn, voc));
                    }
                    stage1.push((o, acc, s, v));
                }
            }
        }
        stage1.sort_by(|a, b| b.0.total_cmp(&a.0));
        report.push_str("## Tuning on DEV\n\nStage 1, evidence accumulation (default scoring), objective = ¼ synthetic @10 s + ¼ synthetic @30 s + ¼ synthetic confident-right + ¼ vocadito end:\n\n| Objective | Tuning | syn @10 s | syn @30 s | syn confident right | voc end | voc time to detect |\n|---|---|---|---|---|---|---|\n");
        for (o, c, s, v) in stage1.iter().take(8) {
            report.push_str(&format!(
                "| {o:.3} | tau {} s, slope {} c/ms, offset {} | {} | {} | {} | {} | {} |\n",
                c.tau_s, c.slope_half, c.offset_track, pct(s.acc[1]), pct(s.acc[3]), pct(s.confident_precision), pct(v.acc[4]), secs(v.ttd_median)
            ));
        }
        let acc = stage1[0].1;
        let (syn, voc) = kept.unwrap_or_default();
        progress("DEV stage 2: scoring grid");
        let grid = scoring_grid(acc);
        let mut stage2 = parallel(grid.len(), opts.threads, |i| {
            let det = KeyDetector::with_config(1.0 / 750.0, grid[i]);
            let s = score(&syn, Some(&det));
            let v = score(&voc, Some(&det));
            (objective(&s, &v), grid[i], s, v)
        });
        stage2.sort_by(|a, b| b.0.total_cmp(&a.0));
        chosen = stage2[0].1;
        report.push_str(&format!("\nStage 2, scoring ({} tunings on the best accumulation). Chosen: `{}` (objective {:.3}).\n\n| Objective | Tuning | syn @10 s | syn @30 s | syn confident right | voc end |\n|---|---|---|---|---|---|\n", grid.len(), describe(&chosen), stage2[0].0));
        for (o, c, s, v) in stage2.iter().take(10) {
            report.push_str(&format!("| {o:.3} | {} | {} | {} | {} | {} |\n", describe(c), pct(s.acc[1]), pct(s.acc[3]), pct(s.confident_precision), pct(v.acc[4])));
        }
        report.push('\n');
    }

    // 2. TEST, once, with the chosen tuning (recorded by the real detector,
    //    so the confirmation logic runs too), against classic profiles.
    progress("TEST");
    let syn = synth_takes(Split::Test, opts.per_key, opts.dur_s, chosen, opts.threads);
    report.push_str(&format!("## TEST: synthetic ({} takes)\n\n{HEADER}", syn.len()));
    let s = score(&syn, None);
    report.push_str(&row("Auto-Key (chosen tuning)", &s));
    for p in Profile::ALL {
        report.push_str(&row(&format!("Krumhansl-Schmuckler, {p:?} profile"), &score_profile(&syn, p)));
    }
    report.push_str(&format!(
        "\nConfirmed key at the end of the take: {} of {} takes, {} right.\n\n",
        s.confirmed, s.takes, s.confirmed_right
    ));
    // By scale family.
    report.push_str("By key type (Auto-Key, end of take / @10 s):\n\n");
    for kind in [ScaleKind::Major, ScaleKind::NaturalMinor, ScaleKind::HarmonicMinor] {
        let sub: Vec<Take> = syn.iter().filter(|t| t.truth_name.is_some_and(|k| k.kind == kind)).cloned().collect();
        let ss = score(&sub, None);
        report.push_str(&format!("- {}: {} / {}\n", kind.name(), pct(ss.acc[4]), pct(ss.acc[1])));
    }
    report.push('\n');
    if have_voc {
        let voc = vocadito_takes(&opts.data, &cache_dir, Split::Test, chosen, opts.threads, progress)?;
        report.push_str(&format!("## TEST: vocadito ({} takes, mean {:.1} s)\n\n{HEADER}", voc.len(), voc.iter().map(|t| t.dur_s).sum::<f32>() / voc.len().max(1) as f32));
        let v = score(&voc, None);
        report.push_str(&row("Auto-Key (chosen tuning)", &v));
        for p in Profile::ALL {
            report.push_str(&row(&format!("Krumhansl-Schmuckler, {p:?} profile"), &score_profile(&voc, p)));
        }
        report.push_str(&format!(
            "\n\"name\" compares with the vocadito evaluation's key (Krumhansl-Kessler on annotator 1's notes). Annotated note time inside the final proposal's set: {} (oracle set: {}). Confirmed at the end: {} of {} takes, {} right.\n\nPer take (truth = oracle set; weights = mean detector confidence / steadiness / in-tune over the voiced time):\n\n| Take | Truth set | KK name | Final proposal | Confidence | Evidence | Voiced | Weights |\n|---|---|---|---|---|---|---|---|\n",
            pct(v.ann_coverage),
            pct(v.oracle_coverage),
            v.confirmed,
            v.takes,
            v.confirmed_right
        ));
        for t in &voc {
            let last = t.snaps.last().map(|s| s.reading).unwrap_or_default();
            let set_name = ScaleKind::identify(t.truth_mask, None).map_or("?".to_string(), |(r, k)| format!("{} {}", NOTE_NAMES[r as usize], k.name()));
            let kk = t.truth_name.map_or("–".to_string(), |k| format!("{} {}", NOTE_NAMES[k.root as usize], k.kind.name()));
            let prop = last.proposal.map_or("–".to_string(), |k| format!("{} {}", NOTE_NAMES[k.root as usize], k.kind.name()));
            let (wc, ws, wt, vs) = t.weights;
            report.push_str(&format!(
                "| {} | {set_name} | {kk} | {prop} | {:.2} | {:.1} s | {vs:.1} s | {wc:.2} / {ws:.2} / {wt:.2} |\n",
                t.name, last.confidence, last.evidence_s
            ));
        }
    }
    std::fs::create_dir_all(&opts.out).map_err(|e| format!("{}: {e}", opts.out.display()))?;
    let path = opts.out.join("report.md");
    std::fs::write(&path, &report).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_takes_stay_in_key_and_are_deterministic() {
        let key = KeyName::new(2, ScaleKind::HarmonicMinor);
        let a = synth_take(key, 7, 20.0, 48_000.0);
        let b = synth_take(key, 7, 20.0, 48_000.0);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.cents.to_bits() == y.cents.to_bits()));
        let voiced = a.iter().filter(|e| e.voiced).count();
        assert!(voiced > a.len() / 2, "mostly singing");
    }

    #[test]
    fn the_oracle_prefers_full_coverage_then_the_smaller_set() {
        // C D E G A only: the pentatonic set covers it exactly.
        let mut h = [0.0; 12];
        for pc in [0, 2, 4, 7, 9] {
            h[pc] = 1.0;
        }
        assert_eq!(oracle_set(&h), ScaleKind::MajorPentatonic.mask(0));
        // Add F and B: C major.
        h[5] = 0.5;
        h[11] = 0.5;
        assert_eq!(oracle_set(&h), ScaleKind::Major.mask(0));
    }
}
