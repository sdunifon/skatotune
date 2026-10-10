//! Training data for the learned switch veto (`skato_eval note-intent dump`
//! and `dump-synth`): one row per control hop with the veto's 58 features of
//! the decided frame and the annotations, in the format
//! `tools/note_intent/train_veto.py` reads (little-endian `f32` rows plus a
//! JSON sidecar with the column names).
//!
//! The features do not depend on the note decisions (the front end alone
//! makes them), so the corrector runs with hard-tune settings and its own
//! decisions only fill the `base_note` column.
//!
//! Synthetic singing (always training-only) comes from the scenario
//! generator with seeds from 50 000 up, disjoint from every evaluation
//! scenario: a chromatic random walk, note offsets uniform in ±48 c (vocadito's
//! notes sit about uniformly around the grid), drift, scoops at phrase
//! onsets, overshoot and glides on legato, vibrato of 15-75 c on longer notes,
//! consonant onsets and level changes. Without it the vocadito-only model
//! wrecked clean legato on the synthetic suite.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use skato_control::intent::{FEATURE_NAMES, FEATURES};
use skato_control::{Corrector, CorrectorConfig};
use skato_core::{Settings, frame_flags};

use super::cache::{self, Config, Fingerprints};
use super::data::{FileCtx, Split, Splits, legal_note};
use super::replay::content_time;
use crate::rng::Rng;
use crate::synth::{Consonant, Melody, Note, NoteKind, SignalBuilder, Vibrato, Voice, VoicedConsonant, Vowel};
use crate::vocadito::{Condition, RunConfig, condition_settings, data as vdata};

/// Columns before the features.
pub const HEAD: [&str; 17] = [
    "file", "singer", "k", "t", "valid", "seg", "r", "fut", "a1_idx", "a1_c", "a2_c", "gt_chrom", "gt2_chrom", "gt_key", "gt2_key", "base_note", "key_mask",
];

/// Columns per row.
pub const NCOL: usize = HEAD.len() + FEATURES;

/// Replay a stream with hard-tune `settings` and call `row(k, corrector, base note)`
/// after every hop.
fn replay_rows(stream: &cache::EstStream, settings: &Settings, mut row: impl FnMut(usize, &mut Corrector, Option<i32>)) {
    let h = stream.h;
    let mut c = Corrector::new(CorrectorConfig {
        sample_rate: h.sample_rate,
        hop: h.hop as usize,
        latency_mode: h.mode,
        lookahead_hops: h.lookahead_hops as usize,
    });
    c.set_settings(&Settings::default());
    c.reset();
    c.set_settings(settings);
    for (k, est) in stream.est.iter().enumerate() {
        let o = c.process_hop(est, None);
        let has = o.frame.flags & frame_flags::VOICED != 0 && o.frame.target.is_finite() && o.frame.target > 0.0;
        let base = has.then(|| (o.frame.target / 100.0).round() as i32);
        row(k, &mut c, base);
    }
}

/// The feature columns of the decided frame (zeros when it is unvoiced):
/// (valid, segment age or −1, r, lookahead hops, features).
fn feature_columns(c: &mut Corrector) -> (f32, f32, f32, f32, [f32; FEATURES]) {
    match c.note_hold_features() {
        Some((r, f, fut, seg)) => (1.0, if seg == u32::MAX { -1.0 } else { seg as f32 }, r, fut as f32, f),
        None => (0.0, -1.0, f32::NAN, 0.0, [0.0; FEATURES]),
    }
}

fn write_rows(out: &Path, rows: &[f32]) -> Result<usize, String> {
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(rows.len() / NCOL)
}

fn write_sidecar(out: &Path, rows: usize, cfg: Config, split: &str, singers: &[String]) -> Result<(), String> {
    let mut cols: Vec<String> = HEAD.iter().map(|s| format!("\"{s}\"")).collect();
    cols.extend(FEATURE_NAMES.iter().map(|s| format!("\"{s}\"")));
    let mut side = String::new();
    let _ = writeln!(
        side,
        "{{\"rows\": {rows}, \"ncol\": {NCOL}, \"config\": \"{}\", \"split\": \"{split}\", \"singers\": [{}], \"columns\": [{}]}}",
        cfg.label(),
        singers.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(","),
        cols.join(",")
    );
    std::fs::write(out.with_extension("json"), side).map_err(|e| e.to_string())
}

/// Dump the rows of every file of `split` in `cfg` to `out` (`.f32`, plus
/// `.json`). The `singer` column indexes every singer of the split file,
/// sorted. Returns the row count.
pub fn dump(data: &Path, cache_dir: &Path, cfg: Config, split: Split, out: &Path, threads: usize) -> Result<usize, String> {
    let splits = Splits::committed();
    let files = splits.files(split);
    let singers: Vec<String> = splits.by_singer(&splits.files(Split::All)).into_keys().collect();
    let fp = Fingerprints::compute(&[cfg]);
    cache::prepare(data, cache_dir, &fp, &files, &[cfg], threads, &|_| {})?;
    let ctxs: Vec<FileCtx> = super::load_contexts(data, cache_dir, &files, &splits, threads)?;
    let per_file = super::parallel(ctxs.len(), threads, |i| -> Result<Vec<f32>, String> {
        let ctx = &ctxs[i];
        let stream = cache::load_stream(cache_dir, &fp, cfg, ctx.id)?;
        let rc = RunConfig::new(cfg.mode, cfg.tier);
        let s_c = condition_settings(Condition::HardChromatic, rc, ctx.key.as_ref()).ok_or("no settings")?;
        let legal_c = crate::scenarios::legal_classes(&s_c);
        let legal_k = condition_settings(Condition::HardKey, rc, ctx.key.as_ref()).map(|s| crate::scenarios::legal_classes(&s));
        let key_mask = legal_k.map_or(0.0, |l| l.iter().enumerate().map(|(i, b)| if *b { 1u32 << i } else { 0 }).sum::<u32>() as f32);
        let sid = singers.iter().position(|s| *s == ctx.singer).unwrap_or(999) as f32;
        let hop_s = f64::from(stream.h.hop) / f64::from(stream.h.sample_rate);
        let gt = |cents: f64, legal: Option<&[bool; 12]>| match legal {
            Some(l) if cents.is_finite() => legal_note(cents, l).map_or(f32::NAN, |n| n as f32),
            _ => f32::NAN,
        };
        let mut v = Vec::with_capacity(stream.est.len() * NCOL);
        replay_rows(&stream, &s_c, |k, c, base| {
            let t = content_time(&stream.h, k) + 0.5 * hop_s;
            let (i1, i2) = (ctx.a1_at(t), ctx.a2_at(t));
            let a1c = i1.map_or(f64::NAN, |i| ctx.a1[i].cents());
            let a2c = i2.map_or(f64::NAN, |i| ctx.a2[i].cents());
            let (valid, seg, r, fut, feats) = feature_columns(c);
            v.extend_from_slice(&[
                ctx.id as f32,
                sid,
                k as f32,
                t as f32,
                valid,
                seg,
                r,
                fut,
                i1.map_or(-1.0, |i| i as f32),
                a1c as f32,
                a2c as f32,
                gt(a1c, Some(&legal_c)),
                gt(a2c, Some(&legal_c)),
                gt(a1c, legal_k.as_ref()),
                gt(a2c, legal_k.as_ref()),
                base.map_or(f32::NAN, |n| n as f32),
                key_mask,
            ]);
            v.extend_from_slice(&feats);
        });
        Ok(v)
    });
    let mut rows = Vec::new();
    for r in per_file {
        rows.extend(r?);
    }
    let n = write_rows(out, &rows)?;
    write_sidecar(out, n, cfg, split.label(), &singers)?;
    Ok(n)
}

/// A random sung line from the synthetic generator (see the module docs).
pub fn synth_melody(seed: u64) -> Melody {
    let mut rng = Rng::new(seed);
    let female = seed % 2 == 1;
    let mut voice = if female { Voice::female() } else { Voice::male() };
    voice.rd = rng.range(0.6, 1.8);
    if rng.uniform() < 0.3 {
        voice = voice.breathy();
    }
    let base = if female { rng.range(57.0, 67.0) } else { rng.range(43.0, 55.0) }.round();
    let n_notes = 18 + (rng.uniform() * 14.0) as usize;
    let vowels = [Vowel::A, Vowel::E, Vowel::I, Vowel::O, Vowel::U];
    let mut m = base;
    let mut notes = Vec::with_capacity(n_notes);
    for i in 0..n_notes {
        let u = rng.uniform();
        let mag = if u < 0.15 {
            0.0
        } else if u < 0.55 {
            1.0
        } else if u < 0.8 {
            2.0
        } else if u < 0.92 {
            rng.range(3.0, 4.99).floor()
        } else {
            rng.range(5.0, 7.99).floor()
        };
        let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        m = (m + sign * mag).clamp(base - 7.0, base + 9.0);
        let dur = (rng.gaussian() * 0.55 + 0.28f64.ln()).exp().clamp(0.07, 1.4);
        let gap = if i == 0 || rng.uniform() < 0.28 { rng.range(0.04, 0.45) } else { 0.0 };
        let mut n = Note::midi(m, dur).vowel(vowels[(rng.uniform() * 5.0) as usize % 5]).glide(rng.range(0.02, 0.12)).detune(rng.range(-48.0, 48.0));
        if gap > 0.0 {
            n = n.gap(gap);
            if rng.uniform() < 0.75 {
                n = n.scoop(rng.range(-160.0, 60.0));
                n.scoop_tau_s = rng.range(0.03, 0.1);
            }
            let o = rng.uniform();
            if o < 0.25 {
                let kind = [VoicedConsonant::M, VoicedConsonant::N, VoicedConsonant::L, VoicedConsonant::V][(rng.uniform() * 4.0) as usize % 4];
                n = n.voiced_onset(kind, rng.range(0.03, 0.08));
            } else if o < 0.4 {
                let kind = [Consonant::S, Consonant::T, Consonant::K, Consonant::Breath][(rng.uniform() * 4.0) as usize % 4];
                n = n.consonant(kind, rng.range(0.03, 0.1), rng.range(-45.0, -30.0)).attack(rng.range(0.005, 0.08));
            } else if o < 0.6 {
                n = n.attack(rng.range(0.03, 0.1));
            }
        } else {
            n = n.overshoot(rng.range(0.0, 45.0));
            if rng.uniform() < 0.25 {
                let kind = [VoicedConsonant::L, VoicedConsonant::N, VoicedConsonant::M][(rng.uniform() * 3.0) as usize % 3];
                n = n.voiced_onset(kind, rng.range(0.03, 0.06));
            }
        }
        if dur > 0.3 && rng.uniform() < 0.6 {
            n = n.vibrato(Vibrato { rate_hz: rng.range(4.5, 7.0), extent_cents: rng.range(15.0, 75.0), delay_s: rng.range(0.05, 0.25), ramp_s: rng.range(0.1, 0.3) });
        }
        n = n.level(rng.range(-6.0, 3.0));
        notes.push(n);
    }
    Melody::new(voice).notes(notes).detune_range(0.0).drift(rng.range(5.0, 18.0), rng.range(0.6, 2.0)).deviation_limit(49.0).seed(seed ^ 0x5a5a)
}

/// Dump `n` synthetic lines (seeds `seed0..seed0 + n`, 44.1 kHz) in `cfg`.
/// The annotation columns hold each note's median sung pitch (f0 truth, 30
/// ms trimmed at each end, like a Tony note median). Returns the row count
/// and the seconds of audio.
pub fn dump_synth(cfg: Config, n: usize, seed0: u64, out: &Path, threads: usize) -> Result<(usize, f64), String> {
    let sr = 44_100.0f32;
    let per = super::parallel(n, threads, |i| -> Result<(Vec<f32>, f64), String> {
        let seed = seed0 + i as u64;
        let sig = SignalBuilder::new(sr).seed(seed ^ 0x77).silence(0.2).melody(synth_melody(seed)).silence(0.25).build();
        let min_f0 = sig.f0_truth.iter().copied().filter(|f| *f > 0.0).fold(f32::INFINITY, f32::min);
        let range = vdata::choose_range(f64::from(min_f0));
        let s_c = condition_settings(Condition::HardChromatic, RunConfig::new(cfg.mode, cfg.tier), None).ok_or("no settings")?;
        let stream = cache::run_detector(&sig.audio, sr, cfg, range, &s_c);
        let g30 = (0.03 * f64::from(sr)) as usize;
        let labels: Vec<(usize, usize, f64)> = sig
            .notes
            .iter()
            .filter(|ns| ns.kind == NoteKind::Note)
            .map(|ns| {
                let (a0, b0) = if ns.len() > 3 * g30 { (ns.start + g30, ns.end - g30) } else { (ns.start, ns.end) };
                let mut v: Vec<f64> = sig.f0_truth[a0..b0].iter().filter(|f| **f > 0.0).map(|f| crate::synth::hz_to_cents64(f64::from(*f))).collect();
                v.sort_by(f64::total_cmp);
                (ns.start, ns.end, if v.is_empty() { f64::NAN } else { v[v.len() / 2] })
            })
            .collect();
        let hop_s = f64::from(stream.h.hop) / f64::from(stream.h.sample_rate);
        let mut v = Vec::with_capacity(stream.est.len() * NCOL);
        replay_rows(&stream, &s_c, |k, c, base| {
            let t = content_time(&stream.h, k) + 0.5 * hop_s;
            let pos = (t * f64::from(sr)).max(0.0) as usize;
            let li = labels.iter().position(|(s0, e0, _)| pos >= *s0 && pos < *e0);
            let a1c = li.map_or(f64::NAN, |j| labels[j].2);
            let gt = if a1c.is_finite() { (a1c / 100.0).round() as f32 } else { f32::NAN };
            let (valid, seg, r, fut, feats) = feature_columns(c);
            v.extend_from_slice(&[
                (1000 + i) as f32,
                (100 + i % 7) as f32,
                k as f32,
                t as f32,
                valid,
                seg,
                r,
                fut,
                li.map_or(-1.0, |j| j as f32),
                a1c as f32,
                a1c as f32,
                gt,
                gt,
                f32::NAN,
                f32::NAN,
                base.map_or(f32::NAN, |n| n as f32),
                0.0,
            ]);
            v.extend_from_slice(&feats);
        });
        Ok((v, sig.duration_s()))
    });
    let mut rows = Vec::new();
    let mut dur = 0.0;
    for r in per {
        let (v, d) = r?;
        rows.extend(v);
        dur += d;
    }
    let count = write_rows(out, &rows)?;
    write_sidecar(out, count, cfg, "synth", &[])?;
    Ok((count, dur))
}
