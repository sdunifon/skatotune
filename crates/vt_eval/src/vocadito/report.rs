//! CSV and Markdown output of the vocadito evaluation.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use vt_core::{EngineConfig, LatencyMode, VoiceRange};

use super::measure::{CorrCounts, DetCounts};
use super::{Condition, CorrRow, DetRow, Outcome, RunConfig, Segment, VocaditoOptions};
use crate::suite::{RANGES, mode_name, range_name};
use crate::wav;

fn f(v: f64, d: usize) -> String {
    if v.is_finite() {
        format!("{v:.d$}")
    } else {
        "-".to_string()
    }
}

fn pct(v: f64, d: usize) -> String {
    f(100.0 * v, d)
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Write files.csv, detector.csv, correction.csv, segments.csv and report.md.
pub fn write_all(out: &Path, opts: &VocaditoOptions, o: &Outcome) -> wav::Result<()> {
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("files.csv"), files_csv(o))?;
    std::fs::write(out.join("detector.csv"), detector_csv(o))?;
    std::fs::write(out.join("correction.csv"), correction_csv(o))?;
    std::fs::write(out.join("segments.csv"), segments_csv(o))?;
    std::fs::write(out.join("report.md"), markdown(opts, o))?;
    Ok(())
}

fn files_csv(o: &Outcome) -> String {
    let mut s = String::from(
        "file,singer,language,avg_pitch_midi,dur_s,range,min_f0_hz,max_f0_hz,voiced_frac,key,key_corr,key_coverage,notes_a1,notes_a2,a1_a2_median_diff_c,a1_matched,stable_frames,moving_frames,ref_rpa50,ref_rpa25,ref_rca50,ref_fine_med_c,ref_fine_p95_c,ref_octave_rate,ref_v_precision,ref_v_recall,ref_v_f1,ref_onset_delay_med_ms,ref_lag_ms\n",
    );
    for x in &o.files {
        let r = &x.reference;
        let _ = writeln!(
            s,
            "{},{},{},{},{:.2},{},{:.1},{:.1},{:.3},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{},{:.4},{:.4},{:.4},{:.4},{},{}",
            x.id,
            csv_field(&x.meta.singer),
            csv_field(&x.meta.language),
            f(x.meta.average_pitch, 0),
            x.dur_s,
            range_name(x.range),
            x.min_f0,
            x.max_f0,
            x.voiced_fraction,
            x.key.map_or("-".to_string(), |k| k.name()),
            x.key.map_or("-".to_string(), |k| f(k.correlation, 3)),
            x.key.map_or("-".to_string(), |k| f(k.coverage, 3)),
            x.notes_a1,
            x.notes_a2,
            f(x.annot_diff_c, 1),
            f(x.annot_matched, 3),
            x.stable_frames,
            x.moving_frames,
            r.rpa50(),
            r.rpa25(),
            r.rca50(),
            f(r.fine.percentile(50.0), 2),
            f(r.fine.percentile(95.0), 2),
            r.octave_rate(),
            r.precision(),
            r.recall(),
            r.f1(),
            f(r.onset_delay_median(), 1),
            f(x.reference_lag_ms, 0),
        );
    }
    s
}

fn detector_csv(o: &Outcome) -> String {
    let mut s = String::from(
        "file,config,range,latency_samples,latency_ms,ref_voiced,both_voiced,rpa50,rpa25,rca50,fine_med_c,fine_p95_c,all_med_c,all_p95_c,octave_up,octave_down,octave_rate,gross_other_rate,v_precision,v_recall,v_f1,v_f1_guarded,onset_delay_med_ms,onset_delay_p90_ms,onsets_missed,onsets_bridged,onsets_checked,onset_spikes,best_lag_ms,rpa50_at_best_lag,measured_lag,measured_peak,transparency_snr_db,info_age_ms,finite,telemetry_ok,render_s,content_segments,content_mean_ms,content_median_ms,content_p95_ms,wander_max_periods\n",
    );
    for d in &o.det {
        let c = &d.counts;
        let sr = o
            .files
            .iter()
            .find(|x| x.id == d.file)
            .map_or(44_100.0, |x| x.sample_rate);
        let _ = writeln!(
            s,
            "{},{},{},{},{:.3},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.5},{:.5},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{:.4},{},{},{},{},{},{},{:.2},{},{},{},{},{}",
            d.file,
            d.config.label(),
            range_name(d.range),
            d.latency,
            f64::from(d.latency) / f64::from(sr) * 1000.0,
            c.ref_voiced,
            c.both,
            c.rpa50(),
            c.rpa25(),
            c.rca50(),
            f(c.fine.percentile(50.0), 2),
            f(c.fine.percentile(95.0), 2),
            f(c.all.percentile(50.0), 2),
            f(c.all.percentile(95.0), 1),
            c.octave_up,
            c.octave_down,
            c.octave_rate(),
            c.gross_other_rate(),
            c.precision(),
            c.recall(),
            c.f1(),
            c.f1_guarded(),
            f(c.onset_delay_median(), 1),
            f(c.onset_delay_p90(), 1),
            c.onsets_missed,
            c.onsets_bridged,
            c.onsets_checked,
            c.onset_spikes,
            f(d.best_lag_ms, 0),
            d.rpa50_at_best,
            d.measured.map_or("-".to_string(), |m| m.lag.to_string()),
            d.measured.map_or("-".to_string(), |m| f(m.peak, 4)),
            f(d.transparency_snr_db, 1),
            f(d.info_age_ms, 2),
            d.finite,
            d.telemetry_ok,
            d.render_s,
            d.content.segments,
            f(d.content.mean_delay_ms(), 3),
            f(d.content.delay_ms(50.0), 3),
            f(d.content.delay_ms(95.0), 3),
            f(d.content.max_abs_wander_periods(), 3),
        );
    }
    s
}

fn correction_csv(o: &Outcome) -> String {
    let mut s = String::from(
        "file,config,condition,range,latency_samples,stable_frames,out_med_c,out_p95_c,out_within5,out_within10,out_within25,in_med_c,in_p95_c,in_within5,in_within10,in_within25,note_out_med_c,note_out_p95_c,note_in_med_c,notes,note_agree_out_pct,note_agree_in_pct,settle_med_ms,settle_p90_ms,settle_missed,onsets_checked,onset_glitches,dropout_pct,dropout_runs,click_events,click_frames,clicks_per_min,click_max_db,lsd_mean_db,lsd_median_db,gross_shifts,out_jumps,warble_episodes,warble_flips,peak_ratio_db,finite,render_s\n",
    );
    for r in &o.corr {
        let c = &r.counts;
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
            r.file,
            r.config.label(),
            r.condition.label(),
            range_name(r.range),
            r.latency,
            c.stable,
            f(c.out.percentile(50.0), 2),
            f(c.out.percentile(95.0), 2),
            f(c.out.within(5.0), 1),
            f(c.out.within(10.0), 1),
            f(c.out.within(25.0), 1),
            f(c.inp.percentile(50.0), 2),
            f(c.inp.percentile(95.0), 2),
            f(c.inp.within(5.0), 1),
            f(c.inp.within(10.0), 1),
            f(c.inp.within(25.0), 1),
            f(crate::dsp::median(&c.note_out), 2),
            f(crate::dsp::percentile(&c.note_out, 95.0), 2),
            f(crate::dsp::median(&c.note_in), 2),
            c.notes,
            f(c.agree_out_pct(), 1),
            f(c.agree_in_pct(), 1),
            f(c.settle_median(), 1),
            f(c.settle_p90(), 1),
            c.settle_missed,
            c.onsets_checked,
            c.onset_glitches,
            f(c.dropout_pct(), 2),
            c.dropout_runs,
            c.click_events,
            c.click_frames,
            f(c.clicks_per_min(), 2),
            f(c.click_max_db, 1),
            f(c.lsd_mean(), 2),
            f(r.lsd_median, 2),
            c.shift_segments,
            c.jumps,
            c.flickers,
            c.flicker_flips,
            f(c.peak_ratio_db, 2),
            r.finite,
            r.render_s,
        );
    }
    s
}

fn segments_csv(o: &Outcome) -> String {
    let mut s = String::from(
        "file,singer,config,condition,kind,start_s,end_s,dur_ms,value,ann_f0_hz,level_dbfs,clarity,tags\n",
    );
    for g in &o.segments {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{:.3},{:.3},{:.0},{},{},{},{},{}",
            g.file,
            csv_field(&g.singer),
            g.config,
            g.condition,
            g.raw.kind,
            g.raw.start_s,
            g.raw.end_s,
            (g.raw.end_s - g.raw.start_s) * 1000.0,
            f(g.raw.value, 1),
            f(g.ann_f0, 1),
            f(g.level_db, 1),
            f(g.clarity, 2),
            g.tags.join("|"),
        );
    }
    s
}

// ---------------------------------------------------------------------------
// Markdown

fn det_header(s: &mut String) {
    s.push_str("| Tracker | RPA 50 c | RPA 25 c | RCA 50 c | Fine med / p95 (c) | Octave err | Other gross | V-prec | V-rec | V-F1 | V-F1 (±20 ms guard) | Onset delay med / p90 (ms) | Onsets missed | Onset spikes > 500 c (of them at f_max) | Ts offset (ms) |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
}

fn det_row(s: &mut String, name: &str, c: &DetCounts, lag: f64) {
    let _ = writeln!(
        s,
        "| {name} | {} % | {} % | {} % | {} / {} | {} % | {} % | {} | {} | {} | {} | {} / {} | {} | {} / {} = {} % ({}) | {} |",
        pct(c.rpa50(), 1),
        pct(c.rpa25(), 1),
        pct(c.rca50(), 1),
        f(c.fine.percentile(50.0), 1),
        f(c.fine.percentile(95.0), 1),
        pct(c.octave_rate(), 2),
        pct(c.gross_other_rate(), 2),
        f(c.precision(), 3),
        f(c.recall(), 3),
        f(c.f1(), 3),
        f(c.f1_guarded(), 3),
        f(c.onset_delay_median(), 1),
        f(c.onset_delay_p90(), 1),
        c.onsets_missed,
        c.onset_spikes,
        c.onsets_checked,
        f(c.onset_spike_pct(), 1),
        c.onset_spikes_fmax,
        if lag.is_finite() {
            format!("{lag:+.0}")
        } else {
            "-".to_string()
        },
    );
}

fn corr_header(s: &mut String) {
    s.push_str("| Config | Condition | Out med / p95 (c) | Out ≤5 / ≤10 / ≤25 c | In med / p95 (c) | In ≤5 / ≤10 / ≤25 c | Note med out / in (c) | Note = annot. out / in | Settle med / p90 (ms) | Onset glitch | Out dropout | Clicks /min | LSD (dB) | Shifts / jumps / warbles (flips) | NaN / blow-up |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
}

fn corr_row(s: &mut String, cfg: &str, cond: &str, c: &CorrCounts) {
    let _ = writeln!(
        s,
        "| {cfg} | {cond} | {} / {} | {} / {} / {} % | {} / {} | {} / {} / {} % | {} / {} | {} / {} % | {} / {} | {} % | {} % | {} | {} | {} / {} / {} ({}) | {} / {} |",
        f(c.out.percentile(50.0), 1),
        f(c.out.percentile(95.0), 1),
        f(c.out.within(5.0), 1),
        f(c.out.within(10.0), 1),
        f(c.out.within(25.0), 1),
        f(c.inp.percentile(50.0), 1),
        f(c.inp.percentile(95.0), 1),
        f(c.inp.within(5.0), 1),
        f(c.inp.within(10.0), 1),
        f(c.inp.within(25.0), 1),
        f(crate::dsp::median(&c.note_out), 1),
        f(crate::dsp::median(&c.note_in), 1),
        f(c.agree_out_pct(), 1),
        f(c.agree_in_pct(), 1),
        f(c.settle_median(), 1),
        f(c.settle_p90(), 1),
        f(c.onset_glitch_pct(), 1),
        f(c.dropout_pct(), 2),
        f(c.clicks_per_min(), 2),
        f(c.lsd_mean(), 2),
        c.shift_segments,
        c.jumps,
        c.flickers,
        c.flicker_flips,
        c.nonfinite,
        c.blowups,
    );
}

fn merged_det<'a>(rows: impl Iterator<Item = &'a DetRow>) -> DetCounts {
    let mut c = DetCounts::default();
    for r in rows {
        c.merge(&r.counts);
    }
    c
}

fn merged_corr<'a>(rows: impl Iterator<Item = &'a CorrRow>) -> CorrCounts {
    let mut c = CorrCounts::default();
    for r in rows {
        c.merge(&r.counts);
    }
    c
}

fn latency_ms(sr: f32, mode: LatencyMode, range: VoiceRange) -> (u32, f64) {
    let cfg = EngineConfig {
        sample_rate: sr,
        latency_mode: mode,
        voice_range: range,
        ..EngineConfig::default()
    };
    let d = vt_engine::latency_samples(&cfg);
    (d, f64::from(d) / f64::from(sr) * 1000.0)
}

fn markdown(opts: &VocaditoOptions, o: &Outcome) -> String {
    let mut s = String::new();
    let total_min = o.files.iter().map(|x| x.dur_s).sum::<f64>() / 60.0;
    let singers: std::collections::BTreeSet<&str> =
        o.files.iter().map(|x| x.meta.singer.as_str()).collect();
    let _ = writeln!(s, "# Vocal Tune on real singing: vocadito\n");
    let _ = writeln!(
        s,
        "Generated by `vt_eval vocadito` in {:.0} s on {} threads: {} tracks, {:.1} min, {} singers. Dataset: vocadito (Bittner et al. 2021, CC BY 4.0), 44.1 kHz solo a-cappella excerpts, F0 annotation (hop 256 samples) and two note annotators. Block size {}. Configs: {}.\n",
        o.elapsed_s,
        o.threads,
        o.files.len(),
        total_min,
        singers.len(),
        opts.blocks.label(),
        opts.configs
            .iter()
            .map(|c| format!("`{}`", c.label()))
            .collect::<Vec<_>>()
            .join(", "),
    );
    s.push_str("Method in brief (details in `crates/vt_eval/src/vocadito/`):\n\n");
    s.push_str("- **Voice range** per file: the highest preset floor at least 10 % below the lowest annotated f0 (Soprano ≥ 154 Hz, Alto/Tenor ≥ 88 Hz, else Low Male).\n");
    s.push_str("- **Detector**: engine at amount 0; telemetry `detected` (the detector's pitch as the control law uses it: lag-compensated `cents`, or in Live Max the pitch at the analysed sample, `f0_hz`; the PSOLA mark pitch is no longer used, vt_control SN-2) mapped to input time with `sample_pos − D`, sampled at each annotation frame (mir_eval conventions: an unvoiced estimate on an annotated-voiced frame counts as a miss). Fine error = |Δ| on frames voiced in both and within 50 c. Octave error = both voiced, |Δ| > 600 c and within 50 c of a multiple of 1200 c. Onset delay = first voiced telemetry frame − annotated voicing onset (onsets after ≥ 100 ms of silence). *Ts offset* = lag (ms) minimising mean min(|Δ|, 50 c); positive means the estimate is stamped late. Baseline: the offline f64 reference tracker (`vt_eval::reference`) on the input.\n");
    s.push_str("- **Correction**: Hard Tune (τ 0, vibrato depth 0, tolerance 5 c) and Natural (τ 25 ms, Humanize 30, Flex 20, tolerance 15 c / 30 ms), chromatic and in the key estimated from annotator 1's notes (Krumhansl-Schmuckler, major/natural minor). Output measured with the reference tracker. *Stable* frames: annotated voiced, ≥ 30 ms inside an annotator-1 note, input voiced, input not moving > 40 c per 20 ms. *Out/In* = distance of the 20 ms-smoothed pitch to the nearest legal note. *Note* = median pitch over a note's stable frames; *Note = annot.* = that note rounds to the annotator's note. *Settle* = onset to within 10 c of a legal note for 20 ms. *Dropout* = stable frames where the output has no pitch. *Clicks* = > 8 kHz output energy > 10 dB over the input's local max (events/min). *LSD* = order-14 LPC envelope distance input vs output. *Shifts* = output − input > 150 c for ≥ 15 ms; *jumps* = output moves > 250 c in 5 ms while the input moves < 80 c; *warbles* = episodes of target A→B→A flips (B < 40 ms, flips within 60 ms merged). *Onset glitch* = the first 100 ms of a note (after ≥ 100 ms silence) has ≥ 5 ms where the input is voiced but the output has no pitch or is > 150 c from the input. *Onset spikes* (detector) = a voiced estimate within [−50, +100] ms of an onset more than 500 c from the onset's annotated pitch.\n\n");

    gates_section(&mut s, o);
    latency_section(&mut s, opts, o);
    detector_section(&mut s, opts, o);
    correction_section(&mut s, opts, o);
    files_section(&mut s, opts, o);
    segments_section(&mut s, opts, o);
    s
}

fn gates_section(s: &mut String, o: &Outcome) {
    s.push_str("## Real-singing gates (Live HQ)\n\nDetector thresholds are relative to the offline reference tracker on the same frames; hard-tune thresholds are regression guards per voice range (`crates/vt_eval/src/vocadito/gates.rs`), shown next to the synthetic DESIGN §4 targets.\n\n| Gate | Target | Measured | Result |\n|---|---|---|---|\n");
    for g in super::gates::gates(o) {
        let res = match g.pass {
            Some(true) => "PASS",
            Some(false) => "**FAIL**",
            None => "n/a",
        };
        let _ = writeln!(s, "| {} | {} | {} | {res} |", g.name, g.target, g.value);
    }
    s.push('\n');
}

fn latency_section(s: &mut String, opts: &VocaditoOptions, o: &Outcome) {
    s.push_str("## Latency\n\n");
    s.push_str("Reported plugin latency D (what the host compensates, and what a performer hears when monitoring through the plugin):\n\n");
    s.push_str("| Mode | Low Male | Alto/Tenor | Soprano | Wide | (at 48 kHz: LM / AT / S / W) |\n|---|---|---|---|---|---|\n");
    for mode in [
        LatencyMode::LiveMin,
        LatencyMode::Live,
        LatencyMode::LivePlus,
        LatencyMode::Studio,
    ] {
        let cells44: Vec<String> = RANGES
            .iter()
            .map(|r| {
                let (d, ms) = latency_ms(44_100.0, mode, *r);
                format!("{ms:.2} ms ({d})")
            })
            .collect();
        let cells48: Vec<String> = RANGES
            .iter()
            .map(|r| format!("{:.2}", latency_ms(48_000.0, mode, *r).1))
            .collect();
        let _ = writeln!(
            s,
            "| {} | {} | {} ms |",
            mode_name(mode),
            cells44.join(" | "),
            cells48.join(" / ")
        );
    }
    s.push_str("\n(44.1 kHz, samples in parentheses; Live+ at its default 10 ms and Studio at its default 40 ms.)\n\n");
    s.push_str("On this corpus (per-file voice range as chosen above):\n\n");
    s.push_str("| Config | Files per range (LM/AT/S) | Mean D | Voiced content delay mean / median / p95 (ms) | Max \\|wander\\| (periods) | Measured audio delay = D (±1) | |measured − D| med / max (samples) | Median xcorr peak | Amount-0 seg. SNR med (dB) | Pitch-info age med (ms) | Detector onset delay med (ms) | Hard-tune settle med / p90 (ms) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for cfg in &opts.configs {
        let rows: Vec<&DetRow> = o.det.iter().filter(|d| d.config == *cfg).collect();
        if rows.is_empty() {
            continue;
        }
        let cnt = |r: VoiceRange| rows.iter().filter(|d| d.range == r).count();
        let mut wsum = 0.0;
        let mut wdur = 0.0;
        for d in &rows {
            if let Some(x) = o.files.iter().find(|x| x.id == d.file) {
                wsum += f64::from(d.latency) / f64::from(x.sample_rate) * 1000.0 * x.dur_s;
                wdur += x.dur_s;
            }
        }
        let ok = rows
            .iter()
            .filter(|d| {
                d.measured
                    .is_some_and(|m| (m.lag as i64 - i64::from(d.latency)).abs() <= 1)
            })
            .count();
        let peaks: Vec<f64> = rows
            .iter()
            .filter_map(|d| d.measured.map(|m| m.peak))
            .collect();
        let offs: Vec<f64> = rows
            .iter()
            .filter_map(|d| {
                d.measured
                    .map(|m| (m.lag as f64 - f64::from(d.latency)).abs())
            })
            .collect();
        let snr: Vec<f64> = rows.iter().map(|d| d.transparency_snr_db).collect();
        // Voiced content delay pooled over every measured segment (ms).
        let delays: Vec<f64> = rows
            .iter()
            .flat_map(|d| {
                let c = &d.content;
                c.wander
                    .iter()
                    .map(move |w| (f64::from(c.reported) + w) / c.sample_rate * 1e3)
            })
            .collect();
        let wander_max = rows
            .iter()
            .map(|d| d.content.max_abs_wander_periods())
            .fold(f64::NAN, f64::max);
        let age: Vec<f64> = rows.iter().map(|d| d.info_age_ms).collect();
        let det = merged_det(rows.iter().copied());
        let hard = merged_corr(
            o.corr
                .iter()
                .filter(|c| c.config == *cfg && c.condition == Condition::HardChromatic),
        );
        let _ = writeln!(
            s,
            "| {} | {}/{}/{} | {} ms | {} / {} / {} | {} | {}/{} | {} / {} | {} | {} | {} | {} | {} / {} |",
            cfg.label(),
            cnt(VoiceRange::LowMale),
            cnt(VoiceRange::AltoTenor),
            cnt(VoiceRange::Soprano),
            f(wsum / wdur, 2),
            f(crate::dsp::mean(&delays), 2),
            f(crate::dsp::median(&delays), 2),
            f(crate::dsp::percentile(&delays, 95.0), 2),
            f(wander_max, 2),
            ok,
            rows.len(),
            f(crate::dsp::median(&offs), 0),
            f(offs.iter().copied().fold(f64::NAN, f64::max), 0),
            f(crate::dsp::median(&peaks), 3),
            f(crate::dsp::median(&snr), 1),
            f(crate::dsp::median(&age), 2),
            f(det.onset_delay_median(), 1),
            f(hard.settle_median(), 1),
            f(hard.settle_p90(), 1),
        );
    }
    let ref_det = {
        let mut c = DetCounts::default();
        for x in &o.files {
            c.merge(&x.reference);
        }
        c
    };
    let _ = writeln!(
        s,
        "\n*Voiced content delay* = how late the amount-0 output carries the sung input on 20 ms segments inside annotator-1 notes (annotated voiced): D plus the PSOLA content wander, by fractional cross-correlation (R§10.3 target: wander ≤ one period). This is what a performer monitoring through the plugin hears on sustained singing; with PDC on, the host removes D and the wander remains. *Pitch-info age* = when a hop's pitch estimate reaches the correction, how far behind the audio being output it is (hop-boundary time − telemetry output time); ≤ 0 means the correction is computed from pitch at or ahead of the audio it is applied to. Detector onset delay of the offline reference tracker on the same onsets: {} ms (the annotation's own onset convention).\n",
        f(ref_det.onset_delay_median(), 1)
    );
}

fn detector_section(s: &mut String, opts: &VocaditoOptions, o: &Outcome) {
    s.push_str("## Detector on real singing\n\n");
    det_header(s);
    let mut ref_c = DetCounts::default();
    for x in &o.files {
        ref_c.merge(&x.reference);
    }
    let lags: Vec<f64> = o.files.iter().map(|x| x.reference_lag_ms).collect();
    det_row(
        s,
        "offline reference (baseline)",
        &ref_c,
        crate::dsp::median(&lags),
    );
    for cfg in &opts.configs {
        let rows: Vec<&DetRow> = o.det.iter().filter(|d| d.config == *cfg).collect();
        if rows.is_empty() {
            continue;
        }
        let lag = crate::dsp::median(&rows.iter().map(|d| d.best_lag_ms).collect::<Vec<_>>());
        det_row(s, &cfg.label(), &merged_det(rows.iter().copied()), lag);
    }
    s.push_str("\nPooled over all annotation frames of all files. *Ts offset* is the median over files.\n\n");
    s.push_str("By voice range (RPA 50 c / octave errors / voicing F1):\n\n");
    s.push_str("| Tracker | Low Male | Alto/Tenor | Soprano |\n|---|---|---|---|\n");
    let cell = |c: &DetCounts| {
        if c.ref_voiced == 0 {
            "-".to_string()
        } else {
            format!(
                "{} % / {} % / {}",
                pct(c.rpa50(), 1),
                pct(c.octave_rate(), 2),
                f(c.f1(), 3)
            )
        }
    };
    let by_range_ref: Vec<String> = [
        VoiceRange::LowMale,
        VoiceRange::AltoTenor,
        VoiceRange::Soprano,
    ]
    .iter()
    .map(|r| {
        let mut c = DetCounts::default();
        for x in o.files.iter().filter(|x| x.range == *r) {
            c.merge(&x.reference);
        }
        cell(&c)
    })
    .collect();
    let _ = writeln!(s, "| offline reference | {} |", by_range_ref.join(" | "));
    for cfg in &opts.configs {
        let cells: Vec<String> = [
            VoiceRange::LowMale,
            VoiceRange::AltoTenor,
            VoiceRange::Soprano,
        ]
        .iter()
        .map(|r| {
            cell(&merged_det(
                o.det.iter().filter(|d| d.config == *cfg && d.range == *r),
            ))
        })
        .collect();
        let _ = writeln!(s, "| {} | {} |", cfg.label(), cells.join(" | "));
    }
    s.push('\n');
}

fn correction_section(s: &mut String, opts: &VocaditoOptions, o: &Outcome) {
    s.push_str("## Correction on real singing\n\n");
    corr_header(s);
    for cfg in &opts.configs {
        for cond in &opts.conditions {
            let rows: Vec<&CorrRow> = o
                .corr
                .iter()
                .filter(|c| c.config == *cfg && c.condition == *cond)
                .collect();
            if rows.is_empty() {
                continue;
            }
            corr_row(
                s,
                &cfg.label(),
                cond.corr_label(),
                &merged_corr(rows.into_iter()),
            );
        }
    }
    s.push_str("\nPooled over files (frame histograms merged). `passthrough` is the amount-0 render measured like a correction (chromatic grid): the floor of the measurement chain (reference-tracker octave flips on the output, PSOLA wander) for shifts, jumps, dropouts and onset glitches. *In* columns are the unprocessed take measured the same way on the same frames (chromatic grid for `*_chromatic`, the estimated scale for `*_key`).\n\n");
    s.push_str("Where the output error sits (stable frames; share of frames and median / p95 distance to the grid, c):\n\n");
    s.push_str("| Config | Condition | First 100 ms of notes | Wobbly input (vibrato, unsteady) | Steady rest |\n|---|---|---|---|---|\n");
    for cfg in &opts.configs {
        for cond in [
            Condition::HardChromatic,
            Condition::HardKey,
            Condition::NaturalKey,
        ] {
            let c = merged_corr(
                o.corr
                    .iter()
                    .filter(|c| c.config == *cfg && c.condition == cond),
            );
            let tot = (c.out_early.len() + c.out_wobbly.len() + c.out_steady.len()).max(1) as f64;
            let cell = |h: &super::measure::Hist| {
                format!(
                    "{:.0} %: {} / {}",
                    100.0 * h.len() as f64 / tot,
                    f(h.percentile(50.0), 1),
                    f(h.percentile(95.0), 1)
                )
            };
            if c.stable > 0 {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | {} |",
                    cfg.label(),
                    cond.label(),
                    cell(&c.out_early),
                    cell(&c.out_wobbly),
                    cell(&c.out_steady)
                );
            }
        }
    }
    s.push('\n');
    // By range, Live HQ and Studio HQ hard_chromatic.
    s.push_str("Hard-tune chromatic by voice range (out med / p95 c, dropout %, clicks/min):\n\n");
    s.push_str("| Config | Low Male | Alto/Tenor | Soprano |\n|---|---|---|---|\n");
    for cfg in &opts.configs {
        let cells: Vec<String> = [
            VoiceRange::LowMale,
            VoiceRange::AltoTenor,
            VoiceRange::Soprano,
        ]
        .iter()
        .map(|r| {
            let c = merged_corr(o.corr.iter().filter(|c| {
                c.config == *cfg && c.condition == Condition::HardChromatic && c.range == *r
            }));
            if c.stable == 0 {
                "-".to_string()
            } else {
                format!(
                    "{} / {} c, {} %, {}",
                    f(c.out.percentile(50.0), 1),
                    f(c.out.percentile(95.0), 1),
                    f(c.dropout_pct(), 2),
                    f(c.clicks_per_min(), 2)
                )
            }
        })
        .collect();
        let _ = writeln!(s, "| {} | {} |", cfg.label(), cells.join(" | "));
    }
    s.push('\n');
}

fn files_section(s: &mut String, opts: &VocaditoOptions, o: &Outcome) {
    s.push_str("## Files\n\n");
    s.push_str("| File | Singer | Lang | Range | f0 min-max (Hz) | Key (coverage) | A1/A2 notes | A1-A2 med diff (c) | Ref RPA50 | ");
    let det_cfgs: Vec<RunConfig> = opts.configs.clone();
    for c in &det_cfgs {
        let _ = write!(s, "{} RPA50 / oct % / V-rec | ", c.label());
    }
    s.push('\n');
    s.push_str("|---|---|---|---|---|---|---|---|---|");
    for _ in &det_cfgs {
        s.push_str("---|");
    }
    s.push('\n');
    for x in &o.files {
        let _ = write!(
            s,
            "| {} | {} | {} | {} | {:.0}-{:.0} | {} | {}/{} | {} | {} % | ",
            x.id,
            x.meta.singer,
            x.meta.language,
            range_name(x.range),
            x.min_f0,
            x.max_f0,
            x.key.map_or("-".to_string(), |k| format!(
                "{} ({:.0} %)",
                k.name(),
                100.0 * k.coverage
            )),
            x.notes_a1,
            x.notes_a2,
            f(x.annot_diff_c, 1),
            pct(x.reference.rpa50(), 1),
        );
        for c in &det_cfgs {
            match o.det.iter().find(|d| d.file == x.id && d.config == *c) {
                Some(d) => {
                    let _ = write!(
                        s,
                        "{} / {} / {} | ",
                        pct(d.counts.rpa50(), 1),
                        pct(d.counts.octave_rate(), 1),
                        f(d.counts.recall(), 3)
                    );
                }
                None => s.push_str("- | "),
            }
        }
        s.push('\n');
    }
    s.push('\n');
    // Worst correction files for the main configs.
    for cfg in opts.configs.iter().filter(|c| {
        c.tier == vt_core::QualityTier::Hq
            && matches!(c.mode, LatencyMode::Live | LatencyMode::Studio)
    }) {
        let mut rows: Vec<&CorrRow> = o
            .corr
            .iter()
            .filter(|c| c.config == *cfg && c.condition == Condition::HardChromatic)
            .collect();
        if rows.is_empty() {
            continue;
        }
        rows.sort_by(|a, b| {
            b.counts
                .out
                .percentile(95.0)
                .total_cmp(&a.counts.out.percentile(95.0))
        });
        let _ = writeln!(
            s,
            "Worst files, `{}` hard-tune chromatic (by output p95 distance to the grid):\n",
            cfg.label()
        );
        s.push_str("| File | Range | Out med / p95 (c) | In med / p95 (c) | Dropout % | Clicks /min | Shifts | Jumps | Warbles | Onset glitch | Settle med (ms) |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for r in rows.iter().take(8) {
            let c = &r.counts;
            let _ = writeln!(
                s,
                "| {} | {} | {} / {} | {} / {} | {} | {} | {} | {} | {} | {}/{} | {} |",
                r.file,
                range_name(r.range),
                f(c.out.percentile(50.0), 1),
                f(c.out.percentile(95.0), 1),
                f(c.inp.percentile(50.0), 1),
                f(c.inp.percentile(95.0), 1),
                f(c.dropout_pct(), 2),
                f(c.clicks_per_min(), 2),
                c.shift_segments,
                c.jumps,
                c.flickers,
                c.onset_glitches,
                c.onsets_checked,
                f(c.settle_median(), 1),
            );
        }
        s.push('\n');
    }
}

fn severity(g: &Segment) -> f64 {
    let dur = (g.raw.end_s - g.raw.start_s) * 1000.0;
    match g.raw.kind {
        "click" => g.raw.value,
        "flicker" => 10.0 * g.raw.value,
        "onset_spike" | "onset_spike_fmax" => 30.0 * (g.raw.value.abs() / 100.0).min(36.0),
        "out_jump" => g.raw.value.abs() / 10.0,
        "gross_shift" | "octave_err" | "gross_err" => dur * (g.raw.value.abs() / 100.0).min(12.0),
        _ => dur,
    }
}

fn segments_section(s: &mut String, opts: &VocaditoOptions, o: &Outcome) {
    s.push_str("## Failure segments\n\n");
    s.push_str("All spans are in `segments.csv`. Counts per kind (sum over files):\n\n");
    let mut kinds: Vec<&str> = o.segments.iter().map(|g| g.raw.kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let mut cols: Vec<(String, &str)> = vec![("reference".to_string(), "detect")];
    for cfg in &opts.configs {
        for cond in &opts.conditions {
            cols.push((cfg.label(), cond.label()));
            if *cond == Condition::Detect {
                cols.push((cfg.label(), cond.corr_label()));
            }
        }
    }
    let mut table: BTreeMap<(&str, usize), usize> = BTreeMap::new();
    for g in &o.segments {
        if let Some(ci) = cols
            .iter()
            .position(|(c, d)| *c == g.config && *d == g.condition)
        {
            *table.entry((g.raw.kind, ci)).or_default() += 1;
        }
    }
    let used: Vec<usize> = (0..cols.len())
        .filter(|ci| kinds.iter().any(|k| table.contains_key(&(*k, *ci))))
        .collect();
    s.push_str("| Kind |");
    for ci in &used {
        let _ = write!(s, " {} {} |", cols[*ci].0, cols[*ci].1);
    }
    s.push_str("\n|---|");
    for _ in &used {
        s.push_str("---|");
    }
    s.push('\n');
    for k in &kinds {
        let _ = write!(s, "| {k} |");
        for ci in &used {
            let _ = write!(s, " {} |", table.get(&(*k, *ci)).copied().unwrap_or(0));
        }
        s.push('\n');
    }
    s.push('\n');

    // Tag frequencies per kind for the Live HQ detector + hard-tune runs.
    let focus = |g: &Segment| {
        g.config == "reference"
            || g.config.starts_with("live_hq")
            || g.config.starts_with("studio_hq")
            || g.config.starts_with("livemin")
    };
    let all_tags = [
        "low_f0",
        "range_edge",
        "aperiodic",
        "quiet",
        "onset",
        "release",
        "fast_pitch",
        "melisma",
        "pitch_present",
        "tail_after_note",
    ];
    s.push_str("Cause tags (share of spans carrying each tag; reference + live_hq + studio_hq + livemin runs):\n\n");
    s.push_str("| Kind | Spans |");
    for t in all_tags {
        let _ = write!(s, " {t} |");
    }
    s.push_str("\n|---|---|");
    for _ in all_tags {
        s.push_str("---|");
    }
    s.push('\n');
    for k in &kinds {
        let spans: Vec<&Segment> = o
            .segments
            .iter()
            .filter(|g| g.raw.kind == *k && focus(g))
            .collect();
        if spans.is_empty() {
            continue;
        }
        let _ = write!(s, "| {k} | {} |", spans.len());
        for t in all_tags {
            let n = spans.iter().filter(|g| g.tags.contains(&t)).count();
            let _ = write!(s, " {:.0} % |", 100.0 * n as f64 / spans.len() as f64);
        }
        s.push('\n');
    }
    s.push_str("\nTags: `low_f0` annotated f0 < 110 Hz; `range_edge` < 1.15 × the preset floor; `aperiodic` input reference clarity < 0.75 (breathy, fry, noise, consonant); `quiet` > 15 dB below the file's voiced median; `onset` / `release` within 80 ms of an annotated voicing onset / offset; `fast_pitch` annotated pitch moving > 1.5 c/ms; `melisma` ≥ 3 notes within ±200 ms with median duration < 200 ms; `pitch_present` (false voicing) the reference tracker hears a clear pitch there; `tail_after_note` false voicing starting at an annotated offset (release / reverb tail).\n\n");

    // Worst spans for the main configs.
    for (cfgs, conds, title) in [
        (
            vec!["live_hq_natural", "studio_hq_natural", "livemin_hq_classic"],
            vec!["detect"],
            "Worst detector spans (Live HQ, Studio HQ, Live-Min)",
        ),
        (
            vec!["live_hq_natural", "studio_hq_natural", "livemin_hq_classic"],
            vec![
                "hard_chromatic",
                "natural_key",
                "hard_key",
                "natural_chromatic",
                "passthrough",
            ],
            "Worst correction spans (Live HQ, Studio HQ, Live-Min; spans also present in the same config's amount-0 pass-through are left out)",
        ),
    ] {
        let control: Vec<&Segment> = o
            .segments
            .iter()
            .filter(|p| p.condition == "passthrough")
            .collect();
        let in_control = |g: &Segment| {
            g.condition != "passthrough"
                && g.condition != "detect"
                && control.iter().any(|p| {
                    p.file == g.file
                        && p.config == g.config
                        && p.raw.kind == g.raw.kind
                        && p.raw.start_s <= g.raw.end_s + 0.03
                        && p.raw.end_s + 0.03 >= g.raw.start_s
                })
        };
        let mut v: Vec<&Segment> = o
            .segments
            .iter()
            .filter(|g| cfgs.contains(&g.config.as_str()) && conds.contains(&g.condition))
            .filter(|g| g.raw.kind != "false_voicing" || severity(g) > 150.0)
            .filter(|g| g.condition != "passthrough" && !in_control(g))
            .collect();
        v.sort_by(|a, b| severity(b).total_cmp(&severity(a)));
        let _ = writeln!(s, "{title}:\n");
        s.push_str("| File (singer) | Config | Condition | Kind | Time (s) | Dur (ms) | Value | Ann f0 (Hz) | Level (dBFS) | Clarity | Tags |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for g in v.iter().take(40) {
            let _ = writeln!(
                s,
                "| {} ({}) | {} | {} | {} | {:.3}-{:.3} | {:.0} | {} | {} | {} | {} | {} |",
                g.file,
                g.singer,
                g.config,
                g.condition,
                g.raw.kind,
                g.raw.start_s,
                g.raw.end_s,
                (g.raw.end_s - g.raw.start_s) * 1000.0,
                f(g.raw.value, 0),
                f(g.ann_f0, 0),
                f(g.level_db, 1),
                f(g.clarity, 2),
                g.tags.join(", ")
            );
        }
        s.push('\n');
    }
    s.push_str("Values: pitch kinds = median Δ (cents, estimate − annotation or output − input); voicing kinds and dropouts = duration (ms); click = peak excess (dB); out_jump = largest 5 ms output move (cents); flicker (warble episode) = number of A→B→A target flips; onset_spike = median Δ of the bad frames (cents); onset_glitch = ms of bad output in the note's first 100 ms.\n");
}
