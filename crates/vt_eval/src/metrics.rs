//! Objective metrics for rendered outputs (DESIGN §3.7, §4; R§10.3).
//!
//! All pitch metrics compare reference-tracker frames of the
//! latency-compensated output against the signal's ground truth. Because the
//! output is aligned by the reported latency D, an output frame at sample n
//! describes the same note as input sample n (PSOLA content wander of up to
//! one period is far inside the ±30 ms boundary exclusion).

use vt_core::{PitchFrame, frame_flags};

use crate::dsp::{BUTTER4_Q, Biquad64, Correlator64, KaiserSinc, decimate, filtfilt_padded, mean, next_pow2, percentile_sorted, rms};
use crate::reference::PitchTrack;
use crate::synth::{NoteKind, NoteSpan, Signal};

/// Summary of an error distribution in cents.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TuningStats {
    /// Frames that entered the statistics.
    pub frames: usize,
    /// Stable truth frames where the output had no pitch (counted separately).
    pub unvoiced: usize,
    pub median_abs: f64,
    pub p95_abs: f64,
    /// Signed mean (bias), cents.
    pub mean: f64,
    /// Percentage of frames within 5 / 10 / 25 c.
    pub within_5: f64,
    pub within_10: f64,
    pub within_25: f64,
}

impl TuningStats {
    pub fn from_errors(errors: &[f64], unvoiced: usize) -> Self {
        let mut abs: Vec<f64> = errors.iter().filter(|e| e.is_finite()).map(|e| e.abs()).collect();
        if abs.is_empty() {
            return Self { unvoiced, median_abs: f64::NAN, p95_abs: f64::NAN, mean: f64::NAN, ..Self::default() };
        }
        abs.sort_by(f64::total_cmp);
        let n = abs.len() as f64;
        let within = |c: f64| 100.0 * abs.iter().filter(|e| **e <= c).count() as f64 / n;
        Self {
            frames: abs.len(),
            unvoiced,
            median_abs: percentile_sorted(&abs, 50.0),
            p95_abs: percentile_sorted(&abs, 95.0),
            mean: mean(&errors.iter().copied().filter(|e| e.is_finite()).collect::<Vec<_>>()),
            within_5: within(5.0),
            within_10: within(10.0),
            within_25: within(25.0),
        }
    }
}

/// Options for [`tuning`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuningOptions {
    /// Exclude this much around every note boundary, s (R§10.3: 30 ms).
    pub boundary_guard_s: f64,
    /// Also score sweeps against their per-sample nearest-semitone target
    /// (the "staircase" a hard-tuned glide should become), excluding the
    /// guard around every target change.
    pub include_sweeps: bool,
}

impl Default for TuningOptions {
    fn default() -> Self {
        Self { boundary_guard_s: 0.03, include_sweeps: false }
    }
}

/// Mean of the piecewise-linear interpolant of `vals` over `width` frames
/// centred on frame `k` (trapezoidal rule); None if any covered frame is
/// missing (NaN) or out of range. A window exactly one vibrato period long
/// cancels a sinusoidal vibrato (residual O((2π/width)²/12)), leaving its centre.
fn boxcar(vals: &[f64], k: usize, width: f64) -> Option<f64> {
    let h = 0.5 * width.max(1.0);
    let (a, b) = (k as f64 - h, k as f64 + h);
    if a < 0.0 {
        return None;
    }
    let (ia, ib) = (a.floor() as usize, b.ceil() as usize);
    let window = vals.get(ia..=ib)?;
    if window.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let at = |x: f64| -> f64 {
        let i = x.floor() as usize;
        let f = x - i as f64;
        let v0 = vals[i];
        if f > 0.0 { v0 + (vals[i + 1] - v0) * f } else { v0 }
    };
    let mut acc = 0.0;
    let mut x0 = a;
    while x0 < b {
        let x1 = (x0.floor() + 1.0).min(b);
        acc += 0.5 * (at(x0) + at(x1)) * (x1 - x0);
        x0 = x1;
    }
    Some(acc / (b - a))
}

/// Cents per frame (NaN when unvoiced).
fn cents_series(track: &PitchTrack) -> Vec<f64> {
    track.frames.iter().map(|f| if f.voiced { f.cents } else { f64::NAN }).collect()
}

/// Frame index range strictly inside `[a, b)` samples.
fn frames_in(track: &PitchTrack, a: usize, b: usize) -> std::ops::Range<usize> {
    let hop = track.hop.max(1);
    let lo = a.div_ceil(hop);
    let hi = b.div_ceil(hop).min(track.frames.len());
    lo..hi.max(lo)
}

fn guard_samples(sig: &Signal, s: f64) -> usize {
    (s * sig.sample_rate as f64).round() as usize
}

/// Tuning error of the output against the intended notes on stable voiced
/// frames: ±guard around every note boundary excluded, sweeps skipped,
/// vibrato notes measured on their centre (one-period boxcar of the output).
pub fn tuning(sig: &Signal, out: &PitchTrack, opts: &TuningOptions) -> TuningStats {
    let guard = guard_samples(sig, opts.boundary_guard_s);
    let cents = cents_series(out);
    let hop_s = out.hop as f64 / out.sample_rate;
    let mut errors = Vec::new();
    let mut unvoiced = 0;
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note && n.target_cents.is_finite()) {
        for k in frames_in(out, note.start + guard, note.end.saturating_sub(guard)) {
            let pos = k * out.hop;
            if !sig.is_voiced(pos) {
                continue;
            }
            let Some(c) = cents.get(k).copied().filter(|c| c.is_finite()) else {
                unvoiced += 1;
                continue;
            };
            let value = match note.vibrato {
                Some(v) if v.rate_hz > 0.0 && (pos - note.start) as f64 / out.sample_rate >= v.delay_s => {
                    // The one-period window must stay inside the note.
                    let width = 1.0 / (v.rate_hz * hop_s);
                    let half = (0.5 * width * out.hop as f64).ceil() as usize + out.hop;
                    if pos < note.start + half || pos + half >= note.end {
                        continue;
                    }
                    match boxcar(&cents, k, width) {
                        Some(m) => m,
                        None => continue,
                    }
                }
                _ => c,
            };
            errors.push(value - note.target_cents);
        }
    }
    if opts.include_sweeps {
        for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Sweep) {
            // Positions where the per-sample target changes.
            let changes: Vec<usize> = (note.start + 1..note.end)
                .filter(|&i| sig.target_cents.get(i) != sig.target_cents.get(i - 1))
                .collect();
            for k in frames_in(out, note.start + guard, note.end.saturating_sub(guard)) {
                let pos = k * out.hop;
                let i = changes.partition_point(|c| *c < pos);
                let near = changes.get(i).is_some_and(|c| c - pos <= guard)
                    || (i > 0 && changes.get(i - 1).is_some_and(|c| pos - c <= guard));
                let target = sig.target_cents.get(pos).copied().unwrap_or(f32::NAN) as f64;
                if near || !target.is_finite() || !sig.is_voiced(pos) {
                    continue;
                }
                match cents.get(k).copied().filter(|c| c.is_finite()) {
                    Some(c) => errors.push(c - target),
                    None => unvoiced += 1,
                }
            }
        }
    }
    TuningStats::from_errors(&errors, unvoiced)
}

/// Settling after note onsets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettlingStats {
    /// Seconds from each evaluated onset until the output stays within
    /// tolerance for the hold time; None if it never did within the note.
    pub per_note: Vec<Option<f64>>,
    pub settled: usize,
    pub median_s: f64,
    pub p90_s: f64,
    pub max_s: f64,
}

/// Time until the output stays within `tol_cents` of the target for `hold_s`
/// (R§10.3: 10 c for 20 ms) after each onset. Evaluates notes that start after
/// a voicing gap (or every note when `include_legato`), without vibrato.
pub fn settling(sig: &Signal, out: &PitchTrack, tol_cents: f64, hold_s: f64, include_legato: bool) -> SettlingStats {
    let cents = cents_series(out);
    let hold = ((hold_s * out.sample_rate) / out.hop as f64).ceil() as usize;
    let mut per_note = Vec::new();
    for note in sig.notes.iter().filter(|n| {
        n.kind == NoteKind::Note && n.vibrato.is_none() && n.target_cents.is_finite() && (include_legato || !n.legato_in)
    }) {
        let range = frames_in(out, note.start, note.end);
        let ok = |k: usize| cents.get(k).is_some_and(|c| (c - note.target_cents).abs() <= tol_cents);
        let mut found = None;
        for k in range.clone() {
            if k + hold >= range.end {
                break;
            }
            if (k..=k + hold).all(ok) {
                found = Some((k * out.hop).saturating_sub(note.start) as f64 / out.sample_rate);
                break;
            }
        }
        per_note.push(found);
    }
    let mut t: Vec<f64> = per_note.iter().flatten().copied().collect();
    t.sort_by(f64::total_cmp);
    SettlingStats {
        settled: t.len(),
        median_s: percentile_sorted(&t, 50.0),
        p90_s: percentile_sorted(&t, 90.0),
        max_s: t.last().copied().unwrap_or(f64::NAN),
        per_note,
    }
}

/// Vibrato measured on one note.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VibratoNote {
    pub start: usize,
    /// Half peak-to-peak extent (√2·RMS of the detrended contour), cents.
    pub in_extent: f64,
    pub out_extent: f64,
    pub in_rate_hz: f64,
    pub out_rate_hz: f64,
    /// Median |output centre − target|, cents.
    pub centre_error: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VibratoStats {
    pub notes: Vec<VibratoNote>,
    /// max |out_extent / in_extent − 1| over the notes.
    pub worst_extent_ratio_error: f64,
    /// Median |output centre − target| over all evaluated frames, cents.
    pub centre_error_median: f64,
    /// max |out_rate − in_rate|, Hz.
    pub worst_rate_error_hz: f64,
}

/// Extent and rate of the detrended contour (cents minus one-period boxcar).
fn vibrato_shape(cents: &[f64], range: std::ops::Range<usize>, period_frames: f64, hop_s: f64) -> Option<(f64, f64, Vec<f64>)> {
    let mut dev = Vec::new();
    let mut times = Vec::new();
    let mut centres = Vec::new();
    for k in range {
        let (Some(c), Some(m)) = (cents.get(k).copied().filter(|c| c.is_finite()), boxcar(cents, k, period_frames)) else {
            continue;
        };
        dev.push(c - m);
        times.push(k as f64 * hop_s);
        centres.push(m);
    }
    if dev.len() < 8 {
        return None;
    }
    let extent = std::f64::consts::SQRT_2 * rms(&dev);
    // Upward zero crossings, linearly interpolated.
    let mut ups = Vec::new();
    for i in 1..dev.len() {
        if dev[i - 1] < 0.0 && dev[i] >= 0.0 && (times[i] - times[i - 1]) < 2.5 * hop_s {
            let f = dev[i - 1] / (dev[i - 1] - dev[i]);
            ups.push(times[i - 1] + f * (times[i] - times[i - 1]));
        }
    }
    let rate = match (ups.first(), ups.last()) {
        (Some(a), Some(b)) if ups.len() >= 2 && b > a => (ups.len() - 1) as f64 / (b - a),
        _ => f64::NAN,
    };
    Some((extent, rate, centres))
}

/// Vibrato preservation: output extent and rate vs the input's (both measured
/// by the same tracker), and the output centre's tuning, on every vibrato note.
pub fn vibrato(sig: &Signal, in_track: &PitchTrack, out_track: &PitchTrack, guard_s: f64) -> VibratoStats {
    let hop_s = out_track.hop as f64 / out_track.sample_rate;
    let cin = cents_series(in_track);
    let cout = cents_series(out_track);
    let mut notes = Vec::new();
    let mut all_centre_err = Vec::new();
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note) {
        let Some(v) = note.vibrato.filter(|v| v.rate_hz > 0.0 && v.extent_cents > 0.0) else { continue };
        let sr = sig.sample_rate as f64;
        let period = 1.0 / v.rate_hz;
        let a = note.start + ((v.delay_s + v.ramp_s + 0.5 * period) * sr) as usize;
        let b = note.end.saturating_sub(((guard_s + 0.5 * period) * sr) as usize);
        if b <= a {
            continue;
        }
        let range = frames_in(out_track, a, b);
        let pf = period / hop_s;
        let (Some((ie, ir, _)), Some((oe, or, oc))) =
            (vibrato_shape(&cin, range.clone(), pf, hop_s), vibrato_shape(&cout, range, pf, hop_s))
        else {
            continue;
        };
        let mut errs: Vec<f64> = oc.iter().map(|c| (c - note.target_cents).abs()).collect();
        all_centre_err.extend(errs.iter().copied());
        errs.sort_by(f64::total_cmp);
        notes.push(VibratoNote {
            start: note.start,
            in_extent: ie,
            out_extent: oe,
            in_rate_hz: ir,
            out_rate_hz: or,
            centre_error: percentile_sorted(&errs, 50.0),
        });
    }
    all_centre_err.sort_by(f64::total_cmp);
    VibratoStats {
        worst_extent_ratio_error: notes.iter().map(|n| (n.out_extent / n.in_extent - 1.0).abs()).fold(f64::NAN, f64::max),
        worst_rate_error_hz: notes.iter().map(|n| (n.out_rate_hz - n.in_rate_hz).abs()).fold(f64::NAN, f64::max),
        centre_error_median: percentile_sorted(&all_centre_err, 50.0),
        notes,
    }
}

/// High-frequency pitch fluctuation on steady notes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterStats {
    pub frames: usize,
    /// RMS of the high-passed output deviation from the target, cents.
    pub out_rms: f64,
    /// Same for the input (the singer's own jitter + tracker floor).
    pub in_rms: f64,
    /// √max(0, out² − in²): what the engine added.
    pub added_rms: f64,
}

/// Added jitter (R§10.3): RMS of the zero-phase high-passed (> `cutoff_hz`)
/// output pitch deviation on steady notes (no vibrato, ±60 ms from the
/// boundaries), against the same measure on the input.
pub fn jitter(sig: &Signal, in_track: &PitchTrack, out_track: &PitchTrack, cutoff_hz: f64) -> JitterStats {
    let frame_rate = out_track.sample_rate / out_track.hop as f64;
    let guard = guard_samples(sig, 0.06);
    let trim = (0.03 * frame_rate).ceil() as usize;
    let min_run = (0.1 * frame_rate).ceil() as usize;
    let hp = [Biquad64::highpass(frame_rate, cutoff_hz, std::f64::consts::FRAC_1_SQRT_2)];
    let cin = cents_series(in_track);
    let cout = cents_series(out_track);
    let (mut so, mut si, mut n) = (0.0, 0.0, 0usize);
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note && n.vibrato.is_none() && n.target_cents.is_finite()) {
        let range = frames_in(out_track, note.start + guard, note.end.saturating_sub(guard));
        // Split into runs where both tracks are voiced.
        let mut k = range.start;
        while k < range.end {
            let both = |i: usize| cin.get(i).is_some_and(|c| c.is_finite()) && cout.get(i).is_some_and(|c| c.is_finite());
            if !both(k) {
                k += 1;
                continue;
            }
            let s = k;
            while k < range.end && both(k) {
                k += 1;
            }
            if k - s < min_run {
                continue;
            }
            let eo: Vec<f64> = cout[s..k].iter().map(|c| c - note.target_cents).collect();
            let ei: Vec<f64> = cin[s..k].iter().map(|c| c - note.target_cents).collect();
            let pad = (3.0 * frame_rate / cutoff_hz).ceil() as usize;
            let (fo, fi) = (filtfilt_padded(&hp, &eo, pad), filtfilt_padded(&hp, &ei, pad));
            for (a, b) in fo.iter().zip(fi.iter()).skip(trim).take((k - s).saturating_sub(2 * trim)) {
                so += a * a;
                si += b * b;
                n += 1;
            }
        }
    }
    if n == 0 {
        return JitterStats { frames: 0, out_rms: f64::NAN, in_rms: f64::NAN, added_rms: f64::NAN };
    }
    let (o, i) = ((so / n as f64).sqrt(), (si / n as f64).sqrt());
    JitterStats { frames: n, out_rms: o, in_rms: i, added_rms: (o * o - i * i).max(0.0).sqrt() }
}

/// Binary voicing agreement (positive class = voiced).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoicingStats {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl VoicingStats {
    fn from_counts(tp: usize, fp: usize, fn_: usize, tn: usize) -> Self {
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { f64::NAN };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { f64::NAN };
        let f1 = if tp > 0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        Self { tp, fp, fn_, tn, precision, recall, f1 }
    }
}

/// Voicing F1 of the engine's telemetry against the truth. Telemetry
/// timestamps are output time, so input time = `sample_pos − latency`; frames
/// within `guard_s` of a truth voicing transition are skipped.
pub fn voicing_f1(sig: &Signal, telemetry: &[PitchFrame], latency: u32, guard_s: f64) -> VoicingStats {
    let guard = guard_samples(sig, guard_s);
    let transitions = sig.voicing_transitions();
    let near = |p: usize| {
        let i = transitions.partition_point(|t| *t < p);
        let after = transitions.get(i).is_some_and(|t| t - p <= guard);
        let before = i > 0 && transitions.get(i - 1).is_some_and(|t| p - t <= guard);
        after || before
    };
    let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
    for f in telemetry {
        let Some(p) = f.sample_pos.checked_sub(latency as u64).map(|p| p as usize) else { continue };
        if p >= sig.len() || near(p) {
            continue;
        }
        let truth = sig.is_voiced(p);
        let pred = f.flags & frame_flags::VOICED != 0;
        match (truth, pred) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    VoicingStats::from_counts(tp, fp, fn_, tn)
}

/// Voicing F1 of a reference track against the truth (same conventions).
pub fn track_voicing_f1(sig: &Signal, track: &PitchTrack, guard_s: f64) -> VoicingStats {
    let frames: Vec<PitchFrame> = track
        .frames
        .iter()
        .map(|f| PitchFrame { sample_pos: f.pos as u64, flags: if f.voiced { frame_flags::VOICED } else { 0 }, ..PitchFrame::default() })
        .collect();
    voicing_f1(sig, &frames, 0, guard_s)
}

/// Result of a cross-correlation delay measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyMeasurement {
    /// Integer lag with the highest normalised correlation, samples.
    pub lag: usize,
    /// Parabolically refined lag, samples.
    pub lag_fine: f64,
    /// Normalised correlation at the peak (1 = exact delayed copy).
    pub peak: f64,
}

/// Delay of `output` relative to `input` (R§10.6): normalised cross-correlation
/// over lags 0..=`max_lag`. Use aperiodic material (noise bursts) so the peak
/// is unique.
pub fn measure_latency(input: &[f32], output: &[f32], max_lag: usize) -> Option<LatencyMeasurement> {
    let a: Vec<f64> = input.iter().map(|v| if v.is_finite() { *v as f64 } else { 0.0 }).collect();
    let b: Vec<f64> = output.iter().map(|v| if v.is_finite() { *v as f64 } else { 0.0 }).collect();
    let e_a: f64 = a.iter().map(|v| v * v).sum();
    if e_a <= 0.0 || b.is_empty() {
        return None;
    }
    let n = next_pow2(a.len() + b.len() + max_lag + 2);
    let mut corr = Correlator64::new(n);
    let mut r = vec![0.0; max_lag + 1];
    corr.cross_correlate(&a, &b, &mut r);
    let mut prefix = vec![0.0; b.len() + 1];
    for (i, v) in b.iter().enumerate() {
        prefix[i + 1] = prefix[i] + v * v;
    }
    let norm: Vec<f64> = r
        .iter()
        .enumerate()
        .map(|(l, rv)| {
            let lo = l.min(b.len());
            let hi = (l + a.len()).min(b.len());
            let e_b = prefix[hi] - prefix[lo];
            if e_b > 0.0 { rv / (e_a * e_b).sqrt() } else { 0.0 }
        })
        .collect();
    let (lag, peak) = norm.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, v)| (i, *v))?;
    let lag_fine = match (lag.checked_sub(1).and_then(|i| norm.get(i)), norm.get(lag + 1)) {
        (Some(&ym1), Some(&yp1)) => {
            let den = ym1 - 2.0 * peak + yp1;
            if den.abs() > 1e-15 { lag as f64 + (0.5 * (ym1 - yp1) / den).clamp(-0.5, 0.5) } else { lag as f64 }
        }
        _ => lag as f64,
    };
    Some(LatencyMeasurement { lag, lag_fine, peak })
}

/// High-frequency click/buzz detector (R§10.3: energy above 8 kHz at grain rate).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClickStats {
    pub frames: usize,
    /// Frames whose >8 kHz energy exceeds the input's by more than 10 dB.
    pub clicks: usize,
    /// Largest excess over the input's local >8 kHz energy, dB.
    pub max_excess_db: f64,
}

/// Compare >8 kHz energy of the aligned output with the input's, frame by
/// frame (2 ms frames, 1 ms hop). The input reference is the maximum over
/// ±`wander_s`, so PSOLA's content wander is not mistaken for clicks.
pub fn clicks(input: &[f32], output_aligned: &[f32], sample_rate: f32, wander_s: f64) -> ClickStats {
    let sr = sample_rate as f64;
    let hf = |x: &[f32]| -> Vec<f64> {
        let mut f: Vec<Biquad64> = BUTTER4_Q.iter().map(|q| Biquad64::highpass(sr, 8000.0, *q)).collect();
        let hp: Vec<f64> = x.iter().map(|v| f.iter_mut().fold(if v.is_finite() { *v as f64 } else { 0.0 }, |a, s| s.process(a))).collect();
        let len = (0.002 * sr) as usize;
        let hop = (0.001 * sr).max(1.0) as usize;
        (0..hp.len().saturating_sub(len) / hop)
            .map(|k| {
                let seg = &hp[k * hop..k * hop + len];
                10.0 * (seg.iter().map(|v| v * v).sum::<f64>() / len as f64).max(1e-20).log10()
            })
            .collect()
    };
    let ein = hf(input);
    let eout = hf(output_aligned);
    let reach = (wander_s * 1000.0).ceil() as usize;
    let mut stats = ClickStats { max_excess_db: f64::NEG_INFINITY, ..ClickStats::default() };
    for (k, eo) in eout.iter().enumerate() {
        let lo = k.saturating_sub(reach);
        let hi = (k + reach + 1).min(ein.len());
        let reference = ein.get(lo..hi).map_or(-200.0, |s| s.iter().cloned().fold(-200.0, f64::max));
        let excess = eo - reference.max(-90.0);
        stats.frames += 1;
        stats.max_excess_db = stats.max_excess_db.max(excess);
        if excess > 10.0 && *eo > -70.0 {
            stats.clicks += 1;
        }
    }
    stats
}

/// LPC-envelope log-spectral distance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LsdStats {
    pub frames: usize,
    pub mean_db: f64,
    pub median_db: f64,
}

/// Autocorrelation-method LPC by Levinson-Durbin; returns (a[0..=p], error).
fn lpc(frame: &[f64], order: usize) -> Option<(Vec<f64>, f64)> {
    let n = frame.len();
    let r: Vec<f64> = (0..=order).map(|k| frame.iter().zip(frame.iter().skip(k)).map(|(a, b)| a * b).sum::<f64>()).collect();
    let r0 = *r.first()?;
    if r0 <= 1e-20 || n <= order {
        return None;
    }
    // Light lag window (white-noise correction) for numerical robustness.
    let mut r = r;
    r[0] *= 1.0 + 1e-9;
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=order {
        let acc: f64 = (1..i).map(|j| a[j] * r[i - j]).sum::<f64>() + r[i];
        let k = -acc / err;
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            return None;
        }
    }
    Some((a, err))
}

/// LPC envelope power (dB) of `a`/`err` on frequencies `freqs` (Hz).
fn lpc_env_db(a: &[f64], err: f64, freqs: &[f64], sr: f64) -> Vec<f64> {
    freqs
        .iter()
        .map(|f| {
            let w = 2.0 * std::f64::consts::PI * f / sr;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, c) in a.iter().enumerate() {
                re += c * (w * k as f64).cos();
                im -= c * (w * k as f64).sin();
            }
            10.0 * (err / (re * re + im * im).max(1e-30)).log10()
        })
        .collect()
}

/// Formant-envelope log-spectral distance between input and aligned output
/// (DESIGN §3.4: < 3 dB for PSOLA). Both are decimated to ~12 kHz, then on
/// 40 ms Hann frames (20 ms hop) fully inside notes (±30 ms excluded) an
/// order-14 LPC envelope is compared on 64 log-spaced points over 100 Hz-5 kHz,
/// gain-normalised (only the shape counts).
pub fn formant_lsd(sig: &Signal, output_aligned: &[f32]) -> LsdStats {
    let sr = sig.sample_rate as f64;
    let factor = ((sr / 12_000.0).round() as usize).max(1);
    let dsr = sr / factor as f64;
    let to64 = |x: &[f32]| -> Vec<f64> { x.iter().map(|v| if v.is_finite() { *v as f64 } else { 0.0 }).collect() };
    let xi = decimate(&to64(&sig.audio), factor);
    let xo = decimate(&to64(output_aligned), factor);
    let flen = (0.04 * dsr) as usize;
    let fhop = (0.02 * dsr) as usize;
    let window: Vec<f64> = (0..flen).map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / flen as f64).sin().powi(2)).collect();
    let freqs: Vec<f64> = (0..64).map(|i| 100.0 * (50.0f64).powf(i as f64 / 63.0)).collect();
    let guard = guard_samples(sig, 0.03);
    let prep = |x: &[f64], s: usize| -> Vec<f64> {
        (0..flen)
            .map(|i| {
                let cur = x.get(s + i).copied().unwrap_or(0.0);
                let prev = if s + i > 0 { x.get(s + i - 1).copied().unwrap_or(0.0) } else { 0.0 };
                (cur - 0.97 * prev) * window[i]
            })
            .collect()
    };
    let mut lsd = Vec::new();
    let mut s = 0usize;
    while s + flen <= xi.len().min(xo.len()) {
        let (a, b) = (s * factor, (s + flen) * factor);
        let inside = sig.notes.iter().any(|n: &NoteSpan| a >= n.start + guard && b + guard <= n.end);
        if inside
            && let (Some((ai, ei)), Some((ao, eo))) = (lpc(&prep(&xi, s), 14), lpc(&prep(&xo, s), 14))
        {
            let di = lpc_env_db(&ai, ei, &freqs, dsr);
            let dout = lpc_env_db(&ao, eo, &freqs, dsr);
            let diff: Vec<f64> = di.iter().zip(dout.iter()).map(|(p, q)| p - q).collect();
            let m = mean(&diff);
            lsd.push(rms(&diff.iter().map(|d| d - m).collect::<Vec<_>>()));
        }
        s += fhop;
    }
    lsd.sort_by(f64::total_cmp);
    LsdStats { frames: lsd.len(), mean_db: mean(&lsd), median_db: percentile_sorted(&lsd, 50.0) }
}

/// Segmental SNR of `test` against `reference` after aligning each segment by
/// the best fractional delay in [`min_lag`, `max_lag`] samples (PSOLA content
/// wander, DESIGN §3.4). Segments of `seg_len` samples with reference energy
/// below −60 dBFS are skipped; per-segment SNR is clamped to [−20, 100] dB.
/// Returns (mean SNR in dB, segments used).
pub fn aligned_segmental_snr(reference: &[f32], test: &[f32], seg_len: usize, min_lag: f64, max_lag: f64) -> (f64, usize) {
    let r: Vec<f64> = reference.iter().map(|v| *v as f64).collect();
    let t: Vec<f64> = test.iter().map(|v| *v as f64).collect();
    let sinc = KaiserSinc::new(16, 8.6);
    let mut snrs = Vec::new();
    let seg_len = seg_len.max(16);
    let mut s = 0;
    while s + seg_len <= t.len().min(r.len()) {
        let seg = &t[s..s + seg_len];
        let e_ref: f64 = r[s..s + seg_len].iter().map(|v| v * v).sum::<f64>() / seg_len as f64;
        if e_ref < 1e-6 {
            s += seg_len;
            continue;
        }
        let shifted = |lag: f64| -> Vec<f64> { (0..seg_len).map(|i| sinc.interp(&r, (s + i) as f64 - lag)).collect() };
        let score = |lag: f64| -> f64 {
            let y = shifted(lag);
            let num: f64 = y.iter().zip(seg).map(|(a, b)| a * b).sum();
            let den = (y.iter().map(|a| a * a).sum::<f64>() * seg.iter().map(|b| b * b).sum::<f64>()).sqrt();
            if den > 0.0 { num / den } else { 0.0 }
        };
        // Integer scan (plain shifts), then golden-section refinement of the
        // sinc-interpolated score around the best lag.
        let lo = min_lag.floor() as i64;
        let hi = max_lag.ceil() as i64;
        let (mut best_l, mut best_v) = (0.0, f64::NEG_INFINITY);
        for l in lo..=hi {
            let (mut num, mut ey, mut et) = (0.0, 0.0, 0.0);
            for (i, b) in seg.iter().enumerate() {
                let idx = (s + i) as i64 - l;
                let a = if idx < 0 { 0.0 } else { r.get(idx as usize).copied().unwrap_or(0.0) };
                num += a * b;
                ey += a * a;
                et += b * b;
            }
            let v = if ey * et > 0.0 { num / (ey * et).sqrt() } else { 0.0 };
            if v > best_v {
                best_v = v;
                best_l = l as f64;
            }
        }
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut a, mut b) = (best_l - 1.0, best_l + 1.0);
        for _ in 0..30 {
            let c = b - g * (b - a);
            let d = a + g * (b - a);
            if score(c) > score(d) {
                b = d;
            } else {
                a = c;
            }
        }
        let y = shifted(0.5 * (a + b));
        let sig_e: f64 = y.iter().map(|v| v * v).sum();
        let noise: f64 = y.iter().zip(seg).map(|(p, q)| (p - q) * (p - q)).sum();
        let snr = if noise > 0.0 { 10.0 * (sig_e / noise).log10() } else { 100.0 };
        snrs.push(snr.clamp(-20.0, 100.0));
        s += seg_len;
    }
    (mean(&snrs), snrs.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::RefFrame;
    use crate::synth::{Vibrato, cents_to_hz64};

    /// Signal with two notes (truth only) and a synthetic track built from a
    /// closure, so metrics can be checked against known answers.
    fn fixture(f: impl Fn(f64) -> Option<f64>) -> (Signal, PitchTrack) {
        let sr = 48_000.0f32;
        let n = 96_000;
        let mut sig = Signal::from_audio(vec![0.0; n], sr);
        let notes = [(4800usize, 43_200usize, 5700.0), (48_000, 91_200, 5900.0)];
        for &(a, b, c) in &notes {
            for i in a..b {
                sig.f0_truth[i] = cents_to_hz64(c) as f32;
                sig.target_cents[i] = c as f32;
            }
            sig.notes.push(NoteSpan {
                start: a,
                end: b,
                target_cents: c,
                detune_cents: 0.0,
                vibrato: None,
                legato_in: false,
                kind: NoteKind::Note,
            });
        }
        let hop = 48;
        let frames = (0..n / hop)
            .map(|k| {
                let pos = k * hop;
                let c = f(pos as f64 / sr as f64).filter(|_| sig.is_voiced(pos));
                RefFrame {
                    pos,
                    voiced: c.is_some(),
                    f0_hz: c.map_or(0.0, cents_to_hz64),
                    cents: c.unwrap_or(f64::NAN),
                    cents_raw: c.unwrap_or(f64::NAN),
                    clarity: 1.0,
                    level_dbfs: -20.0,
                }
            })
            .collect();
        (sig, PitchTrack { sample_rate: sr as f64, hop, frames })
    }

    fn target_at(t: f64) -> f64 {
        if t < 0.95 { 5700.0 } else { 5900.0 }
    }

    #[test]
    fn tuning_constant_offset() {
        let (sig, track) = fixture(|t| Some(target_at(t) + 4.0));
        let s = tuning(&sig, &track, &TuningOptions::default());
        assert!((s.median_abs - 4.0).abs() < 1e-9 && (s.mean - 4.0).abs() < 1e-9);
        assert_eq!(s.within_5, 100.0);
        assert_eq!(s.within_10, 100.0);
        // Boundary guard: notes of 0.8 s and 0.9 s, minus 2 × 30 ms each, at 1 ms frames.
        assert!((s.frames as f64 - 1580.0).abs() <= 4.0, "frames {}", s.frames);
    }

    #[test]
    fn tuning_measures_vibrato_on_centre() {
        let (mut sig, track) = fixture(|t| Some(target_at(t) + 3.0 + 60.0 * (2.0 * std::f64::consts::PI * 5.5 * t).sin()));
        for n in &mut sig.notes {
            n.vibrato = Some(Vibrato::immediate(5.5, 60.0));
        }
        let s = tuning(&sig, &track, &TuningOptions::default());
        assert!((s.median_abs - 3.0).abs() < 0.2, "median {}", s.median_abs);
        assert!(s.p95_abs < 3.5, "p95 {}", s.p95_abs);
    }

    #[test]
    fn settling_finds_decay_time() {
        // Output starts 100 c off and decays with τ = 10 ms: within 10 c after
        // τ·ln(10) ≈ 23 ms.
        let (sig, track) = fixture(|t| {
            let (start, tgt) = if t < 0.95 { (0.1, 5700.0) } else { (1.0, 5900.0) };
            Some(tgt - 100.0 * (-(t - start) / 0.01).exp())
        });
        let s = settling(&sig, &track, 10.0, 0.02, false);
        assert_eq!(s.settled, 2);
        assert!((s.median_s - 0.023).abs() < 0.002, "median {}", s.median_s);
    }

    #[test]
    fn vibrato_extent_ratio() {
        let make = |ext: f64| {
            let (mut sig, track) = fixture(move |t| Some(target_at(t) + ext * (2.0 * std::f64::consts::PI * 6.0 * t).sin()));
            for n in &mut sig.notes {
                n.vibrato = Some(Vibrato::immediate(6.0, 50.0));
            }
            (sig, track)
        };
        let (sig, tin) = make(50.0);
        let (_, tout) = make(45.0);
        let v = vibrato(&sig, &tin, &tout, 0.03);
        assert_eq!(v.notes.len(), 2);
        assert!((v.notes[0].in_extent - 50.0).abs() < 0.5, "{:?}", v.notes[0]);
        assert!((v.worst_extent_ratio_error - 0.1).abs() < 0.01);
        assert!((v.notes[0].in_rate_hz - 6.0).abs() < 0.05);
        assert!(v.centre_error_median < 0.5);
    }

    #[test]
    fn jitter_added_by_high_frequency_wobble() {
        let (sig, tin) = fixture(|t| Some(target_at(t)));
        let (_, tout) = fixture(|t| Some(target_at(t) + 4.0 * (2.0 * std::f64::consts::PI * 40.0 * t).sin() + 20.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin()));
        let j = jitter(&sig, &tin, &tout, 12.0);
        // 4 c peak at 40 Hz → 2.83 c RMS; the 1 Hz drift is rejected.
        assert!((j.added_rms - 2.83).abs() < 0.15, "{j:?}");
        assert!(j.in_rms < 1e-6);
    }

    #[test]
    fn voicing_counts() {
        let (sig, _) = fixture(|_| None);
        let latency = 100u32;
        let frames: Vec<PitchFrame> = (0..2000u64)
            .map(|k| {
                let pos = k * 48;
                let voiced = sig.is_voiced(pos as usize);
                PitchFrame { sample_pos: pos + latency as u64, flags: if voiced { frame_flags::VOICED } else { 0 }, ..PitchFrame::default() }
            })
            .collect();
        let v = voicing_f1(&sig, &frames, latency, 0.02);
        assert_eq!(v.fp + v.fn_, 0);
        assert!((v.f1 - 1.0).abs() < 1e-12);
        let flipped: Vec<PitchFrame> = frames.iter().map(|f| PitchFrame { flags: frame_flags::VOICED, ..*f }).collect();
        let v2 = voicing_f1(&sig, &flipped, latency, 0.02);
        assert!(v2.fp > 0 && v2.precision < 1.0);
    }

    #[test]
    fn latency_of_delayed_noise() {
        let mut r = crate::rng::Rng::new(5);
        let x: Vec<f32> = (0..20_000).map(|i| if (2000..6000).contains(&i) { r.gaussian() as f32 * 0.1 } else { 0.0 }).collect();
        for d in [0usize, 1, 48, 1234] {
            let mut y = vec![0.0f32; x.len() + d];
            y[d..].copy_from_slice(&x);
            let m = measure_latency(&x, &y, 3000).unwrap();
            assert_eq!(m.lag, d);
            assert!(m.peak > 0.999);
        }
    }

    #[test]
    fn click_detector_flags_injected_click() {
        let sr = 48_000.0;
        let x: Vec<f32> = (0..24_000).map(|i| (0.3 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr).sin()) as f32).collect();
        let clean = clicks(&x, &x, sr as f32, 0.02);
        assert_eq!(clean.clicks, 0);
        let mut y = x.clone();
        y[12_000] += 0.5;
        let c = clicks(&x, &y, sr as f32, 0.02);
        assert!(c.clicks >= 1 && c.max_excess_db > 20.0, "{c:?}");
    }

    #[test]
    fn lsd_zero_for_identical_and_positive_for_formant_shift() {
        use crate::synth::{Melody, Note, SignalBuilder, Voice, Vowel};
        let sr = 48_000.0;
        let mk = |v: Vowel| {
            SignalBuilder::new(sr).silence(0.05).melody(Melody::new(Voice::male()).note(Note::midi(50.0, 0.8).vowel(v).detune(0.0))).build()
        };
        let a = mk(Vowel::A);
        let same = formant_lsd(&a, &a.audio);
        assert!(same.frames > 10 && same.mean_db < 1e-6);
        let other = formant_lsd(&a, &mk(Vowel::I).audio);
        assert!(other.mean_db > 4.0, "{other:?}");
    }

    #[test]
    fn segmental_snr_tolerates_fractional_wander() {
        let sr = 48_000.0;
        let x: Vec<f32> = (0..9600).map(|i| (0.3 * (2.0 * std::f64::consts::PI * 300.0 * i as f64 / sr).sin() + 0.1 * (2.0 * std::f64::consts::PI * 1700.0 * i as f64 / sr).sin()) as f32).collect();
        // Delay by 3.4 samples with a sinc interpolator.
        let s = KaiserSinc::new(16, 8.6);
        let xd: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        let y: Vec<f32> = (0..x.len()).map(|i| s.interp(&xd, i as f64 - 3.4) as f32).collect();
        let (snr, n) = aligned_segmental_snr(&x, &y, 960, -2.0, 10.0);
        assert!(n >= 8 && snr > 50.0, "snr {snr}");
    }
}
