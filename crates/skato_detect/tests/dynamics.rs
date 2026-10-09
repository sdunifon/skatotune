//! Time-varying behaviour: step-response lag, vibrato tracking, voicing F1,
//! block-size independence and Studio (centred) alignment (DESIGN §3.2).

mod common;

use common::*;
use skato_core::{QualityTier, VoiceRange};
use skato_detect::Alignment;

/// Long verifier window W = 2.2 / f_min (R§8.4), seconds.
fn long_window_s(range: VoiceRange) -> f64 {
    2.2 / range.f_min() as f64
}

/// Phase-continuous step from `f1` to `f2` at `t_step` seconds.
fn step_f0(fs: f64, f1: f64, f2: f64, t_step: f64, secs: f64) -> Vec<f64> {
    (0..(fs * secs) as usize).map(|i| if (i as f64) < t_step * fs { f1 } else { f2 }).collect()
}

/// Step-response lag (samples): time from the step until the estimate
/// crosses the midpoint (in cents) and stays on the new side for 8 hops,
/// measured against the newest sample (`by_pos = false`) or against
/// `sample_pos` (`by_pos = true`). Also returns the mean reported `info_lag`
/// around the crossing.
fn step_lag(hops: &[Hop], step: u64, c1: f64, c2: f64, by_pos: bool) -> Option<(f64, f64)> {
    let mid = 0.5 * (c1 + c2);
    let up = c2 > c1;
    let past = |h: &Hop| h.est.voiced && ((h.est.cents as f64 > mid) == up);
    let t = |h: &Hop| if by_pos { h.est.sample_pos } else { h.newest };
    for (i, h) in hops.iter().enumerate() {
        if t(h) < step || !past(h) {
            continue;
        }
        if hops[i..(i + 8).min(hops.len())].iter().all(past) {
            let reported =
                hops[i.saturating_sub(4)..(i + 4).min(hops.len())].iter().map(|h| h.est.info_lag as f64).sum::<f64>()
                    / 8.0;
            return Some((t(h) as f64 - step as f64, reported));
        }
    }
    None
}

#[test]
fn step_response_lag_is_below_contract() {
    // DESIGN §3.2: step-response lag at 110/220/440 Hz reported and asserted
    // below 1.5·W/2 (W = 2.2/f_min; Alto/Tenor 27.5 ms → 20.6 ms).
    let fs = 48_000.0f64;
    let range = VoiceRange::AltoTenor;
    let limit_ms = 1.5 * long_window_s(range) / 2.0 * 1000.0;
    let mut failures = Vec::new();
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        for &f1 in &[110.0f64, 220.0, 440.0] {
            // Targets stay inside the Alto/Tenor range (80-1000 Hz).
            for &dc in &[40.0f64, -40.0, 200.0, -200.0, 500.0, -500.0] {
                let f2 = f1 * 2f64.powf(dc / 1200.0);
                let t_step = 0.4;
                let f0 = step_f0(fs, f1, f2, t_step, 0.8);
                let x = Voice::default().render(fs, &f0);
                let mut d = detector(fs as f32, range, Alignment::Causal, tier);
                let hops = run(&mut d, &x);
                let step = (t_step * fs) as u64;
                let (lag, reported) =
                    step_lag(&hops, step, cents(f1), cents(f2), false).expect("estimate never settled");
                let lag_ms = lag / fs * 1000.0;
                eprintln!(
                    "{tier:?} f={f1:5.1} step {dc:+5.0} c: lag {lag_ms:5.2} ms (reported info_lag {:5.2} ms, one period {:5.2} ms)",
                    reported / fs * 1000.0,
                    1000.0 / f1
                );
                if !(lag_ms < limit_ms && lag_ms > 0.0) {
                    failures.push(format!("{tier:?} f={f1} {dc:+} c: lag {lag_ms:.2} ms (limit {limit_ms:.2})"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn vibrato_is_tracked_in_live_hq() {
    // DESIGN §3.2: vibrato (6 Hz, ±50 c) tracked with RMS error < 15 c in
    // Live HQ, measured against the truth at the instant the estimate says
    // it describes (`sample_pos`). Errors against the newest sample (what an
    // uncompensated control law sees) are reported too; Max's lag
    // compensation must reduce them.
    let fs = 48_000.0f64;
    for &f in &[110.0f64, 220.0, 440.0, 880.0] {
        let f0 = vibrato(fs, f, 6.0, 50.0, 1.5);
        let x = Voice { formants: Some(VOWEL_O), ..Voice::default() }.render(fs, &f0);
        let mut rms_now = [0.0f64; 2];
        for (ti, tier) in [QualityTier::Hq, QualityTier::Max].into_iter().enumerate() {
            let mut d = detector(fs as f32, VoiceRange::Wide, Alignment::Causal, tier);
            let hops = run(&mut d, &x);
            let (mut e_pos, mut e_now, mut n, mut total) = (0.0, 0.0, 0usize, 0usize);
            for h in hops.iter().filter(|h| h.newest as f64 > 0.25 * fs) {
                total += 1;
                if !h.est.voiced {
                    continue;
                }
                let c = h.est.cents as f64;
                let a = c - cents(truth_at(&f0, h.est.sample_pos));
                let b = c - cents(truth_at(&f0, h.newest));
                e_pos += a * a;
                e_now += b * b;
                n += 1;
            }
            let rp = (e_pos / n as f64).sqrt();
            let rn = (e_now / n as f64).sqrt();
            rms_now[ti] = rn;
            let voiced = n as f64 / total as f64;
            eprintln!(
                "vibrato f={f:5.1} {tier:?}: RMS vs sample_pos {rp:5.2} c, vs newest {rn:5.2} c, voiced {voiced:.3}"
            );
            assert!(voiced > 0.97, "f={f} {tier:?}: voiced {voiced}");
            if tier == QualityTier::Hq {
                assert!(rp < 15.0, "f={f}: RMS {rp} c");
            }
        }
        assert!(rms_now[1] < rms_now[0], "f={f}: lag compensation did not help ({} vs {})", rms_now[1], rms_now[0]);
    }
}

/// Builds a vowel / silence / noise / sibilant / breath sequence with its
/// voiced mask (true inside vowels).
fn voicing_sequence(fs: f64) -> (Vec<f32>, Vec<bool>) {
    enum Seg {
        Silence(f64),
        Vowel(Formants, f64, f64),
        White(f64, f64),
        Sibilant(f64, f64),
        Breath(f64, f64),
    }
    let segs = [
        Seg::Silence(0.25),
        Seg::Vowel(VOWEL_A, 220.0, 0.4),
        Seg::White(-30.0, 0.25),
        Seg::Vowel(VOWEL_I, 180.0, 0.4),
        Seg::Sibilant(-28.0, 0.2),
        Seg::Vowel(VOWEL_U, 300.0, 0.4),
        Seg::Silence(0.2),
        Seg::Breath(-32.0, 0.2),
        Seg::Vowel(VOWEL_E, 150.0, 0.4),
        Seg::Sibilant(-25.0, 0.15),
        Seg::Vowel(VOWEL_O, 440.0, 0.35),
        Seg::Silence(0.25),
    ];
    let mut x = Vec::new();
    let mut mask = Vec::new();
    let db = |d: f64| 10f64.powf(d / 20.0);
    for (k, s) in segs.iter().enumerate() {
        let (seg, voiced): (Vec<f32>, bool) = match *s {
            Seg::Silence(t) => (vec![0.0; (t * fs) as usize], false),
            Seg::Vowel(fm, f, t) => {
                let f0 = steady(fs, f, t);
                let mut v = Voice { formants: Some(fm), rms: 0.08, jitter: 0.001, seed: k as u64, ..Voice::default() }
                    .render(fs, &f0);
                // 5 ms raised-cosine edges (no clicks).
                let r = (0.005 * fs) as usize;
                let n = v.len();
                for i in 0..r {
                    let g = (0.5 - 0.5 * (std::f64::consts::PI * i as f64 / r as f64).cos()) as f32;
                    v[i] *= g;
                    v[n - 1 - i] *= g;
                }
                (v, true)
            }
            Seg::White(d, t) => (white((t * fs) as usize, db(d), 100 + k as u64), false),
            Seg::Sibilant(d, t) => (sibilant(fs, (t * fs) as usize, db(d), 200 + k as u64), false),
            Seg::Breath(d, t) => (breath(fs, (t * fs) as usize, db(d), 300 + k as u64), false),
        };
        mask.extend(std::iter::repeat_n(voiced, seg.len()));
        x.extend(seg);
    }
    (x, mask)
}

/// Voicing F1 of the estimates against the mask at `sample_pos`.
fn voicing_f1(hops: &[Hop], mask: &[bool], skip: u64) -> (f64, usize, usize, usize) {
    let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
    for h in hops.iter().filter(|h| h.newest >= skip) {
        let truth = mask.get(h.est.sample_pos as usize).copied().unwrap_or(false);
        match (h.est.voiced, truth) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fneg += 1,
            _ => {}
        }
    }
    (2.0 * tp as f64 / (2 * tp + fp + fneg).max(1) as f64, tp, fp, fneg)
}

#[test]
fn voicing_f1_on_mixed_sequence() {
    // DESIGN §3.2 / §4: voicing F1 ≥ 0.97 on vowel/silence/noise/sibilant.
    for &fs in &[44_100.0f64, 48_000.0, 96_000.0] {
        let (x, mask) = voicing_sequence(fs);
        let delay = (0.04 * fs) as usize;
        for (al, name) in [(Alignment::Causal, "live"), (Alignment::Centered { delay }, "studio")] {
            for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
                let mut d = detector(fs as f32, VoiceRange::Wide, al, tier);
                let hops = run(&mut d, &x);
                let (f1, tp, fp, fneg) = voicing_f1(&hops, &mask, (0.06 * fs) as u64);
                eprintln!("fs={fs} {name:6} {tier:?}: voicing F1 {f1:.4} (tp {tp}, fp {fp}, fn {fneg})");
                assert!(f1 >= 0.97, "fs={fs} {name} {tier:?}: F1 {f1}");
                // Sibilant flag shows up on the sibilant segments only.
                let sib = hops.iter().filter(|h| h.est.sibilant).count();
                assert!(sib > 0, "no sibilant frames flagged");
                assert!(hops.iter().filter(|h| h.est.sibilant && h.est.voiced).count() == 0);
            }
        }
    }
}

#[test]
fn block_size_independence() {
    // Same input, analyze() at the same hop boundaries: bit-identical
    // estimates for any host block pattern (DESIGN §2.1 rule 6).
    let fs = 48_000.0f64;
    let mut f0 = vibrato(fs, 196.0, 5.5, 40.0, 0.6);
    f0.extend(std::iter::repeat_n(0.0, (0.1 * fs) as usize));
    f0.extend(steady(fs, 330.0, 0.3));
    let mut x = Voice::default().render(fs, &f0);
    let noise = white(x.len(), 0.003, 9);
    for (a, b) in x.iter_mut().zip(noise) {
        *a += b;
    }
    let mut rng = Rng::new(5);
    let random: Vec<usize> = (0..400).map(|_| 1 + ((rng.next() * 0.5 + 0.5) * 700.0) as usize).collect();
    let patterns: [&[usize]; 7] = [&[64], &[1], &[7], &[32], &[480], &[4096], &random];
    for (al, tier) in [(Alignment::Causal, QualityTier::Hq), (Alignment::Centered { delay: 1920 }, QualityTier::Max)] {
        let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
        let reference = run_blocks(&mut d, &x, patterns[0]);
        let ref_cands = d.candidates().to_vec();
        assert!(reference.iter().filter(|h| h.est.voiced).count() > 100);
        for p in &patterns[1..] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
            let hops = run_blocks(&mut d, &x, p);
            assert_eq!(hops.len(), reference.len());
            for (a, b) in hops.iter().zip(reference.iter()) {
                assert_eq!(a.newest, b.newest);
                assert_eq!(a.est, b.est, "{al:?} pattern {:?} at {}", &p[..p.len().min(3)], a.newest);
            }
            assert_eq!(d.candidates(), &ref_cands[..]);
        }
    }
}

#[test]
fn split_analysis_is_bit_identical_with_newer_samples_pushed() {
    // PERF-7 API: `begin_analysis` at the hop boundary, then the stages run
    // during the next hop while its samples are pushed (an engine spreading
    // one hop's analysis over several callbacks). Every estimate and
    // candidate list must equal the one-shot `analyze` at the boundary.
    use skato_detect::ANALYSIS_STEPS;
    for &fs in &[48_000.0f64, 192_000.0] {
        let mut f0 = vibrato(fs, 196.0, 5.5, 40.0, 0.5);
        f0.extend(std::iter::repeat_n(0.0, (0.08 * fs) as usize));
        f0.extend(steady(fs, 330.0, 0.25));
        let mut x = Voice::default().render(fs, &f0);
        let noise = white(x.len(), 0.002, 4);
        for (a, b) in x.iter_mut().zip(noise) {
            *a += b;
        }
        let delay = (0.03 * fs) as usize;
        for (al, range) in [(Alignment::Causal, VoiceRange::AltoTenor), (Alignment::Centered { delay }, VoiceRange::LowMale)] {
            for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
                let mut a = detector(fs as f32, range, al, tier);
                let mut b = detector(fs as f32, range, al, tier);
                let hop = a.config().hop;
                let mut prev: Option<(skato_core::PitchEstimate, Vec<(f32, f32)>)> = None;
                let mut voiced = 0;
                for (i, chunk) in x.chunks_exact(hop).enumerate() {
                    // b finishes the analysis begun at the previous boundary
                    // while it pushes this hop: one stage per quarter hop, or
                    // (every third hop) the whole hop first, then all stages.
                    let mut got = None;
                    if i % 3 == 2 {
                        b.push(chunk);
                    } else {
                        for piece in chunk.chunks(hop.div_ceil(ANALYSIS_STEPS)) {
                            b.push(piece);
                            if got.is_none() {
                                got = b.step_analysis();
                            }
                        }
                    }
                    if let Some((want, cands)) = prev.take() {
                        let e = match got {
                            Some(e) => e,
                            None => b.finish_analysis(),
                        };
                        assert_eq!(b.analysis_steps_left(), 0);
                        assert_eq!(e, want, "fs={fs} {al:?} {tier:?} hop {i}");
                        assert_eq!(b.candidates(), &cands[..], "fs={fs} {al:?} {tier:?} hop {i}");
                        voiced += usize::from(e.voiced);
                    }
                    b.begin_analysis();
                    assert_eq!(b.analysis_steps_left(), ANALYSIS_STEPS);
                    a.push(chunk);
                    let w = a.analyze();
                    prev = Some((w, a.candidates().to_vec()));
                }
                assert!(voiced > 100, "fs={fs} {al:?} {tier:?}: only {voiced} voiced hops compared");
            }
        }
    }
}

#[test]
fn studio_alignment_describes_the_delayed_instant() {
    // Centered { delay }: sample_pos = newest − delay, info_lag = delay, and
    // the estimate tracks the truth at that instant with no lag (vibrato and
    // steps), better than Live does against its own sample_pos.
    let fs = 48_000.0f64;
    for &delay_ms in &[20.0f64, 40.0, 50.0] {
        let delay = (delay_ms * 0.001 * fs) as usize;
        let f0 = vibrato(fs, 220.0, 6.0, 50.0, 1.2);
        let x = Voice { formants: Some(VOWEL_A), ..Voice::default() }.render(fs, &f0);
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, Alignment::Centered { delay }, tier);
            let hops = run(&mut d, &x);
            for h in &hops {
                if h.newest >= delay as u64 {
                    assert_eq!(h.est.sample_pos, h.newest - delay as u64);
                    assert_eq!(h.est.info_lag, delay as f32);
                }
            }
            let (errs, frac) = cents_errors(&hops, &f0, (0.3 * fs) as u64);
            let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
            eprintln!("studio delay {delay_ms} ms {tier:?}: vibrato RMS {rms:.2} c, voiced {frac:.3}");
            assert!(rms < 6.0 && frac > 0.97, "delay {delay_ms} {tier:?}: RMS {rms}, voiced {frac}");
        }
        // Steps: the crossing (by sample_pos) lands within a few ms of the step.
        for &(f1, dc) in &[(110.0f64, 200.0f64), (220.0, -500.0), (440.0, 40.0)] {
            let f2 = f1 * 2f64.powf(dc / 1200.0);
            let f0 = step_f0(fs, f1, f2, 0.4, 0.8);
            let x = Voice::default().render(fs, &f0);
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, Alignment::Centered { delay }, QualityTier::Hq);
            let hops = run(&mut d, &x);
            let (lag, _) = step_lag(&hops, (0.4 * fs) as u64, cents(f1), cents(f2), true).expect("never settled");
            let lag_ms = lag / fs * 1000.0;
            eprintln!("studio delay {delay_ms} ms step f={f1} {dc:+} c: lag vs sample_pos {lag_ms:+.2} ms");
            assert!(lag_ms.abs() < 6.0, "delay {delay_ms}: residual lag {lag_ms} ms");
        }
    }
}

/// A sung phrase: notes (MIDI, seconds, glide-in seconds, vibrato extent c),
/// with scoops and portamento between legato notes, gaps (silence / breath /
/// sibilant) between phrases. Returns audio, true f0 and a mask of the
/// samples within ±30 ms of a note boundary (excluded from tuning metrics).
fn melody(fs: f64) -> (Vec<f32>, Vec<f64>, Vec<bool>) {
    let notes: [(f64, f64, f64, f64); 9] = [
        (57.0, 0.45, 0.0, 0.0),
        (59.0, 0.30, 0.06, 0.0),
        (60.0, 0.60, 0.05, 45.0),
        (64.0, 0.35, 0.08, 0.0),
        (62.0, 0.40, 0.04, 30.0),
        (69.0, 0.50, 0.10, 50.0), // leap of a fifth
        (67.0, 0.30, 0.05, 0.0),
        (55.0, 0.50, 0.12, 40.0), // octave drop
        (57.0, 0.60, 0.06, 55.0),
    ];
    let hz = |m: f64| 440.0 * 2f64.powf((m - 69.0) / 12.0);
    let mut f0 = Vec::new();
    let mut boundary = Vec::new();
    let mut prev = hz(notes[0].0);
    for (k, &(m, dur, glide, vib)) in notes.iter().enumerate() {
        let target = hz(m);
        let n = (dur * fs) as usize;
        for i in 0..n {
            let t = i as f64 / fs;
            // Glide from the previous note (legato), plus a −60 c scoop on
            // phrase starts.
            let base = if t < glide && k > 0 {
                let u = t / glide;
                prev * (target / prev).powf(0.5 - 0.5 * (std::f64::consts::PI * u).cos())
            } else if k == 0 {
                target * 2f64.powf(-60.0 / 1200.0 * (-t / 0.04).exp())
            } else {
                target
            };
            let v = if t > 0.15 { vib * ((t - 0.15) / 0.1).min(1.0) * (TAU * 5.5 * t).sin() } else { 0.0 };
            f0.push(base * 2f64.powf(v / 1200.0));
            boundary.push(t < glide + 0.03 || t > dur - 0.03);
        }
        prev = target;
    }
    let voice = Voice { formants: Some(VOWEL_O), jitter: 0.0015, ..Voice::default() };
    let (mut x, truth) = voice.render_with_truth(fs, &f0);
    // Insert a breath + sibilant gap after note 4 and a silence after note 7
    // by zeroing the truth there and replacing the audio.
    let mut truth = truth;
    let mut cut = |from: f64, len: f64, noise: Vec<f32>| {
        let a = (from * fs) as usize;
        for (i, v) in noise.into_iter().enumerate().take((len * fs) as usize) {
            if let (Some(xs), Some(t), Some(b)) = (x.get_mut(a + i), truth.get_mut(a + i), boundary.get_mut(a + i)) {
                *xs = v;
                *t = 0.0;
                *b = true;
            }
        }
    };
    let t4: f64 = notes[..4].iter().map(|n| n.1).sum();
    cut(t4 - 0.08, 0.08, sibilant(fs, (0.08 * fs) as usize, 0.03, 4));
    let t7: f64 = notes[..7].iter().map(|n| n.1).sum();
    cut(t7 - 0.1, 0.1, vec![0.0; (0.1 * fs) as usize]);
    (x, truth, boundary)
}

#[test]
fn sung_melody_is_tracked() {
    // Legato glides, scoops, vibrato, leaps of a fifth and an octave, gaps:
    // on stable frames (±30 ms around boundaries excluded) the error vs the
    // truth at `sample_pos` stays small, with no gross (> 50 c) errors, and
    // voicing follows the gaps.
    for &fs in &[44_100.0f64, 48_000.0] {
        let (x, truth, boundary) = melody(fs);
        let mask: Vec<bool> = truth.iter().map(|&f| f > 0.0).collect();
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            for (al, name) in
                [(Alignment::Causal, "live"), (Alignment::Centered { delay: (0.04 * fs) as usize }, "studio")]
            {
                let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
                let hops = run(&mut d, &x);
                let mut errs = Vec::new();
                for h in hops.iter().filter(|h| h.newest as f64 > 0.06 * fs) {
                    let p = h.est.sample_pos as usize;
                    if boundary.get(p).copied().unwrap_or(true)
                        || truth.get(p).copied().unwrap_or(0.0) <= 0.0
                        || !h.est.voiced
                    {
                        continue;
                    }
                    errs.push((h.est.cents as f64 - cents(truth[p])).abs());
                }
                let gross = errs.iter().filter(|&&e| e > 50.0).count();
                let med = median(&errs);
                let p95 = {
                    let mut s = errs.clone();
                    s.sort_by(|a, b| a.total_cmp(b));
                    s[(s.len() as f64 * 0.95) as usize]
                };
                let (f1, ..) = voicing_f1(&hops, &mask, (0.06 * fs) as u64);
                eprintln!(
                    "melody fs={fs} {name:6} {tier:?}: median {med:.2} c, p95 {p95:.2} c, gross {gross}/{}, voicing F1 {f1:.3}",
                    errs.len()
                );
                assert_eq!(gross, 0, "fs={fs} {name} {tier:?}");
                assert!(med < 3.0 && p95 < 12.0, "fs={fs} {name} {tier:?}: median {med} p95 {p95}");
                assert!(f1 > 0.97, "fs={fs} {name} {tier:?}: F1 {f1}");
            }
        }
    }
}
