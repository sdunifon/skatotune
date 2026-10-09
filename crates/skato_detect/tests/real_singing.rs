//! Failure modes found on real singing (the vocadito evaluation and the v1
//! review finding DSP-6), reproduced with synthetic signals so they are
//! guarded without the dataset. The candidate-level decisions measured on
//! vocadito (DSP-5 rough phonation, the harmonic veto, the third harmonic)
//! are unit tests in `src/cands.rs`.

mod common;

use common::*;
use skato_core::{QualityTier, VoiceRange};
use skato_detect::Alignment;

/// Share of the voiced hops after `skip_s` whose estimate is within ±50 c of
/// `want_hz` (by `sample_pos`), and the voiced share of all those hops.
fn share_near(hops: &[Hop], fs: f64, want_hz: f64, skip_s: f64) -> (f64, f64) {
    let skip = (skip_s * fs) as u64;
    let (mut n, mut voiced, mut near) = (0usize, 0usize, 0usize);
    for h in hops.iter().filter(|h| h.newest >= skip) {
        n += 1;
        if h.est.voiced {
            voiced += 1;
            if (h.est.cents as f64 - cents(want_hz)).abs() <= 50.0 {
                near += 1;
            }
        }
    }
    (near as f64 / voiced.max(1) as f64, voiced as f64 / n.max(1) as f64)
}

/// Glottal-like spectral law of the test voice (−12 dB/oct above k = 2).
fn glottal(k: usize) -> f64 {
    1.0 / (1.0 + (k as f64 / 2.0).powi(2))
}

#[test]
fn half_integer_period_tones_keep_their_octave_in_every_tier() {
    // DSP-6: a period of a half-integer number of samples with partials up
    // to 0.45·fs. The full-rate octave check's parabolic vertex under-read
    // the true period's NSDF peak, which then looked aperiodic next to its
    // integer-lag sub-octave: Eco reported the octave below on 289/298 hops
    // (48 kHz, 1391.3 Hz). HQ/Max were only saved by the harmonic veto.
    let cases: [(f32, f64); 5] = [(48_000.0, 1391.3), (48_000.0, 1315.07), (44_100.0, 1400.0), (44_100.0, 1176.0), (44_100.0, 1088.9)];
    let mut failures = Vec::new();
    for &(fs, f) in &cases {
        let f0 = steady(fs as f64, f, 0.4);
        let x = harmonic_tone(fs as f64, &f0, 64, 0.1);
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            for range in [VoiceRange::Soprano, VoiceRange::Wide] {
                let mut d = detector(fs, range, Alignment::Causal, tier);
                let hops = run(&mut d, &x);
                let (near, voiced) = share_near(&hops, fs as f64, f, 0.1);
                eprintln!("fs={fs} f={f} {tier:?} {range:?}: {:.1} % within 50 c, voiced {voiced:.3}", near * 100.0);
                if near < 0.99 || voiced < 0.95 {
                    failures.push(format!("fs={fs} f={f} {tier:?} {range:?}: near {near:.3} voiced {voiced:.3}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Jittered glottal-like voice at `f0` (AR(1) jitter with a 2 ms
/// correlation time, like `Voice`) with a per-harmonic amplitude `amp(k)`,
/// through `formants`, normalised to RMS 0.1.
fn voice(fs: f64, secs: f64, f0: f64, jitter: f64, formants: Formants, amp: impl Fn(usize) -> f64) -> Vec<f32> {
    let n = (fs * secs) as usize;
    let mut rng = Rng::new(3);
    let k_jit = 1.0 / (0.002 * fs);
    let a_jit = jitter * (6.0 / k_jit).sqrt();
    let (mut jit, mut ph) = (0.0f64, 0.0f64); // ph in cycles, unwrapped
    let kmax = (0.45 * fs / f0) as usize;
    let mut filt = FormantFilter::new(fs, &formants);
    let mut x = vec![0.0f64; n];
    for v in x.iter_mut() {
        jit += k_jit * (rng.next() * a_jit - jit);
        ph += f0 * (1.0 + jit) / fs;
        let mut s = 0.0;
        for k in 1..=kmax {
            s += amp(k) * (TAU * k as f64 * ph + 0.15 * k as f64).sin();
        }
        *v = filt.process(s);
    }
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt().max(1e-12);
    x.into_iter().map(|v| (0.1 * v / rms) as f32).collect()
}

/// Smoothstep attack of `attack_s` from `start_s`, silence before.
fn with_attack(x: &mut [f32], fs: f64, start_s: f64, attack_s: f64) {
    for (i, v) in x.iter_mut().enumerate() {
        let t = (i as f64 / fs - start_s) / attack_s;
        let g = if t <= 0.0 { 0.0 } else if t >= 1.0 { 1.0 } else { t * t * (3.0 - 2.0 * t) };
        *v *= g as f32;
    }
}

/// Noise through one resonator (`fc`, `bw` Hz), RMS `rms`.
fn resonant_noise(fs: f64, n: usize, fc: f64, bw: f64, rms: f64, seed: u64) -> Vec<f32> {
    let w = white(n, 1.0, seed);
    let mut f = FormantFilter::new(fs, &[(fc, bw), (0.4 * fs, 4000.0), (0.4 * fs, 4000.0), (0.4 * fs, 4000.0)]);
    let mut x: Vec<f64> = w.iter().map(|&v| f.process(v as f64)).collect();
    let r = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt().max(1e-12);
    x.iter_mut().for_each(|v| *v *= rms / r);
    x.into_iter().map(|v| v as f32).collect()
}

#[test]
fn dominant_third_harmonic_voice_is_tracked_at_its_fundamental() {
    // Vocadito track 21 (a ~160 Hz singer, F1 near 3·f0) was read at the
    // third harmonic for 0.2-0.5 s at a time in every mode: its trough has
    // the shortest lag and wins pYIN's rank prior, and only octave relations
    // were checked. Synthetic: F1 = 3·f0, weak first two harmonics.
    let fs = 44_100.0f64;
    let mut failures = Vec::new();
    for &(f0, bw, h12) in &[(160.0f64, 40.0f64, 0.3f64), (155.0, 50.0, 0.2)] {
        let fm: Formants = [(3.0 * f0, bw), (1800.0, 150.0), (2600.0, 160.0), (3500.0, 250.0)];
        let x = voice(fs, 0.8, f0, 0.003, fm, |k| glottal(k) * if k <= 2 { h12 } else { 1.0 });
        for tier in [QualityTier::Hq, QualityTier::Max] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, Alignment::Causal, tier);
            let hops = run(&mut d, &x);
            let (near, voiced) = share_near(&hops, fs, f0, 0.15);
            eprintln!("f0={f0} F1={} {tier:?}: {:.1} % within 50 c, voiced {voiced:.2}", 3.0 * f0, near * 100.0);
            if near < 0.95 || voiced < 0.95 {
                failures.push(format!("f0={f0} {tier:?}: near {near:.3} voiced {voiced:.3}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn low_voice_onsets_are_voiced_within_two_periods() {
    // Task: Live settling for low voices. A note can only be confirmed once
    // two of its periods are in (R§2.2), but type-I periodicity also reads
    // the attack's amplitude ramp as aperiodicity, which held E2-B2 onsets
    // (20 ms attack) about 6 ms longer: median first correct estimate 12.5 ms
    // after two periods with the v1 detector, 6.2 ms now. No onset may open
    // on a wrong pitch on the way (the first-formant ringing of the first
    // pulses is the trap the faster onsets must avoid).
    let fs = 48_000.0f64;
    let mut excess = Vec::new();
    let mut failures = Vec::new();
    for &f in &[82.41f64, 98.0, 110.0] {
        for vowel in [VOWEL_O, VOWEL_A, VOWEL_U] {
            let start = 0.2;
            let mut x = vec![0.0f32; (start * fs) as usize];
            x.extend_from_slice(&voice(fs, 0.4, f, 0.002, vowel, glottal));
            with_attack(&mut x, fs, start, 0.02);
            for range in [VoiceRange::LowMale, VoiceRange::AltoTenor] {
                if f < 1.1 * range.f_min() as f64 {
                    continue;
                }
                let mut d = detector(fs as f32, range, Alignment::Causal, QualityTier::Hq);
                let hops = run(&mut d, &x);
                let s0 = (start * fs) as u64;
                let ok = |h: &&Hop| (h.est.cents as f64 - cents(f)).abs() < 50.0;
                let first = hops.iter().find(|h| h.newest >= s0 && h.est.voiced && ok(h));
                let wrong = hops.iter().filter(|h| h.newest >= s0 && h.est.voiced && !ok(h)).count();
                let t = first.map_or(f64::INFINITY, |h| (h.newest - s0) as f64 / fs * 1e3);
                eprintln!("f={f} {range:?}: first correct at {t:.1} ms (two periods {:.1} ms), wrong hops {wrong}", 2e3 / f);
                excess.push(t - 2e3 / f);
                if wrong > 0 {
                    failures.push(format!("f={f} {range:?}: {wrong} voiced hops on a wrong pitch"));
                }
            }
        }
    }
    // The common test voice (fixed phase, no jitter), including the 98 Hz /a/
    // whose F1 ringing once opened the note at 749 Hz for two hops.
    for &(f, range) in &[(82.41f64, VoiceRange::LowMale), (98.0, VoiceRange::AltoTenor), (130.8, VoiceRange::AltoTenor), (220.0, VoiceRange::AltoTenor), (440.0, VoiceRange::Soprano)] {
        for vowel in [VOWEL_O, VOWEL_A] {
            let start = 0.2;
            let f0: Vec<f64> = (0..(0.6 * fs) as usize).map(|i| if (i as f64) < start * fs { 0.0 } else { f }).collect();
            let mut x = Voice { formants: Some(vowel), ..Voice::default() }.render(fs, &f0);
            with_attack(&mut x, fs, start, 0.02);
            let mut d = detector(fs as f32, range, Alignment::Causal, QualityTier::Hq);
            let hops = run(&mut d, &x);
            let s0 = (start * fs) as u64;
            let wrong = hops.iter().filter(|h| h.newest >= s0 && h.est.voiced && (h.est.cents as f64 - cents(f)).abs() >= 50.0).count();
            if wrong > 0 {
                failures.push(format!("Voice f={f} {range:?}: {wrong} voiced hops on a wrong pitch"));
            }
        }
    }
    let med = median(&excess);
    eprintln!("median first correct estimate after two periods: {med:.1} ms");
    assert!(med < 8.0, "median {med:.1} ms after two periods");
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn growing_resonant_noise_stays_unvoiced() {
    // Guard for the attack compensation: breath through a resonance, or mic
    // rumble, that grows by 18 dB like a note attack must not be read as
    // periodic (on vocadito, compensating every growing segment raised false
    // voicing far from any note from 3.8 % to 4.7 %). Formant-band
    // resonances stay unvoiced. Low rumble (90 Hz, Q 1.5) is a known weak
    // spot of the detector, old and new alike (v1: 7.5-17 % of the hops,
    // now 10-18 %); the limit there only guards against a large regression.
    let mut failures = Vec::new();
    for &fs in &[44_100.0f64, 48_000.0] {
        for &(fc, bw, limit) in &[(700.0f64, 150.0f64, 0.01f64), (1000.0, 100.0, 0.01), (1500.0, 200.0, 0.01), (90.0, 60.0, 0.2)] {
            let mut x = Vec::new();
            for k in 0..6 {
                x.extend(std::iter::repeat_n(0.0f32, (0.1 * fs) as usize));
                let mut b = resonant_noise(fs, (0.15 * fs) as usize, fc, bw, 0.05, 11 + k);
                for (i, v) in b.iter_mut().enumerate() {
                    let t = i as f64 / fs;
                    *v *= (10f64.powf((-18.0 + 18.0 * (t / 0.04).min(1.0)) / 20.0)) as f32;
                }
                x.extend_from_slice(&b);
            }
            for range in [VoiceRange::AltoTenor, VoiceRange::LowMale, VoiceRange::Soprano] {
                for tier in [QualityTier::Eco, QualityTier::Hq] {
                    let mut d = detector(fs as f32, range, Alignment::Causal, tier);
                    let hops = run(&mut d, &x);
                    let v = hops.iter().filter(|h| h.est.voiced).count() as f64 / hops.len() as f64;
                    eprintln!("fs={fs} fc={fc} bw={bw} {range:?} {tier:?}: voiced {:.2} %", v * 100.0);
                    if v > limit {
                        failures.push(format!("fs={fs} fc={fc} bw={bw} {range:?} {tier:?}: voiced {:.2} %", v * 100.0));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Sample index ranges of the hops whose estimate (by `sample_pos`) lies in
/// `[a, b)` seconds, and the voiced share of them.
fn voiced_share(hops: &[Hop], fs: f64, a: f64, b: f64) -> f64 {
    let (lo, hi) = ((a * fs) as u64, (b * fs) as u64);
    let sel: Vec<&Hop> = hops.iter().filter(|h| h.est.sample_pos >= lo && h.est.sample_pos < hi).collect();
    sel.iter().filter(|h| h.est.voiced).count() as f64 / sel.len().max(1) as f64
}

#[test]
fn release_tails_end_the_voicing_but_soft_notes_and_decrescendos_keep_it() {
    // Vocadito: a note's decaying tail (or the room's) stays periodic at the
    // old pitch, and the detector kept it voiced down to the gate, often
    // through the whole gap before the next note (−25 dB under the note),
    // so the control law carried the old note into the next onset. A tail
    // more than TAIL_DB under its note is not voiced; a soft note after a
    // loud one (a new attack) and a slow decrescendo still are.
    let fs = 48_000.0f64;
    let db = |d: f64| 10f64.powf(d / 20.0);
    // Loud note 220 Hz (0.2-0.6 s), tail decaying 40 dB over 0.3 s, silence,
    // a soft note 330 Hz 14 dB down at 1.1 s.
    let note = |f: f64, secs: f64, seed: u64| {
        Voice { formants: Some(VOWEL_A), jitter: 0.002, seed, ..Voice::default() }.render(fs, &steady(fs, f, secs))
    };
    let mut x = vec![0.0f32; (0.2 * fs) as usize];
    let loud = note(220.0, 0.7, 3);
    let (n_note, n_tail) = ((0.4 * fs) as usize, (0.3 * fs) as usize);
    for (i, &v) in loud.iter().enumerate().take(n_note + n_tail) {
        let g = if i < n_note { 1.0 } else { db(-40.0 * (i - n_note) as f64 / n_tail as f64) };
        x.push(v * g as f32);
    }
    x.resize((1.1 * fs) as usize, 0.0);
    let soft = note(330.0, 0.4, 5);
    let attack = (0.01 * fs) as usize;
    x.extend(soft.iter().enumerate().map(|(i, &v)| v * (db(-14.0) * (i as f64 / attack as f64).min(1.0)) as f32));
    x.resize((1.7 * fs) as usize, 0.0);
    // A decrescendo note: −10 → −28 dB over 1.5 s.
    let decresc = note(262.0, 1.5, 9);
    let n = decresc.len();
    x.extend(decresc.iter().enumerate().map(|(i, &v)| v * db(-18.0 * i as f64 / n as f64) as f32 * 3.16));
    x.resize(x.len() + (0.2 * fs) as usize, 0.0);
    let delay = (0.03 * fs) as usize;
    let mut failures = Vec::new();
    for (al, name) in [(Alignment::Causal, "live"), (Alignment::Centered { delay }, "studio")] {
        for tier in [QualityTier::Eco, QualityTier::Hq] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
            let hops = run(&mut d, &x);
            // Tail: 22-40 dB under the note (0.765-0.9 s).
            let tail = voiced_share(&hops, fs, 0.2 + 0.4 + 0.3 * 22.0 / 40.0, 0.9);
            let loud = voiced_share(&hops, fs, 0.25, 0.58);
            let soft = voiced_share(&hops, fs, 1.13, 1.48);
            let dec = voiced_share(&hops, fs, 1.75, 3.15);
            eprintln!("{name} {tier:?}: note {loud:.3} tail {tail:.3} soft note {soft:.3} decrescendo {dec:.3}");
            if tail > 0.05 || loud < 0.98 || soft < 0.98 || dec < 0.98 {
                failures.push(format!("{name} {tier:?}: note {loud:.3} tail {tail:.3} soft {soft:.3} decrescendo {dec:.3}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Unvoiced hops (by `sample_pos`) inside `[a, b)` seconds.
fn unvoiced_hops(hops: &[Hop], fs: f64, a: f64, b: f64) -> usize {
    let (lo, hi) = ((a * fs) as u64, (b * fs) as u64);
    hops.iter().filter(|h| h.est.sample_pos >= lo && h.est.sample_pos < hi && !h.est.voiced).count()
}

#[test]
fn fast_legato_leaps_stay_voiced() {
    // SN-7: in a fast legato leap the period changes inside the
    // pitch-synchronous refinement window, its clarity collapses for a few
    // ms (0.2-0.3, under the hold threshold) and the detector dropped to
    // unvoiced, which restarted the correction downstream (vocadito: 8.7
    // dropouts per voiced minute, 74 % at note boundaries). A segment as
    // loud as its note now holds through them.
    let fs = 48_000.0f64;
    let mut failures = Vec::new();
    for &(leap, glide) in &[(7.0f64, 0.03f64), (12.0, 0.03), (7.0, 0.015), (-7.0, 0.02), (12.0, 0.02)] {
        let (n_notes, dur) = (6usize, 0.3f64);
        let base = 220.0;
        let hz = |k: usize| if k.is_multiple_of(2) { base } else { base * 2f64.powf(leap / 12.0) };
        let mut f0 = vec![0.0f64; (0.1 * fs) as usize];
        for k in 0..n_notes {
            for i in 0..(dur * fs) as usize {
                let t = i as f64 / fs;
                let f = if k > 0 && t < glide {
                    let u = 0.5 - 0.5 * (std::f64::consts::PI * t / glide).cos();
                    hz(k - 1) * (hz(k) / hz(k - 1)).powf(u)
                } else {
                    hz(k)
                };
                f0.push(f);
            }
        }
        f0.extend(std::iter::repeat_n(0.0, (0.1 * fs) as usize));
        let x = Voice { formants: Some(VOWEL_A), jitter: 0.002, seed: 21, ..Voice::default() }.render(fs, &f0);
        let delay = (0.03 * fs) as usize;
        for (al, name) in [(Alignment::Causal, "live"), (Alignment::Centered { delay }, "studio")] {
            for tier in [QualityTier::Eco, QualityTier::Hq] {
                let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
                let hops = run(&mut d, &x);
                let gaps = unvoiced_hops(&hops, fs, 0.16, 0.1 + n_notes as f64 * dur - 0.02);
                if gaps > 0 {
                    failures.push(format!("{leap:+} st, {:.0} ms glide, {name} {tier:?}: {gaps} unvoiced hops", glide * 1e3));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_brief_resonance_before_a_note_is_not_held() {
    // ON-2 / ON-10: before a note, a breath or consonant resonance passes
    // the onset checks for 2-3 hops; the 20 ms hold then published its
    // (wrong) pitch through the noisy rest of the consonant, and the
    // control law decided the note on it. A segment younger than
    // MIN_HOLD_AGE_S is not held. Synthetic: 4 ms of a clean 900 Hz
    // resonance, 20 ms of it in breath noise (weakly periodic, holdable),
    // 40 ms of silence, then a 196 Hz note.
    let fs = 48_000.0f64;
    let n = |s: f64| (s * fs) as usize;
    let mut x = vec![0.0f32; n(0.2)];
    let tone = harmonic_tone(fs, &steady(fs, 900.0, 0.024), 4, 0.1);
    let noise = breath(fs, n(0.024), 0.1, 17);
    for (i, &t) in tone.iter().enumerate() {
        x.push(if i < n(0.004) { t } else { t + noise[i] });
    }
    x.resize(x.len() + n(0.04), 0.0);
    let note_at = x.len() as f64 / fs;
    x.extend(Voice { formants: Some(VOWEL_O), ..Voice::default() }.render(fs, &steady(fs, 196.0, 0.3)));
    let delay = (0.03 * fs) as usize;
    let mut failures = Vec::new();
    for (al, name) in [(Alignment::Causal, "live"), (Alignment::Centered { delay }, "studio")] {
        for tier in [QualityTier::Eco, QualityTier::Hq] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
            let hops = run(&mut d, &x);
            let wrong = hops
                .iter()
                .filter(|h| h.est.voiced && (h.est.sample_pos as f64) < note_at * fs && (h.est.cents as f64 - cents(196.0)).abs() > 300.0)
                .count();
            let note = voiced_share(&hops, fs, note_at + 0.03, note_at + 0.28);
            eprintln!("{name} {tier:?}: {wrong} voiced hops on the resonance, note voiced {note:.3}");
            if wrong > 3 || note < 0.98 {
                failures.push(format!("{name} {tier:?}: {wrong} voiced hops on the resonance, note voiced {note:.3}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn no_voicing_onset_above_the_range() {
    // ON-2 (f_max-like first estimates): before a note, breath and fricative
    // resonances just above the preset's top pitch opened the voicing there
    // (vocadito: 1040-1220 Hz for 10-20 hops before 135-200 Hz notes, Alto/
    // Tenor preset, f_max 1000 Hz). An onset must start inside the range; a
    // legato glide may still go past it.
    let fs = 48_000.0f64;
    let n = |s: f64| (s * fs) as usize;
    let mut x = vec![0.0f32; n(0.2)];
    x.extend(sine(fs, &steady(fs, 1045.0, 0.04), 0.01));
    x.resize(x.len() + n(0.03), 0.0);
    let note_at = x.len() as f64 / fs;
    x.extend(Voice { formants: Some(VOWEL_A), ..Voice::default() }.render(fs, &steady(fs, 196.0, 0.3)));
    x.resize(x.len() + n(0.1), 0.0);
    // A glide 900 → 1040 Hz inside a voiced note.
    let glide_at = x.len() as f64 / fs;
    let f0: Vec<f64> = (0..n(0.5)).map(|i| 900.0 * (1040.0f64 / 900.0).powf((i as f64 / n(0.2) as f64).min(1.0))).collect();
    x.extend(Voice { formants: Some(VOWEL_A), ..Voice::default() }.render(fs, &f0));
    x.resize(x.len() + n(0.1), 0.0);
    let delay = (0.03 * fs) as usize;
    let mut failures = Vec::new();
    for (al, name) in [(Alignment::Causal, "live"), (Alignment::Centered { delay }, "studio")] {
        for tier in [QualityTier::Eco, QualityTier::Hq] {
            let mut d = detector(fs as f32, VoiceRange::AltoTenor, al, tier);
            let hops = run(&mut d, &x);
            let burst = voiced_share(&hops, fs, 0.2, 0.24);
            let note = voiced_share(&hops, fs, note_at + 0.03, note_at + 0.28);
            let top = voiced_share(&hops, fs, glide_at + 0.25, glide_at + 0.48);
            eprintln!("{name} {tier:?}: resonance voiced {burst:.3}, note {note:.3}, glide top {top:.3}");
            if burst > 0.0 || note < 0.98 || top < 0.98 {
                failures.push(format!("{name} {tier:?}: resonance {burst:.3}, note {note:.3}, glide top {top:.3}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
