//! Steady-tone accuracy and octave robustness (DESIGN §3.2 tests).

mod common;

use common::*;
use skato_core::{QualityTier, VoiceRange};
use skato_detect::Alignment;

/// Median |error| of the voiced hops after 150 ms, voiced fraction, and the
/// count of gross (> 50 c) errors.
fn steady_stats(fs: f32, range: VoiceRange, tier: QualityTier, x: &[f32], f0: &[f64]) -> (f64, f64, usize, f64) {
    let mut d = detector(fs, range, Alignment::Causal, tier);
    let hops = run(&mut d, x);
    let (errs, frac) = cents_errors(&hops, f0, (0.15 * fs) as u64);
    let abs: Vec<f64> = errs.iter().map(|e| e.abs()).collect();
    let gross = abs.iter().filter(|&&e| e > 50.0).count();
    let mean = errs.iter().sum::<f64>() / errs.len().max(1) as f64;
    (median(&abs), frac, gross, mean)
}

#[test]
fn steady_tones_all_rates() {
    // DESIGN §3.2: median error < 1 c on steady tones (< 3 c above 1 kHz),
    // 55-1400 Hz, 44.1/48/96 kHz.
    let freqs = [55.0, 82.4, 110.0, 146.8, 196.0, 261.6, 349.2, 440.0, 587.3, 783.99, 1046.5, 1318.5, 1400.0];
    let mut worst = 0.0f64;
    for &fs in &[44_100.0f32, 48_000.0, 96_000.0] {
        for &f in &freqs {
            let f0 = steady(fs as f64, f, 0.5);
            let signals: [(&str, Vec<f32>); 3] = [
                ("sine", sine(fs as f64, &f0, 0.3)),
                ("harm", harmonic_tone(fs as f64, &f0, 12, 0.1)),
                ("vowel", Voice::default().render(fs as f64, &f0)),
            ];
            for (name, x) in &signals {
                let (med, frac, gross, mean) = steady_stats(fs, VoiceRange::Wide, QualityTier::Hq, x, &f0);
                let limit = if f >= 1000.0 { 3.0 } else { 1.0 };
                eprintln!(
                    "fs={fs} f={f:7.1} {name:5}: median {med:.3} c (mean {mean:+.3}), voiced {frac:.3}, gross {gross}"
                );
                assert!(med < limit, "fs={fs} f={f} {name}: median {med} c");
                assert!(frac > 0.97, "fs={fs} f={f} {name}: voiced fraction {frac}");
                assert_eq!(gross, 0, "fs={fs} f={f} {name}: gross errors");
                worst = worst.max(med / limit);
            }
        }
    }
    eprintln!("worst median / limit = {worst:.3}");
}

#[test]
fn tiers_and_ranges_on_steady_vowels() {
    // Every tier and voice range tracks a vowel inside its range; Max's exact
    // fractional-lag refinement removes the parabolic bias on bright high notes.
    let fs = 48_000.0f32;
    let cases: [(VoiceRange, &[f64]); 4] = [
        (VoiceRange::LowMale, &[55.0, 98.0, 196.0, 440.0, 600.0]),
        (VoiceRange::AltoTenor, &[80.0, 130.8, 261.6, 523.3, 987.8]),
        (VoiceRange::Soprano, &[140.0, 293.7, 587.3, 1174.7, 1400.0]),
        (VoiceRange::Wide, &[55.0, 220.0, 880.0, 1400.0]),
    ];
    for (range, freqs) in cases {
        for &f in freqs {
            let f0 = steady(fs as f64, f, 0.4);
            let bright = harmonic_tone(fs as f64, &f0, 12, 0.1);
            let vowel = Voice { formants: Some(VOWEL_E), ..Voice::default() }.render(fs as f64, &f0);
            for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
                for (name, x) in [("bright", &bright), ("vowel", &vowel)] {
                    let (med, frac, gross, _) = steady_stats(fs, range, tier, x, &f0);
                    // Eco refines below 400 Hz on its 16 kHz stream.
                    let limit = match (tier, f >= 1000.0) {
                        (QualityTier::Max, _) => 0.3,
                        (_, true) => 3.0,
                        (QualityTier::Eco, false) => 1.5,
                        _ => 1.0,
                    };
                    eprintln!(
                        "{range:?} {tier:?} f={f:6.1} {name:6}: median {med:.3} c, voiced {frac:.3}, gross {gross}"
                    );
                    assert!(med < limit, "{range:?} {tier:?} f={f} {name}: median {med} c");
                    assert!(frac > 0.97, "{range:?} {tier:?} f={f} {name}: voiced {frac}");
                    assert_eq!(gross, 0, "{range:?} {tier:?} f={f} {name}");
                }
            }
        }
    }
}

#[test]
fn no_octave_errors_with_strong_f1_near_twice_f0() {
    // DESIGN §3.2: no octave errors on a vowel set with strong F1 near 2·f0
    // (the classic octave-up trap, R§8.5), including a weakened fundamental.
    let fs = 48_000.0f32;
    let mut total = 0usize;
    for &f in &[110.0f64, 147.0, 196.0, 247.0, 294.0, 349.0, 440.0, 523.0] {
        for &(bw, h1) in &[(60.0f64, 1.0f64), (50.0, 0.5), (80.0, 0.3)] {
            for base in VOWELS {
                let mut fm = base;
                fm[0] = (2.0 * f, bw);
                // Keep F2 above F1.
                if fm[1].0 < 2.0 * f + 300.0 {
                    fm[1].0 = 2.0 * f + 400.0;
                }
                let f0 = steady(fs as f64, f, 0.4);
                let x = Voice { formants: Some(fm), h1_gain: h1, ..Voice::default() }.render(fs as f64, &f0);
                for tier in [QualityTier::Eco, QualityTier::Hq] {
                    let mut d = detector(fs, VoiceRange::Wide, Alignment::Causal, tier);
                    let hops = run(&mut d, &x);
                    let (errs, frac) = cents_errors(&hops, &f0, (0.1 * fs) as u64);
                    let octave = errs.iter().filter(|e| e.abs() > 300.0).count();
                    total += errs.len();
                    if octave > 0 || frac < 0.97 {
                        eprintln!("f={f} bw={bw} h1={h1} F={fm:?} {tier:?}: {octave} octave errors, voiced {frac:.3}");
                    }
                    assert_eq!(octave, 0, "f={f} bw={bw} h1={h1} {tier:?}: octave errors");
                    assert!(frac > 0.97, "f={f} {tier:?}: voiced {frac}");
                }
            }
        }
    }
    eprintln!("{total} voiced hops checked");
}

#[test]
fn realistic_voice_with_jitter() {
    // Jittered vowels at 44.1 kHz (0.2 % ≈ 3.5 c std, 2 ms correlation) against
    // the true instantaneous f0: errors stay at the jitter's own scale (the
    // estimate averages ≈ 2 periods), no gross errors, fully voiced.
    let fs = 44_100.0f32;
    for &f in &[98.0f64, 165.0, 262.0, 392.0, 660.0] {
        let f0 = steady(fs as f64, f, 0.5);
        for (i, v) in VOWELS.iter().enumerate() {
            let voice = Voice { formants: Some(*v), jitter: 0.002, seed: 11 + i as u64, ..Voice::default() };
            let (x, truth) = voice.render_with_truth(fs as f64, &f0);
            let (med, frac, gross, _) = steady_stats(fs, VoiceRange::Wide, QualityTier::Hq, &x, &truth);
            eprintln!("jitter f={f} vowel {i}: median {med:.2} c voiced {frac:.3} gross {gross}");
            assert!(med < 4.0 && frac > 0.97 && gross == 0, "f={f} vowel {i}: {med} {frac} {gross}");
        }
    }
}

#[test]
fn additive_noise_robustness() {
    // Vowels with white or breath noise at 20 and 10 dB SNR: no octave
    // errors; accuracy degrades gracefully; mostly voiced at 20 dB.
    let fs = 48_000.0f32;
    for &f in &[110.0f64, 220.0, 440.0] {
        let f0 = steady(fs as f64, f, 0.5);
        let clean = Voice { formants: Some(VOWEL_A), ..Voice::default() }.render(fs as f64, &f0);
        for &snr in &[20.0f64, 10.0] {
            let rms = 0.1 * 10f64.powf(-snr / 20.0);
            for (kind, noise) in
                [("white", white(clean.len(), rms, 3)), ("breath", breath(fs as f64, clean.len(), rms, 4))]
            {
                let x: Vec<f32> = clean.iter().zip(noise.iter()).map(|(a, b)| a + b).collect();
                let (med, frac, _, _) = steady_stats(fs, VoiceRange::Wide, QualityTier::Hq, &x, &f0);
                let mut d = detector(fs, VoiceRange::Wide, Alignment::Causal, QualityTier::Hq);
                let (errs, _) = cents_errors(&run(&mut d, &x), &f0, (0.1 * fs) as u64);
                let octave = errs.iter().filter(|e| e.abs() > 300.0).count();
                eprintln!(
                    "noise f={f} {kind:6} SNR {snr} dB: median {med:.2} c, voiced {frac:.3}, octave errors {octave}"
                );
                assert_eq!(octave, 0, "f={f} {kind} {snr} dB");
                if snr >= 20.0 {
                    assert!(med < 3.0 && frac > 0.9, "f={f} {kind} {snr} dB: {med} {frac}");
                } else {
                    assert!(med < 8.0, "f={f} {kind} {snr} dB: {med}");
                }
            }
        }
    }
}
