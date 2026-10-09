//! Realtime robustness: hostile input, mid-stream tier/settings changes,
//! reset determinism, extreme sample rates and hops (DESIGN §2.1).

mod common;

use common::*;
use vt_core::{PitchEstimate, QualityTier, VoiceRange};
use vt_detect::{Alignment, Detector, DetectorConfig};

fn finite(e: &PitchEstimate) -> bool {
    [e.info_lag, e.f0_hz, e.cents, e.period, e.confidence, e.voiced_prob, e.level_db].iter().all(|v| v.is_finite())
        && (0.0..=1.0).contains(&e.confidence)
        && (0.0..=1.0).contains(&e.voiced_prob)
        && (!e.voiced || (e.f0_hz > 0.0 && e.period > 0.0))
}

#[test]
fn hostile_input_never_panics_and_recovers() {
    let fs = 48_000.0f64;
    let mut x: Vec<f32> = Vec::new();
    x.extend(std::iter::repeat_n(f32::NAN, 3000));
    x.extend(std::iter::repeat_n(f32::INFINITY, 500));
    x.extend(std::iter::repeat_n(f32::NEG_INFINITY, 500));
    x.extend((0..4000).map(|i| if i % 2 == 0 { 1e30 } else { -1e30 }));
    x.extend(std::iter::repeat_n(f32::MIN_POSITIVE * 0.5, 2000)); // subnormal
    x.extend(std::iter::repeat_n(0.9, 4000)); // DC
    x.extend((0..4000).map(|i| if (i / 50) % 2 == 0 { 1.0 } else { -1.0 })); // full-scale square
    let garbage_end = x.len();
    let f0 = steady(fs, 220.0, 0.4);
    x.extend(Voice::default().render(fs, &f0));
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        for al in [Alignment::Causal, Alignment::Centered { delay: 1920 }] {
            let mut d = detector(fs as f32, VoiceRange::Wide, al, tier);
            let hops = run_blocks(&mut d, &x, &[1, 333, 64, 4096]);
            assert!(hops.iter().all(|h| finite(&h.est)), "{tier:?} {al:?}: non-finite estimate");
            // Tracks the vowel within 100 ms after the garbage ends.
            let settle = garbage_end as u64 + (0.1 * fs) as u64 + 1920;
            let good: Vec<f64> = hops
                .iter()
                .filter(|h| h.newest > settle)
                .map(|h| if h.est.voiced { (h.est.cents as f64 - cents(220.0)).abs() } else { f64::INFINITY })
                .collect();
            assert!(!good.is_empty());
            let ok = good.iter().filter(|&&e| e < 5.0).count() as f64 / good.len() as f64;
            assert!(ok > 0.97, "{tier:?} {al:?}: only {ok} of hops recovered");
        }
    }
}

#[test]
fn silence_and_noise_are_unvoiced() {
    let fs = 48_000.0f64;
    let mut d = detector(fs as f32, VoiceRange::Wide, Alignment::Causal, QualityTier::Hq);
    let hops = run(&mut d, &vec![0.0f32; 24_000]);
    assert!(hops.iter().all(|h| !h.est.voiced && h.est.level_db <= -159.0 && !h.est.sibilant));
    assert!(d.candidates().is_empty());
    // White and breath noise well above the gate: unvoiced (after onset).
    for x in [white(48_000, 0.05, 3), breath(fs, 48_000, 0.05, 4)] {
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            let mut d = detector(fs as f32, VoiceRange::Wide, Alignment::Causal, tier);
            let hops = run(&mut d, &x);
            let v = hops.iter().filter(|h| h.est.voiced).count();
            assert!(v * 100 < hops.len(), "{tier:?}: {v} of {} noise hops voiced", hops.len());
        }
    }
}

#[test]
fn tier_and_settings_changes_mid_stream() {
    // set_tier / set_tracking every few hops (RT-safe, soft reset): estimates
    // stay finite and the vibrato keeps being tracked.
    let fs = 48_000.0f64;
    let f0 = vibrato(fs, 262.0, 5.5, 40.0, 1.5);
    let x = Voice { formants: Some(VOWEL_E), ..Voice::default() }.render(fs, &f0);
    let mut d = Detector::new(config(fs as f32, VoiceRange::AltoTenor, Alignment::Causal));
    let hop = d.config().hop;
    let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
    let mut errs = Vec::new();
    for (i, chunk) in x.chunks(hop).enumerate() {
        if i % 37 == 0 {
            d.set_tier(tiers[(i / 37) % 3]);
        }
        if i % 53 == 0 {
            d.set_tracking([0.0, 50.0, 100.0, f32::NAN][(i / 53) % 4], [-55.0, -70.0, f32::NAN][(i / 53) % 3]);
        }
        d.push(chunk);
        let e = d.analyze();
        assert!(finite(&e));
        let newest = ((i + 1) * hop) as u64 - 1;
        if newest as f64 > 0.15 * fs && e.voiced {
            errs.push((e.cents as f64 - cents(truth_at(&f0, e.sample_pos))).abs());
        }
    }
    let voiced = errs.len() as f64 / (x.len() / hop) as f64;
    let big = errs.iter().filter(|&&e| e > 20.0).count();
    assert!(voiced > 0.85 && big * 50 < errs.len(), "voiced {voiced}, {big} big errors of {}", errs.len());
}

#[test]
fn gate_and_tracking_semantics() {
    let fs = 48_000.0f64;
    let f0 = steady(fs, 196.0, 0.5);
    // Quiet vowel (−50 dBFS RMS): voiced at the default −55 dB gate, unvoiced
    // with the gate at −40 dB.
    let quiet = Voice { rms: 10f64.powf(-50.0 / 20.0), ..Voice::default() }.render(fs, &f0);
    let voiced_frac = |x: &[f32], tracking: f32, gate: f32| {
        let mut d = detector(fs as f32, VoiceRange::Wide, Alignment::Causal, QualityTier::Hq);
        d.set_tracking(tracking, gate);
        let hops = run(&mut d, x);
        let tail: Vec<_> = hops.iter().filter(|h| h.newest as f64 > 0.1 * fs).collect();
        tail.iter().filter(|h| h.est.voiced).count() as f64 / tail.len() as f64
    };
    assert!(voiced_frac(&quiet, 50.0, -55.0) > 0.97);
    assert!(voiced_frac(&quiet, 50.0, -40.0) < 0.01);
    // Breathy vowel (noise 6 dB below the voice): relaxed tracking keeps at
    // least as many hops voiced as choosy tracking.
    let mut breathy = Voice::default().render(fs, &f0);
    for (a, b) in breathy.iter_mut().zip(breath(fs, f0.len(), 0.05, 8)) {
        *a += b;
    }
    let relaxed = voiced_frac(&breathy, 0.0, -55.0);
    let normal = voiced_frac(&breathy, 50.0, -55.0);
    let choosy = voiced_frac(&breathy, 100.0, -55.0);
    eprintln!("breathy voiced fraction: relaxed {relaxed:.3}, default {normal:.3}, choosy {choosy:.3}");
    assert!(relaxed >= normal && normal >= choosy, "{relaxed} {normal} {choosy}");
    assert!(relaxed > 0.9);
}

#[test]
fn reset_restores_the_initial_state() {
    let fs = 48_000.0f64;
    let f0 = vibrato(fs, 330.0, 6.0, 30.0, 0.4);
    let x = Voice::default().render(fs, &f0);
    for al in [Alignment::Causal, Alignment::Centered { delay: 2400 }] {
        let mut fresh = detector(fs as f32, VoiceRange::Soprano, al, QualityTier::Max);
        let a = run(&mut fresh, &x);
        let mut reused = detector(fs as f32, VoiceRange::Soprano, al, QualityTier::Max);
        run(&mut reused, &white(30_000, 0.1, 1));
        reused.reset();
        let b = run(&mut reused, &x);
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.est, q.est, "{al:?} after reset at {}", p.newest);
        }
    }
}

#[test]
fn sample_rates_and_hops() {
    // 22.05-192 kHz, explicit and default hops: accurate on a vowel.
    for &(fs, hop) in
        &[(22_050.0f32, 0usize), (44_100.0, 32), (48_000.0, 128), (88_200.0, 0), (176_400.0, 0), (192_000.0, 0)]
    {
        for &f in &[82.0f64, 330.0, 1100.0] {
            let f0 = steady(fs as f64, f, 0.35);
            let x = Voice::default().render(fs as f64, &f0);
            let mut d = Detector::new(DetectorConfig {
                sample_rate: fs,
                voice_range: VoiceRange::Wide,
                hop,
                alignment: Alignment::Causal,
            });
            let expect = if hop == 0 { vt_core::hop_for_rate(fs) } else { hop };
            assert_eq!(d.config().hop, expect, "resolved hop");
            let hops = run(&mut d, &x);
            let (errs, frac) = cents_errors(&hops, &f0, (0.15 * fs) as u64);
            let abs: Vec<f64> = errs.iter().map(|e| e.abs()).collect();
            let med = median(&abs);
            eprintln!("fs={fs} hop={hop} f={f}: median {med:.3} c voiced {frac:.3}");
            assert!(med < 1.0 && frac > 0.97, "fs={fs} hop={hop} f={f}: {med} {frac}");
        }
    }
    // Nonsense rates are sanitised, never panic.
    for fs in [f32::NAN, -1.0, 0.0, 1000.0, f32::INFINITY] {
        let mut d = Detector::new(DetectorConfig {
            sample_rate: fs,
            voice_range: VoiceRange::Wide,
            hop: 64,
            alignment: Alignment::Causal,
        });
        d.push(&[0.1; 1000]);
        assert!(finite(&d.analyze()));
    }
}

#[test]
fn candidates_hook() {
    // ≤ 5 candidates, best first, the best one at the true period.
    let fs = 48_000.0f64;
    let f0 = steady(fs, 247.0, 0.3);
    let x = Voice::default().render(fs, &f0);
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        let mut d = detector(fs as f32, VoiceRange::Wide, Alignment::Causal, tier);
        run(&mut d, &x);
        let c = d.candidates();
        assert!(!c.is_empty() && c.len() <= 5);
        assert!(c.windows(2).all(|w| w[0].1 >= w[1].1), "{c:?}");
        let p = fs as f32 / 247.0;
        assert!((c[0].0 / p - 1.0).abs() < 0.001, "{tier:?}: {c:?}");
    }
}
