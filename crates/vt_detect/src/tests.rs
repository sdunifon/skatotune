//! Unit tests of the assembled detector (signal-level tests live in
//! `tests/`).
use super::*;

#[test]
fn constructs_for_every_configuration() {
    for &sr in &[44_100.0f32, 48_000.0, 96_000.0, 192_000.0] {
        for vr in [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide] {
            for al in [Alignment::Causal, Alignment::Centered { delay: (0.04 * sr) as usize }] {
                let mut d = Detector::new(DetectorConfig { sample_rate: sr, voice_range: vr, hop: 0, alignment: al });
                d.push(&[0.0; 512]);
                let e = d.analyze();
                assert!(!e.voiced);
            }
        }
    }
}

#[test]
fn analyze_before_any_push_and_degenerate_alignments() {
    for al in [
        Alignment::Causal,
        Alignment::Centered { delay: 0 },
        Alignment::Centered { delay: 5 },
        Alignment::Centered { delay: 1 << 20 },
    ] {
        let mut d = Detector::new(DetectorConfig {
            sample_rate: 48_000.0,
            voice_range: VoiceRange::Wide,
            hop: 64,
            alignment: al,
        });
        let e = d.analyze();
        assert!(!e.voiced && e.sample_pos == 0 && e.level_db <= -159.0);
        assert!(d.candidates().is_empty());
        d.push(&[0.3; 100]);
        assert!(d.analyze().info_lag.is_finite());
    }
}

#[test]
fn studio_geometry_splits_delay_between_centring_and_fixed_lag() {
    // Alto/Tenor at 48 kHz: the HQ long frame needs (W + τ_max)/2 ≈ 20.4 ms
    // of centring, the rest of the delay (up to 8 hops) is HMM fixed lag.
    let mk = |delay_ms: f32| {
        let d = Detector::new(DetectorConfig {
            sample_rate: 48_000.0,
            voice_range: VoiceRange::AltoTenor,
            hop: 64,
            alignment: Alignment::Centered { delay: (delay_ms * 48.0) as usize },
        });
        d.studio.unwrap()
    };
    let s = mk(20.0);
    assert_eq!((s.lag_frames, s.d_obs), (0, 960));
    let s = mk(30.0);
    assert!(s.lag_frames > 0 && s.lag_frames < MAX_FIXED_LAG, "{s:?}");
    assert_eq!(s.d_obs + 64 * s.lag_frames, s.delay);
    let s = mk(50.0);
    assert_eq!(s.lag_frames, MAX_FIXED_LAG);
    assert!(
        Detector::new(DetectorConfig {
            sample_rate: 48_000.0,
            voice_range: VoiceRange::Wide,
            hop: 64,
            alignment: Alignment::Causal
        })
        .studio
        .is_none()
    );
}

#[test]
fn tracking_maps_monotonically() {
    let a = TrackingParams::new(0.0);
    let b = TrackingParams::new(50.0);
    let c = TrackingParams::new(100.0);
    assert!(a.eps > b.eps && b.eps > c.eps);
    assert!((a.eps - 0.40).abs() < 1e-6 && (c.eps - 0.05).abs() < 1e-6);
    assert!(a.clarity < b.clarity && b.clarity < c.clarity);
    assert!((b.long_mean - 0.1).abs() < 1e-6 && a.long_mean > c.long_mean);
    assert!((b.switch_ref - 0.01).abs() < 1e-6 && a.switch_ref < c.switch_ref);
    let n = TrackingParams::new(f32::NAN);
    assert_eq!(n.clarity, b.clarity);
}

#[test]
fn search_stream_is_a_cascade_of_the_hq_stream() {
    for fs in [44_100.0f32, 48_000.0, 88_200.0, 96_000.0, 192_000.0] {
        let d = Detector::new(DetectorConfig {
            sample_rate: fs,
            voice_range: VoiceRange::Wide,
            hop: 0,
            alignment: Alignment::Causal,
        });
        assert_eq!(d.search.factor % d.hq.factor, 0);
        assert!((10_000.0..13_000.0).contains(&d.search.rate), "fs={fs}: {}", d.search.rate);
        assert!((20_000.0..26_000.0).contains(&d.hq.rate));
        assert!((14_000.0..17_000.0).contains(&d.eco.rate));
        // Group delay grows through the cascade but stays well under a ms.
        assert!(d.search.gd > d.hq.gd && d.search.gd < 0.001 * fs as f64);
    }
}

#[path = "../tests/common/mod.rs"]
mod common;

/// Dumps per-hop candidate internals for a signal chosen by env vars
/// (debugging aid; run with `--ignored --nocapture`).
#[test]
#[ignore]
fn debug_dump() {
    use common::*;
    let get = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let fs = get("FS", 48000.0);
    let f = get("F0", 247.0);
    let f1 = get("F1", 0.0);
    let bw = get("BW", 50.0);
    let h1 = get("H1", 1.0);
    let tier = match get("TIER", 1.0) as i32 {
        0 => QualityTier::Eco,
        2 => QualityTier::Max,
        _ => QualityTier::Hq,
    };
    let kind = get("KIND", 0.0) as i32;
    let jitter = get("JIT", 0.0);
    let range = match get("RANGE", 3.0) as i32 {
        0 => VoiceRange::LowMale,
        1 => VoiceRange::AltoTenor,
        2 => VoiceRange::Soprano,
        _ => VoiceRange::Wide,
    };
    let f2 = get("F2", 0.0);
    let t_step = get("TSTEP", 0.15);
    let every = get("EVERY", 8.0) as usize;
    let from = get("FROM", 0.0) as usize;
    let f0: Vec<f64> = (0..((t_step + 0.15) * fs) as usize)
        .map(|i| if f2 > 0.0 && i as f64 >= t_step * fs { f2 } else { f })
        .collect();
    let mut fm = VOWEL_A;
    if f1 > 0.0 {
        fm[0] = (f1, bw);
        if fm[1].0 < f1 + 300.0 {
            fm[1].0 = f1 + 400.0;
        }
    }
    let x = match kind {
        1 => harmonic_tone(fs, &f0, 12, 0.1),
        2 => sine(fs, &f0, 0.3),
        _ => Voice { formants: Some(fm), h1_gain: h1, jitter, seed: 11, ..Voice::default() }.render(fs, &f0),
    };
    let mut d = detector(fs as f32, range, Alignment::Causal, tier);
    let hop = d.config().hop;
    for (i, chunk) in x.chunks(hop).enumerate() {
        d.push(chunk);
        let e = d.analyze();
        if i % every == 0 && i >= from {
            let newest = ((i + 1) * hop) as f64;
            println!(
                "hop {i} (t-step {:+.2} ms): v={} c={:.2} err={:+.2} conf={:.3} vp={:.3} per={:.2}",
                (newest - t_step * fs) / fs * 1000.0,
                e.voiced,
                e.cents,
                e.cents as f64 - cents(f),
                e.confidence,
                e.voiced_prob,
                e.period
            );
            for c in d.debug_frame().as_slice() {
                println!(
                    "    {:8.2} Hz p={:.3} (fast {:.3} long {:.3} short {:.3}) g_long={:.3} h_fast={:.3} harm={:.3} seed_fast={}",
                    fs as f32 / c.period,
                    c.p,
                    c.p_fast,
                    c.p_long,
                    c.p_short,
                    c.g_long,
                    c.h_fast,
                    c.harm,
                    c.fast_seed
                );
            }
        }
    }
}
