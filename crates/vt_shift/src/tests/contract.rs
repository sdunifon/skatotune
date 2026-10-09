//! Contract tests: exact dry delay, causality, invariance, API behaviour.

use vt_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};

use super::support::*;

const ALL_MODES: [LatencyMode; 4] =
    [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];

/// Unvoiced input passes as the exact delayed dry signal `x[t − D]`, for
/// every mode, engine and tier, whatever the ratio says (R§8.18).
#[test]
fn unvoiced_is_exact_delay() {
    let x = noise(20_000, 0.5, 7);
    for mode in ALL_MODES {
        for kind in [EngineKind::Natural, EngineKind::Classic] {
            for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, kind, tier);
                let d = sh.latency();
                let ctl = ShiftControl {
                    ratio: 1.37,
                    formant_ratio: 0.8,
                    voiced: false,
                    period_hint: 200.0,
                    confidence: 0.9,
                };
                let y = render_mono(&mut sh, &x, ctl);
                for i in 0..x.len() {
                    let expect = if i >= d { x[i - d] } else { 0.0 };
                    assert_eq!(y[i].to_bits(), expect.to_bits(), "{mode:?} {kind:?} {tier:?} sample {i}");
                }
            }
        }
    }
}

const RANGES: [VoiceRange; 4] = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];

/// A demanding test signal: a vowel gliding 90 → 700 Hz and back with
/// vibrato, with an unvoiced (noise) gap in the middle. Returns the signal
/// and its true f0 per sample (0 in the gap).
fn glide_signal(n: usize, sr: f64) -> (Vec<f32>, Vec<f64>) {
    let f0 = |i: usize| {
        let t = i as f64 / sr;
        let len = n as f64 / sr;
        let base = 90.0 * (700.0f64 / 90.0).powf((std::f64::consts::PI * t / len).sin());
        base * 2f64.powf((2.0 * std::f64::consts::PI * 5.5 * t).sin() * 0.7 / 12.0)
    };
    let mut x = vowel(n, f0, &VOWEL_A, 8000.0);
    let gap = (n * 45 / 100)..(n * 52 / 100);
    let nz = noise(gap.len(), 0.1, 99);
    for (v, z) in x[gap.clone()].iter_mut().zip(nz) {
        *v = z;
    }
    let truth: Vec<f64> = (0..n).map(|i| if gap.contains(&i) { 0.0 } else { f0(i) }).collect();
    (x, truth)
}

/// Control for hop `h` of the glide signal: voiced with the true period,
/// and a ratio sweeping through `ratios` (one value per 50 ms).
fn glide_ctl(truth: &[f64], hop: usize, sr: f32, ratios: &[f32], gamma: f32) -> impl FnMut(usize) -> ShiftControl {
    move |h: usize| {
        let i = (h * hop).min(truth.len() - 1);
        let f = truth[i];
        let r = ratios[(i / (sr as usize / 20)) % ratios.len()];
        ShiftControl {
            ratio: r,
            formant_ratio: gamma,
            voiced: f > 0.0,
            period_hint: if f > 0.0 { (sr as f64 / f) as f32 } else { 0.0 },
            confidence: 1.0,
        }
    }
}

/// No engine ever reads past the newest written input sample, for every
/// mode × voice range × sample rate × engine, with extreme ratios, formant
/// factors, glides and voicing changes (the latency formula covers the
/// lookahead; checked with the read counters).
#[test]
fn never_reads_past_newest_sample() {
    let ratios = [0.25f32, 0.5, 0.94, 1.0, 1.06, 1.5, 2.0, 3.0, 4.0, 1.2];
    for &sr in &[44_100.0f32, 48_000.0, 96_000.0] {
        let n = (sr * 1.2) as usize;
        let (x, truth) = glide_signal(n, sr as f64);
        for mode in ALL_MODES {
            for range in RANGES {
                for kind in [EngineKind::Natural, EngineKind::Classic] {
                    for &gamma in &[0.5f32, 1.0, 1.4, 2.0] {
                        if kind == EngineKind::Classic && gamma != 1.0 {
                            continue;
                        }
                        let cfg = crate::ShifterConfig {
                            sample_rate: sr,
                            channels: 2,
                            hop: vt_core::hop_for_rate(sr),
                            latency_mode: mode,
                            voice_range: range,
                            latency: crate::latency_samples(sr, mode, range, 7.5, 25.0),
                        };
                        let mut sh = crate::Shifter::new(cfg);
                        sh.set_engine(kind, QualityTier::Max);
                        let ctl = glide_ctl(&truth, cfg.hop, sr, &ratios, gamma);
                        let out = render(&mut sh, &[&x, &x], &x, ctl, &[37, 300, 5]);
                        let d = sh.diagnostics();
                        let what = format!("sr={sr} {mode:?} {range:?} {kind:?} γ={gamma}");
                        assert!(d.reads > 0, "{what}: no reads");
                        assert_eq!(d.reads_past_newest, 0, "{what}: {d:?}");
                        assert!(d.max_read_ahead <= 0, "{what}: {d:?}");
                        assert!(out.iter().flatten().all(|v| v.is_finite()), "{what}");
                    }
                }
            }
        }
    }
}

/// Bit-identical output for any host block sizes (hop-aligned controls).
#[test]
fn block_size_invariance() {
    let n = 30_000;
    let (x, truth) = glide_signal(n, SR as f64);
    let ratios = [1.06f32, 0.8, 1.5, 1.0];
    let patterns: [&[usize]; 7] = [&[64], &[1], &[7], &[32], &[480], &[4096], &[13, 1, 250, 64, 3, 999, 2, 77]];
    for mode in ALL_MODES {
        for kind in [EngineKind::Natural, EngineKind::Classic] {
            for tier in [QualityTier::Eco, QualityTier::Hq] {
                let mut reference: Option<Vec<Vec<f32>>> = None;
                for p in patterns {
                    let mut sh = shifter(mode, VoiceRange::Wide, 2, kind, tier);
                    let y = render(&mut sh, &[&x, &x], &x, glide_ctl(&truth, 64, SR, &ratios, 1.2), p);
                    match &reference {
                        None => reference = Some(y),
                        Some(r) => {
                            for ch in 0..2 {
                                for i in 0..n {
                                    assert_eq!(
                                        y[ch][i].to_bits(),
                                        r[ch][i].to_bits(),
                                        "{mode:?} {kind:?} {tier:?} blocks {p:?} ch {ch} sample {i}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Both channels get the same grain timing and ratio: a stereo render's
/// right channel equals a mono render of the right channel driven by the
/// same analysis signal, and identical channels stay identical.
#[test]
fn stereo_channels_share_timing() {
    let n = 24_000;
    let (x, truth) = glide_signal(n, SR as f64);
    let other = steady(n, 330.0, &VOWEL_I);
    let ratios = [1.12f32, 0.9];
    for mode in ALL_MODES {
        for kind in [EngineKind::Natural, EngineKind::Classic] {
            let mut st = shifter(mode, VoiceRange::AltoTenor, 2, kind, QualityTier::Hq);
            let y = render(&mut st, &[&x, &other], &x, glide_ctl(&truth, 64, SR, &ratios, 1.0), &[64]);
            let mut mono = shifter(mode, VoiceRange::AltoTenor, 1, kind, QualityTier::Hq);
            let m = render(&mut mono, &[&other], &x, glide_ctl(&truth, 64, SR, &ratios, 1.0), &[64]);
            let mut mono_l = shifter(mode, VoiceRange::AltoTenor, 1, kind, QualityTier::Hq);
            let ml = render(&mut mono_l, &[&x], &x, glide_ctl(&truth, 64, SR, &ratios, 1.0), &[64]);
            for i in 0..n {
                assert_eq!(y[1][i].to_bits(), m[0][i].to_bits(), "{mode:?} {kind:?} right ch at {i}");
                assert_eq!(y[0][i].to_bits(), ml[0][i].to_bits(), "{mode:?} {kind:?} left ch at {i}");
            }
            let mut same = shifter(mode, VoiceRange::AltoTenor, 2, kind, QualityTier::Hq);
            let z = render(&mut same, &[&x, &x], &x, glide_ctl(&truth, 64, SR, &ratios, 1.0), &[64]);
            assert!(z[0] == z[1], "{mode:?} {kind:?}: identical inputs diverged");
        }
    }
}

/// `mark_cents` reports the input pitch from the mark spacing while locked
/// (within 2 c on a steady vowel, also during vibrato within 15 c of the
/// true instantaneous pitch), and None when unvoiced.
#[test]
fn mark_cents_tracks_input_pitch() {
    for &f0 in &[82.0f64, 150.0, 440.0, 900.0] {
        let n = 24_000;
        let x = steady(n, f0, &VOWEL_A);
        let mut sh = shifter(LatencyMode::Live, VoiceRange::Wide, 1, EngineKind::Natural, QualityTier::Hq);
        let _ = render_mono(&mut sh, &x, voiced(1.0, (SR as f64 / f0) as f32));
        let c = sh.mark_cents().expect("locked");
        let expect = vt_core::hz_to_cents(f0 as f32);
        assert!((c - expect).abs() < 2.0, "f0={f0}: {c} vs {expect}");
    }
    // Vibrato: compare with the true pitch one period before the newest
    // input (the mark spacing measures the latest complete period).
    let n = 30_000;
    let f0 = |i: usize| 220.0 * 2f64.powf((2.0 * std::f64::consts::PI * 6.0 * i as f64 / SR as f64).sin() * 0.5 / 12.0);
    let x = vowel(n, f0, &VOWEL_A, 10_000.0);
    let mut sh = shifter(LatencyMode::Live, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
    let mut worst = 0.0f32;
    let hop = 64;
    for h in 0..n / hop {
        let i = h * hop;
        sh.set_control(&voiced(1.0, (SR as f64 / f0(i)) as f32));
        let mut buf = x[i..i + hop].to_vec();
        let a = buf.clone();
        sh.process(&a, &mut [&mut buf]);
        if h > 40
            && let Some(c) = sh.mark_cents()
        {
            let t_true = SR as f64 / f0(i + hop);
            let truth = vt_core::hz_to_cents(f0((i + hop).saturating_sub((1.2 * t_true) as usize)) as f32);
            worst = worst.max((c - truth).abs());
        }
    }
    assert!(worst < 15.0, "vibrato mark pitch error {worst} c");
    // Unvoiced: None.
    sh.set_control(&ShiftControl::default());
    assert!(sh.mark_cents().is_none());
}

/// DSP-1: Studio's detector is centred on `newest − D`, so its period hint
/// describes input D old. After a sung leap the hint and the newest marks
/// disagree for D; compared naively, every leap above 300 c re-seeded the
/// tracker every two hops and each re-seed swapped the marks under the
/// PSOLA (one-period glitches of 0.2-0.5·T). Now the hint is compared with
/// the marks at the instant it describes: one seed for the whole phrase, no
/// lost lock, and consecutive grains always advance their source by a whole
/// number of glottal cycles (measured on the exact integrated phase).
#[test]
fn studio_leaps_keep_the_marks() {
    let n = 72_000;
    for &leap in &[400.0f64, 500.0, -700.0] {
        // 220 Hz, a 40 ms legato glide to `leap` at 0.5 s and back at 1.0 s.
        let glide = |i: usize, at: usize| {
            let x = ((i as f64 - at as f64) / 1920.0).clamp(0.0, 1.0);
            0.5 - 0.5 * (std::f64::consts::PI * x).cos()
        };
        let f0 = |i: usize| 220.0 * 2f64.powf(leap / 1200.0 * (glide(i, 24_000) - glide(i, 48_000)));
        let x = vowel(n, f0, &VOWEL_A, 8000.0);
        // Integrated phase in cycles (as `vowel` accumulates it).
        let mut phase = vec![0.0f64; n + 1];
        for i in 0..n {
            phase[i + 1] = phase[i] + f0(i) / SR as f64;
        }
        let cycles = |pos: f64| {
            let i = (pos.floor().max(0.0) as usize).min(n - 1);
            phase[i] + (pos - i as f64) * (phase[i + 1] - phase[i])
        };
        let mut sh = shifter(LatencyMode::Studio, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
        let d = sh.latency();
        let (mut grains, mut prev, mut worst) = (0u64, None::<f64>, 0.0f64);
        for (i, &v) in x.iter().enumerate() {
            if i % 64 == 0 {
                let hint = SR as f64 / f0(i.saturating_sub(d));
                sh.set_control(&voiced(1.03, hint as f32));
            }
            let mut b = [v];
            sh.process(&[v], &mut [&mut b[..]]);
            let st = sh.psola.stats;
            if st.grains != grains {
                grains = st.grains;
                if let Some(p) = prev {
                    let k = cycles(st.m_last) - cycles(p);
                    worst = worst.max((k - k.round()).abs());
                }
                prev = Some(st.m_last);
            }
        }
        let dg = sh.diagnostics();
        assert!(dg.grains > 200, "{leap} c: {dg:?}");
        assert_eq!((dg.seeds, dg.lock_losses), (1, 0), "{leap} c: re-seeded");
        assert!(worst < 0.1, "{leap} c: grain source phase jumped by {worst:.2} cycles");
    }
}

/// Engine changes cross-fade over 10 ms without clicks; tier changes
/// cross-fade the kernel; a change back mid-fade is queued and still
/// smooth; Live-Min ignores Natural requests.
#[test]
fn engine_and_tier_switches_are_smooth() {
    let n = 48_000;
    let x = vowel(n, |_| 196.0, &VOWEL_A, 5000.0);
    let period = SR / 196.0;
    for mode in [LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
        let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
        let mut y = x.clone();
        let hop = 64;
        let mut i = 0;
        while i < n {
            let h = i / hop;
            // Classic only at HQ/Max here: Eco Classic is the patent form
            // (hard splices), covered by `eco_classic_is_the_patent_form`.
            match h {
                150 => sh.set_engine(EngineKind::Classic, QualityTier::Hq),
                300 => sh.set_engine(EngineKind::Natural, QualityTier::Max),
                303 => sh.set_engine(EngineKind::Classic, QualityTier::Hq), // change back mid-fade (queued)
                450 => sh.set_engine(EngineKind::Classic, QualityTier::Max),
                500 => sh.set_engine(EngineKind::Natural, QualityTier::Eco),
                600 => sh.set_engine(EngineKind::Natural, QualityTier::Hq),
                601 => sh.set_engine(EngineKind::Natural, QualityTier::Eco), // reverse kernel fade
                _ => {}
            }
            sh.set_control(&voiced(2f32.powf(2.0 / 12.0), period));
            let a = x[i..i + hop].to_vec();
            sh.process(&a, &mut [&mut y[i..i + hop]]);
            i += hop;
        }
        let (c, at) = hf_click_at(&y[4800..], 8000.0);
        assert!(c < -80.0, "{mode:?}: switch click {c:.1} dB at hop {}", (at + 4800) / 64);
        assert_eq!(sh.engine(), EngineKind::Natural);
        // 150, 300, 303 (queued) and 500; the others change only the tier.
        assert_eq!(sh.diagnostics().engine_switches, 4);
    }
    // Live-Min: only Classic.
    let mut lm = shifter(LatencyMode::LiveMin, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
    assert_eq!(lm.engine(), EngineKind::Classic);
    lm.set_engine(EngineKind::Natural, QualityTier::Eco);
    assert_eq!(lm.engine(), EngineKind::Classic);
    assert_eq!(lm.diagnostics().engine_switches, 0);
}

/// DSP-7: an engine switch during a sustained note keeps the level. The
/// incoming engine starts at the content position the outgoing one plays
/// (both wander around the dry position by up to a period); started cold, the
/// equal-gain crossfade summed two copies a fraction of a period apart and
/// dipped by 2-7 dB within the 10 ms fade. Measured as the lowest
/// one-period RMS within 14 ms of the switch against the lower of the steady
/// levels before and after (a render without a switch scores −0.1 dB).
#[test]
fn engine_switches_keep_the_level() {
    let hop = 64;
    let n = 76_800;
    let mut report = String::new();
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        for &f0 in &[110.0f64, 220.0] {
            let x = steady(n, f0, &VOWEL_A);
            let period = SR / f0 as f32;
            let p = period.round() as usize;
            for (from, to) in [(EngineKind::Natural, EngineKind::Classic), (EngineKind::Classic, EngineKind::Natural)] {
                let mut worst = 0.0f64;
                for k in 0..12 {
                    let sw = (38_400 + k * 97) / hop;
                    let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, from, QualityTier::Hq);
                    let mut y = x.clone();
                    for h in 0..n / hop {
                        if h == sw {
                            sh.set_engine(to, QualityTier::Hq);
                        }
                        sh.set_control(&voiced(1.03, period));
                        let a = x[h * hop..(h + 1) * hop].to_vec();
                        sh.process(&a, &mut [&mut y[h * hop..(h + 1) * hop]]);
                    }
                    let rms =
                        |s: usize| (y[s..s + p].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / p as f64).sqrt();
                    let avg = |a: usize, b: usize| {
                        let v: Vec<f64> = (a..b).step_by(24).map(rms).collect();
                        v.iter().sum::<f64>() / v.len() as f64
                    };
                    let s0 = sw * hop;
                    let steady_lvl = avg(s0 - 7200, s0 - 2400).min(avg(s0 + 2400, s0 + 7200));
                    let low = (0..672).step_by(24).map(|j| rms(s0 + j)).fold(f64::INFINITY, f64::min);
                    worst = worst.min(20.0 * (low / steady_lvl).log10());
                }
                if worst < -1.5 {
                    report += &format!("{mode:?} f0={f0} {from:?}->{to:?}: level dip {worst:.2} dB\n");
                }
            }
        }
    }
    assert!(report.is_empty(), "{report}");
}

/// Voiced ↔ unvoiced transitions: equal-gain fades without clicks, and the
/// output is again exactly the dry `x[t − D]` once the fade is over.
#[test]
fn voicing_transitions_are_clean_and_return_to_exact_dry() {
    let n = 48_000;
    let x = vowel(n, |_| 180.0, &VOWEL_A, 5000.0);
    let period = SR / 180.0;
    for mode in ALL_MODES {
        for kind in [EngineKind::Natural, EngineKind::Classic] {
            let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, kind, QualityTier::Hq);
            let d = sh.latency();
            // Voiced in [0.1, 0.4) s and [0.6, 0.8) s, shifted +3 st.
            let voiced_at = |i: usize| (4800..19_200).contains(&i) || (28_800..38_400).contains(&i);
            let y = render(
                &mut sh,
                &[&x],
                &x,
                |h| {
                    let v = voiced_at(h * 64);
                    ShiftControl { voiced: v, ..voiced(2f32.powf(3.0 / 12.0), if v { period } else { 0.0 }) }
                },
                &[64],
            )
            .remove(0);
            let c = hf_click_db(&y[2400..], 8000.0);
            assert!(c < -80.0, "{mode:?} {kind:?}: transition click {c:.1} dB");
            // Exact dry well after the voiced region ended (fade ≤ 4 ms, plus
            // D and a period of grain tail for PSOLA).
            for i in (19_200 + d + 1500)..28_800 {
                assert_eq!(y[i].to_bits(), x[i - d].to_bits(), "{mode:?} {kind:?} sample {i}");
            }
            for i in (38_400 + d + 1500)..n {
                assert_eq!(y[i].to_bits(), x[i - d].to_bits(), "{mode:?} {kind:?} sample {i}");
            }
            // And it was really shifted while voiced.
            let seg = &y[(9600 + d)..19_200];
            let f = median_f0(seg, 120.0, 300.0, 2048, 512).unwrap();
            assert!(cents(f, 180.0 * 2f64.powf(0.25)).abs() < 5.0, "{mode:?} {kind:?}: {f} Hz");
        }
    }
}

/// Garbage in (NaN/Inf samples, NaN/huge/negative controls, empty and
/// mismatched slices) never panics and the output recovers.
#[test]
fn robust_to_garbage() {
    let n = 20_000;
    let mut x = steady(n, 200.0, &VOWEL_A);
    for i in (3000..3100).step_by(7) {
        x[i] = f32::NAN;
    }
    x[5000] = f32::INFINITY;
    x[5001] = f32::NEG_INFINITY;
    x[6000] = 1e30;
    let bad = [
        ShiftControl {
            ratio: f32::NAN,
            formant_ratio: f32::INFINITY,
            voiced: true,
            period_hint: f32::NAN,
            confidence: f32::NAN,
        },
        ShiftControl { ratio: 1e9, formant_ratio: -3.0, voiced: true, period_hint: -5.0, confidence: 2.0 },
        ShiftControl { ratio: 0.0, formant_ratio: 0.0, voiced: true, period_hint: 1e9, confidence: -1.0 },
        voiced(1.3, 240.0),
    ];
    for mode in ALL_MODES {
        for kind in [EngineKind::Natural, EngineKind::Classic] {
            let mut sh = shifter(mode, VoiceRange::Wide, 2, kind, QualityTier::Hq);
            let y = render(&mut sh, &[&x, &x], &x, |h| bad[(h / 10) % bad.len()], &[64, 1, 333]);
            assert!(y.iter().flatten().all(|v| v.is_finite()), "{mode:?} {kind:?}");
            assert!(y[0][15_000..].iter().any(|v| v.abs() > 1e-3), "{mode:?} {kind:?}: dead output");
            // Degenerate calls.
            sh.process(&[], &mut []);
            sh.process(&x[..10], &mut []);
            let mut short = [0.0f32; 5];
            sh.process(&x[..10], &mut [&mut short[..]]);
            let mut a = [0.1f32; 8];
            let mut b = [0.1f32; 3];
            sh.process(&x[..8], &mut [&mut a[..], &mut b[..]]);
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        }
    }
    // Construction with silly configs.
    for cfg in [
        crate::ShifterConfig {
            sample_rate: f32::NAN,
            channels: 0,
            hop: 0,
            latency_mode: LatencyMode::Live,
            voice_range: VoiceRange::Wide,
            latency: 0,
        },
        crate::ShifterConfig {
            sample_rate: 1e9,
            channels: 7,
            hop: 1,
            latency_mode: LatencyMode::Studio,
            voice_range: VoiceRange::LowMale,
            latency: 5,
        },
    ] {
        let mut sh = crate::Shifter::new(cfg);
        let mut y = x.clone();
        sh.set_control(&voiced(1.5, 240.0));
        sh.process(&x, &mut [&mut y[..]]);
        assert!(y.iter().all(|v| v.is_finite()));
        assert_eq!(sh.diagnostics().reads_past_newest, 0);
    }
}

/// `reset()` returns to the freshly built state (same output as a new
/// shifter), keeping the engine choice.
#[test]
fn reset_reproduces_fresh_state() {
    let n = 20_000;
    let (x, truth) = glide_signal(n, SR as f64);
    for kind in [EngineKind::Natural, EngineKind::Classic] {
        let mut a = shifter(LatencyMode::Live, VoiceRange::AltoTenor, 1, kind, QualityTier::Hq);
        let first = render(&mut a, &[&x], &x, glide_ctl(&truth, 64, SR, &[1.2], 1.0), &[64]);
        a.reset();
        let second = render(&mut a, &[&x], &x, glide_ctl(&truth, 64, SR, &[1.2], 1.0), &[64]);
        assert!(first == second, "{kind:?}");
        assert_eq!(a.engine(), kind);
        assert_eq!(a.latency(), latency_samples_for(LatencyMode::Live));
    }
}

fn latency_samples_for(mode: LatencyMode) -> usize {
    crate::latency_samples(SR, mode, VoiceRange::AltoTenor, 10.0, 40.0)
}

/// The content offset (wander) stays within the documented bounds: Live
/// lags D by at most about one period, Studio's nearest-mark choice stays
/// within about ±T/2 of D.
#[test]
fn content_wander_is_bounded() {
    let n = 28_800;
    for &f0 in &[110.0f64, 220.0, 440.0] {
        let x = steady(n, f0, &VOWEL_A);
        let t = SR as f64 / f0;
        for &st in &[-5.0f32, -1.0, 1.0, 5.0] {
            for mode in [LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
                let _ = render_mono(&mut sh, &x, voiced(2f32.powf(st / 12.0), t as f32));
                let d = sh.diagnostics();
                let what = format!(
                    "{mode:?} f0={f0} st={st}: e in [{:.1}, {:.1}], T={t:.1}",
                    d.content_offset_min, d.content_offset_max
                );
                assert!(d.grains > 30, "{what}");
                assert!(d.content_offset_min >= -1.25 * t - 1.0, "{what}");
                assert!(d.content_offset_max <= 0.65 * t + 1.0, "{what}");
                if mode == LatencyMode::Studio {
                    assert!(d.content_offset_min >= -0.65 * t - 1.0, "{what}");
                }
                assert_eq!(d.dry_grains, 0, "{what}");
            }
        }
    }
}

/// Content delay ("heard" latency) of the Natural engine per grain, `D − e`,
/// under a constant non-unity ratio, for Live, Live+ and Studio at
/// 110/220/440 Hz (`VT_REPORT=1` prints mean and p95 against D).
///
/// * Studio's nearest-mark choice is centred on the compensated D:
///   |mean − D| < T/4 in every case, and no bias with the correction's
///   direction (DSP-8): |mean(+100 c) − mean(−100 c)| < 0.05·T and each
///   within 0.05·T of D. A symmetric ±0.6·T keep window put the mean
///   0.1·T early when sharpening and 0.1·T late when flattening.
/// * Live can only use marks already detected, so it averages about
///   D + 0.45·T below ≈ 250 Hz (R§2.1); it never runs ahead of D.
/// * `ShifterDiagnostics::mean_content_delay` agrees with the per-grain mean.
#[test]
fn content_delay_is_centred_in_studio() {
    let n = 57_600;
    let skip = 14_400;
    let mut report = String::new();
    let mut table = String::from("mode      f0    cents   D ms | mean ms  p95 ms | (mean-D)/T\n");
    for &f0 in &[110.0f64, 220.0, 440.0] {
        let x = steady(n, f0, &VOWEL_A);
        let t = SR as f64 / f0;
        for mode in [LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
            let mut means = Vec::new();
            for &c in &[-100.0f64, -35.0, 35.0, 100.0] {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
                let d = sh.latency() as f64;
                sh.set_control(&voiced(2f64.powf(c / 1200.0) as f32, t as f32));
                let mut delays = Vec::new();
                let mut grains = 0;
                for (i, &v) in x.iter().enumerate() {
                    if i % 64 == 0 {
                        sh.set_control(&voiced(2f64.powf(c / 1200.0) as f32, t as f32));
                    }
                    let mut b = [v];
                    sh.process(&[v], &mut [&mut b[..]]);
                    let st = &sh.psola.stats;
                    if st.grains != grains {
                        grains = st.grains;
                        if i >= skip {
                            delays.push(d - st.e_last);
                        }
                    }
                }
                delays.sort_by(|a, b| a.total_cmp(b));
                let mean = delays.iter().sum::<f64>() / delays.len() as f64;
                let p95 = delays[(0.95 * (delays.len() - 1) as f64) as usize];
                let ms = |v: f64| v * 1000.0 / SR as f64;
                let what = format!("{mode:?} f0={f0} {c:+} c");
                table += &format!(
                    "{:<9} {f0:>4} {c:>+6} {:>6.2} | {:>7.2} {:>7.2} | {:+.3}\n",
                    format!("{mode:?}"),
                    ms(d),
                    ms(mean),
                    ms(p95),
                    (mean - d) / t
                );
                let overall = sh.diagnostics().mean_content_delay(sh.latency()).unwrap_or(f64::NAN);
                if (overall - mean).abs() > 0.1 * t {
                    report += &format!("{what}: diagnostics mean {overall:.1} vs per-grain {mean:.1}\n");
                }
                if mode == LatencyMode::Studio && (mean - d).abs() >= 0.25 * t {
                    report += &format!("{what}: Studio mean delay {mean:.1} vs D {d} (T {t:.1})\n");
                }
                if mode == LatencyMode::Live && !(mean >= d - 0.05 * t && mean <= d + 0.75 * t) {
                    report += &format!("{what}: Live mean delay {mean:.1} vs D {d} (T {t:.1})\n");
                }
                if c.abs() == 100.0 {
                    means.push(mean - d);
                }
            }
            if mode == LatencyMode::Studio {
                let (down, up) = (means[0], means[1]);
                if (up - down).abs() >= 0.05 * t || down.abs() >= 0.05 * t || up.abs() >= 0.05 * t {
                    report += &format!(
                        "Studio f0={f0}: mean − D is {:+.3}·T at −100 c and {:+.3}·T at +100 c\n",
                        down / t,
                        up / t
                    );
                }
            }
        }
    }
    if std::env::var("VT_REPORT").is_ok() {
        eprintln!("{table}");
    }
    assert!(report.is_empty(), "{report}");
}

/// The formant knob moves the envelope by γ (grain resampling, R§7.5) while
/// the pitch stays on target.
#[test]
fn formant_ratio_moves_envelope_not_pitch() {
    let n = 28_800;
    let f0 = 150.0;
    let x = steady(n, f0, &VOWEL_A);
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        for &fst in &[-3.0f32, 3.0] {
            let gamma = 2f32.powf(fst / 12.0);
            let r = 2f32.powf(2.0 / 12.0);
            let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
            let ctl = ShiftControl { formant_ratio: gamma, ..voiced(r, SR / f0 as f32) };
            let y = render_mono(&mut sh, &x, ctl);
            let target = f0 * r as f64;
            let got = median_f0(&y[9600..], target / 1.5, target * 1.5, 4096, 1024).unwrap();
            assert!(cents(got, target).abs() < 3.0, "{mode:?} γ={gamma}: pitch {got}");
            // Compare against the input envelope warped by γ: E_out(f) ≈ E_in(f/γ).
            let warped: Vec<(f64, f64)> = VOWEL_A.iter().map(|&(f, b)| (f * gamma as f64, b * gamma as f64)).collect();
            let moved = true_env_lsd_db(&y[9600..], target, &warped, 10_000.0 * gamma as f64, 300.0, 4500.0);
            let unmoved = true_env_lsd_db(&y[9600..], target, &VOWEL_A, 10_000.0, 300.0, 4500.0);
            assert!(
                moved < unmoved - 1.0 && moved < 4.0,
                "{mode:?} γ={gamma}: warped {moved:.2} vs unwarped {unmoved:.2} dB"
            );
        }
    }
}

/// Eco Classic splices whole mark periods under a 0.75 ms raised-cosine
/// crossfade (improvement over R§7.4's patent form, whose hard splices
/// clicked in the evaluation suite): it splices, and by the click metric of
/// `ratio_sweeps_do_not_click` its splices are as clean as HQ's, also in
/// Live-Min (D = 1 ms) and for large upward shifts where the fade must be
/// shortened.
#[test]
fn eco_classic_splices_are_crossfaded() {
    let n = 48_000;
    let mut report = String::new();
    let mut table = String::new();
    for &f0 in &[110.0f64, 196.0, 440.0] {
        // A ±50 c, 5.5 Hz vibrato: the period at a splice differs from the
        // newest mark spacing, which is where hard splices click.
        let vib =
            |i: usize| f0 * 2f64.powf((2.0 * std::f64::consts::PI * 5.5 * i as f64 / SR as f64).sin() * 0.5 / 12.0);
        let x = vowel(n, vib, &VOWEL_A, 5000.0);
        // Up to +7 st: the shifted 5 kHz voice stays below the metric's 8 kHz.
        for &st in &[-5.0f32, 2.0, 7.0] {
            let ctl = voiced(2f32.powf(st / 12.0), SR / f0 as f32);
            for mode in [LatencyMode::LiveMin, LatencyMode::Live] {
                for tier in [QualityTier::Eco, QualityTier::Hq] {
                    let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Classic, tier);
                    let y = render_mono(&mut sh, &x, ctl);
                    let dg = sh.diagnostics();
                    let c = hf_click_db(&y[4800..], 8000.0);
                    let what = format!("{mode:?} {tier:?} f0={f0} st={st}");
                    table +=
                        &format!("{what}: HF {c:.1} dB, {} repeats {} drops\n", dg.classic_repeats, dg.classic_drops);
                    if dg.classic_repeats + dg.classic_drops < 5 {
                        report += &format!("{what}: no splices\n");
                    }
                    if c > -75.0 {
                        report += &format!("{what}: splice click {c:.1} dB\n");
                    }
                }
            }
        }
    }
    if std::env::var("VT_REPORT").is_ok() {
        eprintln!("{table}");
    }
    assert!(report.is_empty(), "{report}");
}

/// DSP-4: Classic in Live-Min (D = 1 ms) never reads past the newest input
/// and stays block-size independent when the ratio jumps arbitrarily from
/// hop to hop (the control law can step a MIDI-target ratio by two octaves
/// at retune 0), including jumps that land while a splice crossfade runs:
/// a rising ratio then makes the *incoming* pointer gain on the input.
#[test]
fn classic_survives_per_hop_ratio_jumps() {
    let mut rng = 0x5EED_1234_ABCD_0001u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for &sr in &[44_100.0f32, 48_000.0, 96_000.0, 192_000.0] {
        let hop = vt_core::hop_for_rate(sr);
        let n = (0.6 * sr) as usize;
        for &f0 in &[220.0f64, 600.0, 900.0] {
            // `vowel` runs at 48 kHz; scale so the period is sr / f0 samples.
            let x = vowel(n, |_| f0 * SR as f64 / sr as f64, &VOWEL_A, 5000.0 * SR as f64 / sr as f64);
            let period = (sr as f64 / f0) as f32;
            // Two octaves either way, one fresh value per hop.
            let ratios: Vec<f32> = (0..n / hop + 1).map(|_| 4f64.powf(2.0 * next() - 1.0) as f32).collect();
            for tier in [QualityTier::Eco, QualityTier::Hq] {
                let cfg = crate::ShifterConfig {
                    sample_rate: sr,
                    channels: 1,
                    hop,
                    latency_mode: LatencyMode::LiveMin,
                    voice_range: VoiceRange::AltoTenor,
                    latency: crate::latency_samples(sr, LatencyMode::LiveMin, VoiceRange::AltoTenor, 10.0, 40.0),
                };
                let mut reference: Option<Vec<f32>> = None;
                for blocks in [&[64usize][..], &[1], &[37, 300, 5]] {
                    let mut sh = crate::Shifter::new(cfg);
                    sh.set_engine(EngineKind::Classic, tier);
                    let y = render(&mut sh, &[&x], &x, |h| voiced(ratios[h.min(ratios.len() - 1)], period), blocks)
                        .remove(0);
                    let d = sh.diagnostics();
                    let what = format!("sr={sr} f0={f0} {tier:?} blocks {blocks:?}");
                    assert!(d.classic_repeats + d.classic_drops > 10, "{what}: {d:?}");
                    assert_eq!(d.reads_past_newest, 0, "{what}: {d:?}");
                    match &reference {
                        None => reference = Some(y),
                        Some(r) => {
                            assert!(r.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: differs")
                        }
                    }
                }
            }
        }
    }
}

/// The shifter can live inside a nice-plug `Plugin` (which must be `Send`).
#[test]
fn shifter_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<crate::Shifter>();
    check::<crate::ShifterConfig>();
    check::<crate::ShifterDiagnostics>();
}

/// A voiced onset starts the PSOLA within about a period of the control
/// turning voiced, also for a low voice in Live where D is only half a
/// period (the seed's interior ±20 % backward NCC peak lets the forward chain
/// start locked, and the first grain may use a short rise), and the grains
/// follow the requested ratio from the first period on.
#[test]
fn onsets_start_promptly_and_follow_the_ratio() {
    let n = 38_400;
    let hop = 64;
    let on = 9_600; // control voiced from 0.2 s
    for (range, f0) in [(VoiceRange::LowMale, 82.4), (VoiceRange::AltoTenor, 130.8), (VoiceRange::AltoTenor, 220.0)] {
        let x = vowel(n, |_| f0, &VOWEL_A, 5000.0);
        let period = SR / f0 as f32;
        let ratio = 2f32.powf(1.0 / 12.0);
        let mut sh = shifter(LatencyMode::Live, range, 1, EngineKind::Natural, QualityTier::Hq);
        let mut y = x.clone();
        let mut onset_at = None;
        for h in 0..n / hop {
            let i = h * hop;
            let ctl = if i >= on { voiced(ratio, period) } else { ShiftControl::default() };
            sh.set_control(&ctl);
            let a = x[i..i + hop].to_vec();
            sh.process(&a, &mut [&mut y[i..i + hop]]);
            if onset_at.is_none() && sh.diagnostics().psola_onsets > 0 {
                onset_at = Some(i + hop);
            }
        }
        let wait = onset_at.expect("PSOLA never started") - on;
        assert!(
            (wait as f32) < 1.2 * period + 2.0 * hop as f32,
            "{f0} Hz: onset {wait} samples after voicing (period {period:.0})"
        );
        // Periods 2-5 of the output after the onset (the first one holds the
        // cross-fade from the dry signal).
        let a = onset_at.unwrap_or(0) + period as usize;
        let seg = &y[a..a + (4.0 * period) as usize];
        let f = nsdf_f0(seg, 0.7 * f0, 1.5 * f0).expect("pitched onset");
        let err = cents(f, f0 * ratio as f64);
        assert!(err.abs() < 10.0, "{f0} Hz: first periods at {err:.1} c from the target");
    }
}

/// ON-4: after silence, the first corrected period comes promptly once the
/// control turns voiced, also for a low voice in Live where the next pitch
/// mark is only known `d_mark` after it. The onset grain may sit on the
/// newest mark at or just before the content position (it reads the dry
/// signal itself, so starting it is seamless); waiting instead for a mark
/// ahead of the content cost up to a period. Measured as the first output
/// sample that differs from the dry signal (the start of the crossfade into
/// the first corrected grain), median over 8 note phases, control voiced
/// 25 ms into the note, +50 c: Live 82 Hz 1.11 → 0.10 periods, Live 110 Hz
/// 1.42 → 0.42. (Studio starts on a mark it already knows; its boundaries
/// are covered by `studio_applies_the_ratio_promptly`.)
#[test]
fn onset_corrects_within_a_period_of_voicing() {
    let hop = 64;
    let ratio = 2f32.powf(50.0 / 1200.0);
    let mut report = String::new();
    for (mode, range, f0, limit) in [
        (LatencyMode::Live, VoiceRange::LowMale, 82.4f64, 0.5),
        (LatencyMode::Live, VoiceRange::AltoTenor, 110.0, 0.75),
    ] {
        let period = SR as f64 / f0;
        let mut waits = Vec::new();
        for k in 0..8 {
            // Silence, then the note with a 5 ms attack, starting at a
            // different phase relative to the hop grid each time.
            let start = 9_600 + k * 37;
            let note = vowel(19_200, |_| f0, &VOWEL_A, 5000.0);
            let mut x = vec![0.0f32; start];
            x.extend(note.iter().enumerate().map(|(i, &v)| v * (i as f32 / 240.0).min(1.0)));
            let mut sh = shifter(mode, range, 1, EngineKind::Natural, QualityTier::Hq);
            let d = sh.latency();
            // Studio's control for content c arrives at output time c + D.
            let von = (start + 1200 + if mode == LatencyMode::Studio { d } else { 0 }).div_ceil(hop) * hop;
            let y = render(
                &mut sh,
                &[&x],
                &x,
                |h| if h * hop >= von { voiced(ratio, period as f32) } else { ShiftControl::default() },
                &[hop],
            )
            .remove(0);
            let wet = (von..x.len()).find(|&t| (y[t] - x[t - d]).abs() > 1e-6).expect("never corrected");
            waits.push((wet - von) as f64 / period);
        }
        waits.sort_by(|a, b| a.total_cmp(b));
        let med = waits[waits.len() / 2];
        if std::env::var("VT_REPORT").is_ok() {
            eprintln!("{mode:?} {f0} Hz: first corrected sample {waits:.2?} periods after voicing");
        }
        if med > limit {
            report += &format!(
                "{mode:?} {f0} Hz: first corrected sample a median {med:.2} periods after voicing (limit {limit}): {waits:.2?}\n"
            );
        }
    }
    assert!(report.is_empty(), "{report}");
}

/// ON-3: the ratio a grain boundary applies is the newest commanded one,
/// decided near the middle of the synthesis period it shapes. A symmetric
/// boundary fixed the next period when the output reached the head grain's
/// centre with the per-sample ramp value, so the realised shift lagged the
/// commanded one by ≈ T/2 plus a hop: on a ±30 c, 4 Hz ratio modulation of a
/// steady vowel (content-aligned like Studio's control), measured from the
/// synthesis periods `s_{k+1} − s_k` against the mark periods, Studio lagged
/// 7.4 ms at 82 Hz, 5.9 ms at 110 Hz and 3.9 ms at 196 Hz; now 0.5, 0.6
/// and 1.1 ms.
#[test]
fn studio_applies_the_ratio_promptly() {
    let hop = 64;
    let n = 76_800;
    let mut report = String::new();
    for (range, f0) in [(VoiceRange::LowMale, 82.4f64), (VoiceRange::AltoTenor, 110.0), (VoiceRange::AltoTenor, 196.0)]
    {
        let x = steady(n, f0, &VOWEL_A);
        let t = SR as f64 / f0;
        let mut sh = shifter(LatencyMode::Studio, range, 1, EngineKind::Natural, QualityTier::Hq);
        let d = sh.latency();
        let w = 2.0 * std::f64::consts::PI * 4.0 / SR as f64;
        let cmd = |c: f64| 30.0 * (w * c).sin();
        let mut cents = Vec::new();
        let (mut grains, mut prev) = (0u64, None::<f64>);
        for h in 0..n / hop {
            let c = (h * hop) as f64 - d as f64;
            sh.set_control(&voiced(2f64.powf(cmd(c) / 1200.0) as f32, t as f32));
            let mut b = x[h * hop..(h + 1) * hop].to_vec();
            let a = b.clone();
            sh.process(&a, &mut [&mut b[..]]);
            let st = sh.psola.stats;
            if st.grains != grains {
                grains = st.grains;
                if let Some(p) = prev {
                    // Realised ratio of the period that just ended, at its middle.
                    cents.push((0.5 * (p + st.s_last), 1200.0 * (t / (st.s_last - p)).log2()));
                }
                prev = Some(st.s_last);
            }
        }
        // Lag of the realised shift: phase of its 4 Hz component.
        let pts: Vec<&(f64, f64)> = cents.iter().filter(|(c, _)| *c > 9600.0).collect();
        let (mut ys, mut yc, mut ss, mut cc, mut sc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &&(c, v) in &pts {
            let (s_, c_) = ((w * c).sin(), (w * c).cos());
            ys += v * s_;
            yc += v * c_;
            ss += s_ * s_;
            cc += c_ * c_;
            sc += s_ * c_;
        }
        let det = ss * cc - sc * sc;
        let (a, b) = ((ys * cc - yc * sc) / det, (yc * ss - ys * sc) / det);
        let lag_ms = -b.atan2(a) / w / SR as f64 * 1000.0;
        let gain = (a * a + b * b).sqrt() / 30.0;
        if std::env::var("VT_REPORT").is_ok() {
            eprintln!("Studio {f0} Hz: ratio lag {lag_ms:.2} ms, gain {gain:.3}");
        }
        if !(lag_ms < 2.0 && (0.9..1.1).contains(&gain)) {
            report += &format!("Studio {f0} Hz: realised shift lags the command by {lag_ms:.2} ms (gain {gain:.3})\n");
        }
    }
    assert!(report.is_empty(), "{report}");
}

/// SN-9: at amount 0 in Live the content offset never falls more than a
/// period behind. When the next mark of the sequence is published late (a
/// re-seed after a one-hop voicing dropout, a widened search), the grain
/// fell back to the latest known mark, up to 1.25 periods back (E_LOST),
/// and jumped forward again a grain later. It now continues on the predicted
/// mark (`e` unchanged, transparent at ratio 1) and a voiced onset starts on
/// the newest mark with e = 0. A vowel gliding 100 → 400 Hz with a one-hop
/// dropout every 97 hops, 6 dropout phases: worst e = −1.14·T (16 grains
/// below −1.02·T); now ≥ −0.9·T.
#[test]
fn content_offset_stays_within_a_period_at_ratio_one() {
    let n = 96_000;
    let f0 = |i: usize| 100.0 * 4f64.powf((std::f64::consts::PI * i as f64 / n as f64).sin());
    let x = vowel(n, f0, &VOWEL_A, 5000.0);
    let hop = 64;
    let mut worst = 0.0f64;
    for ph in 0..6 {
        let mut sh = shifter(LatencyMode::Live, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
        let d = sh.latency();
        let mut grains = 0u64;
        for h in 0..n / hop {
            let i = h * hop;
            let v = (h + ph * 11) % 97 != 0;
            sh.set_control(&ShiftControl { voiced: v, ..voiced(1.0, (SR as f64 / f0(i)) as f32) });
            let mut b = x[i..i + hop].to_vec();
            let a = b.clone();
            sh.process(&a, &mut [&mut b[..]]);
            let st = sh.psola.stats;
            if st.grains != grains {
                grains = st.grains;
                if i > 4800 {
                    worst = worst.min(st.e_last / (SR as f64 / f0(i + hop - d)));
                }
            }
        }
    }
    assert!(worst >= -1.0, "content offset fell {:.2} periods behind", -worst);
}

/// PERF-5: in the Live modes a voiced control's mark-tracker update (a seed
/// at an onset: tens of µs at 192 kHz) runs at the start of the next
/// `process` call, not inside `set_control`, so it never shares a host
/// block with the hop's pitch analysis; the output is identical either way
/// (the analysis ring does not change in between). Studio runs it in
/// `set_control` (vt_engine's pipelined analysis stages start a quarter
/// hop after the boundary). An unvoiced control unlocks at once.
#[test]
fn live_defers_the_tracker_update_to_the_next_block() {
    let x = steady(9_600, 150.0, &VOWEL_A);
    let period = (SR as f64 / 150.0) as f32;
    for (mode, deferred) in [(LatencyMode::Live, true), (LatencyMode::LivePlus, true), (LatencyMode::Studio, false)] {
        let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, QualityTier::Hq);
        // Unvoiced input history, then the first voiced control.
        let mut b = x[..4800].to_vec();
        sh.process(&x[..4800], &mut [&mut b[..]]);
        sh.set_control(&voiced(1.0, period));
        assert_eq!(sh.diagnostics().seeds, u64::from(!deferred), "{mode:?}: seed in set_control");
        let mut b = x[4800..4864].to_vec();
        sh.process(&x[4800..4864], &mut [&mut b[..]]);
        assert_eq!(sh.diagnostics().seeds, 1, "{mode:?}: no seed after process");
        assert!(sh.mark_cents().is_some(), "{mode:?}");
        sh.set_control(&ShiftControl::default());
        assert!(sh.mark_cents().is_none(), "{mode:?}: unvoiced control must unlock at once");
    }
}
