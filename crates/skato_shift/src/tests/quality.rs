//! Sound-quality tests: pitch accuracy, formant preservation, clicks,
//! transparency.

use skato_core::{EngineKind, LatencyMode, QualityTier, VoiceRange};

use super::support::*;

const MODES: [LatencyMode; 3] = [LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];

/// Output f0 of PSOLA and Classic at fixed ratios matches the target within
/// 3 c (DESIGN §3.4), for 100/220/440 Hz vowels and ±1/3/7/12 st.
#[test]
fn fixed_ratio_pitch_accuracy() {
    let n = 28_800; // 0.6 s
    let mut worst = 0.0f64;
    let mut report = String::new();
    for &f0 in &[100.0, 220.0, 440.0] {
        let x = steady(n, f0, &VOWEL_A);
        let period = SR / f0 as f32;
        for &st in &[-12.0f32, -7.0, -3.0, -1.0, 1.0, 3.0, 7.0, 12.0] {
            let r = 2f32.powf(st / 12.0);
            let target = f0 * r as f64;
            let mut cases: Vec<(String, Vec<f32>)> = Vec::new();
            for mode in MODES {
                for tier in [QualityTier::Eco, QualityTier::Hq] {
                    let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, tier);
                    cases.push((format!("psola {mode:?} {tier:?}"), render_mono(&mut sh, &x, voiced(r, period))));
                }
            }
            for mode in [LatencyMode::LiveMin, LatencyMode::Live] {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Classic, QualityTier::Hq);
                cases.push((format!("classic {mode:?}"), render_mono(&mut sh, &x, voiced(r, period))));
            }
            for (name, y) in &cases {
                // Steady part only; long frames for an accurate average.
                let seg = &y[9600..];
                let got = median_f0(seg, target / 1.5, target * 1.5, 4096, 1024).unwrap_or(0.0);
                let err = cents(got, target).abs();
                worst = worst.max(err);
                if err > 3.0 || !err.is_finite() {
                    report += &format!("{name} f0={f0} st={st}: got {got:.3} Hz, target {target:.3} ({err:.2} c)\n");
                }
            }
        }
    }
    assert!(report.is_empty(), "pitch errors > 3 c:\n{report}");
    eprintln!("fixed_ratio_pitch_accuracy: worst {worst:.3} c");
}

/// LPC envelope of a steady segment (order 24, 10 Hz grid to 5.2 kHz).
fn envelope(x: &[f32]) -> Vec<f64> {
    lpc_env_db(x, 24, 520, 10.0)
}

/// PSOLA preserves the formant envelope; Classic moves it with the pitch.
///
/// Three measures (all printed with `SKATO_REPORT=1`):
/// * **Observed-envelope LSD** (asserted): the output's harmonic levels
///   against the input's harmonic levels interpolated (in dB) at the output
///   frequencies. The envelope between input harmonics is unobservable, so
///   this is what an ideal shifter achieves (≈ 0 dB) at any pitch. PSOLA must
///   stay < 3 dB up to ±7 st and < 4.5 dB at ±12 st (where grains of half an
///   input period truncate the formant ringing: the inherent PSOLA high-pitch
///   limit, R§8.19), at 100/220/440 Hz, both vowels, every mode and tier;
///   Classic must exceed 3 dB at ≥ 3 st (its formants move).
/// * **LPC-envelope LSD against the input** (DESIGN §3.4's wording, order
///   24, 400 Hz-5 kHz): asserted < 3 dB where the estimator is reliable,
///   i.e. where an ideal re-synthesis at the target pitch also scores < 2 dB.
///   LPC envelopes of high voices follow the harmonics (≈ 11 harmonics below
///   5 kHz at 440 Hz for order 24), so an ideal shifter itself scores up to
///   11 dB there.
/// * **True-envelope LSD**: against the synthetic vowel's exact envelope at
///   the output harmonics (reported only: it also counts envelope detail no
///   shifter can see).
#[test]
fn formant_envelope_preserved_by_psola_and_moved_by_classic() {
    let n = 28_800;
    let band = 10_000.0;
    let mut report = String::new();
    let mut table = String::new();
    let mut worst = (0.0f64, 0.0f64);
    for &f0 in &[100.0, 220.0, 440.0] {
        for vowel_f in [&VOWEL_A[..], &VOWEL_I[..]] {
            let x = steady(n, f0, vowel_f);
            let env_in = envelope(&x[9600..]);
            let te_in = true_env_lsd_db(&x[9600..], f0, vowel_f, band, 150.0, 5000.0);
            assert!(te_in < 0.05, "measure validation: input scores {te_in}");
            let period = SR / f0 as f32;
            for &st in &[-12.0f32, -7.0, -3.0, -1.0, 1.0, 3.0, 7.0, 12.0] {
                let r = 2f32.powf(st / 12.0);
                let f_out = f0 * r as f64;
                let ideal = steady(n, f_out, vowel_f);
                let bias = lsd_db(&env_in, &envelope(&ideal[9600..]), 10.0, 400.0, 5000.0);
                let limit = if st.abs() > 7.0 { 4.5 } else { 3.0 };
                for mode in MODES {
                    for tier in [QualityTier::Eco, QualityTier::Hq] {
                        let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, tier);
                        let y = render_mono(&mut sh, &x, voiced(r, period));
                        let seg = &y[9600..];
                        let te = true_env_lsd_db(seg, f_out, vowel_f, band, 150.0, 5000.0);
                        let ob = observed_env_lsd_db(&x[9600..], f0, seg, f_out, 150.0, 5000.0);
                        let lpc = lsd_db(&env_in, &envelope(seg), 10.0, 400.0, 5000.0);
                        table += &format!(
                            "psola {mode:?} {tier:?} f0={f0} F1={} st={st}: observed {ob:.2} true-env {te:.2} dB, LPC {lpc:.2} dB (LPC bias {bias:.2})\n",
                            vowel_f[0].0
                        );
                        if st.abs() <= 7.0 {
                            worst.0 = worst.0.max(ob);
                        }
                        if ob >= limit {
                            report += &format!(
                                "psola {mode:?} {tier:?} f0={f0} F1={} st={st}: observed LSD {ob:.2} dB\n",
                                vowel_f[0].0
                            );
                        }
                        if bias < 2.0 {
                            worst.1 = worst.1.max(lpc);
                            if lpc >= 3.0 {
                                report += &format!(
                                    "psola {mode:?} {tier:?} f0={f0} F1={} st={st}: LPC LSD {lpc:.2} dB\n",
                                    vowel_f[0].0
                                );
                            }
                        }
                    }
                }
                if st.abs() >= 3.0 {
                    let mut sh =
                        shifter(LatencyMode::Live, VoiceRange::AltoTenor, 1, EngineKind::Classic, QualityTier::Hq);
                    let y = render_mono(&mut sh, &x, voiced(r, period));
                    let ob = observed_env_lsd_db(&x[9600..], f0, &y[9600..], f_out, 150.0, 5000.0);
                    table += &format!("classic f0={f0} F1={} st={st}: observed {ob:.2} dB\n", vowel_f[0].0);
                    if ob < 3.0 {
                        report +=
                            &format!("classic f0={f0} st={st}: formants did not move (observed LSD {ob:.2} dB)\n");
                    }
                }
            }
        }
    }
    if std::env::var("SKATO_REPORT").is_ok() {
        eprintln!("{table}worst PSOLA (≤ 7 st) observed {:.2} dB, LPC where reliable {:.2} dB", worst.0, worst.1);
    }
    assert!(report.is_empty(), "{report}");
}

/// Ratio 1, formant 1, voiced: PSOLA is near-transparent against x[t − D]
/// after aligning by the measured content offset (segmental SNR ≥ 25 dB,
/// DESIGN §3.4); Classic is exact.
#[test]
fn transparent_at_ratio_one() {
    let n = 24_000;
    for &f0 in &[100.0, 220.0, 440.0] {
        let x = steady(n, f0, &VOWEL_A);
        let period = SR / f0 as f32;
        for mode in MODES {
            for tier in [QualityTier::Eco, QualityTier::Hq] {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Natural, tier);
                let d = sh.latency();
                let y = render_mono(&mut sh, &x, voiced(1.0, period));
                let lo = d.saturating_sub(period as usize + 4);
                let (lag, _) = best_lag(&x, &y, lo, d + period as usize + 4, 6000, n);
                let snr = seg_snr_db(&x, &y, lag, 6000, n);
                assert!(snr >= 25.0, "{mode:?} {tier:?} f0={f0}: segSNR {snr:.1} dB at lag {lag} (D {d})");
                let dg = sh.diagnostics();
                assert_eq!(dg.grain_repeats + dg.grain_skips, 0, "{mode:?} {tier:?} f0={f0}: {dg:?}");
            }
        }
        for mode in [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio] {
            let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, EngineKind::Classic, QualityTier::Hq);
            let d = sh.latency();
            let y = render_mono(&mut sh, &x, voiced(1.0, period));
            for i in 4000..n {
                assert_eq!(y[i].to_bits(), x[i - d].to_bits(), "classic {mode:?} f0={f0} at {i}");
            }
        }
    }
}

/// Vibrato-like ratio sweeps (±1 st at 6 Hz on top of a fixed shift) must
/// not click: the worst 5 ms frame's energy above 8 kHz of a voice
/// band-limited to 5 kHz stays below −75 dB re the mean frame energy (the
/// input itself scores ≈ −100 dB), a level that a 0.001 step or single-sample
/// glitch on a 0.5-peak voice already exceeds.
#[test]
fn ratio_sweeps_do_not_click() {
    let n = 48_000;
    let hop = 64usize;
    let mut report = String::new();
    let mut table = String::new();
    for &f0 in &[110.0, 220.0, 440.0] {
        let x = vowel(n, |_| f0, &VOWEL_A, 5000.0);
        let period = SR / f0 as f32;
        let base = hf_click_db(&x[4800..], 8000.0);
        for &shift in &[-5.0f32, 0.0, 5.0] {
            let ctl = |h: usize| {
                let t = (h * hop) as f32 / SR;
                let st = shift + (2.0 * std::f32::consts::PI * 6.0 * t).sin();
                voiced(2f32.powf(st / 12.0), period)
            };
            let mut cases: Vec<(String, EngineKind, LatencyMode, QualityTier)> = Vec::new();
            for mode in MODES {
                for tier in [QualityTier::Eco, QualityTier::Hq] {
                    cases.push((format!("psola {mode:?} {tier:?}"), EngineKind::Natural, mode, tier));
                }
            }
            for mode in [LatencyMode::LiveMin, LatencyMode::Live] {
                cases.push((format!("classic {mode:?} Hq"), EngineKind::Classic, mode, QualityTier::Hq));
            }
            for (name, kind, mode, tier) in cases {
                let mut sh = shifter(mode, VoiceRange::AltoTenor, 1, kind, tier);
                let y = render(&mut sh, &[&x], &x, ctl, &[64]).remove(0);
                let c = hf_click_db(&y[4800..], 8000.0);
                table += &format!("{name} f0={f0} shift={shift}: {c:.1} dB (input {base:.1})\n");
                if c > -75.0 {
                    report += &format!("{name} f0={f0} shift={shift}: HF {c:.1} dB (input {base:.1})\n");
                }
            }
        }
        // Negative control: a 0.001 step (−54 dB re the 0.5 peak) and a
        // single-sample glitch of that size are caught.
        let mut z = x.clone();
        for v in &mut z[30_000..] {
            *v += 0.001;
        }
        let mut g = x.clone();
        g[30_000] += 0.001;
        assert!(hf_click_db(&z[4800..], 8000.0) > -75.0, "click detector misses a 0.001 step");
        assert!(hf_click_db(&g[4800..], 8000.0) > -75.0, "click detector misses a 0.001 glitch");
    }
    if std::env::var("SKATO_REPORT").is_ok() {
        eprintln!("{table}");
    }
    assert!(report.is_empty(), "{report}");
}

/// Tiers trade accuracy for CPU as specified (R§7.2.2): with fractional
/// reads (a small shift), HQ's 16-tap sinc keeps the 8-14 kHz harmonics of a
/// full-band voice at their input levels where Eco's cubic read rolls them
/// off, and HQ Classic anti-aliases upward shifts (rate-dependent sinc
/// cutoff) where Eco's cubic read folds energy back. (At ratio 1 both are
/// bit-exact: the grains read whole samples.)
#[test]
fn tiers_trade_accuracy() {
    let n = 28_800;
    // The bare glottal source (−6 dB/oct to 16 kHz) has energy up there.
    let x = vowel(n, |_| 230.0, &[], 16_000.0);
    let period = SR / 230.0;
    let r = 2f32.powf(0.5 / 12.0);
    let hf_error = |tier| {
        let mut sh = shifter(LatencyMode::Studio, VoiceRange::AltoTenor, 1, EngineKind::Natural, tier);
        let y = render_mono(&mut sh, &x, voiced(r, period));
        // Level change of the 8-14 kHz harmonics relative to 1-4 kHz.
        let lvl = |s: &[f32], f0: f64, lo: f64, hi: f64| {
            let h = harmonic_amps_db(s, f0, lo, hi);
            h.iter().map(|v| v.1).sum::<f64>() / h.len() as f64
        };
        let f_out = 230.0 * r as f64;
        let tilt_in = lvl(&x[9600..], 230.0, 8000.0, 14_000.0) - lvl(&x[9600..], 230.0, 1000.0, 4000.0);
        let tilt_out = lvl(&y[9600..], f_out, 8000.0, 14_000.0) - lvl(&y[9600..], f_out, 1000.0, 4000.0);
        (tilt_out - tilt_in).abs()
    };
    let (eco, hq) = (hf_error(QualityTier::Eco), hf_error(QualityTier::Hq));
    assert!(hq < 0.4 && eco > 2.0 * hq, "8-14 kHz level error: HQ {hq:.2} dB vs Eco {eco:.2} dB");

    // A 21 kHz tone read at rate 1.5 would land at 31.5 kHz, i.e. alias to
    // 16.5 kHz. (The sinc bank's rate-1.5 table cuts at ≈ 17 kHz with the
    // transition band of a 16-tap kernel, so the tone is placed clearly above
    // it.) The voice is band-limited to 8 kHz (12 kHz after the shift), so
    // 13-19 kHz of the output holds only the alias; splices smear it, hence a
    // band measurement. Levels are relative to the input tone.
    let voice = vowel(n, |_| 230.0, &VOWEL_A, 8000.0);
    let tone: Vec<f32> =
        (0..n).map(|i| 0.3 * (2.0 * std::f64::consts::PI * 21_000.0 * i as f64 / SR as f64).sin() as f32).collect();
    let tone_db = band_energy_db(&tone[8000..], 20_000.0, 22_000.0);
    let mix: Vec<f32> = voice.iter().zip(&tone).map(|(a, b)| a + b).collect();
    let alias = |tier| {
        let mut sh = shifter(LatencyMode::Live, VoiceRange::AltoTenor, 1, EngineKind::Classic, tier);
        let y = render(&mut sh, &[&mix], &voice, |_| voiced(1.5, period), &[64]).remove(0);
        band_energy_db(&y[8000..], 13_000.0, 19_000.0) - tone_db
    };
    // Measured: Eco ≈ −4 dB (cubic folds almost everything back), HQ ≈
    // −22 dB: the 16-tap kernel's transition band is several kHz wide, so it
    // cannot do better this close to Nyquist (voices carry little energy
    // there).
    let (ae, ah) = (alias(QualityTier::Eco), alias(QualityTier::Hq));
    assert!(ah < -18.0 && ah < ae - 14.0, "13-19 kHz alias re tone: HQ {ah:.1} dB vs Eco {ae:.1} dB");
}
