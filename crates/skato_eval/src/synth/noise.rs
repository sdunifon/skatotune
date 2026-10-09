//! Unvoiced material: fricatives, plosive bursts, breaths, calibration noise
//! bursts and click trains.
//!
//! Spectral shapes follow the usual acoustic-phonetics descriptions: /s/ is a
//! narrow high band peaking around 6-8 kHz, /ʃ/ a broader band around 3 kHz,
//! /f/ weak and flat above ~1.5 kHz, plosive bursts are a few ms of broadband
//! noise with a fast exponential decay, and breath is aspiration noise shaped
//! by a neutral, heavily damped tract. Every level is the RMS over the whole
//! event in dBFS, so scenarios can place them relative to the sung level.

use std::f64::consts::PI;

use super::formant::{FormantFilter, Formants, VoiceType};
use crate::dsp::Biquad64;
use crate::rng::Rng;

/// An unvoiced speech sound placed between notes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Consonant {
    /// Alveolar fricative /s/ (sibilant, 4-10 kHz).
    S,
    /// Post-alveolar fricative /ʃ/ (sibilant, 2-6 kHz).
    Sh,
    /// Labiodental fricative /f/ (weak, broadband).
    F,
    /// Alveolar plosive burst /t/.
    T,
    /// Velar plosive burst /k/.
    K,
    /// An audible in-breath.
    Breath,
}

impl Consonant {
    /// Whether a sibilance detector is expected to flag it.
    pub fn is_sibilant(self) -> bool {
        matches!(self, Consonant::S | Consonant::Sh)
    }
}

fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Raised-cosine attack/release envelope over `len` samples.
fn ar_envelope(n: usize, len: usize, attack: usize, release: usize) -> f64 {
    let a = if attack > 0 && n < attack {
        0.5 - 0.5 * (PI * n as f64 / attack as f64).cos()
    } else {
        1.0
    };
    let from_end = len.saturating_sub(n + 1);
    let r = if release > 0 && from_end < release {
        0.5 - 0.5 * (PI * from_end as f64 / release as f64).cos()
    } else {
        1.0
    };
    a * r
}

/// Scale `x` so its RMS equals `level_dbfs`.
fn normalise(x: &mut [f64], level_dbfs: f64) {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt();
    if rms > 1e-30 {
        let g = db_to_lin(level_dbfs) / rms;
        for v in x.iter_mut() {
            *v *= g;
        }
    }
}

/// Render a consonant of `len` samples at RMS `level_dbfs`.
pub(crate) fn render_consonant(
    kind: Consonant,
    len: usize,
    sr: f64,
    level_dbfs: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let ms = |m: f64| ((m * 1e-3 * sr).round() as usize).min(len / 2);
    let mut out = vec![0.0; len];
    match kind {
        Consonant::S | Consonant::Sh | Consonant::F => {
            let mut chain: Vec<Biquad64> = match kind {
                Consonant::S => vec![
                    Biquad64::highpass(sr, 4000.0, 0.707),
                    Biquad64::bandpass(sr, 7000.0, 1.2),
                    Biquad64::lowpass(sr, 11_000.0, 0.707),
                ],
                Consonant::Sh => vec![
                    Biquad64::highpass(sr, 1800.0, 0.707),
                    Biquad64::bandpass(sr, 3200.0, 1.0),
                    Biquad64::lowpass(sr, 7000.0, 0.707),
                ],
                _ => vec![
                    Biquad64::highpass(sr, 1500.0, 0.707),
                    Biquad64::lowpass(sr, 9000.0, 0.707),
                ],
            };
            let (att, rel) = (ms(25.0), ms(35.0));
            for (n, o) in out.iter_mut().enumerate() {
                let mut v = rng.gaussian();
                for f in chain.iter_mut() {
                    v = f.process(v);
                }
                *o = v * ar_envelope(n, len, att, rel);
            }
        }
        Consonant::T | Consonant::K => {
            let (centre, q, tau_ms) = if kind == Consonant::T {
                (4000.0, 0.8, 8.0)
            } else {
                (2000.0, 1.0, 15.0)
            };
            let mut bp = Biquad64::bandpass(sr, centre, q);
            let mut hp = Biquad64::highpass(sr, 500.0, 0.707);
            let att = ms(1.0).max(1);
            let tau = tau_ms * 1e-3 * sr;
            for (n, o) in out.iter_mut().enumerate() {
                let env = if n < att {
                    n as f64 / att as f64
                } else {
                    (-((n - att) as f64) / tau).exp()
                };
                *o = hp.process(bp.process(rng.gaussian())) * env;
            }
        }
        Consonant::Breath => {
            let mut tract =
                FormantFilter::new(sr, &Formants::neutral(VoiceType::Male).widened(3.0));
            let mut hp = Biquad64::highpass(sr, 150.0, 0.707);
            let (att, rel) = (len * 3 / 10, len * 3 / 10);
            for (n, o) in out.iter_mut().enumerate() {
                *o = hp.process(tract.process(rng.gaussian())) * ar_envelope(n, len, att, rel);
            }
        }
    }
    normalise(&mut out, level_dbfs);
    out
}

/// White Gaussian noise burst with 2 ms raised-cosine edges, RMS `level_dbfs`.
/// Used for latency measurement: it is aperiodic, so every engine must pass it
/// as a pure delay.
pub(crate) fn render_noise_burst(len: usize, sr: f64, level_dbfs: f64, rng: &mut Rng) -> Vec<f64> {
    let edge = ((0.002 * sr) as usize).min(len / 2);
    let mut out: Vec<f64> = (0..len)
        .map(|n| rng.gaussian() * ar_envelope(n, len, edge, edge))
        .collect();
    normalise(&mut out, level_dbfs);
    out
}

/// Single-sample clicks every `interval_s` on average; `jitter` (0-1) randomises
/// each interval by up to ±jitter·interval so the train can be made aperiodic.
pub(crate) fn render_pulse_train(
    len: usize,
    sr: f64,
    interval_s: f64,
    jitter: f64,
    amplitude: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut out = vec![0.0; len];
    let base = (interval_s * sr).max(1.0);
    let mut pos = 0.0f64;
    while (pos as usize) < len {
        if let Some(o) = out.get_mut(pos as usize) {
            *o = amplitude;
        }
        let j = jitter.clamp(0.0, 0.95) * rng.range(-1.0, 1.0);
        pos += (base * (1.0 + j)).max(1.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_energy_ratio(x: &[f64], sr: f64, f_split: f64) -> f64 {
        let mut hp = [
            Biquad64::highpass(sr, f_split, 0.541),
            Biquad64::highpass(sr, f_split, 1.307),
        ];
        let mut hi = 0.0;
        let mut tot = 0.0;
        for v in x {
            let h = hp.iter_mut().fold(*v, |a, f| f.process(a));
            hi += h * h;
            tot += v * v;
        }
        hi / tot
    }

    #[test]
    fn sibilants_are_high_frequency_and_levelled() {
        let sr = 48_000.0;
        let mut rng = Rng::new(3);
        let s = render_consonant(Consonant::S, 9600, sr, -30.0, &mut rng);
        let rms = (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt();
        assert!((20.0 * rms.log10() + 30.0).abs() < 0.01);
        assert!(band_energy_ratio(&s, sr, 4000.0) > 0.8);
        let b = render_consonant(Consonant::Breath, 9600, sr, -40.0, &mut rng);
        assert!(band_energy_ratio(&b, sr, 4000.0) < 0.5);
    }

    #[test]
    fn pulse_train_spacing() {
        let mut rng = Rng::new(1);
        let p = render_pulse_train(4800, 48_000.0, 0.01, 0.0, 0.5, &mut rng);
        let idx: Vec<usize> = p
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(idx.len(), 10);
        assert!(idx.windows(2).all(|w| w[1] - w[0] == 480));
    }
}
