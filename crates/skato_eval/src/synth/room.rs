//! Recording-room realism for held-out scenarios: a small diffuse reverb and
//! a background noise floor (EV-4: the clean synthetic scenarios have
//! neither, while real takes always have some).
//!
//! The reverb is a Schroeder/Moorer network (four damped feedback combs in
//! parallel, two allpasses in series) after a short pre-delay. It is a
//! deterministic stand-in for a room, not a model of one: what matters for
//! the harness is that note tails carry decaying pitched energy into the
//! gaps and that the direct sound is smeared by early reflections, as on a
//! roomy vocal take. The dry path is unchanged, so the ground truth still
//! describes the direct sound.

use crate::dsp::Biquad64;
use crate::rng::Rng;

/// Wet reverb signal of `x` for a reverberation time `rt60_s` (seconds to
/// −60 dB). Not normalised; the caller sets the wet level.
pub(crate) fn reverb(x: &[f64], sr: f64, rt60_s: f64) -> Vec<f64> {
    let rt60 = rt60_s.max(0.05);
    let ms = |m: f64| ((m * 1e-3 * sr).round() as usize).max(1);
    let pre = ms(12.0);
    // Mutually prime-ish comb lengths (Schroeder's 29.7-43.7 ms).
    let combs: Vec<usize> = [29.7, 37.1, 41.1, 43.7].iter().map(|m| ms(*m)).collect();
    let mut comb_buf: Vec<Vec<f64>> = combs.iter().map(|l| vec![0.0; *l]).collect();
    let mut comb_pos = vec![0usize; combs.len()];
    let mut comb_lp = vec![0.0f64; combs.len()];
    let gains: Vec<f64> = combs
        .iter()
        .map(|l| 10f64.powf(-3.0 * (*l as f64 / sr) / rt60))
        .collect();
    // High-frequency damping inside the loops (darker tail, as in rooms).
    let damp = 0.35;
    let aps = [(ms(5.0), 0.7), (ms(1.7), 0.7)];
    let mut ap_buf: Vec<Vec<f64>> = aps.iter().map(|(l, _)| vec![0.0; *l]).collect();
    let mut ap_pos = vec![0usize; aps.len()];
    let mut out = vec![0.0; x.len()];
    for (n, o) in out.iter_mut().enumerate() {
        let input = if n >= pre { x[n - pre] } else { 0.0 };
        let mut acc = 0.0;
        for c in 0..combs.len() {
            let buf = &mut comb_buf[c];
            let y = buf[comb_pos[c]];
            comb_lp[c] = (1.0 - damp) * y + damp * comb_lp[c];
            buf[comb_pos[c]] = input + gains[c] * comb_lp[c];
            comb_pos[c] = (comb_pos[c] + 1) % buf.len();
            acc += y;
        }
        let mut v = acc * 0.25;
        for (a, &(_, g)) in aps.iter().enumerate() {
            let buf = &mut ap_buf[a];
            let d = buf[ap_pos[a]];
            let w = v + g * d;
            buf[ap_pos[a]] = w;
            ap_pos[a] = (ap_pos[a] + 1) % buf.len();
            v = d - g * w;
        }
        *o = v;
    }
    out
}

/// Background noise of `len` samples at RMS `level_dbfs`: white noise tilted
/// towards low frequencies (a 1.5 kHz two-pole low-pass mixed with a quarter
/// of the white noise), like room tone and preamp hiss.
pub(crate) fn noise_floor(len: usize, sr: f64, level_dbfs: f64, rng: &mut Rng) -> Vec<f64> {
    let mut lp = Biquad64::lowpass(sr, 1500.0, std::f64::consts::FRAC_1_SQRT_2);
    let mut x: Vec<f64> = (0..len)
        .map(|_| {
            let w = rng.gaussian();
            lp.process(w) + 0.25 * w
        })
        .collect();
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / len.max(1) as f64).sqrt();
    if rms > 0.0 {
        let g = 10f64.powf(level_dbfs / 20.0) / rms;
        x.iter_mut().for_each(|v| *v *= g);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverb_decays_at_its_rt60() {
        let sr = 48_000.0;
        let mut x = vec![0.0; (1.5 * sr) as usize];
        x[0] = 1.0;
        let y = reverb(&x, sr, 0.5);
        let energy = |a: f64, b: f64| -> f64 {
            y[(a * sr) as usize..(b * sr) as usize]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
        };
        // Energy falls 60 dB per RT60: over 0.25 s that is 30 dB; allow the
        // comb/allpass build-up and damping some slack.
        let drop = 10.0 * (energy(0.1, 0.2) / energy(0.35, 0.45)).log10();
        assert!(
            (20.0..45.0).contains(&drop),
            "decay {drop:.1} dB per 0.25 s"
        );
        assert!(
            y[..(0.011 * sr) as usize].iter().all(|v| *v == 0.0),
            "pre-delay"
        );
    }

    #[test]
    fn noise_floor_level() {
        let mut r = Rng::new(3);
        let x = noise_floor(48_000, 48_000.0, -60.0, &mut r);
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        assert!((20.0 * rms.log10() + 60.0).abs() < 1e-6);
    }
}
