//! Vowel formant presets and the cascade formant filter.
//!
//! The vocal tract is modelled as five Klatt (1980) second-order resonators in
//! cascade, each with unity gain at DC, so relative formant amplitudes emerge
//! naturally from the pole positions (as in a real all-pole tract) instead of
//! being hand-set as in a parallel bank.
//!
//! F1-F3 are the Peterson & Barney (1952) averages for adult men and women.
//! F4/F5 and the bandwidths are typical values from the speech-synthesis
//! literature (Klatt 1980; Fant 1960); they mainly shape the top octave.

use std::f64::consts::PI;

/// Which formant table a voice uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoiceType {
    Male,
    Female,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vowel {
    /// /ɑ/ as in "father".
    A,
    /// /ɛ/ as in "bed".
    E,
    /// /i/ as in "beet".
    I,
    /// /ɔ/ as in "bought".
    O,
    /// /u/ as in "boot".
    U,
}

impl Vowel {
    pub const ALL: [Vowel; 5] = [Vowel::A, Vowel::E, Vowel::I, Vowel::O, Vowel::U];

    pub fn formants(self, voice: VoiceType) -> Formants {
        // (F1..F5 Hz, B1..B5 Hz)
        let (freq, bw) = match (voice, self) {
            (VoiceType::Male, Vowel::A) => ([730.0, 1090.0, 2440.0, 3400.0, 4300.0], [80.0, 90.0, 120.0, 160.0, 200.0]),
            (VoiceType::Male, Vowel::E) => ([530.0, 1840.0, 2480.0, 3450.0, 4300.0], [70.0, 90.0, 120.0, 160.0, 200.0]),
            (VoiceType::Male, Vowel::I) => ([270.0, 2290.0, 3010.0, 3500.0, 4400.0], [55.0, 90.0, 120.0, 160.0, 200.0]),
            (VoiceType::Male, Vowel::O) => ([570.0, 840.0, 2410.0, 3350.0, 4250.0], [70.0, 80.0, 120.0, 160.0, 200.0]),
            (VoiceType::Male, Vowel::U) => ([300.0, 870.0, 2240.0, 3300.0, 4200.0], [60.0, 80.0, 120.0, 160.0, 200.0]),
            (VoiceType::Female, Vowel::A) => ([850.0, 1220.0, 2810.0, 4100.0, 4950.0], [90.0, 100.0, 130.0, 170.0, 220.0]),
            (VoiceType::Female, Vowel::E) => ([610.0, 2330.0, 2990.0, 4150.0, 4950.0], [80.0, 100.0, 130.0, 170.0, 220.0]),
            (VoiceType::Female, Vowel::I) => ([310.0, 2790.0, 3310.0, 4200.0, 5000.0], [60.0, 100.0, 130.0, 170.0, 220.0]),
            (VoiceType::Female, Vowel::O) => ([590.0, 920.0, 2710.0, 4050.0, 4950.0], [80.0, 90.0, 130.0, 170.0, 220.0]),
            (VoiceType::Female, Vowel::U) => ([370.0, 950.0, 2670.0, 4000.0, 4900.0], [65.0, 90.0, 130.0, 170.0, 220.0]),
        };
        Formants { freq, bw }
    }
}

/// Five formant frequencies and bandwidths, Hz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Formants {
    pub freq: [f64; 5],
    pub bw: [f64; 5],
}

impl Formants {
    /// Neutral /ə/ (uniform tube) for breath noise.
    pub fn neutral(voice: VoiceType) -> Self {
        match voice {
            VoiceType::Male => Self { freq: [500.0, 1500.0, 2500.0, 3500.0, 4500.0], bw: [80.0, 100.0, 130.0, 170.0, 210.0] },
            VoiceType::Female => {
                Self { freq: [580.0, 1700.0, 2900.0, 4000.0, 5000.0], bw: [90.0, 110.0, 140.0, 180.0, 230.0] }
            }
        }
    }

    /// Linear interpolation of frequencies and bandwidths (t in [0, 1]).
    pub fn lerp(&self, other: &Formants, t: f64) -> Formants {
        let t = t.clamp(0.0, 1.0);
        let mut out = *self;
        for i in 0..5 {
            out.freq[i] = self.freq[i] + (other.freq[i] - self.freq[i]) * t;
            out.bw[i] = self.bw[i] + (other.bw[i] - self.bw[i]) * t;
        }
        out
    }

    /// Same frequencies with every bandwidth multiplied by `k` (breathier tract).
    pub fn widened(mut self, k: f64) -> Formants {
        for b in &mut self.bw {
            *b *= k;
        }
        self
    }

    /// Replace F1 (and its bandwidth), e.g. to place F1 near 2·f0 for octave tests.
    pub fn with_f1(mut self, f1: f64, bw1: f64) -> Formants {
        self.freq[0] = f1;
        self.bw[0] = bw1;
        self
    }

    /// Power gain |H|² and group delay (samples) of the [`FormantFilter`]
    /// cascade at frequency `f`, computed analytically from its poles.
    ///
    /// Each Klatt section is A / ((1 − p·z⁻¹)(1 − p*·z⁻¹)) with
    /// p = r·e^{jθ}, r = e^{−π·BW/fs}, θ = 2π·F/fs; a single pole contributes
    /// group delay (r·cos(ω−θ) − r²) / (1 − 2r·cos(ω−θ) + r²) (positive near
    /// the pole, slightly negative far below it).
    pub fn response(&self, sample_rate: f64, f: f64) -> (f64, f64) {
        let w = 2.0 * PI * f / sample_rate;
        let mut gain = 1.0;
        let mut delay = 0.0;
        for (fr, bw) in self.freq.iter().zip(self.bw.iter()) {
            if fr + bw > 0.47 * sample_rate || *fr <= 0.0 {
                continue; // bypassed section, as in `Resonator::set`
            }
            let r = (-PI * bw / sample_rate).exp();
            let th = 2.0 * PI * fr / sample_rate;
            // DC gain normalisation A = |1 − p|² (unity gain at DC).
            let a = 1.0 - 2.0 * r * th.cos() + r * r;
            let mut den = 1.0;
            for phi in [w - th, w + th] {
                let d = 1.0 - 2.0 * r * phi.cos() + r * r;
                den *= d;
                delay += (r * phi.cos() - r * r) / d;
            }
            gain *= a * a / den;
        }
        (gain, delay)
    }
}

/// Klatt resonator `y = A·x + B·y1 + C·y2` with unity DC gain.
#[derive(Clone, Copy, Debug, Default)]
struct Resonator {
    a: f64,
    b: f64,
    c: f64,
    y1: f64,
    y2: f64,
    bypass: bool,
}

impl Resonator {
    fn set(&mut self, sr: f64, f: f64, bw: f64) {
        // Resonators too close to Nyquist would fold back; skip them (they only
        // matter at very low sample rates, never at 44.1 kHz and up).
        if f + bw > 0.47 * sr || f <= 0.0 {
            self.bypass = true;
            return;
        }
        self.bypass = false;
        let t = 1.0 / sr;
        self.c = -(-2.0 * PI * bw * t).exp();
        self.b = 2.0 * (-PI * bw * t).exp() * (2.0 * PI * f * t).cos();
        self.a = 1.0 - self.b - self.c;
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        if self.bypass {
            return x;
        }
        let y = self.a * x + self.b * self.y1 + self.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Five-resonator cascade with retunable formants.
#[derive(Clone, Debug)]
pub struct FormantFilter {
    sr: f64,
    res: [Resonator; 5],
    current: Option<Formants>,
}

impl FormantFilter {
    pub fn new(sample_rate: f64, formants: &Formants) -> Self {
        let mut f = Self { sr: sample_rate, res: [Resonator::default(); 5], current: None };
        f.set(formants);
        f
    }

    /// Retune (cheap no-op when unchanged). State is kept, so slow formant
    /// glides stay click-free.
    pub fn set(&mut self, formants: &Formants) {
        if self.current.as_ref() == Some(formants) {
            return;
        }
        for (r, (f, b)) in self.res.iter_mut().zip(formants.freq.iter().zip(formants.bw.iter())) {
            r.set(self.sr, *f, *b);
        }
        self.current = Some(*formants);
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        self.res.iter_mut().fold(x, |acc, r| r.process(acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_at(filter: &FormantFilter, sr: f64, f: f64) -> f64 {
        let mut fl = filter.clone();
        let n = (sr * 0.5) as usize;
        let mut peak: f64 = 0.0;
        for i in 0..n {
            let y = fl.process((2.0 * PI * f * i as f64 / sr).sin());
            if i > n / 2 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn unity_dc_gain_and_resonance_peaks() {
        let sr = 48_000.0;
        let fm = Vowel::A.formants(VoiceType::Male);
        let f = FormantFilter::new(sr, &fm);
        let mut dc = f.clone();
        let mut y = 0.0;
        for _ in 0..48_000 {
            y = dc.process(1.0);
        }
        assert!((y - 1.0).abs() < 1e-6, "dc gain {y}");
        // F1 is a local maximum of the response.
        let g = |hz| gain_at(&f, sr, hz);
        assert!(g(730.0) > g(500.0) && g(730.0) > g(950.0));
    }

    /// The analytic gain and group delay match the running filter: measure the
    /// steady-state phase of sinusoids just below and above `f`.
    #[test]
    fn analytic_response_matches_filter() {
        let sr = 48_000.0;
        let fm = Vowel::E.formants(VoiceType::Female);
        let phase_gain = |f: f64| {
            let mut fl = FormantFilter::new(sr, &fm);
            let n = 48_000;
            let (mut i_acc, mut q_acc, mut p_acc) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let a = 2.0 * PI * f * i as f64 / sr;
                let y = fl.process(a.sin());
                if i >= n / 2 {
                    i_acc += y * a.sin();
                    q_acc += y * a.cos();
                    p_acc += y * y;
                }
            }
            (q_acc.atan2(i_acc), 2.0 * p_acc / (n / 2) as f64)
        };
        for f in [200.0, 640.0, 1500.0, 2400.0, 3500.0] {
            let (g, d) = fm.response(sr, f);
            let df = 2.0;
            let (p1, _) = phase_gain(f - df);
            let (p2, g_meas) = phase_gain(f + df);
            let mut dp = p2 - p1;
            while dp > PI {
                dp -= 2.0 * PI;
            }
            while dp < -PI {
                dp += 2.0 * PI;
            }
            let d_meas = -dp / (2.0 * PI * 2.0 * df / sr);
            assert!((d_meas - d).abs() < 0.02 * d.abs().max(1.0), "f {f}: delay {d} vs measured {d_meas}");
            assert!((g_meas / g - 1.0).abs() < 0.05, "f {f}: gain {g} vs measured {g_meas}");
        }
    }

    #[test]
    fn every_vowel_table_is_sorted_and_positive() {
        for v in [VoiceType::Male, VoiceType::Female] {
            for vowel in Vowel::ALL {
                let f = vowel.formants(v);
                assert!(f.freq.windows(2).all(|w| w[0] < w[1]), "{v:?} {vowel:?}");
                assert!(f.bw.iter().all(|b| *b > 0.0));
            }
        }
    }
}
