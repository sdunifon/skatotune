//! Melodies: note lists turned into per-sample pitch, amplitude and vowel
//! trajectories with the imperfections a corrector must handle.
//!
//! The detuning model is the research protocol's (R§10.2): a per-note offset,
//! slow Ornstein-Uhlenbeck drift, onset scoops decaying with τ ≈ 60 ms,
//! overshoot after legato transitions, and natural vibrato. Each note also
//! records the *intended* in-tune pitch, which is what a perfect corrector
//! should output.
//!
//! Pitch is in absolute cents (MIDI × 100, A4 = 6900) throughout.

use std::f64::consts::PI;

use super::formant::{Formants, Vowel};
use super::noise::Consonant;
use super::{NoteKind, NoteSpan, Voice};
use crate::rng::Rng;

/// Natural vibrato on a note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vibrato {
    pub rate_hz: f64,
    /// Half peak-to-peak deviation, cents (a ±50 c vibrato has extent 50).
    pub extent_cents: f64,
    /// Time after the note start before the vibrato begins, s.
    pub delay_s: f64,
    /// Raised-cosine fade-in once it begins, s.
    pub ramp_s: f64,
}

impl Vibrato {
    /// Vibrato with a typical 150 ms onset delay and 200 ms fade-in.
    pub fn new(rate_hz: f64, extent_cents: f64) -> Self {
        Self { rate_hz, extent_cents, delay_s: 0.15, ramp_s: 0.2 }
    }

    /// Fully developed from the first sample (for detector/tracker tests).
    pub fn immediate(rate_hz: f64, extent_cents: f64) -> Self {
        Self { rate_hz, extent_cents, delay_s: 0.0, ramp_s: 0.0 }
    }
}

/// Ornstein-Uhlenbeck pitch drift (stationary standard deviation σ, time
/// constant τ), shared by every note of a melody.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    pub sigma_cents: f64,
    pub tau_s: f64,
}

/// An unvoiced sound placed right before a note's onset (in its gap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetConsonant {
    pub kind: Consonant,
    pub dur_s: f64,
    pub level_dbfs: f64,
}

/// One sung note.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    /// Intended (in-tune) pitch, absolute cents.
    pub cents: f64,
    pub dur_s: f64,
    pub vowel: Vowel,
    /// Fixed offset from the intended pitch; `None` draws from the melody's
    /// `detune_range_cents`.
    pub detune_cents: Option<f64>,
    /// Onset scoop depth (negative = from below). Only applies after a gap.
    pub scoop_cents: f64,
    pub scoop_tau_s: f64,
    /// Overshoot past the new pitch after a legato transition, cents (applied
    /// in the direction of the interval).
    pub overshoot_cents: f64,
    /// Legato transition time from the previous note's pitch, s.
    pub glide_s: f64,
    pub vibrato: Option<Vibrato>,
    /// Silence before the note; 0 slurs it from the previous note (legato).
    pub gap_before_s: f64,
    pub consonant: Option<OnsetConsonant>,
    /// Relative level of this note, dB.
    pub level_db: f64,
    /// Glide linearly (in cents) to this pitch over the note: a sweep. The
    /// per-sample target is then the nearest semitone.
    pub sweep_to_cents: Option<f64>,
}

impl Note {
    pub fn new(cents: f64, dur_s: f64) -> Self {
        Self {
            cents,
            dur_s,
            vowel: Vowel::A,
            detune_cents: None,
            scoop_cents: 0.0,
            scoop_tau_s: 0.06,
            overshoot_cents: 0.0,
            glide_s: 0.06,
            vibrato: None,
            gap_before_s: 0.0,
            consonant: None,
            level_db: 0.0,
            sweep_to_cents: None,
        }
    }

    /// Note from a (possibly fractional) MIDI number.
    pub fn midi(note: f64, dur_s: f64) -> Self {
        Self::new(note * 100.0, dur_s)
    }

    pub fn vowel(mut self, v: Vowel) -> Self {
        self.vowel = v;
        self
    }
    pub fn detune(mut self, cents: f64) -> Self {
        self.detune_cents = Some(cents);
        self
    }
    pub fn scoop(mut self, cents: f64) -> Self {
        self.scoop_cents = cents;
        self
    }
    pub fn overshoot(mut self, cents: f64) -> Self {
        self.overshoot_cents = cents;
        self
    }
    pub fn glide(mut self, s: f64) -> Self {
        self.glide_s = s.max(0.0);
        self
    }
    pub fn vibrato(mut self, v: Vibrato) -> Self {
        self.vibrato = Some(v);
        self
    }
    pub fn gap(mut self, s: f64) -> Self {
        self.gap_before_s = s.max(0.0);
        self
    }
    pub fn consonant(mut self, kind: Consonant, dur_s: f64, level_dbfs: f64) -> Self {
        self.consonant = Some(OnsetConsonant { kind, dur_s, level_dbfs });
        self
    }
    pub fn level(mut self, db: f64) -> Self {
        self.level_db = db;
        self
    }
    pub fn sweep_to(mut self, cents: f64) -> Self {
        self.sweep_to_cents = Some(cents);
        self
    }
}

/// A sung line: notes, the voice singing them, and the shared drift model.
#[derive(Clone, Debug, PartialEq)]
pub struct Melody {
    pub voice: Voice,
    pub notes: Vec<Note>,
    /// Per-note detune drawn uniformly from ±this when a note has none.
    pub detune_range_cents: f64,
    pub drift: Option<Drift>,
    /// Soft limit (tanh) on |detune + drift| so the intended note stays the
    /// nearest legal one in the scenario's scale.
    pub deviation_limit_cents: Option<f64>,
    pub seed: u64,
    /// Amplitude fade-in at each phrase onset, s.
    pub attack_s: f64,
    /// Amplitude fade-out at each phrase end, s.
    pub release_s: f64,
}

impl Melody {
    pub fn new(voice: Voice) -> Self {
        Self {
            voice,
            notes: Vec::new(),
            detune_range_cents: 0.0,
            drift: None,
            deviation_limit_cents: None,
            seed: 1,
            attack_s: 0.02,
            release_s: 0.03,
        }
    }

    pub fn note(mut self, n: Note) -> Self {
        self.notes.push(n);
        self
    }
    pub fn notes(mut self, notes: impl IntoIterator<Item = Note>) -> Self {
        self.notes.extend(notes);
        self
    }
    pub fn detune_range(mut self, cents: f64) -> Self {
        self.detune_range_cents = cents.abs();
        self
    }
    pub fn drift(mut self, sigma_cents: f64, tau_s: f64) -> Self {
        self.drift = Some(Drift { sigma_cents, tau_s });
        self
    }
    pub fn deviation_limit(mut self, cents: f64) -> Self {
        self.deviation_limit_cents = Some(cents.abs().max(1.0));
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// A sustained pitch that alternates between `cents + step` and
    /// `cents − step` every `period_s` (instant steps, no glide): the ±40 c
    /// off-grid step test of R§10.5.
    pub fn steps(voice: Voice, cents: f64, step_cents: f64, period_s: f64, count: usize, vowel: Vowel) -> Self {
        let notes = (0..count).map(|i| {
            let d = if i % 2 == 0 { step_cents } else { -step_cents };
            Note::new(cents, period_s).vowel(vowel).detune(d).glide(0.0)
        });
        Melody::new(voice).notes(notes)
    }

    pub fn duration_s(&self) -> f64 {
        self.notes.iter().map(|n| n.gap_before_s.max(n.consonant.map_or(0.0, |c| c.dur_s)) + n.dur_s).sum()
    }
}

/// A formant trajectory piece: `from` → `to` over `ramp` samples, then `to`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VowelSeg {
    pub start: usize,
    pub from: Formants,
    pub to: Formants,
    pub ramp: usize,
}

/// A consonant to render at `start` (relative to the melody start).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacedConsonant {
    pub start: usize,
    pub len: usize,
    pub kind: Consonant,
    pub level_dbfs: f64,
}

/// Per-sample control trajectories for one melody (positions relative to the
/// melody start). NaN pitch marks unvoiced samples.
pub(crate) struct MelodyLayout {
    pub len: usize,
    pub cents: Vec<f64>,
    pub centre: Vec<f64>,
    pub target: Vec<f64>,
    pub amp: Vec<f64>,
    pub vowels: Vec<VowelSeg>,
    pub notes: Vec<NoteSpan>,
    pub consonants: Vec<PlacedConsonant>,
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    0.5 - 0.5 * (PI * x).cos()
}

/// Overshoot bump: rises to 1 at `u = TP`, then decays (u·e^{1−u/TP}/TP).
fn overshoot_bump(u: f64) -> f64 {
    const TP: f64 = 0.05;
    if u <= 0.0 { 0.0 } else { (u / TP) * (1.0 - u / TP).exp() }
}

impl Melody {
    pub(crate) fn layout(&self, sr: f64) -> MelodyLayout {
        let samples = |s: f64| (s.max(0.0) * sr).round() as usize;

        // Placement: each note's gap must hold its onset consonant.
        let mut spans = Vec::with_capacity(self.notes.len());
        let mut cursor = 0usize;
        for note in &self.notes {
            let cons_len = note.consonant.map_or(0, |c| samples(c.dur_s));
            let gap = samples(note.gap_before_s).max(cons_len);
            let start = cursor + gap;
            let end = start + samples(note.dur_s).max(1);
            spans.push((start, end, gap, cons_len));
            cursor = end;
        }
        let len = cursor;

        let mut rng = Rng::new(self.seed);
        let mut detune_rng = rng.fork(0x0de7);
        let mut drift_rng = rng.fork(0xd71f);

        // OU drift, then two 3 Hz one-poles: the plain OU process is Brownian at
        // short time scales (≈1 c RMS above 12 Hz for σ 15 c, τ 1 s), which
        // would masquerade as jitter. Real drift is slow, so smooth it.
        let mut drift = vec![0.0; len];
        if let Some(d) = self.drift.filter(|d| d.sigma_cents > 0.0 && d.tau_s > 0.0) {
            let a = (-1.0 / (d.tau_s * sr)).exp();
            let kick = d.sigma_cents * (1.0 - a * a).sqrt();
            let lp = 1.0 - (-2.0 * PI * 3.0 / sr).exp();
            let mut x = d.sigma_cents * drift_rng.gaussian();
            let (mut y1, mut y2) = (x, x);
            for v in drift.iter_mut() {
                x = a * x + kick * drift_rng.gaussian();
                y1 += lp * (x - y1);
                y2 += lp * (y1 - y2);
                *v = y2;
            }
        }

        let mut out = MelodyLayout {
            len,
            cents: vec![f64::NAN; len],
            centre: vec![f64::NAN; len],
            target: vec![f64::NAN; len],
            amp: vec![0.0; len],
            vowels: Vec::with_capacity(self.notes.len()),
            notes: Vec::with_capacity(self.notes.len()),
            consonants: Vec::new(),
        };

        let kind = self.voice.kind;
        let mut prev_centre = f64::NAN;
        let mut prev_level = 0.0;
        let mut prev_formants: Option<Formants> = None;
        for (i, (note, &(start, end, gap, cons_len))) in self.notes.iter().zip(spans.iter()).enumerate() {
            let legato = i > 0 && gap == 0;
            let next_legato = spans.get(i + 1).is_some_and(|s| s.2 == 0);
            let detune = note.detune_cents.unwrap_or_else(|| {
                if self.detune_range_cents > 0.0 {
                    detune_rng.range(-self.detune_range_cents, self.detune_range_cents)
                } else {
                    0.0
                }
            });
            let level = 10f64.powf(note.level_db / 20.0);
            let dur = (end - start) as f64 / sr;
            let glide = note.glide_s.min(dur * 0.5);
            let jump = note.cents + detune - prev_centre;
            let jump_sign = if legato && jump.abs() >= 1.0 { jump.signum() } else { 0.0 };
            let attack = samples(self.attack_s).min((end - start) / 2);
            let release = samples(self.release_s).min((end - start) / 2);
            let vib_fade = samples(0.06_f64.min(dur / 4.0));

            for n in start..end {
                let t = (n - start) as f64 / sr;
                let base = match note.sweep_to_cents {
                    Some(to) => note.cents + (to - note.cents) * (t / dur.max(1e-9)),
                    None => note.cents,
                };
                let mut dev = detune + drift.get(n).copied().unwrap_or(0.0);
                if let Some(lim) = self.deviation_limit_cents {
                    dev = lim * (dev / lim).tanh();
                }
                let cur = base + dev;
                let mut p = if legato && prev_centre.is_finite() {
                    let g = if glide > 0.0 { smoothstep(t / glide) } else { 1.0 };
                    prev_centre + (cur - prev_centre) * g
                        + jump_sign * note.overshoot_cents * overshoot_bump(t - glide)
                } else {
                    cur
                };
                if !legato && note.scoop_cents != 0.0 && note.scoop_tau_s > 0.0 {
                    p += note.scoop_cents * (-t / note.scoop_tau_s).exp();
                }
                let vib = note.vibrato.map_or(0.0, |v| {
                    if t < v.delay_s {
                        return 0.0;
                    }
                    let u = t - v.delay_s;
                    let mut a = v.extent_cents * if v.ramp_s > 0.0 { smoothstep(u / v.ramp_s) } else { 1.0 };
                    if next_legato && vib_fade > 0 {
                        a *= smoothstep((end - n) as f64 / vib_fade as f64);
                    }
                    a * (2.0 * PI * v.rate_hz * u).sin()
                });
                out.centre[n] = p;
                out.cents[n] = p + vib;
                out.target[n] = match note.sweep_to_cents {
                    Some(_) => (base / 100.0).round() * 100.0,
                    None => note.cents,
                };

                // Amplitude: phrase attack/release, legato level crossfade.
                let mut amp = if legato && glide > 0.0 {
                    prev_level + (level - prev_level) * smoothstep(t / glide)
                } else {
                    level
                };
                let k = n - start;
                if !legato && k < attack {
                    amp *= smoothstep(k as f64 / attack as f64);
                }
                let from_end = end - 1 - n;
                if !next_legato && from_end < release {
                    amp *= smoothstep(from_end as f64 / release as f64);
                }
                out.amp[n] = amp;
            }

            let formants = note.vowel.formants(kind);
            let (from, ramp) = match prev_formants {
                Some(pf) if legato && pf != formants => (pf, samples(0.06).min((end - start) / 2)),
                _ => (formants, 0),
            };
            out.vowels.push(VowelSeg { start, from, to: formants, ramp });
            if let Some(c) = note.consonant {
                out.consonants.push(PlacedConsonant {
                    start: start - cons_len,
                    len: cons_len,
                    kind: c.kind,
                    level_dbfs: c.level_dbfs,
                });
            }
            out.notes.push(NoteSpan {
                start,
                end,
                target_cents: if note.sweep_to_cents.is_some() { f64::NAN } else { note.cents },
                detune_cents: detune,
                vibrato: note.vibrato,
                legato_in: legato,
                kind: if note.sweep_to_cents.is_some() { NoteKind::Sweep } else { NoteKind::Note },
            });
            prev_centre = out.centre.get(end - 1).copied().unwrap_or(f64::NAN);
            prev_level = level;
            prev_formants = Some(formants);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::Voice;

    const SR: f64 = 48_000.0;

    #[test]
    fn legato_is_continuous_and_gaps_are_unvoiced() {
        let m = Melody::new(Voice::male())
            .note(Note::midi(57.0, 0.3).detune(10.0))
            .note(Note::midi(59.0, 0.3).detune(-10.0).overshoot(30.0))
            .note(Note::midi(60.0, 0.3).gap(0.1).scoop(-100.0));
        let l = m.layout(SR);
        assert_eq!(l.notes.len(), 3);
        let (a, b, c) = (&l.notes[0], &l.notes[1], &l.notes[2]);
        assert_eq!(a.end, b.start);
        assert!(b.legato_in && !c.legato_in);
        assert_eq!(c.start - b.end, (0.1 * SR) as usize);
        // No pitch jump across the legato boundary.
        let jump = (l.centre[b.start] - l.centre[b.start - 1]).abs();
        assert!(jump < 1.0, "jump {jump}");
        // Overshoot: the pitch exceeds the new note + detune after the glide.
        let peak = l.centre[b.start..b.end].iter().cloned().fold(f64::MIN, f64::max);
        assert!(peak > 5900.0 - 10.0 + 25.0, "peak {peak}");
        // Gap is unvoiced; the scoop starts 100 c low.
        assert!(l.cents[b.end + 10].is_nan());
        assert!((l.centre[c.start] - (6000.0 - 100.0)).abs() < 1.0);
        assert!(l.target[c.start] == 6000.0 && l.target[b.end + 10].is_nan());
    }

    #[test]
    fn vibrato_extent_and_rate() {
        let m = Melody::new(Voice::male()).note(Note::midi(57.0, 2.0).detune(0.0).vibrato(Vibrato::new(6.0, 50.0)));
        let l = m.layout(SR);
        let s = (0.6 * SR) as usize;
        let e = (1.9 * SR) as usize;
        let dev: Vec<f64> = (s..e).map(|n| l.cents[n] - l.centre[n]).collect();
        let max = dev.iter().cloned().fold(f64::MIN, f64::max);
        assert!((max - 50.0).abs() < 0.1, "extent {max}");
        let ups = dev.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((ups as f64 - 6.0 * 1.3).abs() <= 1.0, "ups {ups}");
    }

    #[test]
    fn drift_statistics_and_limit() {
        let m = Melody::new(Voice::male()).note(Note::midi(57.0, 60.0).detune(0.0)).drift(15.0, 1.0).seed(9);
        let l = m.layout(8000.0);
        let dev: Vec<f64> = l.centre.iter().map(|c| c - 5700.0).collect();
        let sd = (dev.iter().map(|d| d * d).sum::<f64>() / dev.len() as f64).sqrt();
        assert!(sd > 8.0 && sd < 25.0, "drift sd {sd}");
        let limited = Melody::new(Voice::male())
            .note(Note::midi(57.0, 60.0).detune(30.0))
            .drift(15.0, 1.0)
            .deviation_limit(40.0)
            .seed(9)
            .layout(8000.0);
        assert!(limited.centre.iter().all(|c| (c - 5700.0).abs() < 40.0));
    }

    #[test]
    fn random_detune_respects_range() {
        let m = Melody::new(Voice::male()).notes((0..50).map(|i| Note::midi(50.0 + (i % 12) as f64, 0.05))).detune_range(35.0);
        let l = m.layout(SR);
        assert!(l.notes.iter().all(|n| n.detune_cents.abs() <= 35.0));
        let spread = l.notes.iter().map(|n| n.detune_cents).fold(0.0f64, |a, d| a.max(d.abs()));
        assert!(spread > 20.0);
    }

    #[test]
    fn steps_alternate() {
        let l = Melody::steps(Voice::male(), 5700.0, 40.0, 0.5, 4, Vowel::A).layout(SR);
        let mid = |i: usize| l.centre[(l.notes[i].start + l.notes[i].end) / 2];
        assert!((mid(0) - 5740.0).abs() < 1e-9 && (mid(1) - 5660.0).abs() < 1e-9);
        assert!(l.target.iter().all(|t| *t == 5700.0));
    }

    #[test]
    fn sweep_targets_nearest_semitone() {
        let l = Melody::new(Voice::male()).note(Note::new(4500.0, 1.0).sweep_to(5700.0).detune(0.0)).layout(SR);
        let n = (0.5 * SR) as usize;
        assert!((l.centre[n] - 5100.0).abs() < 1.0);
        assert_eq!(l.target[n], 5100.0);
        assert_eq!(l.notes[0].kind, NoteKind::Sweep);
    }
}
