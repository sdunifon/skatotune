//! Synthetic singing with exact ground truth.
//!
//! Every quality gate is measured against signals from this module, so they
//! must be both realistic enough to exercise the detector and shifter, and
//! *exactly* known:
//!
//! - **Source:** a band-limited LF glottal pulse train ([`glottal`]) driven by
//!   a per-sample f0 trajectory (phase integrated in f64), with per-period
//!   jitter and shimmer and pitch-synchronously modulated aspiration noise.
//! - **Tract:** a five-formant Klatt cascade ([`formant`]) with male and
//!   female vowel presets; legato vowel changes glide over 60 ms.
//! - **Score:** [`Melody`] notes with per-note detune, onset scoops,
//!   overshoot, OU drift, vibrato, legato or gaps; plus unvoiced material
//!   (fricatives, plosives, breaths, noise bursts, click trains) placed by
//!   [`SignalBuilder`].
//! - **Truth:** [`Signal`] carries, per sample, the exact instantaneous f0 the
//!   oscillator used (`f0_truth`, 0 = unvoiced), the intended in-tune note
//!   (`target_cents`, NaN = none) and the vibrato-free sung pitch
//!   (`centre_cents`), plus note and event spans for the metrics.

pub mod formant;
pub mod glottal;
pub mod melody;
pub mod noise;

pub use formant::{FormantFilter, Formants, Vowel, VoiceType};
pub use glottal::{GlottalTable, LfModel, LfShape};
pub use melody::{Drift, Melody, Note, OnsetConsonant, Vibrato};
pub use noise::Consonant;

use crate::rng::Rng;

/// Voice quality of a singer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voice {
    pub kind: VoiceType,
    /// LF shape parameter: 0.3 pressed … 1.0 modal … 2.7 breathy.
    pub rd: f64,
    /// Per-period relative f0 perturbation (standard deviation), e.g. 0.002.
    pub jitter: f64,
    /// Per-period relative amplitude perturbation (standard deviation).
    pub shimmer: f64,
    /// Aspiration noise RMS relative to the glottal source RMS, dB.
    /// `f64::NEG_INFINITY` disables it.
    pub aspiration_db: f64,
}

impl Voice {
    /// Modal male voice with mild, healthy-range jitter and shimmer.
    pub fn male() -> Self {
        Self { kind: VoiceType::Male, rd: 1.0, jitter: 0.002, shimmer: 0.02, aspiration_db: -30.0 }
    }

    /// Slightly laxer (higher Rd) female voice.
    pub fn female() -> Self {
        Self { kind: VoiceType::Female, rd: 1.3, jitter: 0.002, shimmer: 0.02, aspiration_db: -28.0 }
    }

    /// No jitter, shimmer or aspiration: a perfectly periodic source for
    /// accuracy tests.
    pub fn clean(mut self) -> Self {
        self.jitter = 0.0;
        self.shimmer = 0.0;
        self.aspiration_db = f64::NEG_INFINITY;
        self
    }

    pub fn breathy(mut self) -> Self {
        self.rd = 2.2;
        self.aspiration_db = -16.0;
        self
    }
}

/// What a note span represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteKind {
    /// A held note with one intended pitch.
    Note,
    /// A continuous glide; the per-sample target is the nearest semitone.
    Sweep,
}

/// One note in the rendered signal (sample indices into `Signal::audio`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteSpan {
    pub start: usize,
    /// Exclusive.
    pub end: usize,
    /// Intended pitch, absolute cents (NaN for sweeps).
    pub target_cents: f64,
    /// Offset the singer applied (before drift), cents.
    pub detune_cents: f64,
    pub vibrato: Option<Vibrato>,
    /// Slurred from the previous note (no voicing gap).
    pub legato_in: bool,
    pub kind: NoteKind,
}

impl NoteSpan {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn contains(&self, n: usize) -> bool {
        n >= self.start && n < self.end
    }
}

/// Unvoiced material in the rendered signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventKind {
    Consonant(Consonant),
    NoiseBurst,
    PulseTrain,
    /// Raw samples supplied by the scenario (edge cases).
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventSpan {
    pub start: usize,
    pub end: usize,
    pub kind: EventKind,
}

/// A rendered test signal with exact per-sample ground truth.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub sample_rate: f32,
    pub audio: Vec<f32>,
    /// Instantaneous f0 actually synthesised, Hz (includes jitter); 0 = unvoiced.
    pub f0_truth: Vec<f32>,
    /// Intended in-tune note, absolute cents; NaN where there is none.
    pub target_cents: Vec<f32>,
    /// Sung pitch without vibrato and jitter (detune + drift + scoops +
    /// glides), absolute cents; NaN when unvoiced.
    pub centre_cents: Vec<f32>,
    pub notes: Vec<NoteSpan>,
    pub events: Vec<EventSpan>,
}

impl Signal {
    /// Wrap plain audio (e.g. a WAV file) with "unknown" truth.
    pub fn from_audio(audio: Vec<f32>, sample_rate: f32) -> Self {
        let n = audio.len();
        Self {
            sample_rate,
            audio,
            f0_truth: vec![0.0; n],
            target_cents: vec![f32::NAN; n],
            centre_cents: vec![f32::NAN; n],
            notes: Vec::new(),
            events: Vec::new(),
        }
    }

    /// A single sustained note with 60 ms of silence on both sides (for
    /// detector and tracker accuracy tests).
    pub fn sustained(sample_rate: f32, voice: Voice, vowel: Vowel, f0_hz: f64, dur_s: f64, vibrato: Option<Vibrato>) -> Self {
        let cents = hz_to_cents64(f0_hz);
        let mut note = Note::new(cents, dur_s).vowel(vowel).detune(0.0);
        if let Some(v) = vibrato {
            note = note.vibrato(v);
        }
        SignalBuilder::new(sample_rate).silence(0.06).melody(Melody::new(voice).note(note)).silence(0.06).build()
    }

    pub fn len(&self) -> usize {
        self.audio.len()
    }

    pub fn is_empty(&self) -> bool {
        self.audio.is_empty()
    }

    pub fn duration_s(&self) -> f64 {
        self.audio.len() as f64 / self.sample_rate as f64
    }

    pub fn is_voiced(&self, n: usize) -> bool {
        self.f0_truth.get(n).is_some_and(|f| *f > 0.0)
    }

    pub fn note_at(&self, n: usize) -> Option<&NoteSpan> {
        self.notes.iter().find(|s| s.contains(n))
    }

    /// Sample indices where the truth voicing flips.
    pub fn voicing_transitions(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut prev = false;
        for (n, f) in self.f0_truth.iter().enumerate() {
            let v = *f > 0.0;
            if v != prev && n > 0 {
                out.push(n);
            }
            prev = v;
        }
        out
    }

    /// Fraction of samples that are voiced.
    pub fn voiced_fraction(&self) -> f64 {
        if self.f0_truth.is_empty() {
            return 0.0;
        }
        self.f0_truth.iter().filter(|f| **f > 0.0).count() as f64 / self.f0_truth.len() as f64
    }
}

/// Exact Hz → absolute cents in f64.
pub fn hz_to_cents64(hz: f64) -> f64 {
    6900.0 + 1200.0 * (hz / 440.0).log2()
}

/// Exact absolute cents → Hz in f64.
pub fn cents_to_hz64(cents: f64) -> f64 {
    440.0 * ((cents - 6900.0) / 1200.0).exp2()
}

enum Item {
    Silence(f64),
    Melody(Box<Melody>),
    Consonant { kind: Consonant, dur_s: f64, level_dbfs: f64 },
    NoiseBurst { dur_s: f64, level_dbfs: f64 },
    PulseTrain { dur_s: f64, interval_s: f64, jitter: f64, amplitude: f64 },
    Raw(Vec<f32>),
}

/// Sequential timeline of voiced and unvoiced material.
pub struct SignalBuilder {
    sample_rate: f64,
    seed: u64,
    voiced_rms_dbfs: f64,
    items: Vec<Item>,
}

/// Formant ringing rendered past a melody's last sample, s.
const RING_TAIL_S: f64 = 0.05;

impl SignalBuilder {
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate: sample_rate as f64, seed: 0x5eed, voiced_rms_dbfs: -20.0, items: Vec::new() }
    }

    /// Seed for every noise component (aspiration, jitter, consonants).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// RMS of all voiced samples after rendering, dBFS (default −20).
    pub fn voiced_level(mut self, dbfs: f64) -> Self {
        self.voiced_rms_dbfs = dbfs;
        self
    }

    pub fn silence(mut self, secs: f64) -> Self {
        self.items.push(Item::Silence(secs));
        self
    }

    pub fn melody(mut self, m: Melody) -> Self {
        self.items.push(Item::Melody(Box::new(m)));
        self
    }

    /// A standalone consonant (sibilant, plosive or breath) at RMS `level_dbfs`.
    pub fn consonant(mut self, kind: Consonant, secs: f64, level_dbfs: f64) -> Self {
        self.items.push(Item::Consonant { kind, dur_s: secs, level_dbfs });
        self
    }

    /// White noise burst at RMS `level_dbfs` (latency calibration).
    pub fn noise_burst(mut self, secs: f64, level_dbfs: f64) -> Self {
        self.items.push(Item::NoiseBurst { dur_s: secs, level_dbfs });
        self
    }

    /// Click train; `jitter` in [0, 1) randomises the intervals.
    pub fn pulse_train(mut self, secs: f64, interval_s: f64, jitter: f64, amplitude: f64) -> Self {
        self.items.push(Item::PulseTrain { dur_s: secs, interval_s, jitter, amplitude });
        self
    }

    /// Append raw samples (treated as unvoiced, no target).
    pub fn raw(mut self, samples: Vec<f32>) -> Self {
        self.items.push(Item::Raw(samples));
        self
    }

    pub fn build(self) -> Signal {
        let sr = self.sample_rate;
        let samples = |s: f64| (s.max(0.0) * sr).round() as usize;
        let mut master = Rng::new(self.seed);

        // Lay out the items and render the melody controls.
        let mut layouts = Vec::new();
        let mut total = 0usize;
        for item in &self.items {
            let len = match item {
                Item::Silence(s) => samples(*s),
                Item::Melody(m) => {
                    let l = m.layout(sr);
                    let len = l.len;
                    layouts.push((total, l, m.voice));
                    len
                }
                Item::Consonant { dur_s, .. } | Item::NoiseBurst { dur_s, .. } | Item::PulseTrain { dur_s, .. } => {
                    samples(*dur_s)
                }
                Item::Raw(v) => v.len(),
            };
            total += len;
        }

        let mut voiced = vec![0.0f64; total];
        let mut unvoiced = vec![0.0f64; total];
        let mut f0 = vec![0.0f32; total];
        let mut target = vec![f32::NAN; total];
        let mut centre = vec![f32::NAN; total];
        let mut notes = Vec::new();
        let mut events = Vec::new();

        for (offset, layout, voice) in &layouts {
            let mut rng = master.fork(*offset as u64);
            let tail = samples(RING_TAIL_S);
            let (audio, f0_m) = synthesize(layout, voice, sr, tail, &mut rng);
            for (i, a) in audio.iter().enumerate() {
                if let Some(v) = voiced.get_mut(offset + i) {
                    *v += a;
                }
            }
            for (i, f) in f0_m.iter().enumerate() {
                let n = offset + i;
                if n >= total {
                    break;
                }
                f0[n] = *f as f32;
                target[n] = layout.target[i] as f32;
                centre[n] = layout.centre[i] as f32;
            }
            for s in &layout.notes {
                notes.push(NoteSpan { start: s.start + offset, end: s.end + offset, ..*s });
            }
            for c in &layout.consonants {
                let buf = noise::render_consonant(c.kind, c.len, sr, c.level_dbfs, &mut rng);
                add_into(&mut unvoiced, offset + c.start, &buf);
                events.push(EventSpan { start: offset + c.start, end: offset + c.start + c.len, kind: EventKind::Consonant(c.kind) });
            }
        }

        // Standalone unvoiced items.
        let mut cursor = 0usize;
        let mut unv_rng = master.fork(0xbad5);
        for item in &self.items {
            let len = match item {
                Item::Silence(s) => samples(*s),
                Item::Melody(_) => {
                    // Lengths were computed in the same order above.
                    let l = layouts.iter().find(|(o, _, _)| *o == cursor).map_or(0, |(_, l, _)| l.len);
                    cursor += l;
                    continue;
                }
                Item::Consonant { kind, dur_s, level_dbfs } => {
                    let len = samples(*dur_s);
                    let buf = noise::render_consonant(*kind, len, sr, *level_dbfs, &mut unv_rng);
                    add_into(&mut unvoiced, cursor, &buf);
                    events.push(EventSpan { start: cursor, end: cursor + len, kind: EventKind::Consonant(*kind) });
                    len
                }
                Item::NoiseBurst { dur_s, level_dbfs } => {
                    let len = samples(*dur_s);
                    let buf = noise::render_noise_burst(len, sr, *level_dbfs, &mut unv_rng);
                    add_into(&mut unvoiced, cursor, &buf);
                    events.push(EventSpan { start: cursor, end: cursor + len, kind: EventKind::NoiseBurst });
                    len
                }
                Item::PulseTrain { dur_s, interval_s, jitter, amplitude } => {
                    let len = samples(*dur_s);
                    let buf = noise::render_pulse_train(len, sr, *interval_s, *jitter, *amplitude, &mut unv_rng);
                    add_into(&mut unvoiced, cursor, &buf);
                    events.push(EventSpan { start: cursor, end: cursor + len, kind: EventKind::PulseTrain });
                    len
                }
                Item::Raw(v) => {
                    let buf: Vec<f64> = v.iter().map(|x| *x as f64).collect();
                    add_into(&mut unvoiced, cursor, &buf);
                    events.push(EventSpan { start: cursor, end: cursor + v.len(), kind: EventKind::Raw });
                    v.len()
                }
            };
            cursor += len;
        }

        // Calibrate the sung level over voiced samples only.
        let (sum, count) = voiced
            .iter()
            .zip(f0.iter())
            .filter(|(_, f)| **f > 0.0)
            .fold((0.0, 0usize), |(s, c), (v, _)| (s + v * v, c + 1));
        let gain = if count > 0 && sum > 0.0 {
            10f64.powf(self.voiced_rms_dbfs / 20.0) / (sum / count as f64).sqrt()
        } else {
            1.0
        };
        let mut mix: Vec<f64> = voiced.iter().zip(unvoiced.iter()).map(|(v, u)| v * gain + u).collect();
        // Guard against clipping only if a scenario asks for extreme levels.
        let peak = mix.iter().fold(0.0f64, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
        if peak > 0.98 {
            let g = 0.98 / peak;
            for v in &mut mix {
                *v *= g;
            }
        }
        let audio: Vec<f32> = mix.iter().map(|v| *v as f32).collect();
        notes.sort_by_key(|n| n.start);
        events.sort_by_key(|e| e.start);
        Signal { sample_rate: sr as f32, audio, f0_truth: f0, target_cents: target, centre_cents: centre, notes, events }
    }
}

fn add_into(dst: &mut [f64], at: usize, src: &[f64]) {
    for (d, s) in dst.iter_mut().skip(at).zip(src.iter()) {
        *d += s;
    }
}

/// Effective group delay (samples) of the vocal tract for a voice at `f0`, as
/// seen by waveform-matching pitch estimators.
///
/// The radiated sound's pitch lags the source by the tract's group delay, which
/// is dispersive (largest at the formants). Correlation-type estimators (NSDF,
/// YIN, E/H, mark tracking) locate their peak by the *derivative energy* of
/// the waveform (see `reference.rs` stage 3), so the delay they observe is the
/// group delay averaged over harmonics with weights |c_k|²·|H(k·f0)|²·(k·f0)².
fn radiated_delay(table: &GlottalTable, formants: &Formants, f0: f64, sr: f64, f_lo: f64, f_hi: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for k in 1..=table.harmonics() {
        let f = k as f64 * f0;
        if f >= f_hi {
            break;
        }
        let (gain, delay) = formants.response(sr, f);
        let w = table.harmonic_power(k) * GlottalTable::taper(f, f_lo, f_hi).powi(2) * gain * f * f;
        num += w * delay;
        den += w;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Linear interpolation of a NaN-delimited trajectory at fractional `pos`,
/// clamped to the voiced samples on either side.
fn traj_at(traj: &[f64], pos: f64, fallback: f64) -> f64 {
    let i = pos.floor().max(0.0) as usize;
    let frac = pos - pos.floor();
    match (traj.get(i).copied().filter(|v| v.is_finite()), traj.get(i + 1).copied().filter(|v| v.is_finite())) {
        (Some(a), Some(b)) => a + (b - a) * frac,
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => fallback,
    }
}

/// Glottal source + aspiration through the formant cascade for one melody.
/// Returns (audio over `layout.len + tail`, exact radiated f0 per sample over
/// `layout.len`).
///
/// The source runs the pitch trajectory *ahead* by the tract's radiated delay
/// (`radiated_delay`), so the pitch of the output waveform at sample n is the
/// layout's pitch at n: `f0_truth` then describes the audio itself, not the
/// hidden source (they differ by 0.5-2 ms, i.e. up to ~3 c on a fast vibrato).
fn synthesize(layout: &melody::MelodyLayout, voice: &Voice, sr: f64, tail: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
    use std::f64::consts::PI;
    // Harmonics up to 0.45·fs (≤ 20 kHz), tapered over the top 20 %. The taper
    // keeps harmonics from popping in and out as f0 moves.
    let f_hi = (0.45 * sr).min(20_000.0);
    let f_lo = 0.8 * f_hi;
    let lowest_f0 = layout
        .cents
        .iter()
        .filter(|c| c.is_finite())
        .fold(f64::INFINITY, |m, c| m.min(cents_to_hz64(*c) * (1.0 - 6.0 * voice.jitter)));
    let max_h = if lowest_f0.is_finite() && lowest_f0 > 0.0 { (f_hi / lowest_f0).ceil() as usize + 1 } else { 1 };
    let table = GlottalTable::new(LfShape::from_rd(voice.rd), max_h.min(4000));
    let te = table.shape().te;
    let asp_gain = if voice.aspiration_db.is_finite() { table.rms() * 10f64.powf(voice.aspiration_db / 20.0) } else { 0.0 };

    // Per-sample radiated delay: per note from its vowel and nominal pitch,
    // ramped across legato transitions together with the formants.
    let mut delay = vec![0.0f64; layout.len];
    let mut prev_d = 0.0;
    for (note, seg) in layout.notes.iter().zip(layout.vowels.iter()) {
        let mid = layout.centre.get((note.start + note.end) / 2).copied().unwrap_or(f64::NAN);
        let f0 = if mid.is_finite() { cents_to_hz64(mid) } else { 200.0 };
        let d = radiated_delay(&table, &seg.to, f0, sr, f_lo, f_hi);
        let ramp = if note.legato_in { (0.06 * sr) as usize } else { 0 };
        for (i, slot) in delay.iter_mut().enumerate().take(note.end).skip(note.start) {
            let k = i - note.start;
            *slot = if k < ramp { prev_d + (d - prev_d) * (0.5 - 0.5 * (PI * k as f64 / ramp as f64).cos()) } else { d };
        }
        prev_d = d;
    }

    let mut jit_rng = rng.fork(1);
    let mut asp_rng = rng.fork(2);
    let draw_jitter = |rng: &mut Rng| -> (f64, f64) {
        let j = 1.0 + voice.jitter * rng.gaussian().clamp(-4.0, 4.0);
        let s = (1.0 + voice.shimmer * rng.gaussian().clamp(-4.0, 4.0)).clamp(0.5, 1.5);
        (j, s)
    };

    let first = layout.vowels.first().map_or(Formants::neutral(voice.kind), |v| v.from);
    let mut filter = FormantFilter::new(sr, &first);
    let mut seg_idx = 0usize;

    let total = layout.len + tail;
    let mut out = vec![0.0; total];
    // Jitter factor of the source period active at each sample.
    let mut jit_at = vec![1.0f64; layout.len];
    let mut phase = 0.0f64;
    let mut was_voiced = false;
    let (mut jit, mut shim) = (1.0, 1.0);
    for (n, o) in out.iter_mut().enumerate() {
        let cents = layout.cents.get(n).copied().unwrap_or(f64::NAN);
        let mut excitation = 0.0;
        if cents.is_finite() {
            if !was_voiced {
                // Each phrase starts at glottal opening.
                phase = 0.0;
                (jit, shim) = draw_jitter(&mut jit_rng);
            }
            let d = delay.get(n).copied().unwrap_or(0.0);
            let src_cents = traj_at(&layout.cents, n as f64 + d, cents);
            let f0 = cents_to_hz64(src_cents) * jit;
            if let Some(slot) = jit_at.get_mut(n) {
                *slot = jit;
            }
            let amp = layout.amp.get(n).copied().unwrap_or(0.0);
            excitation = amp * shim * table.sample(phase, f0, f_lo, f_hi);
            if asp_gain > 0.0 {
                // Turbulence is strongest while the glottis is open.
                let open = if phase < te { (PI * phase / te).sin().powi(2) } else { 0.0 };
                excitation += asp_gain * amp * (0.25 + 0.75 * open) * asp_rng.gaussian();
            }
            phase += f0 / sr;
            while phase >= 1.0 {
                phase -= 1.0;
                (jit, shim) = draw_jitter(&mut jit_rng);
            }
            was_voiced = true;
        } else {
            was_voiced = false;
        }

        // Formant trajectory.
        while seg_idx + 1 < layout.vowels.len() && layout.vowels.get(seg_idx + 1).is_some_and(|s| n >= s.start) {
            seg_idx += 1;
        }
        if let Some(seg) = layout.vowels.get(seg_idx)
            && n >= seg.start
        {
            if seg.ramp > 0 && n < seg.start + seg.ramp {
                let t = (n - seg.start) as f64 / seg.ramp as f64;
                filter.set(&seg.from.lerp(&seg.to, 0.5 - 0.5 * (PI * t).cos()));
            } else {
                filter.set(&seg.to);
            }
        }
        *o = filter.process(excitation);
    }

    // Radiated truth: the layout pitch at n with the jitter of the source
    // period that radiates at n (emitted `delay` samples earlier).
    let mut f0_out = vec![0.0; layout.len];
    for (n, slot) in f0_out.iter_mut().enumerate() {
        let c = layout.cents.get(n).copied().unwrap_or(f64::NAN);
        if !c.is_finite() {
            continue;
        }
        let back = delay.get(n).copied().unwrap_or(0.0).round() as usize;
        let mut src = n.saturating_sub(back);
        while src < n && !layout.cents.get(src).is_some_and(|v| v.is_finite()) {
            src += 1; // stay inside the phrase at its onset
        }
        *slot = cents_to_hz64(c) * jit_at.get(src).copied().unwrap_or(1.0);
    }
    (out, f0_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_dependent() {
        let build = |seed| {
            SignalBuilder::new(48_000.0)
                .seed(seed)
                .silence(0.05)
                .melody(Melody::new(Voice::male()).note(Note::midi(57.0, 0.3).consonant(Consonant::S, 0.1, -35.0).gap(0.1)))
                .build()
        };
        let (a, b) = (build(1), build(1));
        assert_eq!(a.audio, b.audio);
        assert_eq!(a.f0_truth, b.f0_truth);
        assert_eq!(a.notes, b.notes);
        assert_ne!(a.audio, build(2).audio);
    }

    #[test]
    fn truth_arrays_line_up_with_layout() {
        let s = SignalBuilder::new(48_000.0)
            .silence(0.1)
            .melody(
                Melody::new(Voice::male().clean())
                    .note(Note::midi(57.0, 0.4).detune(12.0))
                    .note(Note::midi(60.0, 0.4).gap(0.2).consonant(Consonant::Sh, 0.15, -35.0)),
            )
            .noise_burst(0.1, -20.0)
            .build();
        assert_eq!(s.notes.len(), 2);
        let a = s.notes[0];
        assert_eq!(a.start, 4800);
        assert!(!s.is_voiced(a.start - 1) && s.is_voiced(a.start) && s.is_voiced(a.end - 1));
        let mid = (a.start + a.end) / 2;
        let f = s.f0_truth[mid] as f64;
        assert!((hz_to_cents64(f) - 5712.0).abs() < 0.01);
        assert_eq!(s.target_cents[mid], 5700.0);
        assert!(s.target_cents[a.end + 10].is_nan());
        assert_eq!(s.events.len(), 2);
        assert_eq!(s.events[0].end, s.notes[1].start);
        assert_eq!(s.voicing_transitions(), vec![a.start, a.end, s.notes[1].start, s.notes[1].end]);
        // Voiced RMS calibrated to −20 dBFS.
        let (sum, cnt) = s.audio.iter().zip(&s.f0_truth).filter(|(_, f)| **f > 0.0).fold((0.0f64, 0), |(a, c), (x, _)| {
            (a + (*x as f64).powi(2), c + 1)
        });
        let db = 10.0 * (sum / cnt as f64).log10();
        assert!((db + 20.0).abs() < 0.5, "voiced level {db}");
    }

    /// Spectrum of a steady clean note at a high, non-integer-ratio f0: every
    /// line must sit on a harmonic, with nothing aliased in between.
    #[test]
    fn source_is_alias_free() {
        for sr in [44_100.0f32, 48_000.0] {
            let f0 = 1234.5;
            let s = Signal::sustained(sr, Voice::female().clean(), Vowel::A, f0, 1.2, None);
            let n = 32_768;
            let start = (0.3 * sr as f64) as usize;
            let x: Vec<f64> = s.audio[start..start + n]
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // 4-term Blackman-Harris window: −92 dB sidelobes.
                    let p = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                    let w = 0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
                    *v as f64 * w
                })
                .collect();
            let mut planner = realfft::RealFftPlanner::<f64>::new();
            let fft = planner.plan_fft_forward(n);
            let mut inp = x.clone();
            let mut spec = fft.make_output_vec();
            fft.process(&mut inp, &mut spec).unwrap();
            let bin_hz = sr as f64 / n as f64;
            let power: Vec<f64> = spec.iter().map(|c| c.norm_sqr()).collect();
            let total: f64 = power.iter().sum();
            let mut off_harmonic = 0.0;
            for (k, p) in power.iter().enumerate() {
                let f = k as f64 * bin_hz;
                let h = (f / f0).round();
                if (f - h * f0).abs() > 6.0 * bin_hz && f > 100.0 {
                    off_harmonic += p;
                }
            }
            let ratio_db = 10.0 * (off_harmonic / total).log10();
            assert!(ratio_db < -80.0, "sr {sr}: off-harmonic energy {ratio_db} dB");
        }
    }

    #[test]
    fn jitter_changes_period_per_cycle_only() {
        let s = Signal::sustained(48_000.0, Voice::male(), Vowel::A, 110.0, 0.5, None);
        let voiced: Vec<f32> = s.f0_truth.iter().copied().filter(|f| *f > 0.0).collect();
        // Piecewise constant per period: count changes ≈ number of periods.
        let changes = voiced.windows(2).filter(|w| w[0] != w[1]).count();
        assert!((changes as f64 - 55.0).abs() < 3.0, "changes {changes}");
        let mean = voiced.iter().map(|f| *f as f64).sum::<f64>() / voiced.len() as f64;
        assert!((mean - 110.0).abs() < 0.2);
    }
}
