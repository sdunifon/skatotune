//! Loading the vocadito dataset (Bittner et al., CC BY 4.0) and the per-file
//! facts derived from its annotations: voice-range preset, estimated key.
//!
//! Layout read (nothing else is touched; the files are data, never executed):
//!
//! ```text
//! <root>/Audio/vocadito_N.wav                     44.1 kHz mono 16-bit
//! <root>/Annotations/F0/vocadito_N_f0.csv         time_s,f0_hz (0 = unvoiced)
//! <root>/Annotations/Notes/vocadito_N_notesA1.csv start_s,pitch_hz,duration_s
//! <root>/Annotations/Notes/vocadito_N_notesA2.csv (second annotator)
//! <root>/vocadito_metadata.csv                    track_id,singer_id,average_pitch,language
//! ```
//!
//! The parsers are defensive: malformed lines are skipped, non-finite or
//! absurd values are dropped (an f0 outside 20 Hz-4 kHz counts as unvoiced),
//! and file sizes are bounded.

use std::path::{Path, PathBuf};

use vt_core::{ScaleKind, VoiceRange};

use crate::synth::hz_to_cents64;
use crate::wav;

/// Largest annotation CSV read, bytes (the real ones are < 200 kB).
const MAX_CSV_BYTES: u64 = 32 << 20;
/// Longest audio accepted, s (the real files are 8-39 s).
const MAX_AUDIO_S: f64 = 600.0;

/// One annotated note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnNote {
    pub start_s: f64,
    pub pitch_hz: f64,
    pub dur_s: f64,
}

impl AnnNote {
    pub fn end_s(&self) -> f64 {
        self.start_s + self.dur_s
    }

    /// Absolute cents (A4 = 6900).
    pub fn cents(&self) -> f64 {
        hz_to_cents64(self.pitch_hz)
    }
}

/// Per-track metadata row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackMeta {
    pub singer: String,
    /// Average pitch, MIDI note number.
    pub average_pitch: f64,
    pub language: String,
}

/// One vocadito track with its annotations.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub id: u32,
    pub meta: TrackMeta,
    pub sample_rate: f32,
    /// Mono audio (multi-channel files are averaged).
    pub audio: Vec<f32>,
    /// F0 annotation `(time_s, f0_hz)`, sorted by time; 0 Hz = unvoiced.
    pub f0: Vec<(f64, f64)>,
    pub notes_a1: Vec<AnnNote>,
    pub notes_a2: Vec<AnnNote>,
}

impl Track {
    pub fn duration_s(&self) -> f64 {
        self.audio.len() as f64 / self.sample_rate as f64
    }

    /// Lowest annotated voiced f0, Hz (NaN if nothing is voiced).
    pub fn min_f0(&self) -> f64 {
        self.f0
            .iter()
            .map(|(_, f)| *f)
            .filter(|f| *f > 0.0)
            .fold(f64::NAN, f64::min)
    }

    /// Highest annotated voiced f0, Hz (NaN if nothing is voiced).
    pub fn max_f0(&self) -> f64 {
        self.f0
            .iter()
            .map(|(_, f)| *f)
            .filter(|f| *f > 0.0)
            .fold(f64::NAN, f64::max)
    }

    /// Fraction of annotation frames that are voiced.
    pub fn voiced_fraction(&self) -> f64 {
        if self.f0.is_empty() {
            return 0.0;
        }
        self.f0.iter().filter(|(_, f)| *f > 0.0).count() as f64 / self.f0.len() as f64
    }
}

fn read_bounded(path: &Path) -> wav::Result<String> {
    let len = std::fs::metadata(path)?.len();
    if len > MAX_CSV_BYTES {
        return Err(format!(
            "{}: {len} bytes is too large for an annotation",
            path.display()
        )
        .into());
    }
    Ok(std::fs::read_to_string(path)?)
}

/// Numeric fields of every line that has at least `n` of them (others skipped).
fn numeric_rows(text: &str, n: usize) -> Vec<Vec<f64>> {
    text.lines()
        .filter_map(|line| {
            let vals: Option<Vec<f64>> = line
                .split(',')
                .take(n)
                .map(|f| f.trim().parse::<f64>().ok().filter(|v| v.is_finite()))
                .collect();
            vals.filter(|v| v.len() == n)
        })
        .collect()
}

/// Parse an F0 annotation: `time_s,f0_hz` rows (header or junk lines skipped).
/// Sorted by time; an f0 outside 20 Hz-4 kHz is unvoiced.
pub fn parse_f0_csv(text: &str) -> Vec<(f64, f64)> {
    let mut rows: Vec<(f64, f64)> = numeric_rows(text, 2)
        .into_iter()
        .filter(|r| r[0] >= 0.0)
        .map(|r| {
            let f = if (20.0..=4000.0).contains(&r[1]) {
                r[1]
            } else {
                0.0
            };
            (r[0], f)
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows.dedup_by(|a, b| a.0 == b.0);
    rows
}

/// Parse a note annotation: `start_s,pitch_hz,duration_s` rows. Notes with a
/// non-positive duration or a pitch outside 20 Hz-4 kHz are dropped.
pub fn parse_notes_csv(text: &str) -> Vec<AnnNote> {
    let mut notes: Vec<AnnNote> = numeric_rows(text, 3)
        .into_iter()
        .filter(|r| r[0] >= 0.0 && r[2] > 0.0 && r[2] < 60.0 && (20.0..=4000.0).contains(&r[1]))
        .map(|r| AnnNote {
            start_s: r[0],
            pitch_hz: r[1],
            dur_s: r[2],
        })
        .collect();
    notes.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    notes
}

/// Parse `vocadito_metadata.csv` into `(track_id, meta)` pairs.
pub fn parse_metadata(text: &str) -> Vec<(u32, TrackMeta)> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            let id = f.first()?.parse::<u32>().ok()?;
            let clean = |s: &str| -> String {
                s.chars()
                    .filter(|c| c.is_alphanumeric() || " +-_/".contains(*c))
                    .take(40)
                    .collect()
            };
            Some((
                id,
                TrackMeta {
                    singer: clean(f.get(1).copied().unwrap_or("")),
                    average_pitch: f
                        .get(2)
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|v| v.is_finite())
                        .unwrap_or(f64::NAN),
                    language: clean(f.get(3).copied().unwrap_or("")),
                },
            ))
        })
        .collect()
}

pub fn audio_path(root: &Path, id: u32) -> PathBuf {
    root.join("Audio").join(format!("vocadito_{id}.wav"))
}

/// Track ids present under `root` (those with an audio file and an F0
/// annotation), ascending.
pub fn list_tracks(root: &Path) -> Vec<u32> {
    let mut ids: Vec<u32> = std::fs::read_dir(root.join("Audio"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            name.strip_prefix("vocadito_")?
                .strip_suffix(".wav")?
                .parse::<u32>()
                .ok()
        })
        .filter(|id| {
            root.join("Annotations/F0")
                .join(format!("vocadito_{id}_f0.csv"))
                .is_file()
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Load one track (audio + F0 + both note annotations + metadata row).
pub fn load_track(root: &Path, id: u32, meta: &[(u32, TrackMeta)]) -> wav::Result<Track> {
    let (chans, sr) = wav::read_wav(&audio_path(root, id))?;
    if !(8_000.0..=384_000.0).contains(&sr) {
        return Err(format!("vocadito_{id}: unsupported sample rate {sr}").into());
    }
    let n = chans.iter().map(|c| c.len()).max().unwrap_or(0);
    if n as f64 > MAX_AUDIO_S * sr as f64 {
        return Err(format!("vocadito_{id}: longer than {MAX_AUDIO_S} s").into());
    }
    let audio: Vec<f32> = if chans.len() <= 1 {
        chans.into_iter().next().unwrap_or_default()
    } else {
        let k = 1.0 / chans.len() as f32;
        (0..n)
            .map(|i| {
                chans
                    .iter()
                    .map(|c| c.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    * k
            })
            .collect()
    }
    .into_iter()
    .map(|v| if v.is_finite() { v } else { 0.0 })
    .collect();
    let ann = root.join("Annotations");
    let f0 = parse_f0_csv(&read_bounded(
        &ann.join("F0").join(format!("vocadito_{id}_f0.csv")),
    )?);
    let notes = |a: &str| -> Vec<AnnNote> {
        read_bounded(
            &ann.join("Notes")
                .join(format!("vocadito_{id}_notes{a}.csv")),
        )
        .map(|t| parse_notes_csv(&t))
        .unwrap_or_default()
    };
    Ok(Track {
        id,
        meta: meta
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, m)| m.clone())
            .unwrap_or_default(),
        sample_rate: sr,
        audio,
        f0,
        notes_a1: notes("A1"),
        notes_a2: notes("A2"),
    })
}

/// Read the metadata table (empty if missing).
pub fn load_metadata(root: &Path) -> Vec<(u32, TrackMeta)> {
    read_bounded(&root.join("vocadito_metadata.csv"))
        .map(|t| parse_metadata(&t))
        .unwrap_or_default()
}

/// The voice-range preset a user would pick for a take: the highest floor
/// (Soprano 140 Hz, Alto/Tenor 80 Hz, else Low Male 55 Hz) that stays at
/// least 10 % below the lowest annotated f0. A higher floor means a lower
/// Live latency, so this is what a latency-aware user would choose.
pub fn choose_range(min_f0_hz: f64) -> VoiceRange {
    const MARGIN: f64 = 1.1;
    if !min_f0_hz.is_finite() {
        VoiceRange::AltoTenor
    } else if min_f0_hz >= MARGIN * VoiceRange::Soprano.f_min() as f64 {
        VoiceRange::Soprano
    } else if min_f0_hz >= MARGIN * VoiceRange::AltoTenor.f_min() as f64 {
        VoiceRange::AltoTenor
    } else {
        VoiceRange::LowMale
    }
}

/// Key/scale estimated from the note annotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyEstimate {
    /// 0 = C .. 11 = B.
    pub key: u8,
    /// `Major` or `NaturalMinor`.
    pub scale: ScaleKind,
    /// Pearson correlation of the duration-weighted pitch-class histogram with
    /// the rotated Krumhansl-Kessler profile.
    pub correlation: f64,
    /// Fraction of annotated note duration whose pitch class is in the scale.
    pub coverage: f64,
}

impl KeyEstimate {
    pub fn name(&self) -> String {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
        ];
        let q = if self.scale == ScaleKind::NaturalMinor {
            "minor"
        } else {
            "major"
        };
        format!("{} {q}", NAMES[self.key as usize % 12])
    }
}

/// Krumhansl-Kessler probe-tone profiles (C major / C minor).
const KK_MAJOR: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KK_MINOR: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Duration-weighted pitch-class histogram (pitch rounded to the nearest
/// equal-tempered semitone, A4 = 440 Hz).
pub fn pitch_class_histogram(notes: &[AnnNote]) -> [f64; 12] {
    let mut h = [0.0; 12];
    for n in notes {
        let midi = (n.cents() / 100.0).round() as i64;
        h[midi.rem_euclid(12) as usize] += n.dur_s;
    }
    h
}

fn pearson(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let ma = a.iter().sum::<f64>() / 12.0;
    let mb = b.iter().sum::<f64>() / 12.0;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..12 {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma).powi(2);
        sbb += (b[i] - mb).powi(2);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

/// Best of the 24 major/minor keys for the notes (Krumhansl-Schmuckler).
pub fn estimate_key(notes: &[AnnNote]) -> Option<KeyEstimate> {
    let h = pitch_class_histogram(notes);
    let total: f64 = h.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut best: Option<KeyEstimate> = None;
    for key in 0..12u8 {
        for (scale, prof) in [
            (ScaleKind::Major, &KK_MAJOR),
            (ScaleKind::NaturalMinor, &KK_MINOR),
        ] {
            let mut rot = [0.0; 12];
            for (pc, r) in rot.iter_mut().enumerate() {
                *r = prof[(pc + 12 - key as usize) % 12];
            }
            let r = pearson(&h, &rot);
            if best.is_none_or(|b| r > b.correlation) {
                let states = scale.note_states(key);
                let inside: f64 = (0..12)
                    .filter(|pc| states[*pc] == vt_core::NoteState::On)
                    .map(|pc| h[pc])
                    .sum();
                best = Some(KeyEstimate {
                    key,
                    scale,
                    correlation: r,
                    coverage: inside / total,
                });
            }
        }
    }
    best
}

/// Median |pitch difference| (cents) between the two annotators' notes,
/// matching each A1 note to the A2 note it overlaps most; and the fraction of
/// A1 notes that found a partner.
pub fn annotator_agreement(a1: &[AnnNote], a2: &[AnnNote]) -> (f64, f64) {
    let mut diffs = Vec::new();
    for n in a1 {
        let best = a2
            .iter()
            .map(|m| {
                let ov = n.end_s().min(m.end_s()) - n.start_s.max(m.start_s);
                (ov, m)
            })
            .filter(|(ov, _)| *ov > 0.5 * n.dur_s.min(0.1))
            .max_by(|x, y| x.0.total_cmp(&y.0));
        if let Some((_, m)) = best {
            diffs.push((n.cents() - m.cents()).abs());
        }
    }
    let matched = if a1.is_empty() {
        0.0
    } else {
        diffs.len() as f64 / a1.len() as f64
    };
    (crate::dsp::median(&diffs), matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f0_csv_parsing_is_defensive() {
        let t = "time,f0\n0.0,0.0\n0.0058,110.5\nbad,line\n0.0116,NaN\n0.0174,99999\n-1,100\n0.0232,220";
        let f = parse_f0_csv(t);
        assert_eq!(
            f,
            vec![(0.0, 0.0), (0.0058, 110.5), (0.0174, 0.0), (0.0232, 220.0)]
        );
    }

    #[test]
    fn notes_csv_parsing() {
        let t = "0.66,143.7,0.29\n1.0,158.4,0.3\n2.0,0,0.5\n3.0,100,-1\nx,y,z\n0.1,200,0.1";
        let n = parse_notes_csv(t);
        assert_eq!(n.len(), 3);
        assert_eq!(n[0].start_s, 0.1);
        assert!((n[1].end_s() - 0.95).abs() < 1e-12);
    }

    #[test]
    fn metadata_parsing() {
        let t = "track_id,singer_id,average_pitch,language\n1,S1,50,Tagalog\n30,S24,56,French+English\n";
        let m = parse_metadata(t);
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].0, 30);
        assert_eq!(m[1].1.language, "French+English");
        assert_eq!(m[0].1.average_pitch, 50.0);
    }

    #[test]
    fn range_choice_leaves_a_margin_below_the_lowest_note() {
        assert_eq!(choose_range(69.0), VoiceRange::LowMale);
        assert_eq!(choose_range(85.0), VoiceRange::LowMale);
        assert_eq!(choose_range(95.0), VoiceRange::AltoTenor);
        assert_eq!(choose_range(150.0), VoiceRange::AltoTenor);
        assert_eq!(choose_range(160.0), VoiceRange::Soprano);
        assert_eq!(choose_range(f64::NAN), VoiceRange::AltoTenor);
    }

    fn note(midi: f64, dur: f64) -> AnnNote {
        AnnNote {
            start_s: 0.0,
            pitch_hz: 440.0 * ((midi - 69.0) / 12.0).exp2(),
            dur_s: dur,
        }
    }

    #[test]
    fn key_of_a_scale_melody() {
        // G major: G A B C D E F#, tonic and dominant weighted.
        let notes: Vec<AnnNote> = [
            (67.0, 2.0),
            (69.0, 0.5),
            (71.0, 1.0),
            (72.0, 0.5),
            (74.0, 1.5),
            (76.0, 0.5),
            (78.0, 0.5),
            (67.0, 1.0),
        ]
        .iter()
        .map(|(m, d)| note(*m, *d))
        .collect();
        let k = estimate_key(&notes).unwrap();
        assert_eq!((k.key, k.scale), (7, ScaleKind::Major), "{}", k.name());
        assert!((k.coverage - 1.0).abs() < 1e-12);
        // A minor triad-heavy melody with the leading tone absent.
        let notes: Vec<AnnNote> = [
            (57.0, 2.0),
            (60.0, 1.0),
            (64.0, 1.5),
            (62.0, 0.5),
            (59.0, 0.5),
            (57.0, 1.0),
            (65.0, 0.5),
        ]
        .iter()
        .map(|(m, d)| note(*m, *d))
        .collect();
        let k = estimate_key(&notes).unwrap();
        assert_eq!(
            (k.key, k.scale),
            (9, ScaleKind::NaturalMinor),
            "{}",
            k.name()
        );
        assert!(estimate_key(&[]).is_none());
    }

    #[test]
    fn annotators_agree_on_identical_notes() {
        let a = vec![
            AnnNote {
                start_s: 0.0,
                pitch_hz: 220.0,
                dur_s: 0.5,
            },
            AnnNote {
                start_s: 0.6,
                pitch_hz: 247.0,
                dur_s: 0.5,
            },
        ];
        let (d, m) = annotator_agreement(&a, &a);
        assert_eq!(d, 0.0);
        assert_eq!(m, 1.0);
    }
}
