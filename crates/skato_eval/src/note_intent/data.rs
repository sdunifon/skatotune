//! The fixed DEV/TEST split and the per-file evaluation context
//! (annotations, the cached reference track of the input, the vocadito
//! `InputSide`, key, voice range and the singer's tuning offset).

use std::collections::BTreeMap;
use std::path::Path;

use skato_core::VoiceRange;

use super::cache;
use crate::reference::PitchTrack;
use crate::vocadito::data::{self as vdata, AnnNote, KeyEstimate};
use crate::vocadito::measure::{InputSide, TimedTrack, nearest_legal};

/// The split, committed with the harness (`crates/skato_eval/data/note_intent_splits.txt`).
/// A singer is DEV when `fnv1a64("noteintent-v1:" + singer) % 100 < 70`:
/// 28 files from 20 singers, TEST 12 files from 9 singers. It never changes;
/// every model and choice is made on DEV, and TEST is read once per release.
pub const SPLITS_TXT: &str = include_str!("../../data/note_intent_splits.txt");

/// Which files to evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Dev,
    Test,
    All,
}

impl Split {
    pub fn parse(s: &str) -> Option<Split> {
        match s {
            "dev" => Some(Split::Dev),
            "test" => Some(Split::Test),
            "all" => Some(Split::All),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Split::Dev => "dev",
            Split::Test => "test",
            Split::All => "all",
        }
    }
}

/// One line of the split file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitEntry {
    pub file: u32,
    pub singer: String,
    pub dev: bool,
}

/// The parsed split.
#[derive(Clone, Debug)]
pub struct Splits {
    pub entries: Vec<SplitEntry>,
}

/// FNV-1a 64 (the split's hash).
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The rule the split was made with.
pub fn singer_is_dev(singer: &str) -> bool {
    fnv1a64(&format!("noteintent-v1:{singer}")) % 100 < 70
}

impl Splits {
    /// The committed split.
    pub fn committed() -> Splits {
        Self::parse(SPLITS_TXT)
    }

    pub fn parse(text: &str) -> Splits {
        let entries = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                let [file, singer, split] = f.as_slice() else { return None };
                Some(SplitEntry { file: file.parse().ok()?, singer: (*singer).to_string(), dev: *split == "dev" })
            })
            .collect();
        Splits { entries }
    }

    pub fn files(&self, split: Split) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|e| match split {
                Split::Dev => e.dev,
                Split::Test => !e.dev,
                Split::All => true,
            })
            .map(|e| e.file)
            .collect()
    }

    pub fn singer(&self, file: u32) -> &str {
        self.entries.iter().find(|e| e.file == file).map_or("", |e| e.singer.as_str())
    }

    /// Files per singer (sorted by singer).
    pub fn by_singer(&self, files: &[u32]) -> BTreeMap<String, Vec<u32>> {
        let mut m: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for &f in files {
            m.entry(self.singer(f).to_string()).or_default().push(f);
        }
        m
    }
}

/// Everything the metrics need about one vocadito file.
#[derive(Clone, Debug)]
pub struct FileCtx {
    pub id: u32,
    pub singer: String,
    pub sr: f32,
    /// Input length, samples.
    pub n_samples: usize,
    pub dur_s: f64,
    /// Annotator 1's and annotator 2's notes.
    pub a1: Vec<AnnNote>,
    pub a2: Vec<AnnNote>,
    pub range: VoiceRange,
    /// Key estimated from annotator 1's notes (the vocadito evaluation's).
    pub key: Option<KeyEstimate>,
    /// Singer tuning offset: the duration-weighted circular mean of A1 note
    /// deviations from A440, cents in [−50, 50).
    pub offset: f64,
    /// Offline reference track of the input (1 ms frames).
    pub in_track: PitchTrack,
    pub input: InputSide,
}

impl FileCtx {
    /// Annotations from the dataset (`data`), the reference track from the
    /// cache (`cache_dir`, built by [`cache::prepare`]).
    pub fn load(data: &Path, cache_dir: &Path, id: u32, splits: &Splits) -> Result<FileCtx, String> {
        let ann_dir = data.join("Annotations");
        let read = |p: std::path::PathBuf| -> Result<String, String> {
            let len = std::fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?.len();
            if len > 32 << 20 {
                return Err(format!("{}: too large", p.display()));
            }
            std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
        };
        let f0 = vdata::parse_f0_csv(&read(ann_dir.join("F0").join(format!("vocadito_{id}_f0.csv")))?);
        let notes = |a: &str| {
            read(ann_dir.join("Notes").join(format!("vocadito_{id}_notes{a}.csv"))).map(|t| vdata::parse_notes_csv(&t)).unwrap_or_default()
        };
        let a1 = notes("A1");
        let a2 = notes("A2");
        let (in_track, n_samples) = cache::read_track(&cache::track_path(cache_dir, id))?;
        let sr = in_track.sample_rate as f32;
        let min_f0 = f0.iter().map(|(_, f)| *f).filter(|f| *f > 0.0).fold(f64::NAN, f64::min);
        let range = vdata::choose_range(min_f0);
        let key = vdata::estimate_key(&a1);
        let ann = TimedTrack::from_annotation(&f0);
        let (onsets, _) = ann.transitions();
        let input = InputSide::new(&in_track, &ann, &a1, &onsets);
        let devs: Vec<(f64, f64)> = a1.iter().map(|n| (n.cents() - 100.0 * (n.cents() / 100.0).round(), n.dur_s)).collect();
        let offset = circ_mean_100(&devs);
        Ok(FileCtx {
            id,
            singer: splits.singer(id).to_string(),
            sr,
            n_samples,
            dur_s: n_samples as f64 / f64::from(sr),
            a1,
            a2,
            range,
            key,
            offset,
            in_track,
            input,
        })
    }

    /// Index of the A1 note covering time `t` (s).
    pub fn a1_at(&self, t: f64) -> Option<usize> {
        note_at(&self.a1, t)
    }

    pub fn a2_at(&self, t: f64) -> Option<usize> {
        note_at(&self.a2, t)
    }
}

/// Last note with `start ≤ t < end`.
pub fn note_at(notes: &[AnnNote], t: f64) -> Option<usize> {
    let i = notes.partition_point(|n| n.start_s <= t);
    (0..i).rev().take(4).find(|&j| t < notes[j].end_s())
}

/// Grid note (cents / 100, rounded) of the nearest legal note to `cents`:
/// the ground truth of an annotated note, as the vocadito evaluation rules.
pub fn legal_note(cents: f64, legal: &[bool; 12]) -> Option<i32> {
    let c = nearest_legal(cents, legal);
    c.is_finite().then(|| (c / 100.0).round() as i32)
}

/// Weighted circular mean of cent deviations on a 100 c circle, in [−50, 50).
fn circ_mean_100(devs: &[(f64, f64)]) -> f64 {
    let (mut s, mut c, mut w) = (0.0, 0.0, 0.0);
    for &(d, wt) in devs {
        if !d.is_finite() || !wt.is_finite() || wt <= 0.0 {
            continue;
        }
        let a = 2.0 * std::f64::consts::PI * d / 100.0;
        s += wt * a.sin();
        c += wt * a.cos();
        w += wt;
    }
    if w <= 0.0 {
        return 0.0;
    }
    s.atan2(c) * 100.0 / (2.0 * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed split is the one the note-intent study used, and the
    /// rule that made it still reproduces every line.
    #[test]
    fn the_committed_split_is_frozen() {
        let s = Splits::committed();
        assert_eq!(s.entries.len(), 40);
        assert_eq!(s.files(Split::Dev).len(), 28);
        assert_eq!(s.files(Split::Test), vec![7, 9, 10, 11, 12, 13, 14, 16, 24, 30, 32, 40]);
        assert_eq!(s.by_singer(&s.files(Split::Dev)).len(), 20);
        let test_singers: Vec<String> = s.by_singer(&s.files(Split::Test)).into_keys().collect();
        assert_eq!(test_singers, ["S10", "S11", "S19", "S24", "S25", "S6", "S7", "S8", "S9"]);
        for e in &s.entries {
            assert_eq!(singer_is_dev(&e.singer), e.dev, "file {}", e.file);
        }
        assert_eq!(fnv1a64(SPLITS_TXT), 0x69b4_dcde_7990_1814, "the split file must never change");
    }
}
