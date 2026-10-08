//! WAV and CSV file helpers for the CLI and suite.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::synth::Signal;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Read any PCM/float WAV into per-channel f32 buffers in [-1, 1].
pub fn read_wav(path: &Path) -> Result<(Vec<Vec<f32>>, f32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let chans = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<std::result::Result<_, _>>()?
        }
    };
    let mut out = vec![Vec::with_capacity(interleaved.len() / chans); chans];
    for (i, v) in interleaved.iter().enumerate() {
        out[i % chans].push(*v);
    }
    Ok((out, spec.sample_rate as f32))
}

/// Write channels (equal lengths; shorter ones are zero-padded) as 32-bit float.
pub fn write_wav(path: &Path, channels: &[&[f32]], sample_rate: f32) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let spec = hound::WavSpec {
        channels: channels.len().max(1) as u16,
        sample_rate: sample_rate.round() as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    let len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    for i in 0..len {
        for c in channels {
            w.write_sample(c.get(i).copied().unwrap_or(0.0))?;
        }
    }
    w.finalize()?;
    Ok(())
}

/// Ground truth at 1 ms resolution: time, f0, target, centre (empty = none).
pub fn write_truth_csv(path: &Path, sig: &Signal) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "time_s,f0_truth_hz,target_cents,centre_cents")?;
    let step = ((sig.sample_rate as f64) / 1000.0).round().max(1.0) as usize;
    let opt = |v: f32| if v.is_finite() { format!("{v:.3}") } else { String::new() };
    for n in (0..sig.len()).step_by(step) {
        writeln!(
            f,
            "{:.4},{:.4},{},{}",
            n as f64 / sig.sample_rate as f64,
            sig.f0_truth[n],
            opt(sig.target_cents[n]),
            opt(sig.centre_cents[n])
        )?;
    }
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip() {
        let dir = std::env::temp_dir().join(format!("vt_eval_wav_{}", std::process::id()));
        let path = dir.join("rt.wav");
        let l: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) - 0.5).collect();
        let r: Vec<f32> = l.iter().map(|v| -v).collect();
        write_wav(&path, &[&l, &r], 44_100.0).unwrap();
        let (ch, sr) = read_wav(&path).unwrap();
        assert_eq!(sr, 44_100.0);
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0], l);
        assert_eq!(ch[1], r);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
