//! Unit tests for the engine's private wiring (the public contract is tested
//! in `tests/`).

use super::*;
use vt_core::{EngineKind, QualityTier, VoiceRange};

fn cfg(mode: LatencyMode) -> EngineConfig {
    EngineConfig { sample_rate: 48_000.0, channels: 1, latency_mode: mode, ..EngineConfig::default() }
}

#[test]
fn sanitize_maps_non_finite_to_silence_and_clamps() {
    assert_eq!(sanitize(f32::NAN), 0.0);
    assert_eq!(sanitize(f32::INFINITY), 0.0);
    assert_eq!(sanitize(f32::NEG_INFINITY), 0.0);
    assert_eq!(sanitize(1.0e30), INPUT_LIMIT);
    assert_eq!(sanitize(-f32::MAX), -INPUT_LIMIT);
    for v in [0.0f32, -0.5, 0.999, 3.0, -1.0e3] {
        assert_eq!(sanitize(v).to_bits(), v.to_bits());
    }
}

#[test]
fn hop_follows_the_sample_rate() {
    for (sr, hop) in
        [(44_100.0f32, 64), (48_000.0, 64), (88_200.0, 128), (96_000.0, 128), (176_400.0, 256), (192_000.0, 256)]
    {
        let e = Engine::new(EngineConfig { sample_rate: sr, ..cfg(LatencyMode::Live) });
        assert_eq!(e.hop, hop, "{sr}");
        assert_eq!(e.analysis.len(), hop);
    }
}

/// CL-3: Studio splits D between the detector's centring/fixed-lag
/// smoothing and the control law's lookahead, and the corrector really gets
/// that lookahead (it used to derive a lead of 0 from a detector centred on
/// `newest − D`). With the RT-2 pipeline one hop of it goes to the deferred
/// analysis.
#[test]
fn studio_splits_its_delay_between_detector_and_corrector() {
    let ranges = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];
    for sr in [44_100.0f32, 48_000.0, 96_000.0, 192_000.0] {
        for range in ranges {
            for studio_ms in [20.0f32, 40.0, 50.0] {
                let c = EngineConfig { sample_rate: sr, voice_range: range, studio_ms, ..cfg(LatencyMode::Studio) };
                let mut e = Engine::new(c);
                let d = e.latency as usize;
                let (delay, lead) = studio_split(sr, e.hop, range, d);
                assert_eq!(delay + lead * e.hop, d);
                assert_eq!(e.detector.config().alignment, Alignment::Centered { delay });
                assert_eq!(e.pipelined, lead >= 1, "{sr} {range:?} {studio_ms}");
                // The corrector's derived lead once its frame ring has filled.
                let mut buf = vec![0.0f32; 64 * e.hop];
                e.process(&mut [&mut buf[..]], &[]);
                assert_eq!(e.corrector.lookahead_lead(), lead.saturating_sub(1), "{sr} {range:?} {studio_ms}");
            }
        }
    }
    // The default Studio setup (40 ms, Alto/Tenor, 48 kHz) has real lookahead.
    let e = Engine::new(cfg(LatencyMode::Studio));
    let (_, lead) = studio_split(48_000.0, e.hop, VoiceRange::AltoTenor, e.latency as usize);
    assert!(lead >= 8, "lead {lead}");
    assert!(e.studio);
    for mode in [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus] {
        let e = Engine::new(cfg(mode));
        assert_eq!(e.detector.config().alignment, Alignment::Causal);
        assert_eq!(e.shifter.latency(), e.latency as usize);
        assert!(!e.pipelined);
    }
}

/// A voiced test signal at `sr`: harmonic notes (with a glide, vibrato and
/// gaps) separated by silence, so the analyses see onsets, releases and
/// moving pitch.
fn voiced(sr: f32, seconds: f32) -> Vec<f32> {
    let n = (seconds * sr) as usize;
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sr);
            let note = (t / 0.35) as usize;
            let local = t - note as f64 * 0.35;
            if local > 0.28 {
                phase = 0.0;
                return 0.0;
            }
            let base = [196.0, 247.0, 165.0, 330.0, 220.0][note % 5];
            let f0 =
                base * 2f64.powf((40.0 * (local / 0.28) + 30.0 * (std::f64::consts::TAU * 5.5 * t).sin()) / 1200.0);
            phase += f0 / f64::from(sr);
            let env = (local / 0.01).min(1.0) * ((0.28 - local) / 0.02).min(1.0);
            let x: f64 = (1..=12).map(|k| (std::f64::consts::TAU * k as f64 * phase).sin() / k as f64).sum();
            (0.25 * env * x) as f32
        })
        .collect()
}

/// Renders `x` through a Studio engine in blocks of `blk` samples, with the
/// tier changing at fixed sample positions (every `tier_every` samples), and
/// returns the output and the telemetry.
fn render_studio(sr: f32, range: VoiceRange, x: &[f32], blk: usize, one_shot: bool) -> (Vec<f32>, Vec<PitchFrame>) {
    let mut e = Engine::new(EngineConfig { sample_rate: sr, voice_range: range, ..cfg(LatencyMode::Studio) });
    assert!(e.pipelined, "{sr} {range:?}");
    e.one_shot = one_shot;
    e.set_telemetry_enabled(true);
    let mut rx = e.take_telemetry().expect("telemetry");
    let tiers = [QualityTier::Hq, QualityTier::Max, QualityTier::Eco];
    let tier_every = (0.3 * sr) as usize + 17;
    let mut out = x.to_vec();
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos < out.len() {
        let seg_end = ((pos / tier_every + 1) * tier_every).min(out.len());
        let tier = tiers[(pos / tier_every) % tiers.len()];
        e.set_settings(&Settings { tier, retune_ms: 0.0, ..Settings::default() });
        while pos < seg_end {
            let len = blk.min(seg_end - pos);
            e.process(&mut [&mut out[pos..pos + len]], &[]);
            pos += len;
            while let Ok(f) = rx.pop() {
                frames.push(f);
            }
        }
    }
    (out, frames)
}

/// PERF-7: the Studio pipeline runs each analysis in stages spread over the
/// next hop (`Detector::begin_analysis` / `step_analysis`), pushing the
/// detector every sample. Its output and telemetry are bit-identical to an
/// analysis run in one go at the boundary (what the pipeline did before,
/// with the new samples held back), for any host block size and with tier
/// changes landing while an analysis is in flight.
#[test]
fn studio_split_analysis_matches_a_one_shot_analysis() {
    for (sr, range) in [
        (44_100.0f32, VoiceRange::AltoTenor),
        (48_000.0, VoiceRange::Soprano),
        (96_000.0, VoiceRange::AltoTenor),
        (192_000.0, VoiceRange::Wide),
    ] {
        let x = voiced(sr, 1.6);
        let (reference, ref_frames) = render_studio(sr, range, &x, 64, true);
        assert!(ref_frames.iter().filter(|f| f.flags & vt_core::frame_flags::VOICED != 0).count() > 100, "{sr}");
        let hop = vt_core::hop_for_rate(sr);
        for blk in [1usize, 7, 16, 32, hop / 8, hop / 2 + 3, hop, 1000] {
            let (out, frames) = render_studio(sr, range, &x, blk, false);
            let first = out.iter().zip(&reference).position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(first, None, "{sr} {range:?} block {blk}: first differing sample");
            assert_eq!(frames, ref_frames, "{sr} {range:?} block {blk}: telemetry");
        }
    }
}

/// Which steps of the Studio pipeline ran in each host block of `blk`
/// samples (the engine's test `work` bits), over 200 hops of voiced input.
fn studio_work_per_block(sr: f32, tier: QualityTier, blk: usize) -> (Engine, Vec<u32>) {
    let mut e =
        Engine::new(EngineConfig { sample_rate: sr, voice_range: VoiceRange::Wide, ..cfg(LatencyMode::Studio) });
    assert!(e.pipelined);
    e.set_settings(&Settings { tier, ..Settings::default() });
    let x = voiced(sr, 200.0 * e.hop as f32 / sr);
    let mut work = Vec::new();
    for c in x.chunks(blk) {
        let mut buf = c.to_vec();
        e.work = 0;
        e.process(&mut [&mut buf[..]], &[]);
        work.push(e.work);
    }
    (e, work)
}

/// RT-2 / PERF-7: in Studio the work of a hop is spread over the next hop
/// at fixed points, so host blocks smaller than a hop share it out. With
/// blocks of an eighth of a hop (192 kHz / 32 samples) no block runs more
/// than one of: the shifter update (mark re-seed), each detector stage, the
/// control law. With blocks of a quarter hop the shifter update still never
/// shares a block with a detector stage, and with blocks up to half a hop
/// the heaviest Studio stage (Track: the pitch HMM's fixed-lag passes)
/// never shares one with the shifter update or with Frames. Every step
/// still runs exactly once per hop. (The whole analysis used to be one
/// chunk: the worst Studio block at 192 kHz/32 was ~106 µs, 63 % of its
/// deadline.)
#[test]
fn studio_analysis_is_spread_over_the_next_hop() {
    const TRACK: u32 = 1 << 2;
    const FRAMES: u32 = 1;
    let heavy = 0b1111 | WORK_CONTROL | WORK_SHIFTER;
    for sr in [48_000.0f32, 96_000.0, 192_000.0] {
        for tier in [QualityTier::Hq, QualityTier::Max] {
            let hop = vt_core::hop_for_rate(sr);
            for blk in [hop / 8, hop / 4, hop / 2, 16, 32] {
                let (e, work) = studio_work_per_block(sr, tier, blk);
                for (i, &w) in work.iter().enumerate() {
                    if blk <= hop / 8 {
                        assert!((w & heavy).count_ones() <= 1, "{sr} {tier:?} block {blk} #{i}: {w:#b}");
                    }
                    if blk <= hop / 4 {
                        assert!(w & WORK_SHIFTER == 0 || w & 0b1111 == 0, "{sr} {tier:?} block {blk} #{i}: {w:#b}");
                    }
                    if blk <= hop / 2 {
                        assert!(
                            w & TRACK == 0 || w & (WORK_SHIFTER | FRAMES) == 0,
                            "{sr} {tier:?} block {blk} #{i}: {w:#b}"
                        );
                    }
                }
                assert_eq!(e.stats.hops, 200);
                // The analysis of the last hop has only begun.
                assert_eq!((e.analyses, e.controls), (199, 199), "{sr} {tier:?} block {blk}");
                assert_eq!(e.detector.analysis_steps_left(), ANALYSIS_STEPS);
            }
        }
    }
}

#[test]
fn shifter_engine_switch_is_immediate_before_the_first_sample() {
    let mut e = Engine::new(cfg(LatencyMode::Live));
    e.set_settings(&Settings { engine: EngineKind::Classic, ..Settings::default() });
    assert_eq!(e.shifter.engine(), EngineKind::Classic);
    assert!(!e.shifter_dirty);
}

#[test]
fn shifter_engine_switch_waits_for_the_hop_boundary() {
    let mut e = Engine::new(cfg(LatencyMode::Live));
    let mut buf = [0.0f32; 10];
    e.process(&mut [&mut buf[..]], &[]);
    e.set_settings(&Settings { engine: EngineKind::Classic, tier: QualityTier::Eco, ..Settings::default() });
    assert_eq!(e.shifter.engine(), EngineKind::Natural, "applied before the boundary");
    assert!(e.shifter_dirty);
    let mut buf = [0.0f32; 53];
    e.process(&mut [&mut buf[..]], &[]);
    assert_eq!(e.shifter.engine(), EngineKind::Natural, "still inside the first hop");
    // The boundary decides the change; the shifter gets it with the first
    // sample after the boundary (Live defers the shifter update, PERF-5).
    let mut buf = [0.0f32; 1];
    e.process(&mut [&mut buf[..]], &[]);
    assert!(!e.shifter_dirty);
    assert!(e.shifter_due.is_some_and(|u| u.engine == Some((EngineKind::Classic, QualityTier::Eco))));
    // A change requested now waits for the next boundary.
    e.set_settings(&Settings { engine: EngineKind::Natural, tier: QualityTier::Eco, ..Settings::default() });
    let mut buf = [0.0f32; 1];
    e.process(&mut [&mut buf[..]], &[]);
    assert_eq!(e.shifter.engine(), EngineKind::Classic);
    assert!(e.shifter_dirty);
}

/// PERF-5: in Live the shifter update of a hop boundary (`set_control`,
/// which re-seeds the mark tracker at a voiced onset: up to ~55 µs at
/// 192 kHz) is applied before the first sample after the boundary instead
/// of at it. With host blocks that end on a boundary it therefore never
/// shares a `process()` call with that hop's analysis (it used to: the
/// worst Live block at 192 kHz/32 was analysis + seed, 89-95 µs), and the
/// output is unchanged (same sample position; block-size invariance and the
/// output hashes in vt_eval are bit-identical).
#[test]
fn live_shifter_update_never_shares_a_block_with_the_analysis() {
    for sr in [48_000.0f32, 96_000.0, 192_000.0] {
        let mut e = Engine::new(EngineConfig { sample_rate: sr, ..cfg(LatencyMode::Live) });
        for blk in [16usize, 32, e.hop / 2] {
            e.reset();
            e.analyses = 0;
            e.shifter_updates = 0;
            let mut both = 0;
            let mut buf = vec![0.0f32; blk];
            for i in 0..(200 * e.hop / blk) {
                for (j, v) in buf.iter_mut().enumerate() {
                    *v = 0.3 * ((i * blk + j) as f32 * 0.03).sin();
                }
                let (a0, u0) = (e.analyses, e.shifter_updates);
                e.process(&mut [&mut buf[..]], &[]);
                if e.analyses > a0 && e.shifter_updates > u0 {
                    both += 1;
                }
            }
            assert_eq!(both, 0, "{sr} block {blk}");
            assert_eq!(e.analyses, 200);
            // The last boundary's update is still pending.
            assert_eq!(e.shifter_updates, 199, "{sr} block {blk}");
        }
    }
}

#[test]
fn reset_applies_a_pending_switch_cleanly() {
    let mut e = Engine::new(cfg(LatencyMode::Live));
    let mut buf = [0.0f32; 10];
    e.process(&mut [&mut buf[..]], &[]);
    e.set_settings(&Settings { engine: EngineKind::Classic, ..Settings::default() });
    e.reset();
    assert_eq!(e.shifter.engine(), EngineKind::Classic);
    assert!(!e.shifter_dirty);
    assert_eq!((e.pos, e.hop_fill), (0, 0));
}

#[test]
fn live_min_forces_classic() {
    let mut e = Engine::new(cfg(LatencyMode::LiveMin));
    assert_eq!(e.shifter.engine(), EngineKind::Classic);
    e.set_settings(&Settings { engine: EngineKind::Natural, ..Settings::default() });
    assert_eq!(e.shifter.engine(), EngineKind::Classic);
}

#[test]
fn segments_never_cross_a_hop_boundary() {
    // After any block the hop accumulator is the sample count modulo hop.
    let mut e = Engine::new(cfg(LatencyMode::Live));
    let mut total = 0usize;
    for n in [1usize, 63, 64, 65, 130, 7, 4096, 3] {
        let mut buf = vec![0.0f32; n];
        e.process(&mut [&mut buf[..]], &[]);
        total += n;
        assert_eq!(e.hop_fill, total % e.hop);
        assert_eq!(e.pos, total as u64);
        assert_eq!(e.stats.hops, (total / e.hop) as u64);
    }
}

#[test]
fn telemetry_frame_is_shifted_to_output_time() {
    let e = Engine::new(EngineConfig { voice_range: VoiceRange::LowMale, ..cfg(LatencyMode::Live) });
    let est = PitchEstimate { sample_pos: 1000, ..PitchEstimate::default() };
    let out = ControlOutput {
        shift: vt_core::ShiftControl::default(),
        shift_cents: 0.0,
        frame: PitchFrame { detected: 5000.0, ..PitchFrame::default() },
    };
    let f = e.telemetry_frame(&est, &out);
    assert_eq!(f.sample_pos, 1000 + u64::from(e.latency));
    assert_eq!(f.detected, 5000.0);
}
