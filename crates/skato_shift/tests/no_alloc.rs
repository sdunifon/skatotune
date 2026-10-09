//! Realtime-safety proof: after `Shifter::new`, no processing call allocates
//! (DESIGN §2.1 rule 1). A counting global allocator records allocations per
//! thread, so tests running in parallel do not interfere.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use skato_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};
use skato_shift::{Shifter, ShifterConfig, latency_samples};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    // `try_with`: the TLS slot may be gone during thread teardown.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards to the system allocator; only adds a thread-local counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

#[test]
fn counter_detects_allocation() {
    // Negative control: the counter must see a real allocation.
    let before = allocs();
    let v: Vec<f32> = std::hint::black_box(vec![1.0; 100]);
    drop(v);
    assert!(allocs() > before);
}

/// A vowel gliding 100 → 500 Hz with a noise gap, so seeds (with backward
/// chains), lock losses, grain repeats/skips and splices all happen.
fn signal(n: usize, sr: f32) -> Vec<f32> {
    let mut ph = 0.0f64;
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let f0 = 100.0 * 5f64.powf((std::f64::consts::PI * t / (n as f64 / sr as f64)).sin());
            ph = (ph + f0 / sr as f64).fract();
            let gap = (n * 4 / 10..n * 5 / 10).contains(&i);
            if gap {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                0.1 * ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            } else {
                let th = 2.0 * std::f64::consts::PI * ph;
                ((1..30).map(|k| (k as f64 * th).cos() / k as f64).sum::<f64>() * 0.05) as f32
            }
        })
        .collect()
}

#[test]
fn processing_does_not_allocate() {
    let sr = 48_000.0f32;
    let n = 48_000;
    let x = signal(n, sr);
    let hop = skato_core::hop_for_rate(sr);
    let modes = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
    for mode in modes {
        // --- construction (allowed to allocate) ---
        let cfg = ShifterConfig {
            sample_rate: sr,
            channels: 2,
            hop,
            latency_mode: mode,
            voice_range: VoiceRange::Wide,
            latency: latency_samples(sr, mode, VoiceRange::Wide, 10.0, 40.0),
        };
        let mut sh = Shifter::new(cfg);
        let mut l = x.clone();
        let mut r = x.clone();
        let blocks = [64usize, 1, 37, 128, 500, 3];
        let before = allocs();
        // --- realtime calls: none may allocate ---
        let mut pos = 0usize;
        let mut bi = 0usize;
        while pos < n {
            if pos.is_multiple_of(hop) {
                let h = pos / hop;
                let i = pos;
                let gap = (n * 4 / 10..n * 5 / 10).contains(&i);
                let t = i as f64 / sr as f64;
                let f0 = 100.0 * 5f64.powf((std::f64::consts::PI * t / (n as f64 / sr as f64)).sin());
                sh.set_control(&ShiftControl {
                    ratio: [1.0f32, 1.3, 0.7, 2.0][(h / 40) % 4],
                    formant_ratio: [1.0f32, 1.2, 0.85][(h / 70) % 3],
                    voiced: !gap,
                    period_hint: if gap { 0.0 } else { (sr as f64 / f0) as f32 },
                    confidence: 0.9,
                });
                match h % 150 {
                    50 => sh.set_engine(EngineKind::Classic, QualityTier::Eco),
                    52 => sh.set_engine(EngineKind::Natural, QualityTier::Max),
                    100 => sh.set_engine(EngineKind::Natural, QualityTier::Hq),
                    _ => {}
                }
                let _ = sh.mark_cents();
                let _ = sh.diagnostics();
            }
            let len = blocks[bi % blocks.len()].min(hop - pos % hop).min(n - pos);
            bi += 1;
            let a = &x[pos..pos + len];
            let (lo, ro) = (&mut l[pos..pos + len], &mut r[pos..pos + len]);
            sh.process(a, &mut [lo, ro]);
            pos += len;
        }
        sh.reset();
        let made = allocs() - before;
        assert_eq!(made, 0, "{mode:?}: {made} allocations on the realtime path");
        let d = sh.diagnostics();
        assert!(d.seeds > 0 && d.marks > 0, "{mode:?}: tracker never ran: {d:?}");
        if mode != LatencyMode::LiveMin {
            assert!(d.grains > 0, "{mode:?}: PSOLA never ran: {d:?}");
        }
    }
}
