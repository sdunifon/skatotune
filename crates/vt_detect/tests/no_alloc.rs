//! Realtime-safety proof: after `Detector::new`, no detector call allocates
//! (DESIGN §2.1 rule 1). A counting global allocator records allocations per
//! thread so parallel tests do not interfere; a negative control shows the
//! counter works.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::*;
use vt_core::{QualityTier, VoiceRange};
use vt_detect::{Alignment, Detector, DetectorConfig};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
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

fn count<F: FnOnce()>(f: F) -> usize {
    let before = ALLOCS.with(|c| c.get());
    f();
    ALLOCS.with(|c| c.get()) - before
}

#[test]
fn counter_detects_allocation() {
    let n = count(|| drop(std::hint::black_box(vec![1.0f32; 64])));
    assert!(n >= 1);
}

#[test]
fn detector_calls_do_not_allocate() {
    let fs = 48_000.0f64;
    // Varied input: silence, vowel with vibrato, noise, sibilant, garbage.
    let mut x = vec![0.0f32; 4800];
    x.extend(Voice::default().render(fs, &vibrato(fs, 196.0, 6.0, 50.0, 0.3)));
    x.extend(white(4800, 0.05, 1));
    x.extend(sibilant(fs, 4800, 0.05, 2));
    x.extend([f32::NAN, f32::INFINITY, 1e30, -1e30].repeat(100));
    x.extend(Voice { formants: Some(VOWEL_I), ..Voice::default() }.render(fs, &steady(fs, 880.0, 0.2)));
    for range in [VoiceRange::LowMale, VoiceRange::Wide] {
        for al in [Alignment::Causal, Alignment::Centered { delay: 2400 }] {
            let mut d =
                Detector::new(DetectorConfig { sample_rate: fs as f32, voice_range: range, hop: 64, alignment: al });
            let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
            let mut sink = 0.0f32;
            let n = count(|| {
                for (i, chunk) in x.chunks(64).enumerate() {
                    if i % 40 == 0 {
                        d.set_tier(tiers[(i / 40) % 3]);
                        d.set_tracking((i % 101) as f32, -55.0 - (i % 7) as f32);
                    }
                    // Every third hop through the split API (PERF-7).
                    let e = if i % 3 == 0 {
                        d.push(chunk);
                        d.analyze()
                    } else {
                        d.begin_analysis();
                        let mut e = d.step_analysis();
                        d.push(chunk);
                        while e.is_none() && d.analysis_steps_left() > 1 {
                            e = d.step_analysis();
                        }
                        e.unwrap_or_else(|| d.finish_analysis())
                    };
                    sink += e.cents + d.candidates().len() as f32;
                }
                d.reset();
                d.push(&x[..1000]);
                sink += d.analyze().level_db;
            });
            std::hint::black_box(sink);
            assert_eq!(n, 0, "{range:?} {al:?}: {n} allocations on the audio path");
        }
    }
}
