//! Realtime-safety proof: after construction, no `vt_core` processing call
//! allocates (DESIGN §2.1 rule 1). A counting global allocator records
//! allocations per thread, so tests running in parallel do not interfere.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use vt_core::fft::{Correlator, FftBackend, FftSizing};
use vt_core::filter::{Biquad, DcBlocker, Decimator, OnePoleLp};
use vt_core::interp::{InterpQuality, Interpolator, cubic};
use vt_core::math;
use vt_core::ring::Ring;
use vt_core::simd;

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

/// Runs `f` and returns how many allocations it made on this thread.
fn count<F: FnOnce()>(f: F) -> usize {
    let before = allocs();
    f();
    allocs() - before
}

#[test]
fn counter_detects_allocation() {
    // Negative control: the counter must see a real allocation.
    let n = count(|| {
        let v: Vec<f32> = std::hint::black_box(vec![1.0; 100]);
        drop(v);
    });
    assert!(n >= 1);
}

#[test]
fn processing_paths_do_not_allocate() {
    // --- construction (allowed to allocate) ---
    let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.031).sin() * 0.5).collect();
    let mut out = vec![0.0f32; 2049];
    let mut y = vec![0.0f32; 4096];
    let mut corrs: Vec<Correlator> =
        [FftBackend::RealFft, FftBackend::Vdsp].iter().map(|&b| Correlator::new(4096, b)).collect();
    // Non-power-of-two (fast) sizes: 3072 = 3·2^10 and 3840 = 15·2^8.
    for b in [FftBackend::RealFft, FftBackend::Vdsp] {
        corrs.push(Correlator::with_sizing(3072, b, FftSizing::Fast));
        corrs.push(Correlator::with_sizing(3800, b, FftSizing::Fast));
    }
    let mut ring = Ring::new(8192);
    let interp = Interpolator::new();
    let mut dec = Decimator::new(48_000.0, 3);
    let mut bq = Biquad::lowpass(48_000.0, 1000.0, 0.7);
    let mut dc = DcBlocker::new(48_000.0);
    let mut op = OnePoleLp::new(100.0);
    let mut dec_out = vec![0.0f32; 2048];

    // --- processing (must not allocate) ---
    let n = count(|| {
        for c in corrs.iter_mut() {
            c.cross_correlate(&x[..2048], &x, &mut out[..2048]);
            c.autocorrelate(&x[..2048], &mut out[..2048]);
            c.power_spectrum(&x, &mut out);
            c.diff_sq(&x, 2048, 0, &mut out[..1024]);
            // Out-of-contract lengths too.
            c.cross_correlate(&x, &x, &mut y);
        }
        ring.extend(&x);
        for &v in &x[..100] {
            ring.push(v);
        }
        let mut acc = 0.0f32;
        for k in 0..2000 {
            let p = 1000.0 + k as f64 * 1.37;
            acc += cubic(&ring, p);
            acc += interp.read(InterpQuality::Sinc8, &ring, p, 1.0);
            acc += interp.read(InterpQuality::Sinc16, &ring, p, 1.7);
            acc += ring.get(k as u64);
        }
        ring.copy_to(10, &mut y[..512]);
        let _ = ring.slices(4000, 64);
        let _ = ring.fetch::<16>(4000);
        acc += simd::dot(&x, &y);
        acc += simd::sum_sq(&x);
        simd::axpy(&mut y, 0.5, &x);
        simd::mul_acc(&mut y, &x, &x);
        simd::diff_sq(&x, 1024, 0, &mut out[..512]);
        simd::diff_sq(&x, 4000, 200, &mut out[..10]); // zero-padded tail path
        let _ = dec.push_block(&x, &mut dec_out);
        for &v in &x {
            acc += bq.process(v) + dc.process(v) + op.process(v);
            if let Some(d) = dec.push(v) {
                acc += d;
            }
            acc += math::fast_exp2(v) + math::fast_log2(v.abs()) + math::fast_tanh(v);
            acc += math::cents_to_ratio(v) + math::lin_to_db(v) + math::db_to_lin(v);
        }
        let _ = simd::active_backend();
        std::hint::black_box(acc);
    });
    assert_eq!(n, 0, "processing allocated {n} times");
}
