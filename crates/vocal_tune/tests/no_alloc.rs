//! The plugin's per-block path must not allocate (DESIGN §2.1): parameter
//! snapshot, Settings resolution, MIDI conversion, engine call, upmix, gain
//! and the CPU meter. A counting global allocator watches only the test's
//! own thread while `process_block` runs. Separate binary because it
//! replaces the global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use nice_plug::midi::{Channel, Key, VoiceID};
use nice_plug::prelude::*;
use vocal_tune::params::{KeyChoice, LatencyModeChoice, MidiModeChoice, ParamValues, ScaleChoice};
use vocal_tune::{BlockHost, VocalTune};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by `f` on this thread.
fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (r, ALLOCS.load(Ordering::Relaxed) - before)
}

/// A host with a fixed event list (no allocation while handing events out).
struct Host {
    events: [NoteEvent<()>; 4],
    next: usize,
    latency_calls: usize,
}

impl BlockHost for Host {
    fn next_event(&mut self) -> Option<NoteEvent<()>> {
        let ev = self.events.get(self.next).copied();
        self.next += 1;
        ev
    }
    fn set_latency_samples(&mut self, _samples: u32) {
        self.latency_calls += 1;
    }
    fn request_restart(&mut self) {}
    fn api(&self) -> PluginApi {
        PluginApi::Vst3
    }
}

fn host() -> Host {
    let note = |timing, key, velocity| NoteEvent::NoteOn {
        timing,
        voice_id: VoiceID::Wildcard,
        channel: Channel::Number(0),
        key: Key::Number(key),
        velocity,
    };
    Host {
        events: [
            note(0, 60, 0.9),
            NoteEvent::MidiPitchBend {
                timing: 10,
                channel: 0,
                value: 0.7,
            },
            NoteEvent::MidiCC {
                timing: 20,
                channel: 0,
                cc: 64,
                value: 1.0,
            },
            note(30, 64, 0.0),
        ],
        next: 0,
        latency_calls: 0,
    }
}

#[test]
fn process_block_does_not_allocate() {
    let mut plugin = VocalTune::default();
    let mut values = ParamValues {
        key: KeyChoice::E,
        scale: ScaleChoice::Major,
        midi_mode: MidiModeChoice::Auto,
        output_gain_db: -3.0,
        ..ParamValues::default()
    };
    plugin.activate_with(&values, 48_000.0, 512, 2);

    let mut left = vec![0.0f32; 512];
    let mut right = vec![0.0f32; 512];
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        *l = (i as f32 * 0.03).sin() * 0.5;
        *r = *l;
    }

    // Warm up once, then measure many blocks, including a structural change
    // (latency report to the host) and odd block sizes.
    {
        let mut io: [&mut [f32]; 2] = [&mut left, &mut right];
        plugin.process_block(&values, &mut io, &mut host());
    }
    let mut latency_calls = 0;
    let (_, allocs) = count(|| {
        for block in 0..200 {
            if block == 100 {
                values.latency_mode = LatencyModeChoice::Studio;
            }
            let n = [512, 1, 7, 64, 480][block % 5];
            let mut h = host();
            let (l, r) = (&mut left[..n], &mut right[..n]);
            let mut io: [&mut [f32]; 2] = [l, r];
            plugin.process_block(&values, &mut io, &mut h);
            latency_calls += h.latency_calls;
        }
    });
    assert_eq!(allocs, 0, "process_block allocated {allocs} times");
    assert_eq!(latency_calls, 1, "the structural change was reported once");
}

#[test]
fn counter_sees_allocations() {
    // Negative control: the counter must notice a real allocation.
    let (v, allocs) = count(|| vec![1u8; 64]);
    assert_eq!(v.len(), 64);
    assert!(allocs >= 1);
}
