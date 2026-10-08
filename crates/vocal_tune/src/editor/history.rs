//! GUI-side telemetry storage and the smooth display clock.
//!
//! [`PitchHistory`] is a fixed-capacity ring of [`PitchFrame`]s, allocated
//! once when the editor first opens and never resized. [`DisplayClock`] turns
//! bursty frame delivery (one burst per host block, 1-85 ms apart) into a
//! steadily advancing "now" so the graph scrolls smoothly at the GUI frame
//! rate (R§8.23: `now_pos = last.sample_pos + elapsed·sr`, clamped).

use std::time::Instant;

use vt_core::PitchFrame;

/// Control hop in samples for a sample rate (DESIGN §2.2): the spacing of
/// telemetry frames. Mirrors `vt_core::hop_for_rate`, which the core crate
/// adds alongside its DSP modules.
pub fn control_hop(sample_rate: f32) -> u64 {
    if sample_rate <= 50_000.0 {
        64
    } else if sample_rate <= 100_000.0 {
        128
    } else {
        256
    }
}

/// Frames kept: > 10 s at the densest telemetry rate (hop 64 at 48 kHz = 750
/// frames/s, hop 256 at 192 kHz likewise), comfortably more than the 6 s view.
pub const HISTORY_CAPACITY: usize = 16_384;

/// Fixed-capacity ring of telemetry frames in arrival (= output time) order.
#[derive(Default)]
pub struct PitchHistory {
    buf: Vec<PitchFrame>,
    /// Index of the oldest frame.
    head: usize,
    len: usize,
}

impl PitchHistory {
    /// Allocate the ring (GUI thread, once). Pushes before this are ignored.
    pub fn ensure_allocated(&mut self) {
        if self.buf.is_empty() {
            self.buf = vec![PitchFrame::default(); HISTORY_CAPACITY];
            self.head = 0;
            self.len = 0;
        }
    }

    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Append a frame, overwriting the oldest when full. A frame older than
    /// the newest one means the engine was reset or rebuilt (positions start
    /// again at 0), so the stale history is dropped.
    pub fn push(&mut self, frame: PitchFrame) {
        let cap = self.buf.len();
        if cap == 0 {
            return;
        }
        if self
            .last()
            .is_some_and(|last| frame.sample_pos < last.sample_pos)
        {
            self.clear();
        }
        let idx = (self.head + self.len) % cap;
        if let Some(slot) = self.buf.get_mut(idx) {
            *slot = frame;
        }
        if self.len < cap {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % cap;
        }
    }

    /// Frame `i` counted from the oldest (0) to the newest (`len - 1`).
    pub fn get(&self, i: usize) -> Option<&PitchFrame> {
        if i >= self.len {
            return None;
        }
        self.buf.get((self.head + i) % self.buf.len())
    }

    pub fn last(&self) -> Option<&PitchFrame> {
        self.len.checked_sub(1).and_then(|i| self.get(i))
    }

    /// Index of the first frame with `sample_pos >= pos` (`len` if none).
    /// Positions are monotonic, so this is a binary search.
    pub fn lower_bound(&self, pos: u64) -> usize {
        let (mut lo, mut hi) = (0, self.len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.get(mid) {
                Some(f) if f.sample_pos < pos => lo = mid + 1,
                _ => hi = mid,
            }
        }
        lo
    }

    /// Iterate frames `from..to` (logical indices, clamped).
    pub fn iter_range(&self, from: usize, to: usize) -> impl Iterator<Item = &PitchFrame> + '_ {
        (from..to.min(self.len)).filter_map(move |i| self.get(i))
    }
}

/// Seconds the display may run ahead of the newest frame. Covers the gap
/// between bursts for host blocks up to ~85 ms; when audio stops the graph
/// coasts this far and then halts instead of scrolling into emptiness.
const MAX_LEAD_S: f64 = 0.1;
/// Time constant of the clock's correction toward the frame timeline.
const SLEW_TAU_S: f64 = 0.15;
/// Errors larger than this are jumps (seek, restart), not jitter: snap.
const SNAP_S: f64 = 0.25;

/// Smoothly advancing display position in output samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct DisplayClock {
    pos: f64,
    newest: Option<(u64, Instant)>,
    last_tick: Option<Instant>,
}

impl DisplayClock {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record the newest frame position whenever frames arrive.
    pub fn on_frames(&mut self, newest_pos: u64, now: Instant) {
        if self.newest.is_none() {
            self.pos = newest_pos as f64;
        }
        self.newest = Some((newest_pos, now));
    }

    /// Advance to `now` and return the display position (output samples).
    pub fn tick(&mut self, now: Instant, sample_rate: f64) -> f64 {
        let dt = self
            .last_tick
            .map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f64())
            .min(0.25);
        self.last_tick = Some(now);
        let Some((newest, arrived)) = self.newest else {
            return self.pos;
        };
        let sr = sample_rate.max(1.0);
        // Free-run at real-time speed...
        self.pos += dt * sr;
        // ...and steer toward where the frame timeline says "now" is.
        let since = now
            .saturating_duration_since(arrived)
            .as_secs_f64()
            .min(MAX_LEAD_S);
        let target = newest as f64 + since * sr;
        let err = target - self.pos;
        if err.abs() > SNAP_S * sr {
            self.pos = target;
        } else {
            self.pos += err * (1.0 - (-dt / SLEW_TAU_S).exp());
        }
        // Never run past the lead limit even when the slew lags.
        self.pos = self.pos.min(newest as f64 + MAX_LEAD_S * sr);
        self.pos
    }
}

/// Vertical view of the pitch graph, in absolute cents.
#[derive(Clone, Copy, Debug)]
pub struct GraphView {
    pub centre: f32,
    pub span: f32,
    /// Follow the sung pitch automatically.
    pub auto: bool,
    goal: f32,
}

pub const DEFAULT_SPAN: f32 = 1600.0;
pub const MIN_SPAN: f32 = 500.0;
pub const MAX_SPAN: f32 = 4800.0;
/// Follow time constant, seconds.
const FOLLOW_TAU_S: f32 = 0.35;

impl Default for GraphView {
    fn default() -> Self {
        // C4..E5 region: the middle of most singing voices.
        Self {
            centre: 6400.0,
            span: DEFAULT_SPAN,
            auto: true,
            goal: 6400.0,
        }
    }
}

impl GraphView {
    pub fn lo(&self) -> f32 {
        self.centre - 0.5 * self.span
    }

    pub fn hi(&self) -> f32 {
        self.centre + 0.5 * self.span
    }

    /// Auto-follow: move only when the pitch leaves the middle 50 % of the
    /// view (hysteresis keeps the lanes still while singing around a note),
    /// then glide there.
    pub fn follow(&mut self, latest: Option<f32>, dt: f32) {
        if !self.auto {
            return;
        }
        if let Some(p) = latest.filter(|p| p.is_finite() && *p > 0.0) {
            let margin = 0.25 * self.span;
            if p < self.lo() + margin || p > self.hi() - margin {
                self.goal = p;
            }
        }
        let a = 1.0 - (-dt.max(0.0) / FOLLOW_TAU_S).exp();
        self.centre += a * (self.goal - self.centre);
    }

    /// Manual pan by `cents` (disables auto-follow).
    pub fn pan(&mut self, cents: f32) {
        self.auto = false;
        self.centre = (self.centre + cents).clamp(1200.0, 11_000.0);
        self.goal = self.centre;
    }

    /// Zoom by `factor` (> 1 zooms in) around the centre.
    pub fn zoom(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.span = (self.span / factor).clamp(MIN_SPAN, MAX_SPAN);
        }
    }

    /// Back to auto-follow at the default zoom.
    pub fn reset(&mut self) {
        self.auto = true;
        self.span = DEFAULT_SPAN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn frame(pos: u64) -> PitchFrame {
        PitchFrame {
            sample_pos: pos,
            ..PitchFrame::default()
        }
    }

    #[test]
    fn ring_wraps_and_keeps_order() {
        let mut h = PitchHistory::default();
        h.push(frame(1)); // ignored: not allocated
        assert!(h.is_empty());
        h.ensure_allocated();
        let cap = h.capacity();
        for i in 0..(cap as u64 + 10) {
            h.push(frame(i * 64));
        }
        assert_eq!(h.len(), cap);
        assert_eq!(h.get(0).unwrap().sample_pos, 10 * 64);
        assert_eq!(h.last().unwrap().sample_pos, (cap as u64 + 9) * 64);
        assert!(h.get(cap).is_none());
    }

    #[test]
    fn lower_bound_finds_window_start() {
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        for i in 0..100u64 {
            h.push(frame(i * 64));
        }
        assert_eq!(h.lower_bound(0), 0);
        assert_eq!(h.lower_bound(64 * 50), 50);
        assert_eq!(h.lower_bound(64 * 50 + 1), 51);
        assert_eq!(h.lower_bound(u64::MAX), 100);
        assert_eq!(h.iter_range(98, 1000).count(), 2);
    }

    #[test]
    fn backwards_positions_clear_the_history() {
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        h.push(frame(1000));
        h.push(frame(2000));
        h.push(frame(64)); // engine reset
        assert_eq!(h.len(), 1);
        assert_eq!(h.last().unwrap().sample_pos, 64);
    }

    #[test]
    fn clock_advances_smoothly_between_bursts() {
        let sr = 48_000.0;
        let mut c = DisplayClock::default();
        let t0 = Instant::now();
        c.on_frames(48_000, t0);
        let mut last = c.tick(t0, sr);
        assert_eq!(last, 48_000.0);
        // 60 fps GUI, frames arriving in 512-sample bursts (~10.7 ms).
        let mut newest = 48_000u64;
        let mut next_burst = 0.0107;
        for k in 1..300 {
            let t = k as f64 / 60.0;
            while next_burst <= t {
                newest += 512;
                c.on_frames(newest, t0 + Duration::from_secs_f64(next_burst));
                next_burst += 512.0 / sr;
            }
            let now = c.tick(t0 + Duration::from_secs_f64(t), sr);
            let step = now - last;
            // Monotonic, close to one frame's worth of samples (800).
            assert!(step > 0.0, "clock went backwards at frame {k}");
            if k > 30 {
                assert!(
                    (step - 800.0).abs() < 120.0,
                    "uneven step {step} at frame {k}"
                );
            }
            last = now;
        }
    }

    #[test]
    fn clock_stops_when_audio_stops() {
        let sr = 48_000.0;
        let mut c = DisplayClock::default();
        let t0 = Instant::now();
        c.on_frames(10_000, t0);
        let mut pos = 0.0;
        for k in 0..120 {
            pos = c.tick(t0 + Duration::from_millis(k * 16), sr);
        }
        assert!(pos <= 10_000.0 + MAX_LEAD_S * sr + 1.0);
    }

    #[test]
    fn view_follow_has_hysteresis() {
        let mut v = GraphView::default();
        let c0 = v.centre;
        // Inside the middle half: no movement.
        for _ in 0..100 {
            v.follow(Some(c0 + 200.0), 1.0 / 60.0);
        }
        assert!((v.centre - c0).abs() < 1e-3);
        // Outside: glide to it.
        for _ in 0..300 {
            v.follow(Some(c0 + 700.0), 1.0 / 60.0);
        }
        assert!((v.centre - (c0 + 700.0)).abs() < 1.0);
        v.pan(-100.0);
        assert!(!v.auto);
        v.zoom(100.0);
        assert_eq!(v.span, MIN_SPAN);
        v.reset();
        assert!(v.auto);
        assert_eq!(v.span, DEFAULT_SPAN);
    }
}
