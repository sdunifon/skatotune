//! Release freeze, the DSP half of the Labs "learned note hold"
//! (`docs/research/note-intent.md` §5.1): no downward note switch while the
//! input level has fallen by more than 4 dB over the last 40 ms.
//!
//! At the end of a phrase the voice decays and most singers droop below the
//! note's midpoint; switching down there is a flip on a dying note, not a new
//! note. A real step down keeps (or regains) its level, so the freeze lets it
//! through as soon as the level stops falling. On vocadito this removed about
//! a third of the hard-tune warble on top of the learned veto.
//!
//! The level comes from the detector (`PitchEstimate::level_db`) of the
//! newest hop. In Studio that is the newest frame of the lookahead (the
//! comparison then looks slightly ahead of the frame being decided, as
//! evaluated). Realtime: a fixed 64-value ring.

/// Hops of level history kept.
const RING: usize = 64;
/// A downward switch waits while the level dropped more than this...
pub(crate) const DROP_DB: f32 = 4.0;
/// ... over this long.
pub(crate) const WINDOW_MS: f32 = 40.0;
/// Level used for a non-finite estimate, dBFS.
const FLOOR_DB: f32 = -160.0;

/// Ring of recent input levels and the freeze test.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReleaseFreeze {
    ring: [f32; RING],
    head: usize,
    len: usize,
    /// The comparison's age, hops (40 ms, at most `RING − 1`).
    hops: usize,
}

impl ReleaseFreeze {
    pub fn new(window_hops: u32) -> Self {
        Self { ring: [f32::NAN; RING], head: 0, len: 0, hops: (window_hops as usize).min(RING - 1) }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Record the newest hop's level (every hop, voiced or not).
    #[inline]
    pub fn push(&mut self, level_db: f32) {
        self.head = (self.head + 1) % RING;
        if let Some(slot) = self.ring.get_mut(self.head) {
            *slot = if level_db.is_finite() { level_db } else { FLOOR_DB };
        }
        self.len = (self.len + 1).min(RING);
    }

    /// Level `age` hops old (NaN when not recorded).
    #[inline]
    fn at(&self, age: usize) -> f32 {
        if age >= self.len { f32::NAN } else { self.ring.get((self.head + RING - age) % RING).copied().unwrap_or(f32::NAN) }
    }

    /// Whether a switch from note `cur` to note `m` waits: it goes down while
    /// the level is falling (compared with the oldest level within the
    /// window while the ring is still filling).
    pub fn blocks(&self, cur: usize, m: usize) -> bool {
        if m >= cur {
            return false;
        }
        let now = self.at(0);
        let then = self.at(self.hops.min(self.len.saturating_sub(1)));
        now.is_finite() && then.is_finite() && then - now > DROP_DB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOPS: u32 = 30; // 40 ms at 48 kHz

    fn feed(r: &mut ReleaseFreeze, levels: impl IntoIterator<Item = f32>) {
        for l in levels {
            r.push(l);
        }
    }

    #[test]
    fn blocks_only_downward_switches_while_the_level_falls() {
        let mut r = ReleaseFreeze::new(HOPS);
        assert!(!r.blocks(61, 60), "no history");
        feed(&mut r, (0..40).map(|i| -10.0 - 0.3 * i as f32)); // −9 dB over 30 hops
        assert!(r.blocks(61, 60), "down while decaying");
        assert!(r.blocks(64, 60), "any step down");
        assert!(!r.blocks(60, 61), "up is never frozen");
        assert!(!r.blocks(60, 60));
        // Exactly 4 dB is not more than 4 dB.
        let mut r = ReleaseFreeze::new(HOPS);
        feed(&mut r, (0..=30).map(|i| -10.0 - 4.0 * i as f32 / 30.0));
        assert!(!r.blocks(61, 60));
    }

    /// A real step down that plateaus is released once the level stops
    /// falling (here 40 ms after the decay ends).
    #[test]
    fn a_plateau_releases_the_freeze() {
        let mut r = ReleaseFreeze::new(HOPS);
        feed(&mut r, (0..20).map(|_| -12.0));
        feed(&mut r, (0..10).map(|i| -12.0 - i as f32)); // fall 9 dB
        assert!(r.blocks(61, 60));
        feed(&mut r, (0..30).map(|_| -21.0));
        assert!(!r.blocks(61, 60), "level steady for the whole window");
        feed(&mut r, [-15.0]);
        assert!(!r.blocks(61, 60), "rising");
    }

    #[test]
    fn non_finite_levels_count_as_silence_and_reset_clears() {
        let mut r = ReleaseFreeze::new(HOPS);
        feed(&mut r, (0..31).map(|_| -20.0));
        feed(&mut r, [f32::NAN]);
        assert!(r.blocks(61, 60), "NaN reads as −160 dB: a fall");
        r.reset();
        assert!(!r.blocks(61, 60));
        let wide = ReleaseFreeze::new(500);
        assert_eq!(wide.hops, RING - 1);
    }
}
