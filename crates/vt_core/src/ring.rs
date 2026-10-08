//! Power-of-two circular sample buffer addressed by *absolute* sample index.
//!
//! Every DSP stage in Vocal Tune talks about samples by their absolute index
//! since `reset()` (a `u64`, R§8 notation), not by a position inside some
//! buffer. [`Ring`] stores the most recent `capacity()` samples and translates
//! absolute indices with a bit mask, so readers (pitch-mark tracker, PSOLA
//! grains, the Classic read pointer, the detector windows) can ask for "sample
//! 1 234 567" directly. Reads outside the retained window return zeros instead
//! of panicking, which keeps every realtime path total even on bad positions.

/// Power-of-two circular buffer addressed by absolute sample index.
#[derive(Clone, Debug)]
pub struct Ring {
    buf: Box<[f32]>,
    /// `capacity − 1`; capacity is a power of two.
    mask: usize,
    /// Total samples pushed since construction / `clear()`. The newest sample
    /// has index `written − 1`.
    written: u64,
}

impl Ring {
    /// Allocates a zeroed ring holding at least `min_capacity` samples
    /// (rounded up to a power of two, minimum 1).
    pub fn new(min_capacity: usize) -> Self {
        let cap = min_capacity.max(1).next_power_of_two();
        Self { buf: vec![0.0; cap].into_boxed_slice(), mask: cap - 1, written: 0 }
    }

    /// Number of samples retained.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Total samples pushed; the newest sample has index `written() − 1`.
    #[inline]
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Oldest absolute index still stored (equals `written()` when empty).
    #[inline]
    pub fn oldest(&self) -> u64 {
        self.written.saturating_sub(self.capacity() as u64)
    }

    /// Whether absolute `index` is currently stored.
    #[inline]
    pub fn contains(&self, index: u64) -> bool {
        index < self.written && self.written - index <= self.capacity() as u64
    }

    /// Appends one sample.
    #[inline]
    pub fn push(&mut self, x: f32) {
        let i = (self.written as usize) & self.mask;
        if let Some(slot) = self.buf.get_mut(i) {
            *slot = x;
        }
        self.written = self.written.wrapping_add(1);
    }

    /// Appends a block of samples (at most two `memcpy`s). If `xs` is longer
    /// than the capacity only its newest `capacity()` samples are kept, but
    /// `written()` still advances by `xs.len()`.
    pub fn extend(&mut self, xs: &[f32]) {
        let cap = self.capacity();
        let skip = xs.len().saturating_sub(cap);
        let tail = xs.get(skip..).unwrap_or(&[]);
        let start = (self.written.wrapping_add(skip as u64) as usize) & self.mask;
        let first = tail.len().min(cap - start);
        let (a, b) = tail.split_at(first);
        if let Some(dst) = self.buf.get_mut(start..start + a.len()) {
            dst.copy_from_slice(a);
        }
        if let Some(dst) = self.buf.get_mut(..b.len()) {
            dst.copy_from_slice(b);
        }
        self.written = self.written.wrapping_add(xs.len() as u64);
    }

    /// Sample at absolute `index`; 0.0 if not yet written or already
    /// overwritten. Never panics.
    #[inline]
    pub fn get(&self, index: u64) -> f32 {
        if self.contains(index) { self.buf.get((index as usize) & self.mask).copied().unwrap_or(0.0) } else { 0.0 }
    }

    /// Sample at a signed absolute index (negative = before the stream began,
    /// reads as 0.0).
    #[inline]
    pub fn get_signed(&self, index: i64) -> f32 {
        if index < 0 { 0.0 } else { self.get(index as u64) }
    }

    /// Copies `out.len()` samples starting at absolute `start`, writing zeros
    /// wherever a sample is unavailable.
    pub fn copy_to(&self, start: u64, out: &mut [f32]) {
        let end = start.saturating_add(out.len() as u64);
        // Valid absolute range [lo, hi) ∩ [start, end).
        let lo = self.oldest().max(start);
        let hi = self.written.min(end);
        if lo >= hi {
            out.fill(0.0);
            return;
        }
        let head = (lo - start) as usize;
        let len = (hi - lo) as usize;
        let (pre, rest) = out.split_at_mut(head.min(out.len()));
        pre.fill(0.0);
        let (mid, post) = rest.split_at_mut(len.min(rest.len()));
        post.fill(0.0);
        self.copy_contiguous(lo, mid);
    }

    /// Up to two contiguous slices covering `len` samples from absolute
    /// `start`, or `None` if any of them is unavailable. The first slice holds
    /// the older samples. Useful for SIMD dot products over a ring window.
    pub fn slices(&self, start: u64, len: usize) -> Option<(&[f32], &[f32])> {
        if len == 0 {
            return Some((&[], &[]));
        }
        let last = start.checked_add(len as u64 - 1)?;
        if len > self.capacity() || !self.contains(start) || !self.contains(last) {
            return None;
        }
        let s = (start as usize) & self.mask;
        let first = len.min(self.capacity() - s);
        let a = self.buf.get(s..s + first)?;
        let b = self.buf.get(..len - first)?;
        Some((a, b))
    }

    /// `N` consecutive samples from signed absolute index `start` (zeros where
    /// unavailable). The fast path (whole window stored) is a plain copy; the
    /// interpolators use this for their tap windows.
    #[inline]
    pub fn fetch<const N: usize>(&self, start: i64) -> [f32; N] {
        let mut out = [0.0f32; N];
        if start >= 0 {
            let s = start as u64;
            if N > 0 && N <= self.capacity() && self.contains(s) && self.contains(s + (N as u64 - 1)) {
                let i = (s as usize) & self.mask;
                if let Some(src) = self.buf.get(i..i + N) {
                    out.copy_from_slice(src);
                } else {
                    for (k, o) in out.iter_mut().enumerate() {
                        *o = self.buf.get((i + k) & self.mask).copied().unwrap_or(0.0);
                    }
                }
                return out;
            }
        }
        for (k, o) in out.iter_mut().enumerate() {
            *o = self.get_signed(start.saturating_add(k as i64));
        }
        out
    }

    /// Zeros the contents and resets `written()` to 0.
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
        self.written = 0;
    }

    /// Copies `out.len()` stored samples from absolute `start` (caller ensures
    /// the whole range is available).
    fn copy_contiguous(&self, start: u64, out: &mut [f32]) {
        let s = (start as usize) & self.mask;
        let first = out.len().min(self.capacity() - s);
        let (a, b) = out.split_at_mut(first);
        if let Some(src) = self.buf.get(s..s + a.len()) {
            a.copy_from_slice(src);
        }
        if let Some(src) = self.buf.get(..b.len()) {
            b.copy_from_slice(src);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_up() {
        assert_eq!(Ring::new(0).capacity(), 1);
        assert_eq!(Ring::new(1).capacity(), 1);
        assert_eq!(Ring::new(5).capacity(), 8);
        assert_eq!(Ring::new(1024).capacity(), 1024);
        assert_eq!(Ring::new(1025).capacity(), 2048);
    }

    #[test]
    fn push_get_and_wrap() {
        let mut r = Ring::new(8);
        assert_eq!(r.get(0), 0.0);
        for i in 0..20u64 {
            r.push(i as f32 + 1.0);
        }
        assert_eq!(r.written(), 20);
        assert_eq!(r.oldest(), 12);
        // Retained: indices 12..20.
        for i in 12..20u64 {
            assert_eq!(r.get(i), i as f32 + 1.0);
        }
        // Overwritten and not-yet-written read as zero.
        assert_eq!(r.get(11), 0.0);
        assert_eq!(r.get(0), 0.0);
        assert_eq!(r.get(20), 0.0);
        assert_eq!(r.get(u64::MAX), 0.0);
        assert_eq!(r.get_signed(-1), 0.0);
        assert_eq!(r.get_signed(19), 20.0);
    }

    #[test]
    fn extend_matches_push() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin()).collect();
        for chunk in [1usize, 3, 7, 8, 13, 64, 100, 999] {
            let mut a = Ring::new(64);
            let mut b = Ring::new(64);
            for c in data.chunks(chunk) {
                a.extend(c);
            }
            for &x in &data {
                b.push(x);
            }
            assert_eq!(a.written(), b.written());
            for i in 0..1100u64 {
                assert_eq!(a.get(i), b.get(i), "chunk={chunk} i={i}");
            }
        }
    }

    #[test]
    fn extend_longer_than_capacity() {
        let mut r = Ring::new(16);
        r.push(-1.0);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        r.extend(&data);
        assert_eq!(r.written(), 51);
        for i in 35..51u64 {
            assert_eq!(r.get(i), (i - 1) as f32);
        }
        assert_eq!(r.get(34), 0.0);
    }

    #[test]
    fn copy_to_zero_fills_unavailable() {
        let mut r = Ring::new(16);
        for i in 0..40u64 {
            r.push(i as f32);
        }
        // Stored: 24..40.
        let mut out = [9.0f32; 30];
        r.copy_to(20, &mut out);
        for (k, &v) in out.iter().enumerate() {
            let idx = 20 + k as u64;
            let expect = if (24..40).contains(&idx) { idx as f32 } else { 0.0 };
            assert_eq!(v, expect, "idx={idx}");
        }
        // Entirely out of range, both sides, and near u64::MAX.
        let mut out = [9.0f32; 8];
        r.copy_to(0, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut out = [9.0f32; 8];
        r.copy_to(100, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut out = [9.0f32; 8];
        r.copy_to(u64::MAX - 3, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut empty: [f32; 0] = [];
        r.copy_to(30, &mut empty);
    }

    #[test]
    fn copy_to_across_wrap_point() {
        let mut r = Ring::new(8);
        for i in 0..13u64 {
            r.push(i as f32);
        }
        let mut out = [0.0f32; 8];
        r.copy_to(5, &mut out);
        assert_eq!(out, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn slices_and_fetch() {
        let mut r = Ring::new(8);
        for i in 0..13u64 {
            r.push(i as f32);
        }
        let (a, b) = r.slices(6, 6).unwrap();
        let joined: Vec<f32> = a.iter().chain(b).copied().collect();
        assert_eq!(joined, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        assert!(r.slices(4, 3).is_none()); // 4 overwritten
        assert!(r.slices(11, 3).is_none()); // 13 not written
        assert!(r.slices(5, 9).is_none()); // longer than capacity
        assert_eq!(r.slices(3, 0), Some((&[][..], &[][..])));
        assert_eq!(r.fetch::<4>(9), [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(r.fetch::<4>(11), [11.0, 12.0, 0.0, 0.0]);
        assert_eq!(r.fetch::<4>(3), [0.0, 0.0, 5.0, 6.0]);
        assert_eq!(r.fetch::<3>(-2), [0.0, 0.0, 0.0]);
        let mut fresh = Ring::new(8);
        fresh.extend(&[1.0, 2.0, 3.0]);
        assert_eq!(fresh.fetch::<4>(-1), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(r.fetch::<4>(i64::MAX), [0.0; 4]);
    }

    #[test]
    fn clear_resets() {
        let mut r = Ring::new(4);
        r.extend(&[1.0, 2.0, 3.0]);
        r.clear();
        assert_eq!(r.written(), 0);
        assert_eq!(r.get(0), 0.0);
        r.push(5.0);
        assert_eq!(r.get(0), 5.0);
    }
}
