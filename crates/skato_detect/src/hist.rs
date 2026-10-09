//! Mirrored sample history for the analysis streams.
//!
//! Every analysis stage reads *windows* of recent samples (E/H lag products,
//! verifier frames, refinement windows, level/sibilance windows). A plain ring
//! splits a window in two at the wrap point; [`Mirror`] instead writes each
//! sample twice (at `i` and `i + capacity`), so any window of up to
//! `capacity` recent samples is one contiguous slice that SIMD kernels and FFT
//! copies can consume directly. The cost is one extra store per sample.
//!
//! Indices are *stream* sample numbers since the last reset (`i64`, so the
//! analysis maths can look "before the start" without underflow). The buffer
//! starts out as if `capacity` zeros had been written before sample 0, so
//! windows reaching back before the stream began read as silence.

/// Power-of-two mirrored history buffer (see the module docs).
#[derive(Clone, Debug)]
pub(crate) struct Mirror {
    /// `2 × capacity` samples; slot `k` and `k + capacity` hold the same value.
    buf: Box<[f32]>,
    /// `capacity − 1`.
    mask: usize,
    /// Internal write counter, starting at `capacity` (the virtual zero prefix).
    written: u64,
}

impl Mirror {
    /// History holding at least `min_capacity` samples (rounded up to a power
    /// of two, minimum 2).
    pub(crate) fn new(min_capacity: usize) -> Self {
        let cap = min_capacity.max(2).next_power_of_two();
        Self { buf: vec![0.0; 2 * cap].into_boxed_slice(), mask: cap - 1, written: cap as u64 }
    }

    /// Number of samples retained.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Stream index of the next sample to be written (= samples pushed since
    /// the last reset).
    #[inline]
    pub(crate) fn len(&self) -> i64 {
        (self.written - self.capacity() as u64) as i64
    }

    /// Appends one sample.
    #[inline]
    pub(crate) fn push(&mut self, x: f32) {
        let cap = self.capacity();
        let i = (self.written as usize) & self.mask;
        if let Some(s) = self.buf.get_mut(i) {
            *s = x;
        }
        if let Some(s) = self.buf.get_mut(i + cap) {
            *s = x;
        }
        self.written += 1;
    }

    /// The `len` samples with stream indices `end − len .. end` as one slice
    /// (oldest first), or `None` if any of them is in the future or already
    /// overwritten.
    #[inline]
    pub(crate) fn window(&self, end: i64, len: usize) -> Option<&[f32]> {
        let cap = self.capacity();
        // Internal (always positive) index of the window end.
        let e = end.checked_add(cap as i64)?;
        if e < 0 || len > cap || e as u64 > self.written {
            return None;
        }
        let e = e as u64;
        if e < len as u64 || e - (len as u64) < self.written.saturating_sub(cap as u64) {
            return None;
        }
        let p = (e as usize) & self.mask;
        self.buf.get(p + cap - len..p + cap)
    }

    /// The `len` samples ending at `end` (exclusive), newest first, copied into
    /// `out[..len]`. Unavailable windows produce zeros. Returns the number of
    /// samples written (`min(len, out.len())`).
    pub(crate) fn copy_reversed(&self, end: i64, len: usize, out: &mut [f32]) -> usize {
        let len = len.min(out.len());
        let dst = &mut out[..len];
        match self.window(end, len) {
            Some(src) => {
                for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                    *d = *s;
                }
            }
            None => dst.fill(0.0),
        }
        len
    }

    /// Forgets all samples (back to the zero prefix).
    pub(crate) fn clear(&mut self) {
        self.buf.fill(0.0);
        self.written = self.capacity() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_contiguous_across_the_wrap() {
        let mut m = Mirror::new(8);
        assert_eq!(m.capacity(), 8);
        // Before anything is written, windows reaching back read zeros.
        assert_eq!(m.window(0, 4), Some(&[0.0f32; 4][..]));
        for i in 0..21 {
            m.push(i as f32 + 1.0);
        }
        assert_eq!(m.len(), 21);
        for end in 13..=21i64 {
            for len in 0..=8usize {
                if end - (len as i64) < 13 {
                    continue;
                }
                let w = m.window(end, len).expect("window in range");
                let expect: Vec<f32> = ((end - len as i64)..end).map(|i| i as f32 + 1.0).collect();
                assert_eq!(w, &expect[..], "end={end} len={len}");
            }
        }
        // Overwritten, future and oversize windows are refused.
        assert!(m.window(21, 9).is_none());
        assert!(m.window(22, 1).is_none());
        assert!(m.window(12, 1).is_none());
        assert!(m.window(i64::MAX, 1).is_none());
        assert!(m.window(i64::MIN, 1).is_none());
    }

    #[test]
    fn reversed_copy_and_clear() {
        let mut m = Mirror::new(16);
        for i in 0..10 {
            m.push(i as f32);
        }
        let mut out = [0.0f32; 4];
        assert_eq!(m.copy_reversed(10, 4, &mut out), 4);
        assert_eq!(out, [9.0, 8.0, 7.0, 6.0]);
        // Out-of-range window → zeros.
        out = [1.0; 4];
        m.copy_reversed(100, 4, &mut out);
        assert_eq!(out, [0.0; 4]);
        m.clear();
        assert_eq!(m.len(), 0);
        assert_eq!(m.window(0, 16), Some(&[0.0f32; 16][..]));
    }
}
