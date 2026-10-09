//! Minimal hand-written FFI to Apple Accelerate's vDSP real DFT
//! (`vDSP_DFT_zrop_*`), plus the packed-spectrum maths the [`super::Correlator`]
//! needs.
//!
//! vDSP conventions (from `vDSP.h`), for a real transform of length N:
//! * **Input split:** the real signal is viewed as N/2 complex numbers, even
//!   samples in `realp`, odd samples in `imagp`.
//! * **Packed output:** `realp[0] = H[0]` (DC), `imagp[0] = H[N/2]`
//!   (Nyquist; both are real), `realp[k] + i·imagp[k] = H[k]` for 0 < k < N/2.
//! * **Scale:** the forward transform returns **2×** the mathematical DFT; the
//!   inverse is unscaled (`Σ_k H[k]·e^{+2πijk/N}`), producing the time signal
//!   in the same even/odd split. A forward→inverse round trip is thus 2N×.
//!
//! The setups are created in [`Plan::new`] and destroyed in `Drop`; execution
//! (`vDSP_DFT_Execute`) does not allocate.
//!
//! **Alignment and determinism.** vDSP picks differently rounded code paths
//! depending on the alignment of the split buffers: with the imaginary half
//! at 32 mod 64 bytes the correlation differed in the last bits from a fully
//! aligned plan (measured). Plain `Box<[f32]>` buffers land wherever the
//! allocator puts them, so two plans fed the same input could disagree,
//! breaking DESIGN §2.1 rule 6 across engine instances and sessions. Every
//! split buffer is therefore an [`Aligned`] block starting on a 128-byte
//! boundary (the M4 cache line), which makes the result independent of the
//! instance.

use core::ptr::NonNull;

/// Opaque `struct vDSP_DFT_SetupStruct`.
#[repr(C)]
struct SetupStruct {
    _private: [u8; 0],
}

/// `vDSP_DFT_Direction` (`typedef vDSP_ENUM(int, …)`).
const FORWARD: i32 = 1;
const INVERSE: i32 = -1;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_DFT_zrop_CreateSetup(previous: *mut SetupStruct, length: usize, direction: i32) -> *mut SetupStruct;
    fn vDSP_DFT_Execute(setup: *const SetupStruct, ir: *const f32, ii: *const f32, or: *mut f32, oi: *mut f32);
    fn vDSP_DFT_DestroySetup(setup: *mut SetupStruct);
}

/// Alignment of every split buffer, bytes (one M4 cache line).
const ALIGN: usize = 128;

/// One 128-byte line of floats; `Aligned` is built from these so its first
/// element always sits on an [`ALIGN`] boundary.
#[derive(Clone, Copy)]
#[repr(C, align(128))]
struct Line([f32; ALIGN / 4]);

// The `Deref` impls below rely on `Line` being exactly ALIGN bytes, padding
// free, at ALIGN alignment (the attribute cannot name the constant).
const _: () = assert!(core::mem::size_of::<Line>() == ALIGN && core::mem::align_of::<Line>() == ALIGN);

/// Zero-initialised heap buffer of `len` floats starting on an [`ALIGN`]
/// boundary, used as `&[f32]` / `&mut [f32]` through `Deref`.
struct Aligned {
    lines: Box<[Line]>,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let lines = vec![Line([0.0; ALIGN / 4]); len.div_ceil(ALIGN / 4)].into_boxed_slice();
        Self { lines, len }
    }
}

impl core::ops::Deref for Aligned {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: `Line` is `repr(C)` around `[f32; 32]` with size 128 = its
        // alignment, so `lines` is `lines.len() · 32` contiguous, initialised
        // f32 values; `len <= lines.len() · 32` by construction.
        unsafe { core::slice::from_raw_parts(self.lines.as_ptr().cast::<f32>(), self.len) }
    }
}

impl core::ops::DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as in `deref`; `&mut self` makes the borrow unique.
        unsafe { core::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast::<f32>(), self.len) }
    }
}

/// Forward + inverse setups and split-complex scratch for one length.
pub(super) struct Plan {
    n: usize,
    fwd: NonNull<SetupStruct>,
    inv: NonNull<SetupStruct>,
    /// Spectrum / signal A in split form (n/2 each).
    a_re: Aligned,
    a_im: Aligned,
    /// Spectrum B in split form (n/2 each).
    b_re: Aligned,
    b_im: Aligned,
}

// SAFETY: the setup objects are immutable after creation; vDSP documents that
// a setup may be used by `vDSP_DFT_Execute` from any thread, and this type
// only executes through `&mut self`, so no two executions on one plan overlap.
// The owned buffers are ordinary heap memory.
unsafe impl Send for Plan {}
// SAFETY: `&Plan` exposes no way to execute or mutate (all methods that touch
// the setups or buffers take `&mut self`).
unsafe impl Sync for Plan {}

impl Plan {
    /// Creates setups for real length `n`: a power of two ≥ 2, or f·2^k with
    /// f ∈ {3, 5, 15} and k ≥ 4 (the lengths `vDSP.h` documents for
    /// `vDSP_DFT_zrop_CreateSetup`). Returns `None` for any other length or if
    /// vDSP declines it (callers fall back to realfft).
    pub(super) fn new(n: usize) -> Option<Self> {
        if !documented_len(n) {
            return None;
        }
        // SAFETY: plain FFI calls; a null `previous` asks for fresh memory.
        // The inverse shares twiddle data with the forward setup.
        let fwd = NonNull::new(unsafe { vDSP_DFT_zrop_CreateSetup(core::ptr::null_mut(), n, FORWARD) })?;
        // SAFETY: as above; `fwd` is a valid setup.
        let inv = match NonNull::new(unsafe { vDSP_DFT_zrop_CreateSetup(fwd.as_ptr(), n, INVERSE) }) {
            Some(inv) => inv,
            None => {
                // SAFETY: `fwd` was created above and is not used afterwards.
                unsafe { vDSP_DFT_DestroySetup(fwd.as_ptr()) };
                return None;
            }
        };
        let h = n / 2;
        let z = || Aligned::new(h);
        Some(Self { n, fwd, inv, a_re: z(), a_im: z(), b_re: z(), b_im: z() })
    }

    /// Splits `x` (zero-padded to n) into even/odd halves.
    fn load(x: &[f32], re: &mut [f32], im: &mut [f32]) {
        let (pairs, odd) = x.as_chunks::<2>();
        let used = pairs.len().min(re.len()).min(im.len());
        for ((r, i), p) in re.iter_mut().zip(im.iter_mut()).zip(pairs) {
            *r = p[0];
            *i = p[1];
        }
        let mut next = used;
        if let (Some(&last), Some(r), Some(i)) = (odd.first(), re.get_mut(used), im.get_mut(used)) {
            *r = last;
            *i = 0.0;
            next += 1;
        }
        if let Some(r) = re.get_mut(next..) {
            r.fill(0.0);
        }
        if let Some(i) = im.get_mut(next..) {
            i.fill(0.0);
        }
    }

    /// In-place forward transform of the split buffers `re`/`im`.
    fn forward(setup: NonNull<SetupStruct>, re: &mut [f32], im: &mut [f32]) {
        // SAFETY: `re`/`im` have length n/2 (allocated in `new`), as the setup
        // requires; vDSP allows in-place operation (Or == Ir, Oi == Ii).
        unsafe {
            let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
            vDSP_DFT_Execute(setup.as_ptr(), pr, pi, pr, pi);
        }
    }

    /// Forward transforms `x` into the A buffers.
    fn forward_a(&mut self, x: &[f32]) {
        let x = x.get(..self.n).unwrap_or(x);
        Self::load(x, &mut self.a_re, &mut self.a_im);
        Self::forward(self.fwd, &mut self.a_re, &mut self.a_im);
    }

    /// Forward transforms `x` into the B buffers.
    fn forward_b(&mut self, x: &[f32]) {
        let x = x.get(..self.n).unwrap_or(x);
        Self::load(x, &mut self.b_re, &mut self.b_im);
        Self::forward(self.fwd, &mut self.b_re, &mut self.b_im);
    }

    /// Inverse-transforms the A buffers in place and writes
    /// `out[τ] = scale · signal[τ]`.
    fn inverse_a_to(&mut self, out: &mut [f32], scale: f32) {
        // SAFETY: as in `forward`, with the inverse setup.
        unsafe {
            let (pr, pi) = (self.a_re.as_mut_ptr(), self.a_im.as_mut_ptr());
            vDSP_DFT_Execute(self.inv.as_ptr(), pr, pi, pr, pi);
        }
        // Time sample 2m is in re[m], 2m + 1 in im[m].
        let (pairs, odd) = out.as_chunks_mut::<2>();
        for ((p, &r), &i) in pairs.iter_mut().zip(self.a_re.iter()).zip(self.a_im.iter()) {
            p[0] = r * scale;
            p[1] = i * scale;
        }
        if let (Some(o), Some(&r)) = (odd.first_mut(), self.a_re.get(pairs.len())) {
            *o = r * scale;
        }
    }

    /// `out[τ] = Σ a[j]·b[j+τ]` (circular in n; see `Correlator`).
    pub(super) fn cross_correlate(&mut self, a: &[f32], b: &[f32], out: &mut [f32]) {
        self.forward_a(a);
        self.forward_b(b);
        // conj(A)·B on the packed spectrum. Element 0 holds two real values
        // (DC in re, Nyquist in im) that multiply independently; elements
        // 1.. are complex bins. The bins run over flat sub-slices: the
        // earlier `.skip(1)` on a nested zip did not vectorise and took
        // 1.7-2.3 µs at n = 1536-2048 (12-16× slower), i.e. a third of the
        // whole correlation.
        if let (Some((ar0, ar)), Some((ai0, ai)), Some((&br0, br)), Some((&bi0, bi))) =
            (self.a_re.split_first_mut(), self.a_im.split_first_mut(), self.b_re.split_first(), self.b_im.split_first())
        {
            *ar0 *= br0;
            *ai0 *= bi0;
            for (((xr, xi), &yr), &yi) in ar.iter_mut().zip(ai.iter_mut()).zip(br).zip(bi) {
                let (r, i) = (*xr, *xi);
                *xr = r * yr + i * yi;
                *xi = r * yi - i * yr;
            }
        }
        // Forward gives 2·DFT for each operand (4× in the product), the
        // inverse a further N×: undo both.
        let scale = 1.0 / (4.0 * self.n as f32);
        self.inverse_a_to(out, scale);
    }

    /// `out[τ] = Σ x[j]·x[j+τ]`.
    pub(super) fn autocorrelate(&mut self, x: &[f32], out: &mut [f32]) {
        self.forward_a(x);
        // |A|² on the packed spectrum (flat sub-slices, as in
        // `cross_correlate`).
        if let (Some((ar0, ar)), Some((ai0, ai))) = (self.a_re.split_first_mut(), self.a_im.split_first_mut()) {
            *ar0 *= *ar0;
            *ai0 *= *ai0;
            for (r, i) in ar.iter_mut().zip(ai.iter_mut()) {
                *r = *r * *r + *i * *i;
                *i = 0.0;
            }
        }
        let scale = 1.0 / (4.0 * self.n as f32);
        self.inverse_a_to(out, scale);
    }

    /// `out[k] = |X[k]|²` for k in 0..=n/2 (true DFT scale).
    pub(super) fn power_spectrum(&mut self, x: &[f32], out: &mut [f32]) {
        self.forward_a(x);
        let h = self.n / 2;
        // |2X|² / 4.
        if let (Some(o), Some(&r)) = (out.first_mut(), self.a_re.first()) {
            *o = 0.25 * r * r;
        }
        if let (Some(o), Some(&i)) = (out.get_mut(h), self.a_im.first()) {
            *o = 0.25 * i * i;
        }
        // Bins 1..h (flat sub-slices; `out` may be shorter than n/2 + 1).
        let bins = out.get_mut(1..h.min(out.len())).unwrap_or_default();
        let (re, im) = (self.a_re.get(1..).unwrap_or_default(), self.a_im.get(1..).unwrap_or_default());
        for ((o, &r), &i) in bins.iter_mut().zip(re).zip(im) {
            *o = 0.25 * (r * r + i * i);
        }
    }
}

/// Lengths vDSP documents for the real DFT: 2^k (k ≥ 1) or f·2^k with
/// f ∈ {3, 5, 15}, k ≥ 4. (It also accepts a few undocumented ones, such as
/// 120 or 180, which are deliberately not used.)
pub(super) fn documented_len(n: usize) -> bool {
    if n >= 2 && n.is_power_of_two() {
        return true;
    }
    [3usize, 5, 15].into_iter().any(|f| {
        let m = n / f;
        n.is_multiple_of(f) && m >= 16 && m.is_power_of_two()
    })
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: both setups were created in `new`, are destroyed exactly
        // once, and no execution can be in flight (we hold `&mut self`).
        unsafe {
            vDSP_DFT_DestroySetup(self.inv.as_ptr());
            vDSP_DFT_DestroySetup(self.fwd.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the alignment-dependent rounding described in the
    /// module docs: every split buffer of every plan starts on an `ALIGN`
    /// boundary, wherever the allocator places it.
    #[test]
    fn split_buffers_are_line_aligned() {
        let mut spacers = Vec::new();
        let mut plans = Vec::new();
        for (k, n) in [64usize, 768, 1024, 1536, 2048, 3072, 4096].into_iter().cycle().take(140).enumerate() {
            spacers.push(vec![0u8; 1 + (k * 37) % 4093]);
            let p = Plan::new(n).expect("documented length");
            for b in [&p.a_re, &p.a_im, &p.b_re, &p.b_im] {
                assert_eq!(b.as_ptr() as usize % ALIGN, 0, "n={n} instance {k}");
                assert_eq!(b.len(), n / 2);
            }
            plans.push(p);
        }
        // Odd lengths round up to whole lines but expose exactly `len`.
        let a = Aligned::new(33);
        assert_eq!((a.len(), a.lines.len()), (33, 2));
        assert!(a.iter().all(|&v| v == 0.0));
    }
}
