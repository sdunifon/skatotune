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

/// Forward + inverse setups and split-complex scratch for one length.
pub(super) struct Plan {
    n: usize,
    fwd: NonNull<SetupStruct>,
    inv: NonNull<SetupStruct>,
    /// Spectrum / signal A in split form (n/2 each).
    a_re: Box<[f32]>,
    a_im: Box<[f32]>,
    /// Spectrum B in split form (n/2 each).
    b_re: Box<[f32]>,
    b_im: Box<[f32]>,
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
    /// Creates setups for real length `n` (power of two ≥ 2). Returns `None`
    /// if vDSP has no implementation for that length (callers fall back to
    /// realfft).
    pub(super) fn new(n: usize) -> Option<Self> {
        if n < 2 || !n.is_power_of_two() {
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
        let z = || vec![0.0f32; h].into_boxed_slice();
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
        // (DC in re, Nyquist in im) that multiply independently.
        if let (Some(ar), Some(ai), Some(&br), Some(&bi)) =
            (self.a_re.first_mut(), self.a_im.first_mut(), self.b_re.first(), self.b_im.first())
        {
            *ar *= br;
            *ai *= bi;
        }
        let rest = self.a_re.iter_mut().zip(self.a_im.iter_mut()).zip(self.b_re.iter().zip(self.b_im.iter())).skip(1);
        for ((ar, ai), (&br, &bi)) in rest {
            let (xr, xi) = (*ar, *ai);
            *ar = xr * br + xi * bi;
            *ai = xr * bi - xi * br;
        }
        // Forward gives 2·DFT for each operand (4× in the product), the
        // inverse a further N×: undo both.
        let scale = 1.0 / (4.0 * self.n as f32);
        self.inverse_a_to(out, scale);
    }

    /// `out[τ] = Σ x[j]·x[j+τ]`.
    pub(super) fn autocorrelate(&mut self, x: &[f32], out: &mut [f32]) {
        self.forward_a(x);
        if let (Some(ar), Some(ai)) = (self.a_re.first_mut(), self.a_im.first_mut()) {
            *ar *= *ar;
            *ai *= *ai;
        }
        for (ar, ai) in self.a_re.iter_mut().zip(self.a_im.iter_mut()).skip(1) {
            *ar = *ar * *ar + *ai * *ai;
            *ai = 0.0;
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
        let bins = out.iter_mut().take(h).skip(1);
        for ((o, &r), &i) in bins.zip(self.a_re.iter().skip(1)).zip(self.a_im.iter().skip(1)) {
            *o = 0.25 * (r * r + i * i);
        }
    }
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
