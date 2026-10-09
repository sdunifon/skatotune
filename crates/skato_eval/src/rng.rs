//! Deterministic pseudo-random numbers for the synthetic signals.
//!
//! Every generated signal must be bit-reproducible from its seed on every
//! platform, so the harness carries its own tiny generator instead of pulling
//! in `rand` (whose algorithms may change between versions): xoshiro256** for
//! the stream, seeded through SplitMix64, plus Box-Muller Gaussians.

/// xoshiro256** generator (Blackman & Vigna), seeded with SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    /// Second Box-Muller output, returned by the next `gaussian()` call.
    spare: Option<f64>,
}

impl Rng {
    /// Generator for `seed`. Equal seeds give equal streams on every platform.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            // SplitMix64 step: decorrelates nearby seeds before they reach xoshiro.
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let s = [next(), next(), next(), next()];
        Self { s, spare: None }
    }

    /// Derive an independent child stream (e.g. one per signal component), so
    /// adding draws to one component never shifts the others.
    pub fn fork(&mut self, salt: u64) -> Rng {
        Rng::new(self.next_u64() ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) with 53 bits of resolution.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal deviate (Box-Muller, both outputs used).
    pub fn gaussian(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // u1 in (0, 1] so ln(u1) is finite.
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let mut c = Rng::new(43);
        let va: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let vb: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        let vc: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
        assert_eq!(va, vb);
        assert_ne!(va, vc);
    }

    #[test]
    fn uniform_and_gaussian_moments() {
        let mut r = Rng::new(7);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            s1 += u;
        }
        assert!((s1 / n as f64 - 0.5).abs() < 0.005);
        s1 = 0.0;
        for _ in 0..n {
            let g = r.gaussian();
            s1 += g;
            s2 += g * g;
        }
        let mean = s1 / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn forks_are_independent_of_parent_draws() {
        let mut p1 = Rng::new(1);
        let mut p2 = Rng::new(1);
        let mut f1 = p1.fork(5);
        let mut f2 = p2.fork(5);
        assert_eq!(f1.next_u64(), f2.next_u64());
        let mut f3 = p1.fork(6);
        assert_ne!(f3.next_u64(), f2.next_u64());
    }
}
