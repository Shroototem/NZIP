//! ncore: the GPU entropy core (Phase 2) and its bit-exact CPU reference.
//!
//! Coding model: the input is split into `lanes` (contiguous segments) that are coded
//! independently and in parallel: one GPU warp per lane, or one CPU task per lane. For
//! every byte, a per-lane model produces a 256-way distribution, which is quantised to
//! integer frequencies summing exactly to 2^16 ([`quantize`]) and coded with rANS.
//! This is the same interface the neural tier will use (network logits -> quantised
//! frequencies), so the core is reusable as is.
//!
//! **Everything is integer arithmetic.** The CPU reference ([`cpu`]) is the
//! specification; the CUDA kernels must produce byte-identical streams, and either side
//! can decode what the other encoded.

pub mod cpu;
pub mod gpu;
pub mod model;
pub mod rans;
pub mod stream;

/// Probability precision: frequencies sum to 2^SCALE_BITS.
pub const SCALE_BITS: u32 = 16;
pub const TOTAL: u32 = 1 << SCALE_BITS;
/// Symbols per rANS frame. Each frame is encoded backwards from a buffer of its
/// (cum, freq) pairs and flushed separately, which bounds encoder scratch memory.
pub const FRAME: usize = 1 << 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Corrupt(&'static str),
    Gpu(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Corrupt(s) => write!(f, "corrupt ncore stream: {s}"),
            Error::Gpu(s) => write!(f, "GPU error: {s}"),
        }
    }
}
impl std::error::Error for Error {}

/// Quantise non-negative weights to frequencies >= 1 that sum exactly to [`TOTAL`].
///
/// `scale = floor((TOTAL - 256) * 2^32 / sum(r))` (one division per call), then
/// `f[s] = 1 + floor(r[s] * scale / 2^32)`, which never exceeds the exact share, so the
/// remainder `TOTAL - sum(f)` is >= 0; it goes to the symbol with the largest weight
/// (lowest index on ties). All-zero weights give a uniform distribution. Requires
/// `sum(r) < 2^32`. The GPU implements exactly this, so results match bit for bit.
pub fn quantize(r: &[u32; 256], f: &mut [u32; 256]) {
    let sum: u64 = r.iter().map(|&v| v as u64).sum();
    if sum == 0 {
        f.fill(TOTAL / 256);
        return;
    }
    debug_assert!(sum < 1 << 32);
    let scale = (((TOTAL - 256) as u64) << 32) / sum; // r[s] <= sum keeps r*scale < 2^48
    let mut tot = 0u32;
    let mut best = 0usize;
    for s in 0..256 {
        f[s] = 1 + ((r[s] as u64 * scale) >> 32) as u32;
        tot += f[s];
        if r[s] > r[best] {
            best = s;
        }
    }
    f[best] += TOTAL - tot;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_sums_exactly() {
        let mut f = [0u32; 256];
        for seed in 0..200u32 {
            let mut r = [0u32; 256];
            let mut x = seed.wrapping_mul(2654435761) | 1;
            for v in r.iter_mut() {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                *v = if x % 3 == 0 { 0 } else { x % (1 << (seed % 24 + 1)) };
            }
            quantize(&r, &mut f);
            assert_eq!(f.iter().sum::<u32>(), TOTAL);
            assert!(f.iter().all(|&v| v >= 1));
        }
        quantize(&[0; 256], &mut f);
        assert!(f.iter().all(|&v| v == 256));
    }
}
