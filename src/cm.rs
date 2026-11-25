//! Fast context-mixing primitives (integer only, identical on every CPU).
//!
//! * `Ctr`: a 16-bit bit probability with a fixed adaptation rate (cheapest to update, and
//!   on binary data as good as count-based rates).
//! * `Mix`: a small mixer with `K` inputs held in registers (no SIMD gather/scatter, which
//!   costs more than it saves at this size), one weight row selected per bit.

use crate::tables::{squash, stretch};


/// First step of a young counter: 1/2^YOUNG_START.
pub const YOUNG_START: u32 = 2;

#[derive(Clone, Copy)]
pub struct Ctr {
    p: u16,
}

impl Ctr {
    pub const INIT: Ctr = Ctr { p: 32768 };

    /// Logit (scaled by 256) of P(1).
    #[inline(always)]
    pub fn st(self) -> i32 {
        stretch(self.p as i32 >> 4)
    }

    #[inline(always)]
    pub fn p(self) -> u32 {
        self.p as u32
    }

    /// Like `update`, but a young counter learns fast: the low 3 bits count the bits seen,
    /// and the step is 1/2^(count + START) until it reaches 1/2^rate. Models that restart
    /// often (every LZ sub-block) lose much less to the cold start.
    #[inline(always)]
    pub fn update_young(&mut self, y: u32, rate: u32) {
        let n = (self.p & 7) as u32;
        let r = (n + YOUNG_START).min(rate);
        let t = ((y << 16) - y) as i32;
        let p = (self.p & !7) as i32;
        let q = (p + ((t - p) >> r)) as u16;
        self.p = (q & !7) | (n + 1).min(7) as u16;
    }

    /// Moves P(1) toward the coded bit by 1/2^rate.
    #[inline(always)]
    pub fn update(&mut self, y: u32, rate: u32) {
        let t = ((y << 16) - y) as i32;
        let p = self.p as i32;
        self.p = (p + ((t - p) >> rate)) as u16;
    }
}

/// Inputs and weights live in 8 lanes (unused lanes stay 0), so the dot product and the
/// update compile to a few AVX2 instructions whatever `K` is; results equal K-lane math.
#[derive(Clone)]
pub struct Mix<const K: usize> {
    w: Vec<[i32; 8]>,
    x: [i32; 8],
    row: usize,
    pr: i32,
    lr: i32,
}

impl<const K: usize> Mix<K> {
    pub fn new(rows: usize, lr: i32) -> Self {
        assert!(K <= 8);
        let mut init = [0i32; 8];
        for v in init.iter_mut().take(K) {
            *v = (65536 / K as i32) * 2;
        }
        Mix { w: vec![init; rows], x: [0; 8], row: 0, pr: 2048, lr }
    }

    /// Last prediction as a 12-bit probability.
    #[inline(always)]
    pub fn pr12(&self) -> i32 {
        self.pr
    }

    /// P(1) scaled by 65536 (clamped away from 0 and 1).
    #[inline(always)]
    pub fn predict(&mut self, x: [i32; K], row: usize) -> u32 {
        let mut xs = [0i32; 8];
        xs[..K].copy_from_slice(&x);
        self.x = xs;
        self.row = row;
        let w = &self.w[row];
        let mut s = 0i64;
        for i in 0..8 {
            s += xs[i] as i64 * w[i] as i64;
        }
        self.pr = squash((s >> 16) as i32);
        ((self.pr << 4) as u32).clamp(48, 65535 - 48)
    }

    #[inline(always)]
    pub fn update(&mut self, y: u32) {
        let err = (((y as i32) << 12) - self.pr) * self.lr;
        let w = &mut self.w[self.row];
        for i in 0..8 {
            w[i] = (w[i] + ((self.x[i] * err) >> 13)).clamp(-(1 << 22), 1 << 22);
        }
    }
}

/// 16 counters for the 4 bits of one nibble (tree nodes 1..15; slot 0 unused), in one
/// half cache line: a byte touches two groups per table instead of up to 8 lines.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct G(pub [Ctr; 16]);

impl G {
    pub const INIT: G = G([Ctr::INIT; 16]);
}

/// Groups per context for direct tables: one for the first nibble, 16 for the second.
pub const NG: usize = 17;

#[inline(always)]
pub fn hg(v: u32, bits: u32) -> usize {
    (v.wrapping_mul(0x9E37_79B1) >> (32 - bits)) as usize
}
