//! Fast context-mixing primitives (integer only, identical on every CPU).
//!
//! * `Ctr`: a 16-bit bit probability with a fixed adaptation rate (cheapest to update, and
//!   on binary data as good as count-based rates).
//! * `Mix`: a small mixer with `K` inputs held in registers (no SIMD gather/scatter, which
//!   costs more than it saves at this size), one weight row selected per bit.

use crate::tables::{squash, stretch};


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

    /// Moves P(1) toward the coded bit by 1/2^rate.
    #[inline(always)]
    pub fn update(&mut self, y: u32, rate: u32) {
        let t = ((y << 16) - y) as i32;
        let p = self.p as i32;
        self.p = (p + ((t - p) >> rate)) as u16;
    }
}

/// Rates 1/(n + 1.5) for a counter seen n times.
const RATE: [i32; 16] = {
    let mut r = [0i32; 16];
    let mut n = 0;
    while n < 16 {
        r[n] = (65536 * 2 / (2 * n as i32 + 3)) as i32;
        n += 1;
    }
    r
};

impl Ctr {
    /// Adaptive-rate update: the low 4 bits count updates (up to `lim` <= 15), the rate
    /// starts fast and settles at 1/(lim + 1.5). Probability precision is 12 bits.
    #[inline(always)]
    pub fn update_n(&mut self, y: u32, lim: u32) {
        let n = (self.p & 15) as usize;
        let p = (self.p >> 4) as i32;
        let t = if y == 1 { 4095 } else { 0 };
        let p = p + (((t - p) * RATE[n]) >> 16);
        let n = (n + 1).min(lim as usize);
        self.p = ((p as u16) << 4) | n as u16;
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

/// Mixer with 32-bit arithmetic over `K` inputs (pad unused inputs with 0) so the dot
/// product and the update vectorize; weights 16384 = 1.0, kept within +-8 so a dot product
/// of 8 inputs of |x| <= 2047 cannot overflow.
#[derive(Clone)]
pub struct MixV<const K: usize> {
    w: Vec<[i32; K]>,
    x: [i32; K],
    row: usize,
    pr: i32,
    lr: i32,
}

const WMAX: i32 = 130_000;

impl<const K: usize> MixV<K> {
    pub fn new(rows: usize, lr: i32, inputs: usize) -> Self {
        let mut init = [0i32; K];
        for v in init.iter_mut().take(inputs) {
            *v = (16384 / inputs as i32) * 2;
        }
        MixV { w: vec![init; rows], x: [0; K], row: 0, pr: 2048, lr }
    }

    #[inline(always)]
    fn dot(x: &[i32; K], w: &[i32; K]) -> i32 {
        let mut s = 0i32;
        for i in 0..K {
            s = s.wrapping_add(x[i].wrapping_mul(w[i]));
        }
        s
    }

    /// P(1) as a 12-bit probability; remembers the inputs for `update`.
    #[inline(always)]
    pub fn predict12(&mut self, x: [i32; K], row: usize) -> i32 {
        self.x = x;
        self.row = row;
        self.pr = squash(Self::dot(&x, &self.w[row]) >> 14);
        self.pr
    }

    /// Same prediction without remembering anything.
    #[inline(always)]
    pub fn peek12(&self, x: &[i32; K], row: usize) -> i32 {
        squash(Self::dot(x, &self.w[row]) >> 14)
    }

    #[inline(always)]
    pub fn update(&mut self, y: u32) {
        let err = (((y as i32) << 12) - self.pr) * self.lr;
        let w = &mut self.w[self.row];
        for i in 0..K {
            w[i] = (w[i] + ((self.x[i] * err) >> 15)).clamp(-WMAX, WMAX);
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
