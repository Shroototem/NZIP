//! Fast context-mixing primitives (integer only, identical on every CPU).
//!
//! * `Ctr`: a bit probability whose adaptation rate starts fast and slows down as the
//!   context is seen more often (1/(n+1.5), down to 1/limit).
//! * `Mix`: a small mixer with `K` inputs held in registers (no SIMD gather/scatter, which
//!   costs more than it saves at this size), one weight row selected per bit.

use crate::tables::{squash, stretch};

/// 65536 / (n + 1.5)
const RATE: [u16; 1024] = {
    let mut t = [0u16; 1024];
    let mut n = 0;
    while n < 1024 {
        t[n] = (131072 / (2 * n as u32 + 3)) as u16;
        n += 1;
    }
    t
};

#[derive(Clone, Copy)]
pub struct Ctr {
    p: u16,
    n: u16,
}

impl Ctr {
    pub const INIT: Ctr = Ctr { p: 32768, n: 0 };

    /// Logit (scaled by 256) of P(1).
    #[inline(always)]
    pub fn st(self) -> i32 {
        stretch(self.p as i32 >> 4)
    }

    #[inline(always)]
    pub fn p(self) -> u32 {
        self.p as u32
    }

    #[inline(always)]
    pub fn update(&mut self, y: u32, limit: u16) {
        let t = ((y << 16) - y) as i32;
        let p = self.p as i32;
        self.p = (p + (((t - p) * RATE[self.n as usize] as i32) >> 16)) as u16;
        if self.n < limit {
            self.n += 1;
        }
    }
}

pub struct Mix<const K: usize> {
    w: Vec<[i32; K]>,
    x: [i32; K],
    row: usize,
    pr: i32,
    lr: i32,
}

impl<const K: usize> Mix<K> {
    pub fn new(rows: usize, lr: i32) -> Self {
        Mix { w: vec![[(65536 / K as i32) * 2; K]; rows], x: [0; K], row: 0, pr: 2048, lr }
    }

    /// P(1) scaled by 65536 (clamped away from 0 and 1).
    #[inline(always)]
    pub fn predict(&mut self, x: [i32; K], row: usize) -> u32 {
        self.x = x;
        self.row = row;
        let w = &self.w[row];
        let mut s = 0i64;
        for i in 0..K {
            s += x[i] as i64 * w[i] as i64;
        }
        self.pr = squash((s >> 16) as i32);
        ((self.pr << 4) as u32).clamp(48, 65535 - 48)
    }

    #[inline(always)]
    pub fn update(&mut self, y: u32) {
        let err = (((y as i32) << 12) - self.pr) * self.lr;
        let w = &mut self.w[self.row];
        for i in 0..K {
            w[i] = (w[i] + ((self.x[i] * err) >> 11)).clamp(-(1 << 22), 1 << 22);
        }
    }
}
