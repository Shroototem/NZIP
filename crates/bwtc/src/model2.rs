//! Model v2: rank-flag decomposition for BWT output (GPU-friendly, integer-only).
//!
//! BWT output is dominated by runs and by recurrences of a few recent symbols, which is
//! what bsc's QLFC exploits. Each byte is coded as a short chain of binary decisions:
//!   is it the most recent distinct byte (rank 0)?  -> 1 decision for most bytes
//!   else rank 1? else rank 2? else escape -> full byte via an order-1 bit model.
//! Rank flags use contexts built from the current run length, the previous byte's rank
//! and the rank-0 byte itself; each is a pair of adaptive counters mixed by a tiny
//! logistic mixer, then refined by an APM. Fixed instruction sequence per decision, so
//! GPU lanes stay in lockstep apart from the (rare) escape path.

use crate::model::Tables;

const K: usize = 3; // MTF ranks modelled with flags

#[derive(Clone, Copy, Debug)]
pub struct Config2 {
    pub fast: u32,
    pub slow: u32,
    pub lr: i32,
}

impl Default for Config2 {
    fn default() -> Self {
        Self { fast: 4, slow: 7, lr: 6 }
    }
}

#[inline]
fn upd(v: &mut u16, bit: u32, rate: u32) {
    if bit == 1 {
        *v += ((65535 - *v as u32) >> rate) as u16;
    } else {
        *v -= (*v as u32 >> rate) as u16;
    }
}

/// One binary decision with 3 contexts (each: fast+slow counter), mixer, APM.
struct Bin {
    c: [Vec<[u16; 2]>; 3],
    w: Vec<[i32; 7]>,
    apm: Vec<u16>,
}

impl Bin {
    fn new(sizes: [usize; 3], wsets: usize, apm_ctx: usize, t: &Tables) -> Self {
        let mut apm = vec![0u16; apm_ctx * 33];
        for c in 0..apm_ctx {
            for j in 0..33 {
                apm[c * 33 + j] = (t.sq((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        Self {
            c: [vec![[32768; 2]; sizes[0]], vec![[32768; 2]; sizes[1]], vec![[32768; 2]; sizes[2]]],
            w: vec![[1 << 14, 1 << 14, 1 << 14, 1 << 14, 1 << 14, 1 << 14, 0]; wsets],
            apm,
        }
    }

    /// Returns p(bit=1) in 12 bits and updates with `bit`.
    #[inline]
    fn code(&mut self, t: &Tables, ctx: [usize; 3], wset: usize, actx: usize, bit: Option<u32>, cfg: &Config2, decide: &mut dyn FnMut(u32, Option<u32>) -> u32) -> u32 {
        let mut x = [0i32; 7];
        for k in 0..3 {
            let v = self.c[k][ctx[k]];
            x[2 * k] = t.st((v[0] >> 4) as u32);
            x[2 * k + 1] = t.st((v[1] >> 4) as u32);
        }
        x[6] = 256;
        let w = &self.w[wset];
        let dot: i64 = (0..7).map(|k| x[k] as i64 * w[k] as i64).sum();
        let pm = t.sq((dot >> 16) as i32);
        let s = t.st(pm) + 2048;
        let lo = (s >> 7) as usize;
        let wlo = (s & 127) as u32;
        let base = actx * 33 + lo;
        let pa = (self.apm[base] as u32 * (128 - wlo) + self.apm[base + 1] as u32 * wlo) >> 11;
        let p = ((pm + 3 * pa) >> 2).clamp(1, 4095);
        let b = match bit {
            Some(b) => {
                decide(p, Some(b));
                b
            }
            None => decide(p, None),
        };
        for k in 0..3 {
            let v = &mut self.c[k][ctx[k]];
            upd(&mut v[0], b, cfg.fast);
            upd(&mut v[1], b, cfg.slow);
        }
        let err = ((b << 12) as i32 - pm as i32) * cfg.lr;
        let w = &mut self.w[wset];
        for k in 0..7 {
            w[k] += (x[k] * err) >> 10;
        }
        let target = if b == 1 { 65535i32 } else { 0 };
        let a0 = &mut self.apm[base];
        *a0 = (*a0 as i32 + (((target - *a0 as i32) * (128 - wlo as i32)) >> 13)) as u16;
        let a1 = &mut self.apm[base + 1];
        *a1 = (*a1 as i32 + (((target - *a1 as i32) * wlo as i32) >> 13)) as u16;
        b
    }
}

pub struct Lane2 {
    flags: [Bin; K],
    lit: Bin, // escape byte: order-1 bit model, ctx = (c1, node), (node), (r0, node)
    mtf: [u8; 256],
    run: u32,       // length of current rank-0 run
    prev_rank: u32, // rank of previous byte (K = escape)
    cfg: Config2,
}

fn runq(r: u32) -> usize {
    // 0,1,2,3,4-5,6-7,8-11,12-15,... capped at 15
    if r < 4 { r as usize } else { (2 * (31 - r.leading_zeros()) as usize + ((r >> (30 - r.leading_zeros())) & 1) as usize).min(15) }
}

impl Lane2 {
    pub fn new(cfg: Config2, t: &Tables) -> Self {
        let mut mtf = [0u8; 256];
        for (i, m) in mtf.iter_mut().enumerate() {
            *m = i as u8;
        }
        let mk = || Bin::new([16 * (K + 1), 16 * 256, 256 * (K + 1)], 16, 16 * (K + 1), t);
        Self {
            flags: [mk(), mk(), mk()],
            lit: Bin::new([256 * 256, 256, 256 * 256], 8, 256, t),
            mtf,
            run: 0,
            prev_rank: 0,
            cfg,
        }
    }

    /// Code one byte. `decide(p12_of_one, bit)` is called for every binary decision: when
    /// encoding (`byte` is Some) it gets `Some(bit)`; when decoding it gets `None` and must
    /// return the decoded bit. Returns the byte.
    pub fn code(&mut self, t: &Tables, byte: Option<u8>, decide: &mut dyn FnMut(u32, Option<u32>) -> u32) -> u8 {
        let rank = byte.map(|b| self.mtf.iter().position(|&m| m == b).unwrap());
        let rq = runq(self.run);
        let pr = self.prev_rank as usize;
        let r0 = self.mtf[0] as usize;
        let cfg = self.cfg;
        let mut got: Option<usize> = None;
        for k in 0..K {
            let ctx = [rq * (K + 1) + pr, rq * 256 + r0, r0 * (K + 1) + pr];
            let bit = rank.map(|r| (r == k) as u32);
            let b = self.flags[k].code(t, ctx, rq, rq * (K + 1) + pr, bit, &cfg, decide);
            if b == 1 {
                got = Some(k);
                break;
            }
        }
        let (value, rank_v) = match got {
            Some(k) => (self.mtf[k], k),
            None => {
                // escape: code the byte with an order-1 bit model (ranks < K can't occur)
                let c1 = r0;
                let mut node = 1usize;
                for i in (0..8).rev() {
                    let bit = byte.map(|b| ((b >> i) & 1) as u32);
                    let ctx = [c1 * 256 + node, node, (self.mtf[1] as usize) * 256 + node];
                    let b = self.lit.code(t, ctx, 7 - i, c1, bit, &cfg, decide);
                    node = (node << 1) | b as usize;
                }
                let v = (node & 0xFF) as u8;
                (v, self.mtf.iter().position(|&m| m == v).unwrap())
            }
        };
        // move to front
        self.mtf.copy_within(0..rank_v, 1);
        self.mtf[0] = value;
        self.run = if rank_v == 0 { self.run + 1 } else { 0 };
        self.prev_rank = rank_v.min(K) as u32;
        value
    }
}
