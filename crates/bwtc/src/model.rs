//! Bit-level context-mixing model for BWT output. All arithmetic is integer, so the CPU
//! and GPU implementations produce identical probabilities (required for decoding).
//!
//! Per lane (an independent segment of BWT output):
//! * order-0 and order-1 contexts (previous byte), indexed by the partial-byte node,
//!   each with a fast and a slow adaptive counter (16-bit probabilities);
//! * a small logistic mixer (weights selected by bit position);
//! * an APM/SSE stage on (previous byte, mixed probability).
//!
//! The order-1 counters may be initialised from a per-block static prior, removing most
//! of the cold-start cost of short segments.

pub const PBITS: u32 = 12; // probability precision handed to the entropy coder

/// stretch(p) = ln(p/(1-p)) scaled, for 12-bit p; squash is its inverse.
pub struct Tables {
    stretch: Vec<i16>,
    squash: Vec<u16>,
}

impl Tables {
    pub fn new() -> Self {
        // squash(d) for d in [-2047, 2047], output in [1, 4095]
        let squash: Vec<u16> = (-2047..=2047)
            .map(|d: i32| {
                let v = 4096.0 / (1.0 + (-(d as f64) / 256.0).exp());
                (v.round() as i32).clamp(1, 4095) as u16
            })
            .collect();
        // stretch = inverse of squash (monotone), built by scanning
        let mut stretch = vec![0i16; 4096];
        let mut pi = 0usize;
        for d in -2047..=2047i32 {
            let v = squash[(d + 2047) as usize] as usize;
            while pi <= v {
                stretch[pi] = d as i16;
                pi += 1;
            }
        }
        while pi < 4096 {
            stretch[pi] = 2047;
            pi += 1;
        }
        Self { stretch, squash }
    }
    #[inline]
    pub fn st(&self, p12: u32) -> i32 {
        self.stretch[p12 as usize] as i32
    }
    #[inline]
    pub fn sq(&self, d: i32) -> u32 {
        self.squash[(d.clamp(-2047, 2047) + 2047) as usize] as u32
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub fast_rate: u32,
    pub slow_rate: u32,
    pub mix_lr: i32,
    pub use_apm: bool,
    pub use_order0: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { fast_rate: 4, slow_rate: 7, mix_lr: 6, use_apm: true, use_order0: true }
    }
}

const NIN: usize = 5; // o0 fast, o0 slow, o1 fast, o1 slow, bias

/// Per-lane adaptive state.
pub struct Lane {
    o0: Vec<[u16; 2]>,      // [node] -> (fast, slow)
    o1: Vec<[u16; 2]>,      // [c1*256 + node]
    w: Vec<[i32; NIN]>,     // [bitpos] mixer weights, 16.16
    apm: Vec<u16>,          // [c1*33 + bucket]
    c1: u32,
    cfg: Config,
}

impl Lane {
    /// `prior`: optional order-1 initial probabilities, [c1*256 + node] (16-bit).
    pub fn new(cfg: Config, prior: Option<&[u16]>) -> Self {
        let o1 = match prior {
            Some(p) => p.iter().map(|&v| [v, v]).collect(),
            None => vec![[32768u16, 32768u16]; 256 * 256],
        };
        let mut apm = vec![0u16; 256 * 33];
        let t = Tables::new();
        for c in 0..256 {
            for j in 0..33 {
                apm[c * 33 + j] = (t.sq((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        let w0 = if cfg.use_order0 { 1 << 14 } else { 0 };
        Self {
            o0: vec![[32768, 32768]; 256],
            o1,
            w: vec![[w0, w0, 1 << 15, 1 << 15, 0]; 8],
            apm,
            c1: 0,
            cfg,
        }
    }

    /// Code (or measure) one byte: calls `bit(p12_of_one, bit)` for each of 8 bits MSB-first.
    #[inline]
    pub fn code_byte<F: FnMut(u32, u32)>(&mut self, t: &Tables, byte: u32, mut bit: F) {
        let mut node = 1u32;
        for i in (0..8).rev() {
            let b = (byte >> i) & 1;
            let p = self.predict_update(t, node, 7 - i, b);
            bit(p, b);
            node = (node << 1) | b;
        }
        self.c1 = byte;
    }

    /// Decode-side variant: `decide(p12)` returns the decoded bit.
    #[inline]
    pub fn decode_byte<F: FnMut(u32) -> u32>(&mut self, t: &Tables, mut decide: F) -> u32 {
        let mut node = 1u32;
        for bp in 0..8 {
            let p = self.predict(t, node, bp);
            let b = decide(p.0);
            self.update(t, node, bp, b, p);
            node = (node << 1) | b;
        }
        let byte = node & 0xFF;
        self.c1 = byte;
        byte
    }

    #[inline]
    fn predict_update(&mut self, t: &Tables, node: u32, bp: u32, b: u32) -> u32 {
        let p = self.predict(t, node, bp);
        self.update(t, node, bp, b, p);
        p.0
    }

    /// Returns (final p12, mixed p12, inputs, apm index, apm weight lo).
    #[inline]
    fn predict(&self, t: &Tables, node: u32, bp: u32) -> (u32, u32, [i32; NIN], usize, u32) {
        let a = self.o0[node as usize];
        let c = self.o1[(self.c1 * 256 + node) as usize];
        let x = [
            t.st((a[0] >> 4) as u32),
            t.st((a[1] >> 4) as u32),
            t.st((c[0] >> 4) as u32),
            t.st((c[1] >> 4) as u32),
            256,
        ];
        let w = &self.w[bp as usize];
        let mut dot: i64 = 0;
        for k in 0..NIN {
            dot += x[k] as i64 * w[k] as i64;
        }
        let pm = t.sq((dot >> 16) as i32);
        if !self.cfg.use_apm {
            return (pm, pm, x, 0, 0);
        }
        // APM: interpolate between two buckets of stretch(pm)
        let s = t.st(pm) + 2048; // 1..4095
        let lo = (s >> 7) as usize; // 0..31
        let wlo = (s & 127) as u32;
        let base = (self.c1 as usize) * 33 + lo;
        let pa = (self.apm[base] as u32 * (128 - wlo) + self.apm[base + 1] as u32 * wlo) >> 11; // 16->12 bit, /128
        let pf = ((pm + 3 * pa) >> 2).clamp(1, 4095);
        (pf, pm, x, base, wlo)
    }

    #[inline]
    fn update(&mut self, t: &Tables, node: u32, bp: u32, b: u32, p: (u32, u32, [i32; NIN], usize, u32)) {
        let (_, pm, x, base, wlo) = p;
        let (fr, sr) = (self.cfg.fast_rate, self.cfg.slow_rate);
        let upd = |v: &mut u16, rate: u32| {
            if b == 1 {
                *v += ((65535 - *v as u32) >> rate) as u16;
            } else {
                *v -= (*v as u32 >> rate) as u16;
            }
        };
        {
            let a = &mut self.o0[node as usize];
            upd(&mut a[0], fr);
            upd(&mut a[1], sr);
        }
        {
            let c = &mut self.o1[(self.c1 * 256 + node) as usize];
            upd(&mut c[0], fr);
            upd(&mut c[1], sr);
        }
        // mixer: w += lr * err * x
        let err = ((b << 12) as i32 - pm as i32) * self.cfg.mix_lr;
        let w = &mut self.w[bp as usize];
        for k in 0..NIN {
            w[k] += (x[k] * err) >> 10;
        }
        if self.cfg.use_apm {
            let target = if b == 1 { 65535u32 } else { 0 };
            let a0 = &mut self.apm[base];
            let d0 = ((target as i32 - *a0 as i32) * (128 - wlo as i32)) >> (7 + 6);
            *a0 = (*a0 as i32 + d0) as u16;
            let a1 = &mut self.apm[base + 1];
            let d1 = ((target as i32 - *a1 as i32) * wlo as i32) >> (7 + 6);
            *a1 = (*a1 as i32 + d1) as u16;
        }
        let _ = t;
    }
}

/// Build a per-block order-1 prior from byte-pair counts: P(bit=1 | c1, node), 16-bit,
/// computed from counts quantised to 8 bits per (c1, byte) so it can be transmitted cheaply.
pub fn prior_from_counts(q: &[u8]) -> Vec<u16> {
    // q[c1*256 + byte] = quantised log-count (0 = never seen)
    let mut prior = vec![32768u16; 256 * 256];
    for c1 in 0..256 {
        let mut cnt = [0u64; 256];
        for b in 0..256 {
            let v = q[c1 * 256 + b] as u32;
            cnt[b] = if v == 0 { 0 } else { dequant(v) };
        }
        // subtree sums over the binary tree of byte values
        let mut sum = [0u64; 512];
        for b in 0..256 {
            sum[256 + b] = cnt[b];
        }
        for n in (1..256).rev() {
            sum[n] = sum[2 * n] + sum[2 * n + 1];
        }
        for node in 1..256 {
            let (z, o) = (sum[2 * node], sum[2 * node + 1]);
            // Laplace-ish smoothing, clamp away from 0/1
            let p = ((o * 2 + 1) << 16) / ((z + o) * 2 + 2);
            prior[c1 * 256 + node] = p.clamp(1024, 64511) as u16;
        }
    }
    prior
}

/// 8-bit log-scale quantisation of a count (0 reserved for zero).
pub fn quant(c: u64) -> u8 {
    if c == 0 {
        return 0;
    }
    // q = 1 + round(16 * log2(c)), capped
    let l = 64 - c.leading_zeros() as u64 - 1; // floor log2
    let frac = if l >= 4 { (c >> (l - 4)) & 15 } else { (c << (4 - l)) & 15 };
    (1 + l * 16 + frac).min(255) as u8
}

pub fn dequant(q: u32) -> u64 {
    let v = q - 1;
    let (l, frac) = ((v / 16) as u64, (v % 16) as u64);
    if l >= 4 { (16 + frac) << (l - 4) } else { (16 + frac) >> (4 - l) }.max(1)
}
