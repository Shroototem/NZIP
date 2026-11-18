//! mixlab: how much do adaptive context models add to a pretrained neural model?
//!
//! Reads a file and the neural model's next-byte distributions (`train/dump_probs.py`:
//! 256 x u16 per byte, rows summing to 65536) and measures, in bits per byte:
//!   nn    the neural model alone (exact cost of its distribution),
//!   cm    a paq-style bitwise context-mixing model alone (orders 0-4, 6, match model),
//!   nn+cm both, mixed by an online-learned logistic mixer with an SSE stage.
//! Every model (and the mixer) restarts at each lane boundary, like the real coder.
//! Float arithmetic: this estimates achievable ratio; a codec needs the integer version.
//!
//! Usage: mixlab <lane-bytes> <file> [<file> ...]   (each file needs <file>.p16 beside it)

use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;

#[inline]
fn stretch(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}
#[inline]
fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn clampp(p: f32) -> f32 {
    p.clamp(1.0 / 4096.0, 4095.0 / 4096.0)
}

/// Adaptive bit probability with a count-based learning rate (fast at first, then slow).
#[derive(Clone, Copy)]
struct Counter {
    p: f32,
    n: u16,
}
impl Default for Counter {
    fn default() -> Self {
        Self { p: 0.5, n: 0 }
    }
}
impl Counter {
    #[inline]
    fn update(&mut self, bit: u32, limit: u16) {
        let r = 1.0 / (self.n as f32 + 1.5);
        self.p += (bit as f32 - self.p) * r;
        if self.n < limit {
            self.n += 1;
        }
    }
}

const HBITS: u32 = 22;
const ORDERS: [usize; 6] = [1, 2, 3, 4, 6, 8];

struct Mixer {
    w: Vec<f32>, // [ctx][n]
    n: usize,
    lr: f32,
}
impl Mixer {
    fn new(n: usize, ctxs: usize, init: f32) -> Self {
        Self { w: vec![init; n * ctxs], n, lr: 0.02 }
    }
    #[inline]
    fn dot(&self, ctx: usize, x: &[f32]) -> f32 {
        let w = &self.w[ctx * self.n..(ctx + 1) * self.n];
        w.iter().zip(x).map(|(a, b)| a * b).sum()
    }
    #[inline]
    fn learn(&mut self, ctx: usize, x: &[f32], err: f32) {
        let lr = self.lr;
        let w = &mut self.w[ctx * self.n..(ctx + 1) * self.n];
        for (a, b) in w.iter_mut().zip(x) {
            *a += lr * err * b;
        }
    }
}

/// SSE / APM: refines a probability given a small context, interpolating 33 buckets.
struct Apm {
    t: Vec<f32>,
}
impl Apm {
    fn new(ctxs: usize) -> Self {
        let mut t = vec![0.0; ctxs * 33];
        for c in 0..ctxs {
            for j in 0..33 {
                t[c * 33 + j] = squash((j as f32 - 16.0) / 2.0);
            }
        }
        Self { t }
    }
    #[inline]
    fn refine(&self, p: f32, ctx: usize) -> (f32, usize, f32) {
        let s = (stretch(p).clamp(-7.999, 7.999) + 8.0) * 2.0;
        let j = s.floor() as usize;
        let f = s - j as f32;
        let i = ctx * 33 + j;
        (self.t[i] * (1.0 - f) + self.t[i + 1] * f, i, f)
    }
    #[inline]
    fn update(&mut self, i: usize, f: f32, bit: u32) {
        let y = bit as f32;
        self.t[i] += (y - self.t[i]) * 0.02 * (1.0 - f);
        self.t[i + 1] += (y - self.t[i + 1]) * 0.02 * f;
    }
}

struct Cm {
    use_nn: bool,
    use_cm: bool,
    tables: Vec<Vec<Counter>>, // one per order (order 0 = node only, then ORDERS)
    o0: Vec<Counter>,
    match_tab: Vec<u32>,
    match_conf: Vec<Counter>, // [min(len,31)*2 + expected bit]
    mix1: Mixer,              // ctx: bit position x match-length bucket
    mix2: Mixer,              // ctx: previous byte
    apm1: Apm,                // ctx: previous byte x node
    apm2: Apm,                // ctx: hash of two previous bytes
}

impl Cm {
    fn new(use_nn: bool, use_cm: bool) -> Self {
        let n_in = Self::n_inputs(use_nn, use_cm);
        Self {
            use_nn,
            use_cm,
            tables: ORDERS.iter().map(|_| vec![Counter::default(); 1 << HBITS]).collect(),
            o0: vec![Counter::default(); 256],
            match_tab: vec![u32::MAX; 1 << 20],
            match_conf: vec![Counter::default(); 64],
            mix1: Mixer::new(n_in, 8 * 4, 0.3),
            mix2: Mixer::new(n_in, 256, 0.3),
            apm1: Apm::new(256 * 256),
            apm2: Apm::new(1 << 16),
        }
    }
    fn n_inputs(use_nn: bool, use_cm: bool) -> usize {
        1 + if use_nn { 2 } else { 0 } + if use_cm { 1 + ORDERS.len() + 1 } else { 0 }
    }

    /// Bits for coding `data` (one lane), given the neural distributions `nn` (256 per byte).
    fn run(&mut self, data: &[u8], nn: &[u16]) -> f64 {
        let mut bits = 0.0f64;
        let mut hist: u64 = 0; // last 8 bytes
        let mut ctx_hash = [0u32; ORDERS.len()];
        let (mut mptr, mut mlen) = (u32::MAX, 0u32);
        let mut x = Vec::with_capacity(16);
        let mut cum = [0u32; 257];
        for (i, &byte) in data.iter().enumerate() {
            let row = &nn[i * 256..(i + 1) * 256];
            if self.use_nn {
                for s in 0..256 {
                    cum[s + 1] = cum[s] + row[s] as u32;
                }
            }
            // Per-byte context hashes.
            for (k, &o) in ORDERS.iter().enumerate() {
                let mask = if o >= 8 { u64::MAX } else { (1u64 << (8 * o)) - 1 };
                let h = ((hist & mask).wrapping_add(o as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                ctx_hash[k] = (h >> (64 - HBITS)) as u32;
            }
            let expected = if mptr != u32::MAX { Some(data[mptr as usize]) } else { None };
            let c1 = (hist & 0xFF) as usize;
            let c2 = ((hist >> 8) & 0xFF) as usize;
            let mut node = 1usize;
            let mut lo = 0usize; // symbol range of the current node: [lo, lo + span)
            let mut span = 256usize;
            let mut miss = false;
            for bp in 0..8 {
                let bit = ((byte >> (7 - bp)) & 1) as u32;
                x.clear();
                x.push(1.0); // bias
                if self.use_nn {
                    let half = span / 2;
                    let all = (cum[lo + span] - cum[lo]) as f32;
                    let one = (cum[lo + span] - cum[lo + half]) as f32;
                    let p = clampp(one / all.max(1.0));
                    let s = stretch(p);
                    x.push(s);
                    x.push(s.abs() * s.signum() * 0.5); // second weight: lets the mixer scale confidence
                }
                let mut midx = usize::MAX;
                let mut ebit = 0;
                if self.use_cm {
                    x.push(stretch(clampp(self.o0[node].p)));
                    for k in 0..ORDERS.len() {
                        let idx = ((ctx_hash[k] as usize) ^ (node * 0x9E37)) & ((1 << HBITS) - 1);
                        x.push(stretch(clampp(self.tables[k][idx].p)));
                    }
                    match expected {
                        Some(e) if !miss => {
                            ebit = ((e >> (7 - bp)) & 1) as usize;
                            midx = (mlen.min(31) as usize) * 2 + ebit;
                            let conf = clampp(self.match_conf[midx].p);
                            x.push(stretch(conf));
                        }
                        _ => x.push(0.0),
                    }
                }
                let mb = if mlen == 0 || miss { 0 } else if mlen < 16 { 1 } else if mlen < 32 { 2 } else { 3 };
                let ctx1 = bp * 4 + mb;
                let d = (self.mix1.dot(ctx1, &x) + self.mix2.dot(c1, &x)) * 0.5;
                let p_mix = clampp(squash(d));
                let (pa1, ia1, fa1) = self.apm1.refine(p_mix, c1 * 256 + node);
                let (pa2, ia2, fa2) = self.apm2.refine(p_mix, ((c2 << 8) | c1) ^ (node << 4) & 0xFFFF);
                let p = clampp((2.0 * p_mix + pa1 + pa2) * 0.25);
                bits -= if bit == 1 { (p as f64).log2() } else { ((1.0 - p) as f64).log2() };
                // Learn.
                let err = bit as f32 - p_mix;
                self.mix1.learn(ctx1, &x, err);
                self.mix2.learn(c1, &x, err);
                self.apm1.update(ia1, fa1, bit);
                self.apm2.update(ia2, fa2, bit);
                if self.use_cm {
                    self.o0[node].update(bit, 60);
                    for k in 0..ORDERS.len() {
                        let idx = ((ctx_hash[k] as usize) ^ (node * 0x9E37)) & ((1 << HBITS) - 1);
                        self.tables[k][idx].update(bit, 250);
                    }
                    if midx != usize::MAX {
                        // match_conf predicts "next bit is 1"; train toward the actual bit.
                        self.match_conf[midx].update(bit, 1000);
                        if bit as usize != ebit {
                            miss = true;
                        }
                    }
                }
                node = (node << 1) | bit as usize;
                let half = span / 2;
                if bit == 1 {
                    lo += half;
                }
                span = half;
            }
            // Match model: follow or re-find via a hash of the last 6 bytes.
            hist = (hist << 8) | byte as u64;
            if mptr != u32::MAX && !miss {
                mptr += 1;
                mlen += 1;
            } else {
                mptr = u32::MAX;
                mlen = 0;
            }
            if i >= 6 {
                let h = (((hist & 0xFFFF_FFFF_FFFF) * 0x2545_F491_4F6C_DD1D) >> 44) as usize;
                if mptr == u32::MAX {
                    let cand = self.match_tab[h];
                    if cand != u32::MAX {
                        mptr = cand;
                        mlen = 1;
                    }
                }
                self.match_tab[h] = (i + 1) as u32;
            }
            if mptr as usize >= data.len() {
                mptr = u32::MAX;
                mlen = 0;
            }
        }
        bits
    }
}

fn nn_only(data: &[u8], nn: &[u16]) -> f64 {
    data.iter().enumerate().map(|(i, &b)| -((nn[i * 256 + b as usize] as f64) / 65536.0).log2()).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lane: usize = args[1].parse().expect("lane bytes");
    let files = &args[2..];
    println!("lane = {lane} bytes; bits per byte (lower is better)");
    println!("| file | nn | cm | nn+cm |");
    println!("|---|---:|---:|---:|");
    let rows: Vec<String> = files.par_iter().map(|f| {
        let data = std::fs::read(f).expect("file");
        let pf = File::open(format!("{f}.p16")).expect("p16");
        let map = unsafe { Mmap::map(&pf) }.expect("map");
        let nn: &[u16] = unsafe { std::slice::from_raw_parts(map.as_ptr() as *const u16, map.len() / 2) };
        assert_eq!(nn.len(), data.len() * 256, "p16 length");
        let mut res = [0.0f64; 3];
        let cfgs = [(true, false), (false, true), (true, true)];
        let out: Vec<f64> = cfgs.par_iter().map(|&(un, uc)| {
            let mut total = 0.0;
            for s in (0..data.len()).step_by(lane) {
                let e = (s + lane).min(data.len());
                if un && !uc {
                    total += nn_only(&data[s..e], &nn[s * 256..e * 256]);
                } else {
                    total += Cm::new(un, uc).run(&data[s..e], &nn[s * 256..e * 256]);
                }
            }
            total / data.len() as f64
        }).collect();
        res.copy_from_slice(&out);
        let name = std::path::Path::new(f).file_name().unwrap().to_string_lossy().to_string();
        format!("| {name} | {:.3} | {:.3} | {:.3} |", res[0], res[1], res[2])
    }).collect();
    for r in rows {
        println!("{r}");
    }
}
