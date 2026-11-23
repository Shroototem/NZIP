//! Entropy coder for BWT output.
//!
//! BWT output is mostly runs. Each byte is coded as a short decision tree:
//!   1. same as the previous byte?           (run model)
//!   2. else, same as the last *other* byte?  (two-symbol move-to-front)
//!   3. else, the byte itself, as 8 binary decisions (order-1/2 context mixing)
//! so most bytes cost one or two binary decisions instead of eight. Every decision is
//! predicted by adaptive counters mixed by a small online neural mixer, then coded with the
//! integer arithmetic coder. One generic function drives both encoder and decoder, so the
//! two can never disagree.

use crate::cm::{Ctr, G, Mix, NG, hg};

/// Adaptation rate (shift) of the literal counters.
const LR: u32 = 3;
use crate::coder::{BitIo, Decoder, Dual, Encoder};
use crate::mixer::knob;


const O2_BITS: usize = 16;

#[inline(always)]
fn to16(p12: i32) -> u32 {
    ((p12 << 4) + 8).clamp(32, 65535 - 32) as u32
}

#[inline(always)]
fn runq(run: usize) -> usize {
    if run < 12 { run } else { (12 + (usize::BITS - run.leading_zeros()) as usize - 4).min(23) }
}

struct Model {
    c1: usize,
    c2: usize,
    d2: usize,
    run: usize,
    hist_a: usize,
    hist_b: usize,
    h2: usize,
    // decision 1: same as c1
    a_run: Vec<Dual>,
    a_hist: Vec<Dual>,
    a_o2: Vec<Ctr>,
    a_hc: Vec<Ctr>,
    ma: Mix<7>,
    // decision 2: same as d2
    b_cd: Vec<Dual>,
    b_hist: Vec<Dual>,
    b_run: Vec<Dual>,
    mb: Mix<7>,
    // literal bits
    t0: Vec<G>,
    t1: Vec<G>,
    t2: Vec<G>,
    t3: Vec<G>,
    ml: Mix<5>,
    lim: u32,
    lf: u32,
}

impl Model {
    fn new() -> Model {
        let ma = Mix::new(24 * 4, 2);
        let mb = Mix::new(24, 2);
        let ml = Mix::new(256, 2);
        Model {
            c1: 0,
            c2: 0,
            d2: 0,
            run: 0,
            hist_a: 0,
            hist_b: 0,
            h2: 0,
            a_run: vec![Dual::INIT; 24 * 256],
            a_hist: vec![Dual::INIT; 1 << 12],
            a_o2: vec![Ctr::INIT; 1 << 16],
            a_hc: vec![Ctr::INIT; 1 << 16],
            ma,
            b_cd: vec![Dual::INIT; 65536],
            b_hist: vec![Dual::INIT; 256 * 24],
            b_run: vec![Dual::INIT; 24 * 256],
            mb,
            t0: vec![G::INIT; NG],
            t1: vec![G::INIT; 256 * NG],
            t2: vec![G::INIT; 1 << (O2_BITS - 4)],
            t3: vec![G::INIT; 256 * NG],
            ml,
            lim: knob(1) as u32,
            lf: knob(2) as u32,
        }
    }

    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, c: usize) -> usize {
        let lim = self.lim;
        let lf = self.lf;
        let rq = runq(self.run);
        let c1 = self.c1;

        // 1. same as previous byte?
        let ia = rq * 256 + c1;
        let ih = self.hist_a & 0xfff;
        let io2 = self.h2;
        let ihc = ((self.hist_a & 0xff) << 8) | c1;
        let (r0, r1) = self.a_run[ia].st2();
        let (h0, h1) = self.a_hist[ih].st2();
        self.ma.predict([r0, r1, h0, h1, self.a_o2[io2].st(), self.a_hc[ihc].st(), 256], rq * 4 + (self.hist_a & 3));
        let pr = self.ma.pr12();
        let p = pr * 16;
        let y = io.code((c == c1) as u32, (p.clamp(32, 65535 - 32)) as u32);
        self.a_run[ia].update(y, lf, lim);
        self.a_hist[ih].update(y, lf, lim);
        self.a_o2[io2].update(y, LR);
        self.a_hc[ihc].update(y, LR);
        self.ma.update(y);
        self.hist_a = (self.hist_a << 1) | y as usize;
        if y == 1 {
            self.run += 1;
            self.push(c1);
            return c1;
        }

        // 2. same as the last other byte?
        let d2 = self.d2;
        let icd = (c1 << 8) | d2;
        let ihb = ((self.hist_b & 0xff) * 24) + rq;
        let irb = rq * 256 + d2;
        let (d0, d1) = self.b_cd[icd].st2();
        let (h0, h1) = self.b_hist[ihb].st2();
        let (r0, r1) = self.b_run[irb].st2();
        self.mb.predict([d0, d1, h0, h1, r0, r1, 256], rq);
        let pr = self.mb.pr12();
        let y = io.code((c == d2) as u32, to16(pr));
        self.b_cd[icd].update(y, lf, lim);
        self.b_hist[ihb].update(y, lf, lim);
        self.b_run[irb].update(y, lf, lim);
        self.mb.update(y);
        self.hist_b = (self.hist_b << 1) | y as usize;
        let sym = if y == 1 {
            d2
        } else {
            // 3. literal, MSB first, one counter group per nibble and context
            let mut c0 = 1usize;
            let c2 = (self.h2 as u32) * NG as u32;
            for half in 0..2 {
                let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
                let g0 = &mut self.t0[sel].0;
                let g1 = &mut self.t1[c1 * NG + sel].0;
                let g2 = &mut self.t2[hg(c2 + sel as u32, O2_BITS as u32 - 4)].0;
                let g3 = &mut self.t3[d2 * NG + sel].0;
                let mut j = 1usize;
                for q in 0..4 {
                    let i = 7 - (half * 4 + q);
                    let p = self.ml.predict([g3[j].st(), g0[j].st(), g1[j].st(), g2[j].st(), 256], c0);
                    let bit = io.code(((c >> i) & 1) as u32, p);
                    g0[j].update(bit, LR);
                    g1[j].update(bit, LR);
                    g2[j].update(bit, LR);
                    g3[j].update(bit, LR);
                    self.ml.update(bit);
                    j = (j << 1) | bit as usize;
                    c0 = (c0 << 1) | bit as usize;
                }
            }
            c0 & 255
        };
        self.run = 0;
        self.d2 = c1;
        self.push(sym);
        sym
    }

    #[inline(always)]
    fn push(&mut self, c: usize) {
        self.c2 = self.c1;
        self.c1 = c;
        self.h2 = (self.c2 << 8) | self.c1;
    }
}

#[inline(always)]
fn encode_impl(data: &[u8]) -> Vec<u8> {
    let mut m = Model::new();
    let mut enc = Encoder::new(data.len() / 4 + 64);
    for &c in data {
        m.code(&mut enc, c as usize);
    }
    enc.finish()
}

#[inline(always)]
fn decode_impl(src: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let mut m = Model::new();
    let mut dec = Decoder::new(src);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(m.code(&mut dec, 0) as u8);
    }
    if dec.overrun() {
        return Err("BWT stream truncated".into());
    }
    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn encode_avx2(data: &[u8]) -> Vec<u8> {
    encode_impl(data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn decode_avx2(src: &[u8], n: usize) -> Result<Vec<u8>, String> {
    decode_impl(src, n)
}

fn encode_one(data: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { encode_avx2(data) };
    }
    encode_impl(data)
}

fn decode_one(src: &[u8], n: usize) -> Result<Vec<u8>, String> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { decode_avx2(src, n) };
    }
    decode_impl(src, n)
}

/// Segments are coded independently so one block can use every core in both directions:
/// about 2 MiB each, rounded to a whole number of rounds over `threads` so no round runs
/// with idle cores (at least 256 KiB each, at most 1024 segments).
fn segments_for(n: usize, threads: usize) -> usize {
    let t = threads.max(1);
    let rounds = n.div_ceil(t * (2 << 20)).max(1);
    (rounds * t).min(n / (256 << 10)).clamp(1, MAX_SEGS)
}

const MAX_SEGS: usize = 1024;

/// `segs:varint {comp_len:varint}* streams...`
pub fn encode(data: &[u8], threads: usize) -> Vec<u8> {
    let k = segments_for(data.len(), threads);
    let seg = data.len().div_ceil(k).max(1);
    let parts: Vec<&[u8]> = if data.is_empty() { vec![data] } else { data.chunks(seg).collect() };
    let coded = crate::par::map(&parts, threads, |p| {
        let c = encode_one(p);
        crate::progress::work(p.len());
        c
    });
    let mut out = Vec::with_capacity(coded.iter().map(|c| c.len()).sum::<usize>() + 16 * k);
    crate::container::write_varint(&mut out, coded.len() as u64);
    for c in &coded {
        crate::container::write_varint(&mut out, c.len() as u64);
    }
    for c in coded {
        out.extend_from_slice(&c);
    }
    out
}

pub fn decode(src: &[u8], n: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt BWT coder stream".to_string();
    let mut r = src;
    let k = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
    if k == 0 || k > MAX_SEGS || k > n.max(1) {
        return Err(e());
    }
    let mut lens = Vec::with_capacity(k);
    for _ in 0..k {
        lens.push(crate::container::read_varint(&mut r).map_err(|_| e())? as usize);
    }
    let seg = n.div_ceil(k).max(1);
    let mut jobs = Vec::with_capacity(k);
    let mut off = 0usize;
    for (i, &l) in lens.iter().enumerate() {
        let s = r.get(off..off + l).ok_or_else(e)?;
        off += l;
        let len = seg.min(n.saturating_sub(i * seg));
        jobs.push((s, len));
    }
    let parts = crate::par::map(&jobs, threads, |&(s, len)| {
        let d = decode_one(s, len);
        crate::progress::work(len);
        d
    });
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend_from_slice(&p?);
    }
    Ok(out)
}

/// Memory the coder needs (both directions), excluding data buffers.
pub const MEM: u64 = (4 << O2_BITS) + 2 * 65536 * 33 * 2 + (2 << 20);

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut v = Vec::new();
        let mut r = 1u32;
        for i in 0..200_000u32 {
            r = r.wrapping_mul(1103515245).wrapping_add(12345);
            let run = (r >> 28) as usize;
            for _ in 0..run {
                v.push(((r >> 16) % 7) as u8 + if i % 3 == 0 { 0 } else { 200 });
            }
        }
        let c = super::encode(&v, 4);
        assert_eq!(super::decode(&c, v.len(), 3).unwrap(), v);
        assert!(c.len() < v.len() / 2);
    }
}
