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

use crate::coder::{Decoder, Encoder};
use crate::mixer::{Mixer, knob};
use crate::tables::{squash, stretch};

const O2_BITS: usize = 16;

/// Two probabilities on one context: one adapts fast (rate 1/8), one slowly (1/128).
/// Pure shifts: cheap, and identical on every CPU.
#[derive(Clone, Copy)]
struct Dual([u16; 2]);

impl Dual {
    const INIT: Dual = Dual([32768; 2]);
    #[inline(always)]
    fn add(self, m: &mut Mixer) {
        m.add(stretch(self.0[0] as i32 >> 4));
        m.add(stretch(self.0[1] as i32 >> 4));
    }
    #[inline(always)]
    fn update(&mut self, y: u32, fast: u32, slow: u32) {
        let t = ((y << 16) - y) as i32;
        let a = self.0[0] as i32;
        let b = self.0[1] as i32;
        self.0[0] = (a + ((t - a) >> fast)) as u16;
        self.0[1] = (b + ((t - b) >> slow)) as u16;
    }
}

struct Apm {
    t: Vec<u16>,
    idx: usize,
}

impl Apm {
    fn new(n: usize) -> Apm {
        let mut t = vec![0u16; n * 33];
        for i in 0..n {
            for j in 0..33 {
                t[i * 33 + j] = (squash((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        Apm { t, idx: 0 }
    }
    #[inline(always)]
    fn pp(&mut self, pr: i32, cx: usize) -> i32 {
        let s = stretch(pr) + 2048;
        let lo = (s >> 7) as usize;
        let w = s & 127;
        let base = cx * 33 + lo;
        self.idx = base + ((w >> 6) as usize);
        ((self.t[base] as i32) * (128 - w) + (self.t[base + 1] as i32) * w) >> 7
    }
    #[inline(always)]
    fn update(&mut self, y: u32) {
        const R: u32 = 7;
        let g = ((y << 16) + (y << R) - y - y) as i32;
        let v = self.t[self.idx] as i32;
        self.t[self.idx] = (v + ((g - v) >> R)) as u16;
    }
}

/// Encoder or decoder behind one interface.
trait BitIo {
    /// Codes `bit` (ignored when decoding) with P(1) = p/65536; returns the actual bit.
    fn code(&mut self, bit: u32, p: u32) -> u32;
}

impl BitIo for Encoder {
    #[inline(always)]
    fn code(&mut self, bit: u32, p: u32) -> u32 {
        self.encode(bit, p);
        bit
    }
}

impl BitIo for Decoder<'_> {
    #[inline(always)]
    fn code(&mut self, _bit: u32, p: u32) -> u32 {
        self.decode(p)
    }
}

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
    a_o2: Vec<Dual>,
    a_hc: Vec<Dual>,
    ma: Mixer,
    apm_a: Apm,
    // decision 2: same as d2
    b_cd: Vec<Dual>,
    b_hist: Vec<Dual>,
    b_run: Vec<Dual>,
    mb: Mixer,
    // literal bits
    t0: Vec<Dual>,
    t1: Vec<Dual>,
    t2: Vec<Dual>,
    t3: Vec<Dual>,
    ml: Mixer,
    apm_l: Apm,
    lim: u32,
    lf: u32,
}

impl Model {
    fn new() -> Model {
        let ma = Mixer::new(9, 24 * 4);
        let mb = Mixer::new(7, 24);
        let ml = Mixer::new(9, 256);
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
            a_o2: vec![Dual::INIT; 1 << 16],
            a_hc: vec![Dual::INIT; 1 << 16],
            ma,
            apm_a: Apm::new(24 * 16),
            b_cd: vec![Dual::INIT; 65536],
            b_hist: vec![Dual::INIT; 256 * 24],
            b_run: vec![Dual::INIT; 24 * 256],
            mb,
            t0: vec![Dual::INIT; 256],
            t1: vec![Dual::INIT; 65536],
            t2: vec![Dual::INIT; 1 << O2_BITS],
            t3: vec![Dual::INIT; 65536],
            ml,
            apm_l: Apm::new(256),
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
        let io2 = (self.h2 >> 6) & 0xffff;
        let ihc = ((self.hist_a & 0xff) << 8) | c1;
        let m = &mut self.ma;
        self.a_run[ia].add(m);
        self.a_hist[ih].add(m);
        self.a_o2[io2].add(m);
        self.a_hc[ihc].add(m);
        m.add(256);
        m.select(rq * 4 + (self.hist_a & 3));
        let pr = m.predict();
        let pa = self.apm_a.pp(pr, rq * 16 + (self.hist_a & 15));
        let p = (pr * 16 + pa) >> 1;
        let y = io.code((c == c1) as u32, (p.clamp(32, 65535 - 32)) as u32);
        self.a_run[ia].update(y, lf, lim);
        self.a_hist[ih].update(y, lf, lim);
        self.a_o2[io2].update(y, lf, lim);
        self.a_hc[ihc].update(y, lf, lim);
        self.ma.update(y);
        self.apm_a.update(y);
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
        let m = &mut self.mb;
        self.b_cd[icd].add(m);
        self.b_hist[ihb].add(m);
        self.b_run[irb].add(m);
        m.add(256);
        m.select(rq);
        let pr = m.predict();
        let y = io.code((c == d2) as u32, to16(pr));
        self.b_cd[icd].update(y, lf, lim);
        self.b_hist[ihb].update(y, lf, lim);
        self.b_run[irb].update(y, lf, lim);
        self.mb.update(y);
        self.hist_b = (self.hist_b << 1) | y as usize;
        let sym = if y == 1 {
            d2
        } else {
            // 3. literal, MSB first
            let mut c0 = 1usize;
            for i in (0..8).rev() {
                let i1 = (c1 << 8) | c0;
                let i2 = self.h2 | c0;
                let i3 = (d2 << 8) | c0;
                let m = &mut self.ml;
                self.t3[i3].add(m);
                self.t0[c0].add(m);
                self.t1[i1].add(m);
                self.t2[i2].add(m);
                m.add(256);
                m.select(c0);
                let pr = m.predict();
                let pl = self.apm_l.pp(pr, c0);
                let p = (pr * 16 + pl * 3) >> 2;
                let bit = io.code(((c >> i) & 1) as u32, p.clamp(32, 65535 - 32) as u32);
                self.t0[c0].update(bit, lf, lim);
                self.t1[i1].update(bit, lf, lim);
                self.t2[i2].update(bit, lf, lim);
                self.t3[i3].update(bit, lf, lim);
                self.ml.update(bit);
                self.apm_l.update(bit);
                c0 = (c0 << 1) | bit as usize;
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
        let h = ((self.c2 << 8) | self.c1) as u32;
        self.h2 = ((h.wrapping_mul(0x9E37_79B1) >> (32 - (O2_BITS - 8))) as usize) << 8;
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

/// Segments are coded independently so one block can use every core in both directions.
/// The count depends only on the data length, never on the machine.
pub fn segments_for(n: usize) -> usize {
    (n / (2 << 20)).clamp(1, 256)
}

/// `segs:varint {comp_len:varint}* streams...`
pub fn encode(data: &[u8], threads: usize) -> Vec<u8> {
    let k = segments_for(data.len());
    let seg = data.len().div_ceil(k).max(1);
    let parts: Vec<&[u8]> = if data.is_empty() { vec![data] } else { data.chunks(seg).collect() };
    let coded = crate::par::map(&parts, threads, |p| encode_one(p));
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
    if k != segments_for(n) && !(n == 0 && k == 1) {
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
    let parts = crate::par::map(&jobs, threads, |&(s, len)| decode_one(s, len));
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
