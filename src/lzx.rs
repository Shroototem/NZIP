//! Binary engine: LZ77 with an optimal parser and a context-mixing token coder.
//!
//! BWT is weak on machine code and other fixed-layout binary data; LZ77 keeps that layout.
//! * Match finder: hash chains over 4-byte prefixes (+ a 3-byte table for near matches).
//! * Parser: forward dynamic programming over windows of up to 2048 positions. Every
//!   position's literal, short-rep, rep0..3 and match options are priced with the coder's
//!   current probabilities (like LZMA's optimal parser); long matches are taken at once.
//! * Coder: LZMA-style token structure (12-state machine, rep distances, length and
//!   distance trees) on the integer binary arithmetic coder. Literals are predicted by an
//!   online mixer over order-1, order-2, match-byte and position contexts.
//! Data is coded in independent sub-blocks so both directions use every core.
//!
//! Prices only steer the encoder's choices; the decoder never computes them.

use crate::cm::{Ctr, Mix};
use crate::coder::{BitIo, Decoder, Encoder};

/// Largest independent sub-block (sub-blocks run in parallel in both directions).
pub const SUB_BLOCK: usize = 64 << 20;

/// Sub-block size for `n` bytes on `threads` threads: enough sub-blocks to keep every
/// thread busy in both directions, but at least 8 MiB (smaller ones lose ratio).
/// Dev override: NZ_SUB in MiB.
fn sub_size(n: usize, threads: usize) -> usize {
    if let Some(v) = std::env::var("NZ_SUB").ok().and_then(|v| v.parse::<usize>().ok()) {
        return (v << 20).clamp(1 << 20, SUB_BLOCK);
    }
    let even = n.div_ceil(threads.max(1)).next_multiple_of(1 << 20);
    even.clamp(8 << 20, SUB_BLOCK).min(n.max(1))
}
const MAX_LEN: usize = 273;
const WIN: usize = 2048;
const H4_BITS: u32 = 22;
const H3_BITS: u32 = 16;
/// Binary-tree search depth and "nice" length: deeper / longer = smaller but slower.
const DEPTH: usize = 48;
const NICE: usize = 64;
const STATES: usize = 12;

// ---------------------------------------------------------------- bit models

#[derive(Clone, Copy)]
struct Prob(u16);

impl Prob {
    const INIT: Prob = Prob(32768);
    #[inline(always)]
    fn p(self) -> u32 {
        (self.0 as u32).clamp(64, 65535 - 64)
    }
    #[inline(always)]
    fn update(&mut self, b: u32) {
        let v = self.0 as i32;
        let t = ((b << 16) - b) as i32;
        self.0 = (v + ((t - v) >> 5)) as u16;
    }
}

#[inline(always)]
fn bit<IO: BitIo>(io: &mut IO, p: &mut Prob, b: u32) -> u32 {
    let r = io.code(b, p.p());
    p.update(r);
    r
}

#[inline(always)]
fn tree<IO: BitIo>(io: &mut IO, probs: &mut [Prob], nbits: u32, val: u32) -> u32 {
    let mut m = 1usize;
    for i in (0..nbits).rev() {
        let r = bit(io, &mut probs[m], (val >> i) & 1);
        m = (m << 1) | r as usize;
    }
    (m - (1 << nbits)) as u32
}

#[inline(always)]
fn rtree<IO: BitIo>(io: &mut IO, probs: &mut [Prob], nbits: u32, val: u32) -> u32 {
    let mut m = 1usize;
    let mut out = 0;
    for i in 0..nbits {
        let r = bit(io, &mut probs[m], (val >> i) & 1);
        m = (m << 1) | r as usize;
        out |= r << i;
    }
    out
}

#[inline(always)]
fn direct<IO: BitIo>(io: &mut IO, nbits: u32, val: u32) -> u32 {
    let mut out = 0;
    for i in (0..nbits).rev() {
        out |= io.code((val >> i) & 1, 32768) << i;
    }
    out
}

// ---------------------------------------------------------------- prices (encoder only)

/// Cost in 1/16 bit of coding a bit whose probability is p/4096.
static PRICE: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

fn price_table() -> &'static [u32] {
    PRICE.get_or_init(|| (0..4097u32).map(|i| (-(i.max(1) as f64 / 4096.0).log2() * 16.0).round() as u32).collect())
}

#[inline(always)]
fn pr(t: &[u32], p: Prob, b: u32) -> u32 {
    let p1 = p.p();
    t[((if b == 1 { p1 } else { 65536 - p1 }) >> 4) as usize]
}

fn tree_price(t: &[u32], probs: &[Prob], nbits: u32, val: u32) -> u32 {
    let mut m = 1usize;
    let mut s = 0;
    for i in (0..nbits).rev() {
        let b = (val >> i) & 1;
        s += pr(t, probs[m], b);
        m = (m << 1) | b as usize;
    }
    s
}

fn rtree_price(t: &[u32], probs: &[Prob], nbits: u32, val: u32) -> u32 {
    let mut m = 1usize;
    let mut s = 0;
    for i in 0..nbits {
        let b = (val >> i) & 1;
        s += pr(t, probs[m], b);
        m = (m << 1) | b as usize;
    }
    s
}

// ---------------------------------------------------------------- lengths and distances

struct LenCoder {
    choice: Prob,
    choice2: Prob,
    low: [[Prob; 8]; 4],
    mid: [[Prob; 8]; 4],
    high: [Prob; 256],
}

impl LenCoder {
    fn new() -> LenCoder {
        LenCoder { choice: Prob::INIT, choice2: Prob::INIT, low: [[Prob::INIT; 8]; 4], mid: [[Prob::INIT; 8]; 4], high: [Prob::INIT; 256] }
    }
    /// `l` = length - 2 (0..=271).
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, l: u32, ps: usize) -> u32 {
        if bit(io, &mut self.choice, (l >= 8) as u32) == 0 {
            tree(io, &mut self.low[ps], 3, l)
        } else if bit(io, &mut self.choice2, (l >= 16) as u32) == 0 {
            8 + tree(io, &mut self.mid[ps], 3, l.wrapping_sub(8))
        } else {
            16 + tree(io, &mut self.high, 8, l.wrapping_sub(16))
        }
    }
    fn prices(&self, t: &[u32], out: &mut [[u32; MAX_LEN - 1]; 4]) {
        for (ps, row) in out.iter_mut().enumerate() {
            for (l, v) in row.iter_mut().enumerate() {
                let l = l as u32;
                *v = if l < 8 {
                    pr(t, self.choice, 0) + tree_price(t, &self.low[ps], 3, l)
                } else if l < 16 {
                    pr(t, self.choice, 1) + pr(t, self.choice2, 0) + tree_price(t, &self.mid[ps], 3, l - 8)
                } else {
                    pr(t, self.choice, 1) + pr(t, self.choice2, 1) + tree_price(t, &self.high, 8, l - 16)
                };
            }
        }
    }
}

#[inline(always)]
fn slot_of(d: u32) -> u32 {
    if d < 4 {
        d
    } else {
        let n = 31 - d.leading_zeros();
        (n << 1) | ((d >> (n - 1)) & 1)
    }
}

struct DistCoder {
    slot: [[Prob; 64]; 4],
    spec: [[Prob; 64]; 14],
    align: [Prob; 16],
}

impl DistCoder {
    fn new() -> DistCoder {
        DistCoder { slot: [[Prob::INIT; 64]; 4], spec: [[Prob::INIT; 64]; 14], align: [Prob::INIT; 16] }
    }
    /// `d` = distance - 1.
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, d: u32, ls: usize) -> u32 {
        let s = tree(io, &mut self.slot[ls], 6, slot_of(d));
        if s < 4 {
            return s;
        }
        let footer = (s >> 1) - 1;
        let base = (2 | (s & 1)) << footer;
        let r = d.wrapping_sub(base);
        if s < 14 {
            base + rtree(io, &mut self.spec[s as usize], footer, r)
        } else {
            let hi = direct(io, footer - 4, r >> 4);
            base + (hi << 4) + rtree(io, &mut self.align, 4, r & 15)
        }
    }
}

// ---------------------------------------------------------------- literal model

struct Lit {
    t1: Vec<Ctr>,
    t2: Vec<Ctr>,
    tm: Vec<Ctr>,
    tp: Vec<Ctr>,
    tx: Vec<Ctr>,
    m: Mix<6>,
}

const TX_BITS: u32 = 14;
/// Counter adaptation limit for literal contexts.
const LIM: u16 = 60;

/// If `pos` is inside the rel32 operand of an x86 CALL/JMP/Jcc, returns (operand byte
/// index 1..=4, context): the filtered operand is a big-endian absolute target, so its
/// high byte tracks the position and each later byte the ones before it.
#[inline(always)]
fn op_ctx(buf: &[u8], pos: usize, base: usize) -> (usize, u32) {
    for k in 1..=4usize {
        if pos < k {
            break;
        }
        let i = pos - k;
        let op = (buf[i] & 0xFE) == 0xE8 || (i >= 1 && buf[i - 1] == 0x0F && (buf[i] & 0xF0) == 0x80);
        if op {
            let abs = base + pos;
            let ctx = match k {
                1 => ((abs >> 16) & 0xff) as u32,
                2 => ((buf[pos - 1] as u32) << 8) | ((abs >> 8) & 0xff) as u32,
                3 => ((buf[pos - 2] as u32) << 8) | buf[pos - 1] as u32,
                _ => buf[pos - 3] as u32,
            };
            return (k, ctx);
        }
    }
    (0, 0)
}

impl Lit {
    fn new() -> Lit {
        Lit {
            t1: vec![Ctr::INIT; 1 << 16],
            t2: vec![Ctr::INIT; 1 << 18],
            tm: vec![Ctr::INIT; 1 << 16],
            tp: vec![Ctr::INIT; 1 << 12],
            tx: vec![Ctr::INIT; 1 << (TX_BITS + 8)],
            m: Mix::new(256 * 2 * 5, 1),
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, byte: u32, prev: u32, h2: usize, mb: u32, pos: usize, matched: bool, op: (usize, u32)) -> u32 {
        let mut c0 = 1usize;
        let pctx = (pos & 15) << 8;
        let (k, octx) = op;
        let xbase = if k == 0 { 0 } else { (((k as u32) << 16 | octx).wrapping_mul(0x9E37_79B1) >> (32 - TX_BITS)) as usize } << 8;
        let row = ((matched as usize) << 8) | (k << 9);
        let (b1, bm) = ((prev as usize) << 8, (mb as usize) << 8);
        for i in (0..8).rev() {
            let (i1, i2, im, ip, ix) = (b1 | c0, h2 | c0, bm | c0, pctx | c0, xbase | c0);
            let x = [self.t1[i1].st(), self.t2[i2].st(), self.tm[im].st(), self.tp[ip].st(), self.tx[ix].st(), 256];
            let p = self.m.predict(x, c0 | row);
            let b = io.code((byte >> i) & 1, p);
            self.m.update(b);
            self.t1[i1].update(b, LIM);
            self.t2[i2].update(b, LIM);
            self.tm[im].update(b, LIM);
            self.tp[ip].update(b, LIM);
            self.tx[ix].update(b, LIM);
            c0 = (c0 << 1) | b as usize;
        }
        (c0 & 255) as u32
    }

    /// Approximate cost (1/16 bit) from the slow order-1 counters.
    #[inline(always)]
    fn price(&self, t: &[u32], byte: u32, prev: u32) -> u32 {
        let mut c0 = 1usize;
        let mut s = 0;
        for i in (0..8).rev() {
            let b = (byte >> i) & 1;
            let p = Prob(self.t1[((prev as usize) << 8) | c0].p() as u16);
            s += pr(t, p, b);
            c0 = (c0 << 1) | b as usize;
        }
        s
    }
}

// ---------------------------------------------------------------- tokens and model

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tok {
    Lit,
    Short,
    Rep(u8, u16),
    Match(u16, u32),
}

impl Tok {
    fn len(self) -> usize {
        match self {
            Tok::Lit | Tok::Short => 1,
            Tok::Rep(_, l) | Tok::Match(l, _) => l as usize,
        }
    }
}

#[inline(always)]
fn next_state(s: usize, t: Tok) -> usize {
    match t {
        Tok::Lit => {
            if s < 4 {
                0
            } else if s < 10 {
                s - 3
            } else {
                s - 6
            }
        }
        Tok::Match(..) => {
            if s < 7 {
                7
            } else {
                10
            }
        }
        Tok::Rep(..) => {
            if s < 7 {
                8
            } else {
                11
            }
        }
        Tok::Short => {
            if s < 7 {
                9
            } else {
                11
            }
        }
    }
}

#[inline(always)]
fn next_reps(r: [u32; 4], t: Tok) -> [u32; 4] {
    match t {
        Tok::Match(_, d) => [d, r[0], r[1], r[2]],
        Tok::Rep(1, _) => [r[1], r[0], r[2], r[3]],
        Tok::Rep(2, _) => [r[2], r[0], r[1], r[3]],
        Tok::Rep(3, _) => [r[3], r[0], r[1], r[2]],
        _ => r,
    }
}

struct Model {
    /// Offset of this sub-block inside its block (x86 targets are block-relative).
    base: usize,
    /// Model inline x86 branch operands (data filtered in place, not split).
    x86ctx: bool,
    state: usize,
    reps: [u32; 4],
    is_match: [[Prob; 4]; STATES],
    is_rep: [Prob; STATES],
    is_rep0: [Prob; STATES],
    is_rep1: [Prob; STATES],
    is_rep2: [Prob; STATES],
    is_rep0_long: [[Prob; 4]; STATES],
    mlen: LenCoder,
    rlen: LenCoder,
    dist: DistCoder,
    lit: Lit,
}

#[inline(always)]
fn h2_of(buf: &[u8], pos: usize) -> usize {
    let a = if pos >= 2 { buf[pos - 2] as u32 } else { 0 };
    let b = if pos >= 1 { buf[pos - 1] as u32 } else { 0 };
    ((((a << 8) | b).wrapping_mul(0x9E37_79B1) >> 22) as usize) << 8
}

impl Model {
    fn new(base: usize, x86ctx: bool) -> Model {
        Model {
            base,
            x86ctx,
            state: 0,
            reps: [0; 4],
            is_match: [[Prob::INIT; 4]; STATES],
            is_rep: [Prob::INIT; STATES],
            is_rep0: [Prob::INIT; STATES],
            is_rep1: [Prob::INIT; STATES],
            is_rep2: [Prob::INIT; STATES],
            is_rep0_long: [[Prob::INIT; 4]; STATES],
            mlen: LenCoder::new(),
            rlen: LenCoder::new(),
            dist: DistCoder::new(),
            lit: Lit::new(),
        }
    }

    /// Codes one token at `pos`. `buf` holds every byte before `pos` (and, when encoding,
    /// the bytes after it). Returns the token and, for a literal, its byte.
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, buf: &[u8], pos: usize, t: Tok) -> (Tok, u8) {
        let s = self.state;
        let ps = pos & 3;
        let lit_byte = if matches!(t, Tok::Lit) && pos < buf.len() { buf[pos] as u32 } else { 0 };
        let out;
        let mut byte = 0u8;
        if bit(io, &mut self.is_match[s][ps], (t != Tok::Lit) as u32) == 0 {
            let prev = if pos > 0 { buf[pos - 1] as u32 } else { 0 };
            let r0 = self.reps[0] as usize + 1;
            let mb = if pos >= r0 { buf[pos - r0] as u32 } else { 0 };
            let op = if self.x86ctx { op_ctx(buf, pos, self.base) } else { (0, 0) };
            byte = self.lit.code(io, lit_byte, prev, h2_of(buf, pos), mb, pos, s >= 7, op) as u8;
            out = Tok::Lit;
        } else if bit(io, &mut self.is_rep[s], matches!(t, Tok::Rep(..) | Tok::Short) as u32) == 0 {
            let (l, d) = if let Tok::Match(l, d) = t { (l as u32, d) } else { (2, 0) };
            let l = self.mlen.code(io, l - 2, ps) + 2;
            let d = self.dist.code(io, d, (l as usize - 2).min(3));
            out = Tok::Match(l as u16, d);
        } else {
            let k = match t {
                Tok::Short => 0,
                Tok::Rep(k, _) => k,
                _ => 0,
            };
            if bit(io, &mut self.is_rep0[s], (k != 0) as u32) == 0 {
                if bit(io, &mut self.is_rep0_long[s][ps], (t != Tok::Short) as u32) == 0 {
                    out = Tok::Short;
                } else {
                    let l = if let Tok::Rep(_, l) = t { l as u32 } else { 2 };
                    out = Tok::Rep(0, (self.rlen.code(io, l - 2, ps) + 2) as u16);
                }
            } else {
                let kk = if bit(io, &mut self.is_rep1[s], (k != 1) as u32) == 0 {
                    1
                } else if bit(io, &mut self.is_rep2[s], (k != 2) as u32) == 0 {
                    2
                } else {
                    3
                };
                let l = if let Tok::Rep(_, l) = t { l as u32 } else { 2 };
                out = Tok::Rep(kk, (self.rlen.code(io, l - 2, ps) + 2) as u16);
            }
        }
        self.state = next_state(s, out);
        self.reps = next_reps(self.reps, out);
        (out, byte)
    }
}

// ---------------------------------------------------------------- match finder

/// Binary-tree match finder (like LZMA's bt4): every position is inserted into a binary
/// search tree of suffixes rooted at its 4-byte hash bucket, so a short walk finds the
/// longest earlier matches; a 3-byte table catches nearby short matches.
struct Mf {
    depth: usize,
    nice: usize,
    head4: Vec<u32>,
    head3: Vec<u32>,
    son: Vec<u32>,
}

#[inline(always)]
fn match_len(buf: &[u8], a: usize, b: usize, limit: usize) -> usize {
    match_len_from(buf, a, b, 0, limit)
}

/// Length of the common prefix of buf[a..] and buf[b..] (a < b), known to be >= `start`.
#[inline(always)]
fn match_len_from(buf: &[u8], a: usize, b: usize, start: usize, limit: usize) -> usize {
    let mut l = start;
    while l + 8 <= limit {
        let x = u64::from_le_bytes(buf[a + l..a + l + 8].try_into().unwrap());
        let y = u64::from_le_bytes(buf[b + l..b + l + 8].try_into().unwrap());
        let z = x ^ y;
        if z != 0 {
            return l + (z.trailing_zeros() / 8) as usize;
        }
        l += 8;
    }
    while l < limit && buf[a + l] == buf[b + l] {
        l += 1;
    }
    l
}

impl Mf {
    fn new(n: usize) -> Mf {
        Mf { depth: DEPTH, nice: NICE, head4: vec![0; 1 << H4_BITS], head3: vec![0; 1 << H3_BITS], son: vec![0; 2 * n] }
    }

    #[inline(always)]
    fn hashes(buf: &[u8], p: usize) -> (usize, usize) {
        let v = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
        let h4 = (v.wrapping_mul(0x9E37_79B1) >> (32 - H4_BITS)) as usize;
        let h3 = ((v & 0xFF_FFFF).wrapping_mul(0x85EB_CA77) >> (32 - H3_BITS)) as usize;
        (h4, h3)
    }

    /// Inserts `p` into the tree; with `out`, also returns matches (len, dist-1) with
    /// strictly increasing length. `buf` holds `n` bytes plus NICE + 8 bytes of padding, so
    /// every tree comparison covers exactly `nice` bytes: the tree stays ordered whatever
    /// order positions are inserted in.
    #[inline(always)]
    fn walk(&mut self, buf: &[u8], n: usize, p: usize, mut out: Option<&mut Vec<(u32, u32)>>) {
        if p + 4 > n {
            return;
        }
        let real = (n - p).min(MAX_LEN);
        let lim = self.nice;
        let (h4, h3) = Self::hashes(buf, p);
        let c3 = self.head3[h3] as usize;
        let mut cur = self.head4[h4] as usize;
        self.head4[h4] = p as u32 + 1;
        self.head3[h3] = p as u32 + 1;
        let mut best = 2usize;
        let mut best_q = usize::MAX;
        if let Some(o) = out.as_deref_mut() {
            if c3 > 0 {
                let q = c3 - 1;
                if p - q <= 1 << 16 {
                    let l = match_len(buf, q, p, real);
                    if l > best {
                        best = l;
                        best_q = q;
                        o.push((l as u32, (p - q - 1) as u32));
                    }
                }
            }
        }
        let (mut p0, mut p1) = (2 * p + 1, 2 * p);
        let (mut len0, mut len1) = (0usize, 0usize);
        let mut depth = self.depth;
        loop {
            if cur == 0 || depth == 0 {
                self.son[p0] = 0;
                self.son[p1] = 0;
                break;
            }
            depth -= 1;
            let q = cur - 1;
            let len = match_len_from(buf, q, p, len0.min(len1), lim);
            if let Some(o) = out.as_deref_mut() {
                let rl = len.min(real);
                if rl > best {
                    best = rl;
                    best_q = q;
                    o.push((rl as u32, (p - q - 1) as u32));
                }
            }
            if len >= lim {
                // identical up to the limit: p replaces q in the tree
                self.son[p1] = self.son[2 * q];
                self.son[p0] = self.son[2 * q + 1];
                break;
            }
            if buf[q + len] < buf[p + len] {
                self.son[p1] = cur as u32;
                p1 = 2 * q + 1;
                cur = self.son[p1] as usize;
                len1 = len;
            } else {
                self.son[p0] = cur as u32;
                p0 = 2 * q;
                cur = self.son[p0] as usize;
                len0 = len;
            }
        }
        // the tree stops comparing at `nice`; extend the longest match to its real length
        if let Some(o) = out {
            if best >= lim && lim < real && best_q != usize::MAX {
                let l = match_len_from(buf, best_q, p, best, real);
                if let Some(last) = o.last_mut() {
                    last.0 = l as u32;
                }
            }
        }
    }

    #[inline(always)]
    fn insert(&mut self, buf: &[u8], n: usize, p: usize) {
        self.walk(buf, n, p, None);
    }

    #[inline(always)]
    fn find(&mut self, buf: &[u8], n: usize, p: usize, out: &mut Vec<(u32, u32)>) {
        out.clear();
        self.walk(buf, n, p, Some(out));
    }
}

// ---------------------------------------------------------------- optimal parser

#[derive(Clone, Copy)]
struct Node {
    price: u32,
    from: u32,
    tok: Tok,
    reps: [u32; 4],
    state: u8,
}

const INF: u32 = u32::MAX;

struct Prices {
    mlen: Box<[[u32; MAX_LEN - 1]; 4]>,
    rlen: Box<[[u32; MAX_LEN - 1]; 4]>,
    slot: [[u32; 64]; 4],
    small: [[u32; 128]; 4],
    align: [u32; 16],
}

impl Prices {
    fn new() -> Prices {
        Prices {
            mlen: Box::new([[0; MAX_LEN - 1]; 4]),
            rlen: Box::new([[0; MAX_LEN - 1]; 4]),
            slot: [[0; 64]; 4],
            small: [[0; 128]; 4],
            align: [0; 16],
        }
    }
    fn update(&mut self, m: &Model, t: &[u32]) {
        m.mlen.prices(t, &mut self.mlen);
        m.rlen.prices(t, &mut self.rlen);
        for ls in 0..4 {
            for s in 0..64u32 {
                self.slot[ls][s as usize] = tree_price(t, &m.dist.slot[ls], 6, s);
            }
            for d in 0..128u32 {
                let s = slot_of(d);
                let mut p = self.slot[ls][s as usize];
                if s >= 4 {
                    let footer = (s >> 1) - 1;
                    let base = (2 | (s & 1)) << footer;
                    p += rtree_price(t, &m.dist.spec[s as usize], footer, d - base);
                }
                self.small[ls][d as usize] = p;
            }
        }
        for a in 0..16u32 {
            self.align[a as usize] = rtree_price(t, &m.dist.align, 4, a);
        }
    }
    #[inline(always)]
    fn dist(&self, d: u32, len: usize) -> u32 {
        let ls = (len - 2).min(3);
        if d < 128 {
            self.small[ls][d as usize]
        } else {
            let s = slot_of(d);
            let footer = (s >> 1) - 1;
            self.slot[ls][s as usize] + (footer - 4) * 16 + self.align[(d & 15) as usize]
        }
    }
}

struct Parser {
    nodes: Vec<Node>,
    matches: Vec<(u32, u32)>,
    prices: Prices,
    since: usize,
}

impl Parser {
    fn new() -> Parser {
        Parser {
            nodes: vec![Node { price: INF, from: 0, tok: Tok::Lit, reps: [0; 4], state: 0 }; WIN + MAX_LEN + 2],
            matches: Vec::with_capacity(64),
            prices: Prices::new(),
            since: usize::MAX / 2,
        }
    }

    /// Chooses tokens for buf[pos0..end); returns them in order.
    /// `buf` = the `n` coded bytes plus the match finder's padding.
    fn window(&mut self, m: &Model, mf: &mut Mf, buf: &[u8], n: usize, pos0: usize, toks: &mut Vec<Tok>) {
        let t = price_table();
        if self.since >= 1024 {
            self.prices.update(m, t);
            self.since = 0;
        }
        let w = WIN.min(n - pos0);
        let span = (w + MAX_LEN + 1).min(n - pos0 + 1);
        let last = span.min(self.nodes.len() - 1);
        for nd in &mut self.nodes[..=last] {
            nd.price = INF;
        }
        self.nodes[0] = Node { price: 0, from: 0, tok: Tok::Lit, reps: m.reps, state: m.state as u8 };
        let mut end = w;
        let mut i = 0;
        while i < end {
            let nd = self.nodes[i];
            let pos = pos0 + i;
            mf.find(buf, n, pos, &mut self.matches);
            if nd.price == INF {
                i += 1;
                continue;
            }
            let s = nd.state as usize;
            let ps = pos & 3;
            let room = n - pos;
            let relax = |nodes: &mut Vec<Node>, j: usize, price: u32, tok: Tok| {
                if price < nodes[j].price {
                    nodes[j] = Node {
                        price,
                        from: i as u32,
                        tok,
                        reps: next_reps(nd.reps, tok),
                        state: next_state(s, tok) as u8,
                    };
                }
            };
            // literal
            let prev = if pos > 0 { buf[pos - 1] as u32 } else { 0 };
            let lp = nd.price + pr(t, m.is_match[s][ps], 0) + m.lit.price(t, buf[pos] as u32, prev);
            relax(&mut self.nodes, i + 1, lp, Tok::Lit);
            let mprice = nd.price + pr(t, m.is_match[s][ps], 1);
            let rprice = mprice + pr(t, m.is_rep[s], 1);
            // short rep
            let r0 = nd.reps[0] as usize + 1;
            if pos >= r0 && buf[pos] == buf[pos - r0] {
                let p = rprice + pr(t, m.is_rep0[s], 0) + pr(t, m.is_rep0_long[s][ps], 0);
                relax(&mut self.nodes, i + 1, p, Tok::Short);
            }
            // rep matches
            let mut longest = 0usize;
            let mut long_tok = Tok::Lit;
            for k in 0..4 {
                let r = nd.reps[k] as usize + 1;
                if pos < r || room < 2 {
                    continue;
                }
                let len = match_len(buf, pos - r, pos, room.min(MAX_LEN));
                if len < 2 {
                    continue;
                }
                let base = rprice
                    + match k {
                        0 => pr(t, m.is_rep0[s], 0) + pr(t, m.is_rep0_long[s][ps], 1),
                        1 => pr(t, m.is_rep0[s], 1) + pr(t, m.is_rep1[s], 0),
                        2 => pr(t, m.is_rep0[s], 1) + pr(t, m.is_rep1[s], 1) + pr(t, m.is_rep2[s], 0),
                        _ => pr(t, m.is_rep0[s], 1) + pr(t, m.is_rep1[s], 1) + pr(t, m.is_rep2[s], 1),
                    };
                if len >= mf.nice {
                    if len > longest {
                        longest = len;
                        long_tok = Tok::Rep(k as u8, len as u16);
                    }
                    continue;
                }
                for l in 2..=len {
                    relax(&mut self.nodes, i + l, base + self.prices.rlen[ps][l - 2], Tok::Rep(k as u8, l as u16));
                }
            }
            // normal matches
            let mbase = mprice + pr(t, m.is_rep[s], 0);
            let mut from_len = 3usize;
            for &(len, d) in &self.matches {
                let len = len as usize;
                if len >= mf.nice {
                    if len > longest {
                        longest = len;
                        long_tok = Tok::Match(len as u16, d);
                    }
                    break;
                }
                for l in from_len..=len {
                    relax(&mut self.nodes, i + l, mbase + self.prices.mlen[ps][l - 2] + self.prices.dist(d, l), Tok::Match(l as u16, d));
                }
                from_len = len + 1;
            }
            if longest > 0 {
                // take the long match: the window ends where it ends
                let j = i + longest;
                let p = nd.price; // cheap enough that nothing else should compete
                self.nodes[j] = Node { price: p, from: i as u32, tok: long_tok, reps: next_reps(nd.reps, long_tok), state: next_state(s, long_tok) as u8 };
                for q in pos + 1..pos + longest {
                    mf.insert(buf, n, q);
                }
                end = j;
                break;
            }
            i += 1;
        }
        // backtrack from `end`
        let start = toks.len();
        let mut j = end;
        while j > 0 {
            let nd = self.nodes[j];
            toks.push(nd.tok);
            j = nd.from as usize;
        }
        toks[start..].reverse();
        self.since += end;
    }
}

// ---------------------------------------------------------------- sub-block coding

/// Codes `n` bytes; `buf` = those bytes + NICE + 8 bytes of padding for the match finder.
fn encode_sub(buf: &[u8], n: usize, base: usize, x86ctx: bool) -> Vec<u8> {
    let mut m = Model::new(base, x86ctx);
    let mut mf = Mf::new(n);
    let mut parser = Parser::new();
    let mut enc = Encoder::new(n / 3 + 64);
    let mut toks = Vec::with_capacity(WIN);
    let mut pos = 0;
    while pos < n {
        toks.clear();
        parser.window(&m, &mut mf, buf, n, pos, &mut toks);
        for &t in &toks {
            let (out, _) = m.code(&mut enc, &buf[..n], pos, t);
            debug_assert_eq!(out, t);
            pos += t.len();
        }
    }
    enc.finish()
}

fn decode_sub(src: &[u8], n: usize, base: usize, x86ctx: bool) -> Result<Vec<u8>, String> {
    let mut m = Model::new(base, x86ctx);
    let mut dec = Decoder::new(src);
    let mut out: Vec<u8> = Vec::with_capacity(n);
    while out.len() < n {
        let pos = out.len();
        let (t, b) = m.code(&mut dec, &out, pos, Tok::Lit);
        if let Tok::Lit = t {
            out.push(b);
        } else {
            let d = match t {
                Tok::Match(_, d) => d as usize,
                _ => m.reps[0] as usize, // reps already rotated: the used distance is now reps[0]
            } + 1;
            let len = t.len();
            if d > pos || pos + len > n {
                return Err("corrupt LZ stream".into());
            }
            let start = pos - d;
            if d >= len {
                out.extend_from_within(start..start + len);
            } else {
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
        }
        if dec.overrun() {
            return Err("LZ stream truncated".into());
        }
    }
    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn encode_avx2(buf: &[u8], n: usize, base: usize, x: bool) -> Vec<u8> {
    encode_sub(buf, n, base, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn decode_avx2(src: &[u8], n: usize, base: usize, x: bool) -> Result<Vec<u8>, String> {
    decode_sub(src, n, base, x)
}

fn encode_one(buf: &[u8], n: usize, base: usize, x: bool) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { encode_avx2(buf, n, base, x) };
    }
    encode_sub(buf, n, base, x)
}

fn decode_one(src: &[u8], n: usize, base: usize, x: bool) -> Result<Vec<u8>, String> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { decode_avx2(src, n, base, x) };
    }
    decode_sub(src, n, base, x)
}

/// `sub_size:varint subs:varint {comp_len:varint}* streams...`
/// `x86ctx`: the data holds x86 code filtered in place (model branch operands).
pub fn encode(data: &[u8], threads: usize, x86ctx: bool) -> Vec<u8> {
    let sub = sub_size(data.len(), threads);
    let parts: Vec<(usize, &[u8])> =
        if data.is_empty() { vec![(0, data)] } else { data.chunks(sub).enumerate().map(|(i, c)| (i * sub, c)).collect() };
    let coded = crate::par::map(&parts, threads, |&(base, p)| {
        crate::par::with_memory(mem_for(p.len()), || {
            let mut buf = Vec::with_capacity(p.len() + NICE + 8);
            buf.extend_from_slice(p);
            buf.resize(p.len() + NICE + 8, 0);
            encode_one(&buf, p.len(), base, x86ctx)
        })
    });
    let mut out = Vec::with_capacity(coded.iter().map(|c| c.len()).sum::<usize>() + 8 * coded.len());
    crate::container::write_varint(&mut out, sub as u64);
    crate::container::write_varint(&mut out, coded.len() as u64);
    for c in &coded {
        crate::container::write_varint(&mut out, c.len() as u64);
    }
    for c in coded {
        out.extend_from_slice(&c);
    }
    out
}

pub fn decode(src: &[u8], n: usize, threads: usize, x86ctx: bool) -> Result<Vec<u8>, String> {
    let e = || "corrupt LZ block".to_string();
    let mut r = src;
    let sub = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
    if sub == 0 || sub > SUB_BLOCK {
        return Err(e());
    }
    let k = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
    if k != n.div_ceil(sub).max(1) {
        return Err(e());
    }
    let mut lens = Vec::with_capacity(k);
    for _ in 0..k {
        lens.push(crate::container::read_varint(&mut r).map_err(|_| e())? as usize);
    }
    let mut jobs = Vec::with_capacity(k);
    let mut off = 0usize;
    for (i, &l) in lens.iter().enumerate() {
        let s = r.get(off..off + l).ok_or_else(e)?;
        off += l;
        jobs.push((s, sub.min(n - (i * sub).min(n)), i * sub));
    }
    let parts = crate::par::map(&jobs, threads, |&(s, len, base)| crate::par::with_memory(len as u64 + (24 << 20), || decode_one(s, len, base, x86ctx)));
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend_from_slice(&p?);
    }
    Ok(out)
}

/// Memory to encode a sub-block of `len` bytes (match finder + model + parser).
fn mem_for(len: usize) -> u64 {
    (4 << H4_BITS) + (4 << H3_BITS) + 8 * len as u64 + (24 << 20)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut v: Vec<u8> = Vec::new();
        let mut r = 7u32;
        for i in 0..400_000u32 {
            r = r.wrapping_mul(1103515245).wrapping_add(12345);
            if i % 1000 < 600 {
                v.push((r >> 24) as u8 & 0x0f);
            } else {
                let k = v.len().saturating_sub((r >> 20) as usize % 5000 + 1);
                let b = v.get(k).copied().unwrap_or(0);
                v.push(b);
            }
        }
        v.extend(std::iter::repeat_n(7u8, 5000));
        for d in [&v[..], b"", b"a", b"abcabcabcabcabcabcabcabcabcabc"] {
            for x in [false, true] {
                let c = super::encode(d, 4, x);
                assert_eq!(super::decode(&c, d.len(), 3, x).unwrap(), d, "len {}", d.len());
            }
        }
    }
}

#[cfg(test)]
mod perf {
    #[test]
    #[ignore]
    fn split_timing() {
        let p = std::env::var("NZ_FILE").unwrap();
        let buf = std::fs::read(p).unwrap();
        let buf = &buf[..buf.len().min(8 << 20)];
        let t = std::time::Instant::now();
        let mut pb = buf.to_vec();
        pb.resize(buf.len() + super::NICE + 8, 0);
        let mut mf = super::Mf::new(buf.len());
        let mut out = Vec::new();
        let mut cnt = 0usize;
        for p in 0..buf.len() {
            mf.find(&pb, buf.len(), p, &mut out);
            cnt += out.len();
        }
        let tf = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let c = super::encode_sub(&pb, buf.len(), 0, false);
        let te = t.elapsed().as_secs_f64();
        eprintln!("find-all {tf:.2}s ({cnt} matches) | full encode {te:.2}s -> {} | {:.1} MB/s", c.len(), buf.len() as f64 / 1e6 / te);
    }
}

#[cfg(test)]
mod litperf {
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn enc(buf: &[u8]) -> Vec<u8> {
        let mut lit = super::Lit::new();
        let mut enc = crate::coder::Encoder::new(buf.len());
        for pos in 0..buf.len() {
            let prev = if pos > 0 { buf[pos - 1] as u32 } else { 0 };
            let mb = if pos >= 4 { buf[pos - 4] as u32 } else { 0 };
            lit.code(&mut enc, buf[pos] as u32, prev, super::h2_of(buf, pos), mb, pos, false, (0, 0));
        }
        enc.finish()
    }
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn dec(c: &[u8], n: usize) -> Vec<u8> {
        let mut lit = super::Lit::new();
        let mut dec = crate::coder::Decoder::new(c);
        let mut out = Vec::with_capacity(n);
        for pos in 0..n {
            let prev = if pos > 0 { out[pos - 1] as u32 } else { 0 };
            let mb = if pos >= 4 { out[pos - 4] as u32 } else { 0 };
            let b = lit.code(&mut dec, 0, prev, super::h2_of(&out, pos), mb, pos, false, (0, 0));
            out.push(b as u8);
        }
        out
    }
    /// Literal model alone over a file: `NZ_FILE=... cargo test --release lit_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn lit_timing() {
        let p = std::env::var("NZ_FILE").unwrap();
        let buf = std::fs::read(p).unwrap();
        let buf = &buf[..buf.len().min(4 << 20)];
        let t = std::time::Instant::now();
        let c = unsafe { enc(buf) };
        let te = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let out = unsafe { dec(&c, buf.len()) };
        let td = t.elapsed().as_secs_f64();
        assert!(out == buf);
        eprintln!("lit: {} -> {} | enc {:.0} ns/B | dec {:.0} ns/B", buf.len(), c.len(), te * 1e9 / buf.len() as f64, td * 1e9 / buf.len() as f64);
    }
}
