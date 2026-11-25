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

use crate::cm::{G, NG, hg};
use crate::coder::{BitIo, Decoder, Encoder};

/// Largest independent sub-block (sub-blocks run in parallel in both directions).
pub const SUB_BLOCK: usize = 64 << 20;

/// History each sub-block may match into from the sub-blocks before it. Decoding stays
/// parallel: a first pass decodes every sub-block at once with bytes from before it read
/// as zero (the encoder models the same view), a second, sequential pass copies them in.
const DICT: usize = 16 << 20;

/// Bytes of history before sub-block start `base`.
#[inline]
fn dict_len(base: usize) -> usize {
    base.min(DICT)
}

/// The decoder's first-pass view: literals as coded, copies copied from the view itself,
/// history before `d0` as zeros. Contexts come from it on both sides.
struct View {
    v: Vec<u8>,
}

impl View {
    fn new(n: usize, d0: usize) -> View {
        View { v: vec![0; n.max(d0)] }
    }
    /// After coding token `t` at `pos` (the model's reps already updated).
    #[inline(always)]
    fn apply(&mut self, buf: &[u8], pos: usize, t: Tok, reps0: u32) {
        match t {
            Tok::Lit => self.v[pos] = buf[pos],
            _ => {
                let d = reps0 as usize + 1;
                for k in pos..pos + t.len() {
                    self.v[k] = self.v[k - d];
                }
            }
        }
    }
}

/// Largest sub-block for effort-capped streams (see `encode`).
const FAST_SUB: usize = 16 << 20;

/// Bytes of work per thread in the batch being compressed (set by the container for each
/// wave of blocks; 0 = unknown).
static SHARE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_share(bytes: usize) {
    SHARE.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Arrivals the parser keeps per position (1, 2 or 4), from the threads compressing: more
/// paths give ~1% smaller output for 1.4-2.5x the parse time, so they are only used where
/// there are cores to spare. Dev override: NZ_ARR.
static ARRIVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);

pub fn set_effort(threads: usize) {
    let a = if threads >= 16 { 4 } else if threads >= 6 { 2 } else { 1 };
    ARRIVALS.store(a, std::sync::atomic::Ordering::Relaxed);
}

/// Arrivals for a stream: `arr` caps the machine's effort (0 = no cap).
fn effort(arr: usize) -> usize {
    if arr == 0 { arrivals() } else { arrivals().min(arr) }
}

fn arrivals() -> usize {
    match crate::sys::dev_var("NZ_ARR").and_then(|v| v.parse::<usize>().ok()) {
        Some(a) => a,
        None => ARRIVALS.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Sub-block size for a stream of `n` bytes: about one thread's share of the whole batch,
/// two for machine code (it reuses distant history, so bigger sub-blocks pay off more);
/// 256 KiB - 64 MiB. Dev override: NZ_SUB in MiB.
fn sub_size(n: usize, threads: usize, code: bool) -> usize {
    if let Some(v) = crate::sys::dev_var("NZ_SUB").and_then(|v| v.parse::<usize>().ok()) {
        return (v << 20).clamp(1 << 20, SUB_BLOCK);
    }
    let share = match SHARE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => n.div_ceil(threads.max(1)),
        s => s,
    };
    // machine code: two shares, but at least 4 pieces once the stream is big, so one
    // block's decode never becomes the long pole
    let share = if !code { share } else if n > 32 << 20 { (2 * share).min(n.div_ceil(4)) } else { 2 * share };
    share.next_multiple_of(64 << 10).clamp(256 << 10, SUB_BLOCK).min(n.max(1))
}
const MAX_LEN: usize = 273;
const WIN: usize = 2048;
const H4_BITS: u32 = 22;
const H3_BITS: u32 = 16;
/// Binary-tree search depth and "nice" length: deeper / longer = smaller but slower.
const DEPTH: usize = 48;
const NICE: usize = 64;
const STATES: usize = 12;
/// Parser also tries match/rep + literal + rep0 (LZMA's lookahead).
const LOOKAHEAD: bool = true;

// ---------------------------------------------------------------- bit models

#[derive(Clone, Copy)]
struct Prob(u16);

impl Prob {
    const INIT: Prob = Prob(32768);
    #[inline(always)]
    fn p(self) -> u32 {
        (self.0 as u32).clamp(64, 65535 - 64)
    }
    /// Rate 1/32, faster while young (count in the low 3 bits, as `Ctr::update_young`).
    #[inline(always)]
    fn update(&mut self, b: u32) {
        let n = (self.0 & 7) as u32;
        let r = (n + crate::cm::YOUNG_START).min(5);
        let v = (self.0 & !7) as i32;
        let t = ((b << 16) - b) as i32;
        let q = (v + ((t - v) >> r)) as u16;
        self.0 = (q & !7) | (n + 1).min(7) as u16;
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
    if nbits <= 16 {
        return io.direct(val & ((1 << nbits) - 1), nbits);
    }
    let lo = nbits - 16;
    let hi = io.direct((val >> lo) & 0xFFFF, 16);
    (hi << lo) | io.direct(val & ((1 << lo) - 1), lo)
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

#[derive(Clone)]
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

#[derive(Clone)]
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

/// Literal model: two counters per bit, a fast one and a slow twin (rates LRATE and
/// LRATE + 2), averaged. Their context is the match byte for the literal right after a
/// match, the operand context inside the rel32 target of an x86 branch, else the previous
/// byte; the second nibble's counters are also keyed by the first nibble.
#[derive(Clone)]
struct Lit {
    t1: Vec<G>,
    s1: Vec<G>,
    tm: Vec<G>,
    sm: Vec<G>,
    tx: Vec<G>,
    sx: Vec<G>,
}

/// Counter rate (shift) of the fast counters.
const LRATE: u32 = 4;
const TX_BITS: u32 = 18;

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
            t1: vec![G::INIT; 256 * NG],
            s1: vec![G::INIT; 256 * NG],
            tm: vec![G::INIT; 256 * NG],
            sm: vec![G::INIT; 256 * NG],
            tx: vec![G::INIT; 1 << TX_BITS],
            sx: vec![G::INIT; 1 << TX_BITS],
        }
    }

    /// Index of the counter groups for nibble selector `sel`, and which pair of tables
    /// holds them (0 = previous byte, 1 = match byte, 2 = x86 operand).
    #[inline(always)]
    fn slot(sel: usize, prev: u32, mb: u32, matched: bool, xc: Option<u32>) -> (u8, usize) {
        match xc {
            _ if matched => (1, mb as usize * NG + sel),
            Some(x) => (2, hg(x + sel as u32, TX_BITS)),
            None => (0, prev as usize * NG + sel),
        }
    }

    #[inline(always)]
    fn pair(&self, t: u8, i: usize) -> (&G, &G) {
        match t {
            1 => (&self.tm[i], &self.sm[i]),
            2 => (&self.tx[i], &self.sx[i]),
            _ => (&self.t1[i], &self.s1[i]),
        }
    }

    #[inline(always)]
    fn pair_mut(&mut self, t: u8, i: usize) -> (&mut G, &mut G) {
        match t {
            1 => (&mut self.tm[i], &mut self.sm[i]),
            2 => (&mut self.tx[i], &mut self.sx[i]),
            _ => (&mut self.t1[i], &mut self.s1[i]),
        }
    }

    /// Hash seed of the x86 operand context (None outside branch targets).
    #[inline(always)]
    fn xc(op: (usize, u32)) -> Option<u32> {
        let (k, octx) = op;
        (k != 0).then(|| (((k as u32) << 16) | octx).wrapping_mul(NG as u32))
    }

    #[inline(always)]
    fn p(a: &G, b: &G, j: usize) -> u32 {
        ((a.0[j].p() + b.0[j].p()) >> 1).clamp(48, 65535 - 48)
    }

    /// Codes (or decodes, per `io`) one literal; `mb` = match byte, `op` = op_ctx.
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, byte: u32, prev: u32, mb: u32, matched: bool, op: (usize, u32)) -> u32 {
        let xc = Self::xc(op);
        let mut c0 = 1usize;
        for half in 0..2 {
            let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
            let (t, i) = Self::slot(sel, prev, mb, matched, xc);
            let (fa, sa) = self.pair_mut(t, i);
            let mut j = 1usize;
            for q in 0..4 {
                let b = io.code((byte >> (7 - (half * 4 + q))) & 1, Self::p(fa, sa, j));
                fa.0[j].update_young(b, LRATE);
                sa.0[j].update_young(b, LRATE + 2);
                j = (j << 1) | b as usize;
                c0 = (c0 << 1) | b as usize;
            }
        }
        (c0 & 255) as u32
    }

    /// Cost (1/16 bit) of `byte` under the model as it stands (no updates).
    #[inline(always)]
    fn cost(&self, t: &[u32], byte: u32, prev: u32, mb: u32, matched: bool, op: (usize, u32)) -> u32 {
        let xc = Self::xc(op);
        let mut c0 = 1usize;
        let mut s = 0;
        for half in 0..2 {
            let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
            let (tb, i) = Self::slot(sel, prev, mb, matched, xc);
            let (fa, sa) = self.pair(tb, i);
            let mut j = 1usize;
            for q in 0..4 {
                let b = (byte >> (7 - (half * 4 + q))) & 1;
                s += pr(t, Prob(Self::p(fa, sa, j) as u16), b);
                j = (j << 1) | b as usize;
                c0 = (c0 << 1) | b as usize;
            }
        }
        s
    }
}

// ---------------------------------------------------------------- dense segments

/// Segments of 2^SEG_BITS bytes are coded either normally (LZ tokens) or, when they hold
/// no repeats (packed weights, already-compressed data...), as literals only: the encoder
/// then skips match finding and parsing. One adaptive bit per segment says which.
const SEG_BITS: u32 = 16;

/// Encoder: which segments of `buf` to code as dense (few repeats, high entropy).
fn dense_segments(buf: &[u8]) -> Vec<bool> {
    const H: u32 = 16;
    let seg = 1usize << SEG_BITS;
    let mut table = vec![u32::MAX; 1 << H];
    let mut covered = vec![0usize; buf.len().div_ceil(seg)];
    let mut p = 0usize;
    while p + 8 <= buf.len() {
        let v = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
        let h = (v.wrapping_mul(0x9E37_79B1) >> (32 - H)) as usize;
        let c = table[h] as usize;
        table[h] = p as u32;
        if c != u32::MAX as usize && buf[c..c + 5] == buf[p..p + 5] {
            let l = match_len(buf, c, p, (buf.len() - p).min(256));
            covered[p >> SEG_BITS] += l;
            p += l;
        } else {
            p += 1;
        }
    }
    covered
        .iter()
        .enumerate()
        .map(|(i, &cov)| {
            let s = &buf[i * seg..((i + 1) * seg).min(buf.len())];
            cov * 16 < s.len() && crate::codec::entropy_milli(s) > 4500
        })
        .collect()
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

#[derive(Clone)]
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
    /// dense-segment state: flags (encoder only), end of the current segment, its mode
    flags: Vec<bool>,
    seg_end: usize,
    dense: bool,
    is_dense: [Prob; 2],
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
            flags: Vec::new(),
            seg_end: 0,
            dense: false,
            is_dense: [Prob::INIT; 2],
        }
    }

    /// Codes one token at `pos`. `buf` holds every byte before `pos` (and, when encoding,
    /// the bytes after it). Returns the token and, for a literal, its byte.
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, buf: &[u8], pos: usize, t: Tok) -> (Tok, u8) {
        let s = self.state;
        let ps = pos & 3;
        let lit_byte = if matches!(t, Tok::Lit) && pos < buf.len() { buf[pos] as u32 } else { 0 };
        if pos >= self.seg_end {
            let seg = pos >> SEG_BITS;
            self.seg_end = (seg + 1) << SEG_BITS;
            let want = self.flags.get(seg).copied().unwrap_or(false) as u32;
            self.dense = bit(io, &mut self.is_dense[self.dense as usize], want) == 1;
        }
        let out;
        let mut byte = 0u8;
        if self.dense || bit(io, &mut self.is_match[s][ps], (t != Tok::Lit) as u32) == 0 {
            let prev = if pos > 0 { buf[pos - 1] as u32 } else { 0 };
            let r0 = self.reps[0] as usize + 1;
            let mb = if pos >= r0 { buf[pos - r0] as u32 } else { 0 };
            let op = if self.x86ctx { op_ctx(buf, pos, self.base) } else { (0, 0) };
            byte = self.lit.code(io, lit_byte, prev, mb, s >= 7, op) as u8;
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
    from_a: u8,
    tok: Tok,
    reps: [u32; 4],
    state: u8,
    /// reached by `first`, one literal, then `tok` (a rep0), when set
    chain: bool,
    first: Tok,
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

struct Parser<const A: usize> {
    /// `A` cheapest arrivals per position, sorted by price
    nodes: Vec<[Node; A]>,
    matches: Vec<(u32, u32)>,
    prices: Prices,
    since: usize,
}

/// Arrivals dearer than the best one at a position by more than this (1/16 bit) are dropped.
const TOL: u32 = 256;
/// Arrivals that also try normal matches.
const MATCH_ARR: usize = 1;
/// Arrivals that also try the literal + rep0 lookahead.
const LOOK_ARR: usize = 1;

const EMPTY: Node = Node { price: INF, from: 0, from_a: 0, tok: Tok::Lit, reps: [0; 4], state: 0, chain: false, first: Tok::Lit };

/// Inserts an arrival into a position's sorted list (one arrival per (rep0, state)).
#[inline(always)]
fn insert<const A: usize>(list: &mut [Node; A], nd: Node) {
    // dearer than every kept arrival (including any with the same state): nothing to do
    if nd.price >= list[A - 1].price {
        return;
    }
    let mut at = A;
    for (k, e) in list.iter().enumerate() {
        if e.price == INF {
            break;
        }
        if e.reps[0] == nd.reps[0] && e.state == nd.state {
            if nd.price >= e.price {
                return;
            }
            at = k;
            break;
        }
    }
    if at == A {
        if nd.price >= list[A - 1].price {
            return;
        }
        at = A - 1;
    }
    // remove slot `at`, then insert in order
    let mut k = at;
    while k > 0 && list[k - 1].price > nd.price {
        list[k] = list[k - 1];
        k -= 1;
    }
    list[k] = nd;
}

impl<const A: usize> Parser<A> {
    fn new() -> Parser<A> {
        Parser { nodes: vec![[EMPTY; A]; WIN + 2 * MAX_LEN + 4], matches: Vec::with_capacity(64), prices: Prices::new(), since: usize::MAX / 2 }
    }

    /// Chooses tokens for buf[pos0..end); returns them in order.
    /// `buf` = the `n` coded bytes plus the match finder's padding.
    /// Tokens start before `limit` (the next dense segment).
    #[allow(clippy::too_many_arguments)]
    fn window(&mut self, m: &Model, src: &mut MatchSrc, buf: &[u8], n: usize, limit: usize, pos0: usize, toks: &mut Vec<Tok>) {
        let t = price_table();
        if self.since >= 1024 {
            self.prices.update(m, t);
            self.since = 0;
        }
        let w = WIN.min(limit - pos0);
        let span = (w + MAX_LEN + 1).min(n - pos0 + 1);
        let last = span.min(self.nodes.len() - 1);
        for nd in &mut self.nodes[..=last] {
            *nd = [EMPTY; A];
        }
        self.nodes[0][0] = Node { price: 0, reps: m.reps, state: m.state as u8, ..EMPTY };
        let mut end = w;
        let mut i = 0;
        while i < end {
            let pos = pos0 + i;
            match src {
                MatchSrc::Live(mf) => mf.find(buf, n, pos, &mut self.matches),
                MatchSrc::Lists(l) => {
                    self.matches.clear();
                    for &(len, d) in l.at(pos) {
                        let len = len.min((n - pos) as u32);
                        if len >= 3 && self.matches.last().is_none_or(|x| x.0 < len) {
                            self.matches.push((len, d));
                        }
                    }
                }
            }
            if self.nodes[i][0].price == INF {
                i += 1;
                continue;
            }
            let ps = pos & 3;
            let room = n - pos;
            let mut longest = 0usize;
            let mut long_tok = Tok::Lit;
            let prev = if pos > 0 { buf[pos - 1] as u32 } else { 0 };
            let mut lit_memo = [(u32::MAX, 0u32); A];
            let mut rep_memo = [(0usize, 0usize); 8];
            let mut memo_n = 0usize;
            for a in 0..A {
                let nd = self.nodes[i][a];
                if nd.price == INF {
                    break;
                }
                let s = nd.state as usize;
                let prices = &self.prices;
                let nodes = &mut self.nodes;
                let relax = |nodes: &mut Vec<[Node; A]>, j: usize, price: u32, tok: Tok| {
                    // far dearer than the best way here: cannot win later
                    if price >= nodes[j][A - 1].price || price > nodes[j][0].price.saturating_add(TOL) {
                        return;
                    }
                    insert(&mut nodes[j], Node { price, from: i as u32, from_a: a as u8, tok, reps: next_reps(nd.reps, tok), state: next_state(s, tok) as u8, chain: false, first: Tok::Lit });
                };
                // `first`, one literal, then rep0 again: LZMA's two-step lookahead
                let lookahead = |nodes: &mut Vec<[Node; A]>, first: Tok, base: u32| {
                    if !LOOKAHEAD || a >= LOOK_ARR {
                        return;
                    }
                    let l = first.len();
                    let at = pos + l; // the literal
                    if at + 3 > n {
                        return;
                    }
                    let reps1 = next_reps(nd.reps, first);
                    let d = reps1[0] as usize + 1;
                    if at < d || buf[at] == buf[at - d] {
                        return;
                    }
                    let cap = (n - at - 1).min(MAX_LEN).min(last.saturating_sub(i + l + 1));
                    if cap < 2 {
                        return;
                    }
                    let l2 = match_len(buf, at + 1 - d, at + 1, cap);
                    if l2 < 2 {
                        return;
                    }
                    let s1 = next_state(s, first);
                    let lit = m.lit.cost(t, buf[at] as u32, buf[at - 1] as u32, buf[at - d] as u32, true, if m.x86ctx { op_ctx(buf, at, m.base) } else { (0, 0) });
                    let s2 = next_state(s1, Tok::Lit);
                    let ps2 = (at + 1) & 3;
                    let price = base
                        + pr(t, m.is_match[s1][at & 3], 0)
                        + lit
                        + pr(t, m.is_match[s2][ps2], 1)
                        + pr(t, m.is_rep[s2], 1)
                        + pr(t, m.is_rep0[s2], 0)
                        + pr(t, m.is_rep0_long[s2][ps2], 1)
                        + prices.rlen[ps2][l2 - 2];
                    let tok = Tok::Rep(0, l2 as u16);
                    insert(
                        &mut nodes[i + l + 1 + l2],
                        Node { price, from: i as u32, from_a: a as u8, tok, reps: next_reps(next_reps(reps1, Tok::Lit), tok), state: next_state(s2, tok) as u8, chain: true, first },
                    );
                };
                // literal (its price depends only on the match byte and the matched flag)
                let lit = {
                    let r0 = nd.reps[0] as usize + 1;
                    let mb = if pos >= r0 { buf[pos - r0] as u32 } else { 0 };
                    let key = mb | ((s >= 7) as u32) << 8;
                    match lit_memo.iter().find(|e| e.0 == key) {
                        Some(e) => e.1,
                        None => {
                            let op = if m.x86ctx { op_ctx(buf, pos, m.base) } else { (0, 0) };
                            let c = m.lit.cost(t, buf[pos] as u32, prev, mb, s >= 7, op);
                            lit_memo[a] = (key, c);
                            c
                        }
                    }
                };
                relax(nodes, i + 1, nd.price + pr(t, m.is_match[s][ps], 0) + lit, Tok::Lit);
                let mprice = nd.price + pr(t, m.is_match[s][ps], 1);
                let rprice = mprice + pr(t, m.is_rep[s], 1);
                // short rep
                let r0 = nd.reps[0] as usize + 1;
                if pos >= r0 && buf[pos] == buf[pos - r0] {
                    relax(nodes, i + 1, rprice + pr(t, m.is_rep0[s], 0) + pr(t, m.is_rep0_long[s][ps], 0), Tok::Short);
                }
                // rep matches
                for k in 0..4 {
                    let r = nd.reps[k] as usize + 1;
                    if pos < r || room < 2 {
                        continue;
                    }
                    let len = match rep_memo.iter().find(|e| e.0 == r) {
                        Some(e) => e.1,
                        None => {
                            let l = match_len(buf, pos - r, pos, room.min(MAX_LEN));
                            rep_memo[memo_n & 7] = (r, l);
                            memo_n += 1;
                            l
                        }
                    };
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
                    if len >= NICE {
                        if a == 0 && len > longest {
                            longest = len;
                            long_tok = Tok::Rep(k as u8, len as u16);
                        }
                        continue;
                    }
                    for l in 2..=len {
                        relax(nodes, i + l, base + prices.rlen[ps][l - 2], Tok::Rep(k as u8, l as u16));
                    }
                    lookahead(nodes, Tok::Rep(k as u8, len as u16), base + prices.rlen[ps][len - 2]);
                }
                // normal matches: from the cheapest arrival only (they all lead to the
                // same state; only the price differs)
                let mbase = mprice + pr(t, m.is_rep[s], 0);
                let mut from_len = 3usize;
                let none: &[(u32, u32)] = &[];
                for &(len, d) in if a < MATCH_ARR { &self.matches[..] } else { none } {
                    let len = len as usize;
                    if len >= NICE {
                        if a == 0 && len > longest {
                            longest = len;
                            long_tok = Tok::Match(len as u16, d);
                        }
                        break;
                    }
                    for l in from_len..=len {
                        relax(nodes, i + l, mbase + prices.mlen[ps][l - 2] + prices.dist(d, l), Tok::Match(l as u16, d));
                    }
                    if from_len <= len {
                        lookahead(nodes, Tok::Match(len as u16, d), mbase + prices.mlen[ps][len - 2] + prices.dist(d, len));
                    }
                    from_len = len + 1;
                }
                if longest > 0 {
                    break;
                }
            }
            if longest > 0 {
                // take the long match from the cheapest arrival: the window ends where it ends
                let nd = self.nodes[i][0];
                let j = i + longest;
                let s = nd.state as usize;
                self.nodes[j] = [EMPTY; A];
                self.nodes[j][0] = Node { price: nd.price, from: i as u32, from_a: 0, tok: long_tok, reps: next_reps(nd.reps, long_tok), state: next_state(s, long_tok) as u8, chain: false, first: Tok::Lit };
                if let MatchSrc::Live(mf) = src {
                    for q in pos + 1..pos + longest {
                        mf.insert(buf, n, q);
                    }
                }
                end = j;
                break;
            }
            i += 1;
        }
        // backtrack from the cheapest arrival at `end`
        let start = toks.len();
        let (mut j, mut a) = (end, 0usize);
        while j > 0 {
            let nd = self.nodes[j][a];
            toks.push(nd.tok);
            if nd.chain {
                toks.push(Tok::Lit);
                toks.push(nd.first);
            }
            j = nd.from as usize;
            a = nd.from_a as usize;
        }
        toks[start..].reverse();
        self.since += end;
    }
}

// ---------------------------------------------------------------- sub-block coding

/// Where the parser gets its matches: a live match finder, or lists found ahead of time.
enum MatchSrc<'a> {
    Live(&'a mut Mf),
    Lists(&'a Lists),
}

/// Match lists (len, dist - 1) for positions [start, start + idx.len() - 1).
struct Lists {
    start: usize,
    idx: Vec<u32>,
    m: Vec<(u32, u32)>,
}

impl Lists {
    #[inline(always)]
    fn at(&self, pos: usize) -> &[(u32, u32)] {
        let i = pos - self.start;
        &self.m[self.idx[i] as usize..self.idx[i + 1] as usize]
    }
}

/// Bit sink that only replays the model (prices for a parse that is coded later).
struct NullIo;

impl BitIo for NullIo {
    #[inline(always)]
    fn code(&mut self, bit: u32, _p: u32) -> u32 {
        bit
    }
    #[inline(always)]
    fn direct(&mut self, v: u32, _k: u32) -> u32 {
        v
    }
}

/// A parsed token with an explicit distance (dist - 1); 0-length = literal.
#[derive(Clone, Copy)]
struct Out {
    len: u16,
    d: u32,
    /// the token the parser chose (kept when the coder's rep distances agree)
    tok: Tok,
}

/// Segment size for parallel parsing inside one sub-block.
const PSEG: usize = 2 << 20;
/// Segments whose match lists are found per round (bounds memory).
const PROUND: usize = 4;

/// Finds the match lists of positions [s, e) (dense segments get empty lists).
fn find_lists(mf: &mut Mf, buf: &[u8], n: usize, flags: &[bool], s: usize, e: usize) -> Lists {
    let mut idx = Vec::with_capacity(e - s + 1);
    let mut m = Vec::with_capacity((e - s) * 2);
    let mut tmp = Vec::with_capacity(64);
    idx.push(0);
    // inside a long match (>= NICE) the parser takes that match and skips ahead, so the
    // positions it covers need no search: they continue the same match (same distance,
    // the remaining length), and only go into the tree
    let mut run: Option<(usize, u32)> = None;
    for p in s..e {
        if !flags[p >> SEG_BITS] {
            match run {
                Some((end, d)) if p + NICE <= end => {
                    mf.insert(buf, n, p);
                    m.push(((end - p) as u32, d));
                }
                _ => {
                    mf.find(buf, n, p, &mut tmp);
                    m.extend_from_slice(&tmp);
                    run = tmp.last().filter(|x| x.0 as usize >= 2 * NICE).map(|x| (p + x.0 as usize, x.1));
                }
            }
        }
        idx.push(m.len() as u32);
    }
    Lists { start: s, idx, m }
}

/// Parses [s, e) with its own pricing model (warmed on the bytes just before `s`) and
/// returns tokens with explicit distances; the caller codes them in order.
fn parse_seg(buf: &[u8], n: usize, start: &Model, lists: &Lists, s: usize, e: usize, arr: usize) -> Vec<Out> {
    match effort(arr) {
        4 => parse_seg_a::<4>(buf, n, start, lists, s, e),
        2 => parse_seg_a::<2>(buf, n, start, lists, s, e),
        _ => parse_seg_a::<1>(buf, n, start, lists, s, e),
    }
}

fn parse_seg_a<const A: usize>(buf: &[u8], n: usize, start: &Model, lists: &Lists, s: usize, e: usize) -> Vec<Out> {
    // prices come from the coder's model as it stood when this round began
    let mut m = start.clone();
    let flags = &start.flags;
    let mut io = NullIo;
    m.seg_end = s; // the dense bit is replayed from `s`
    let mut parser = Parser::<A>::new();
    let mut toks = Vec::with_capacity(WIN);
    let mut out = Vec::with_capacity((e - s) / 3);
    let mut pos = s;
    while pos < e {
        let seg = pos >> SEG_BITS;
        if flags[seg] {
            let end = ((seg + 1) << SEG_BITS).min(e);
            while pos < end {
                m.code(&mut io, &buf[..n], pos, Tok::Lit);
                out.push(Out { len: 0, d: 0, tok: Tok::Lit });
                pos += 1;
            }
            continue;
        }
        let limit = flags[seg..].iter().position(|&d| d).map_or(n, |k| (seg + k) << SEG_BITS).min(e);
        toks.clear();
        parser.window(&m, &mut MatchSrc::Lists(lists), buf, e, limit, pos, &mut toks);
        for &t in &toks {
            let o = match t {
                Tok::Lit => Out { len: 0, d: 0, tok: t },
                Tok::Short => Out { len: 1, d: m.reps[0], tok: t },
                Tok::Rep(k, l) => Out { len: l, d: m.reps[k as usize], tok: t },
                Tok::Match(l, d) => Out { len: l, d, tok: t },
            };
            m.code(&mut io, &buf[..n], pos, t);
            out.push(o);
            pos += t.len();
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn parse_seg_avx2(buf: &[u8], n: usize, start: &Model, lists: &Lists, s: usize, e: usize, arr: usize) -> Vec<Out> {
    parse_seg(buf, n, start, lists, s, e, arr)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn encode_mt_avx2(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool, threads: usize, arr: usize) -> Vec<u8> {
    encode_sub_mt(buf, n, d0, base, x86ctx, threads, arr)
}


/// Like `encode_sub`, but the parse runs on several threads: match lists are found in
/// order (the match finder sees all earlier data), segments of PSEG bytes are parsed in
/// parallel, and one final pass codes every token with the real model. The decoder is
/// the same; only the encoder's choices can differ slightly at segment starts.
/// `buf[..d0]` is history from earlier sub-blocks (matched into, never coded); bytes
/// [d0, n) are coded. `base` = stream offset of buf[0].
fn encode_sub_mt(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool, threads: usize, arr: usize) -> Vec<u8> {
    let flags = dense_segments(&buf[..n]);
    let mut m = Model::new(base, x86ctx);
    m.flags = flags.clone();
    let mut enc = Encoder::new((n - d0) / 3 + 64);
    let mut view = View::new(n, d0);
    let segs: Vec<(usize, usize)> = (0..(n - d0).div_ceil(PSEG)).map(|i| (d0 + i * PSEG, (d0 + (i + 1) * PSEG).min(n))).collect();
    let mut pos = d0;
    // the GPU finds every position's matches at once when it can (else the CPU match
    // finder runs ahead on its own thread); history segments need no lists
    let gpu = if !crate::sys::dev_flag("NZ_NOGPUMF") {
        let skip: Vec<bool> = flags.iter().enumerate().map(|(i, &f)| f || (i + 1) << SEG_BITS <= d0).collect();
        crate::gpu::match_lists(&buf[..n], SEG_BITS, &skip).map(|(idx, m)| Lists { start: 0, idx, m })
    } else {
        None
    };
    std::thread::scope(|sc| {
        // the match finder runs ahead on its own thread (it only depends on the data)
        let (tx, rx) = std::sync::mpsc::sync_channel::<Lists>(PROUND);
        let fl = &flags;
        let sg = &segs;
        let on_gpu = gpu.is_some();
        sc.spawn(move || {
            if on_gpu {
                return;
            }
            let mut mf = Mf::new(n);
            for p in 0..d0 {
                if !fl[p >> SEG_BITS] {
                    mf.insert(buf, n, p);
                }
            }
            for &(s, e) in sg {
                if tx.send(find_lists(&mut mf, buf, n, fl, s, e)).is_err() {
                    return;
                }
            }
        });
        for round in segs.chunks(PROUND) {
            let lists: Vec<Lists> = if on_gpu { Vec::new() } else { round.iter().map(|_| rx.recv().expect("match finder stopped")).collect() };
            let gl = gpu.as_ref();
            let jobs: Vec<usize> = (0..round.len()).collect();
            let start = &m;
            let outs = crate::par::map(&jobs, threads, |&i| {
                #[cfg(target_arch = "x86_64")]
                if crate::codec::has_avx2() {
                    return unsafe { parse_seg_avx2(buf, n, start, gl.unwrap_or_else(|| &lists[i]), round[i].0, round[i].1, arr) };
                }
                parse_seg(buf, n, start, gl.unwrap_or_else(|| &lists[i]), round[i].0, round[i].1, arr)
            });
            for o in outs.iter().flatten() {
                let t = match o.tok {
                    // the parser's own choice when it is still valid here
                    Tok::Lit | Tok::Match(..) => o.tok,
                    Tok::Short if m.reps[0] == o.d => o.tok,
                    Tok::Rep(k, _) if m.reps[k as usize] == o.d => o.tok,
                    // segment start: the coder's rep distances differ from the parser's
                    Tok::Short => Tok::Lit,
                    _ => match m.reps.iter().position(|&r| r == o.d) {
                        Some(k) => Tok::Rep(k as u8, o.len),
                        None => Tok::Match(o.len, o.d),
                    },
                };
                if t == Tok::Lit {
                    view.v[pos] = buf[pos];
                }
                m.code(&mut enc, &view.v[..n], pos, t);
                view.apply(buf, pos, t, m.reps[0]);
                pos += t.len();
            }
        }
    });
    debug_assert_eq!(pos, n);
    enc.finish()
}

/// Codes `n` bytes; `buf` = those bytes + NICE + 8 bytes of padding for the match finder.
fn encode_sub(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool, arr: usize) -> Vec<u8> {
    match effort(arr) {
        4 => encode_sub_a::<4>(buf, n, d0, base, x86ctx),
        2 => encode_sub_a::<2>(buf, n, d0, base, x86ctx),
        _ => encode_sub_a::<1>(buf, n, d0, base, x86ctx),
    }
}

fn encode_sub_a<const A: usize>(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool) -> Vec<u8> {
    let mut m = Model::new(base, x86ctx);
    let mut mf = Mf::new(n);
    let mut parser = Parser::<A>::new();
    let mut enc = Encoder::new((n - d0) / 3 + 64);
    let mut toks = Vec::with_capacity(WIN);
    let mut view = View::new(n, d0);
    m.flags = dense_segments(&buf[..n]);
    for p in 0..d0 {
        if !m.flags[p >> SEG_BITS] {
            mf.insert(buf, n, p);
        }
    }
    let mut pos = d0;
    while pos < n {
        let seg = pos >> SEG_BITS;
        if m.flags[seg] {
            let end = ((seg + 1) << SEG_BITS).min(n);
            while pos < end {
                view.v[pos] = buf[pos];
                m.code(&mut enc, &view.v[..n], pos, Tok::Lit);
                pos += 1;
            }
            continue;
        }
        let limit = m.flags[seg..].iter().position(|&d| d).map_or(n, |k| (seg + k) << SEG_BITS);
        toks.clear();
        parser.window(&m, &mut MatchSrc::Live(&mut mf), buf, n, limit, pos, &mut toks);
        for &t in &toks {
            if t == Tok::Lit {
                view.v[pos] = buf[pos];
            }
            let (out, _) = m.code(&mut enc, &view.v[..n], pos, t);
            debug_assert_eq!(out, t);
            view.apply(buf, pos, t, m.reps[0]);
            pos += t.len();
        }
    }
    enc.finish()
}

/// A copy the first decoding pass could not finish (its source holds bytes from before
/// the sub-block): (offset in the sub-block, distance, length).
type Fixup = (u32, u32, u32);

/// First pass for one sub-block of `n` bytes with `d0` bytes of history before it (read as
/// zeros): returns its bytes and the copies the second pass must redo, in order.
fn decode_sub(src: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool) -> Result<(Vec<u8>, Vec<Fixup>), String> {
    let mut m = Model::new(base, x86ctx);
    let mut dec = Decoder::new(src);
    let total = d0 + n;
    let mut out: Vec<u8> = Vec::with_capacity(total);
    out.resize(d0, 0);
    // which bytes hold their real value (only tracked when there is history)
    let mut known: Vec<u8> = if d0 > 0 { vec![0u8; d0] } else { Vec::new() };
    let mut fix: Vec<Fixup> = Vec::new();
    while out.len() < total {
        let pos = out.len();
        let (t, b) = m.code(&mut dec, &out, pos, Tok::Lit);
        if let Tok::Lit = t {
            out.push(b);
            if d0 > 0 {
                known.push(1);
            }
        } else {
            let d = match t {
                Tok::Match(_, d) => d as usize,
                _ => m.reps[0] as usize, // reps already rotated: the used distance is now reps[0]
            } + 1;
            let len = t.len();
            if d > pos || pos + len > total {
                return Err("corrupt LZ stream".into());
            }
            let start = pos - d;
            if d >= len {
                out.extend_from_within(start..start + len);
            } else {
                // overlapping copy: repeat the d-byte period
                out.resize(pos + len, 0);
                for k in pos..pos + len {
                    out[k] = out[k - d];
                }
            }
            if d0 > 0 {
                if d >= len {
                    known.extend_from_within(start..start + len);
                } else {
                    known.resize(pos + len, 0);
                    for k in pos..pos + len {
                        known[k] = known[k - d];
                    }
                }
                if known[pos..pos + len].contains(&0) {
                    fix.push(((pos - d0) as u32, d as u32, len as u32));
                }
            }
        }
        if dec.overrun() {
            return Err("LZ stream truncated".into());
        }
    }
    out.drain(..d0);
    Ok((out, fix))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn encode_avx2(buf: &[u8], n: usize, d0: usize, base: usize, x: bool, arr: usize) -> Vec<u8> {
    encode_sub(buf, n, d0, base, x, arr)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn decode_avx2(src: &[u8], n: usize, d0: usize, base: usize, x: bool) -> Result<(Vec<u8>, Vec<Fixup>), String> {
    decode_sub(src, n, d0, base, x)
}

fn encode_one(buf: &[u8], n: usize, d0: usize, base: usize, x: bool, arr: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { encode_avx2(buf, n, d0, base, x, arr) };
    }
    encode_sub(buf, n, d0, base, x, arr)
}

fn decode_one(src: &[u8], n: usize, d0: usize, base: usize, x: bool) -> Result<(Vec<u8>, Vec<Fixup>), String> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { decode_avx2(src, n, d0, base, x) };
    }
    decode_sub(src, n, d0, base, x)
}

/// `sub_size:varint subs:varint {comp_len:varint}* streams...`
/// `x86ctx`: the data holds x86 code filtered in place (model branch operands).
/// `code`: the data is machine code (sub-blocks may be larger).
pub fn encode(data: &[u8], threads: usize, x86ctx: bool, code: bool, arr: usize) -> Vec<u8> {
    // code that is mostly packed data (weights, resources) decodes slowly and gains
    // little from history: it keeps normal sub-blocks
    let code = code && {
        let chunks: Vec<&[u8]> = data.chunks(4 << 20).collect();
        let dense: usize = crate::par::map(&chunks, threads, |c| dense_segments(c).iter().filter(|&&d| d).count()).iter().sum();
        dense * 2 < data.len().div_ceil(1 << SEG_BITS)
    };
    // capped effort marks bulk data that must go fast: smaller sub-blocks spread it over
    // more threads (and keep one block from running alone at the end)
    let sub = if arr != 0 && !crate::sys::dev_flag("NZ_SUB") { sub_size(data.len(), threads, code).min(FAST_SUB) } else { sub_size(data.len(), threads, code) };
    let parts: Vec<(usize, &[u8])> =
        if data.is_empty() { vec![(0, data)] } else { data.chunks(sub).enumerate().map(|(i, c)| (i * sub, c)).collect() };
    let coded = crate::par::map(&parts, threads, |&(base, p)| {
        let d0 = dict_len(base);
        let n = d0 + p.len();
        crate::par::with_memory(mem_for(n), || {
            // history from the sub-blocks before, then this one, then match finder padding
            let mut buf = Vec::with_capacity(n + NICE + 8);
            buf.extend_from_slice(&data[base - d0..base + p.len()]);
            buf.resize(n + NICE + 8, 0);
            let b0 = base - d0;
            let c = if !crate::sys::dev_flag("NZ_NOMT") && p.len() >= 2 * PSEG {
                #[cfg(target_arch = "x86_64")]
                let c = if crate::codec::has_avx2() { unsafe { encode_mt_avx2(&buf, n, d0, b0, x86ctx, threads, arr) } } else { encode_sub_mt(&buf, n, d0, b0, x86ctx, threads, arr) };
                #[cfg(not(target_arch = "x86_64"))]
                let c = encode_sub_mt(&buf, n, d0, b0, x86ctx, threads, arr);
                c
            } else {
                encode_one(&buf, n, d0, b0, x86ctx, arr)
            };
            crate::progress::work(p.len());
            c
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
    // pass 1: every sub-block at once (history read as zeros)
    let parts = crate::par::map(&jobs, threads, |&(s, len, base)| {
        let d0 = dict_len(base);
        crate::par::with_memory(2 * (d0 + len) as u64 + (24 << 20), || {
            let d = decode_one(s, len, d0, base - d0, x86ctx);
            crate::progress::work(len);
            d
        })
    });
    // pass 2, in order: the copies that reached into earlier sub-blocks
    let mut out = Vec::with_capacity(n);
    for p in parts {
        let (bytes, fix) = p?;
        let at = out.len();
        out.extend_from_slice(&bytes);
        for &(pos, d, len) in &fix {
            let dst = at + pos as usize;
            let d = d as usize;
            if d > dst || dst + len as usize > out.len() {
                return Err(e());
            }
            for k in dst..dst + len as usize {
                out[k] = out[k - d];
            }
        }
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
                let c = super::encode(d, 4, x, x, 0);
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
        let c = super::encode_sub(&pb, buf.len(), 0, 0, false, 0);
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
            lit.code(&mut enc, buf[pos] as u32, prev, mb, false, (0, 0));
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
            let b = lit.code(&mut dec, 0, prev, mb, false, (0, 0));
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
