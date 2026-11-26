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

use crate::cm::{G, MixV, NG, hg};
use crate::coder::{BitIo, Decoder, Encoder};

/// Largest independent sub-block (sub-blocks run in parallel in both directions).
pub const SUB_BLOCK: usize = 64 << 20;

/// History each sub-block may match into from the sub-blocks before it. Decoding stays
/// parallel: a first pass decodes every sub-block at once with bytes from before it read
/// as zero (the encoder models the same view), a second, sequential pass copies them in.
pub const DICT: usize = 16 << 20;

/// Bytes of history before sub-block start `base`.
#[inline]
fn dict_len(base: usize) -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    base.min(*D.get_or_init(|| crate::sys::dev_var("NZ_DICT").and_then(|v| v.parse::<usize>().ok()).map_or(DICT, |m| m << 20)))
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

/// Greedy LZ pass that counts (literals, matches) per 64 KiB chunk of `data`: decode time
/// is about a fixed cost per literal plus one per match, whatever the bytes, so this tells
/// where sub-block cuts give every decoder thread the same work. Regions run in parallel,
/// each with the DICT bytes before it sparsely in its table.
fn work_map(data: &[u8], threads: usize) -> Vec<(u32, u32)> {
    const TB: u32 = 20;
    const MIN: usize = 6;
    let n = data.len();
    let chunks = n.div_ceil(1 << SEG_BITS);
    let region = (n.div_ceil(threads.max(1))).next_multiple_of(1 << SEG_BITS).max(4 << 20);
    let regions: Vec<usize> = (0..n.div_ceil(region)).collect();
    #[inline(always)]
    fn key(d: &[u8], p: usize) -> usize {
        let v = u64::from_le_bytes(d[p..p + 8].try_into().unwrap()) << 16;
        (v.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - TB)) as usize
    }
    let parts = crate::par::map(&regions, threads, |&r| {
        let (r0, r1) = (r * region, ((r + 1) * region).min(n));
        let mut out = vec![(0u32, 0u32); (r1 - r0).div_ceil(1 << SEG_BITS)];
        let mut tab = vec![0u32; 1 << TB];
        let end = n.saturating_sub(8);
        for p in (r0.saturating_sub(DICT)..r0.min(end)).step_by(4) {
            tab[key(data, p)] = p as u32 + 1;
        }
        let mut p = r0;
        while p < r1 {
            let c = (p - r0) >> SEG_BITS;
            if p >= end {
                out[c].0 += 1;
                p += 1;
                continue;
            }
            let h = key(data, p);
            let cand = tab[h] as usize;
            tab[h] = p as u32 + 1;
            if cand != 0 && p + 1 - cand <= DICT && data[cand - 1..cand - 1 + MIN] == data[p..p + MIN] {
                let src = cand - 1;
                let mut l = MIN;
                while p + l < n && l < 4096 && data[src + l] == data[p + l] {
                    l += 1;
                }
                out[c].1 += 1;
                if l > 16 && p + l / 2 < end {
                    tab[key(data, p + l / 2)] = (p + l / 2) as u32 + 1;
                }
                p += l;
            } else {
                out[c].0 += 1;
                p += 1;
            }
        }
        out
    });
    let mut all: Vec<(u32, u32)> = parts.into_iter().flatten().collect();
    all.truncate(chunks);
    all
}

/// Decode work of one chunk in work_map units (calibrated on the decoder).
#[inline]
fn work_of(c: (u32, u32)) -> u64 {
    LIT_WORK * c.0 as u64 + MATCH_WORK * c.1 as u64 + (1 << SEG_BITS) / 16
}
const LIT_WORK: u64 = 15;
const MATCH_WORK: u64 = 0;

/// Work density (units per byte) of the data coded so far, x1024: sub-blocks of a block
/// denser than that are cut smaller, so pieces of every block take about the same time.
static WORK: std::sync::Mutex<(u64, u64, f64, usize)> = std::sync::Mutex::new((0, 0, 0.0, 0));

/// Sub-block starts (and `data.len()` last): `sub` bytes each on average, cut where every
/// piece holds the same decode work.
fn piece_cuts(data: &[u8], sub: usize, threads: usize) -> Vec<usize> {
    let n = data.len();
    if n <= sub.max(2 << SEG_BITS) || crate::sys::dev_flag("NZ_BYTECUTS") {
        if n <= sub {
            let mut g = WORK.lock().unwrap();
            g.2 += n as f64 / sub as f64;
            g.3 += 1;
        }
        let mut c: Vec<usize> = (0..n.div_ceil(sub.max(1)).max(1)).map(|i| i * sub).collect();
        c.push(n);
        return c;
    }
    let w: Vec<u64> = work_map(data, threads).into_iter().map(work_of).collect();
    let total: u64 = w.iter().sum();
    // cuts: a blend of equal work (decode balance) and equal bytes (every piece's model
    // gets as much data to learn from)
    let blend: u64 = crate::sys::dev_var("NZ_WBLEND").and_then(|v| v.parse().ok()).unwrap_or(50);
    let cw: Vec<u64> = w.iter().enumerate().map(|(i, &x)| {
        let bytes = ((i + 1) << SEG_BITS).min(n) - (i << SEG_BITS);
        (x * blend * n as u64 / total.max(1) + bytes as u64 * (100 - blend)) / 100
    }).collect();
    let ctotal: u64 = cw.iter().sum();
    // pieces: this block's work over a share's worth at the density seen so far
    // pieces: this block's work over a share's worth at the density seen so far; the
    // blocks' fractions add up (rather than each rounding up), so a batch of blocks
    // makes about one piece per thread
    let k = {
        let mut g = WORK.lock().unwrap();
        g.0 += total;
        g.1 += n as u64;
        let dens = g.0 as f64 / g.1 as f64;
        let own = total as f64 / n as f64;
        let ideal = n as f64 / sub as f64 * own / dens;
        let (lo, hi) = (n.div_ceil(SUB_BLOCK).max(1), n.div_ceil(sub).max(1));
        let k = ((g.2 + ideal + 0.2).floor() as i64 - g.3 as i64).clamp(lo as i64, hi as i64) as usize;
        // pieces saved earlier come only out of blocks at or below the average density:
        // a denser block sets the decode time, so it keeps its own share
        let k = if own > dens { k.max((ideal.round() as usize).clamp(lo, hi)) } else { k };
        g.2 += ideal;
        g.3 += k;
        if crate::sys::dev_flag("NZ_WTRACE") {
            eprintln!("  dens own {own:.1} global {dens:.1} ideal {ideal:.2} n/sub {:.2}", n as f64 / sub as f64);
        }
        if crate::sys::dev_flag("NZ_KBYTES") { hi } else { k }
    };
    let mut cuts = vec![0];
    let mut acc = 0u64;
    let mut next = 1;
    for (i, &x) in cw.iter().enumerate() {
        acc += x;
        let at = (i + 1) << SEG_BITS;
        if next < k && acc * k as u64 >= ctotal * next as u64 && at < n {
            cuts.push(at);
            next += 1;
        } else if at - cuts[cuts.len() - 1] >= SUB_BLOCK && at < n {
            cuts.push(at);
        }
    }
    cuts.push(n);
    if crate::sys::dev_flag("NZ_WTRACE") {
        eprintln!("  cuts of {} MB: {} pieces (by bytes {}), MB {:?}", n >> 20, cuts.len() - 1, n.div_ceil(sub), cuts.windows(2).map(|c| (c[1] - c[0]) >> 20).collect::<Vec<_>>());
    }
    cuts
}

/// Largest sub-block for effort-capped streams (see `encode`).
const FAST_SUB: usize = 16 << 20;

/// Bytes of work per thread in the batch being compressed (set by the container for each
/// wave of blocks; 0 = unknown).
static SHARE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_share(bytes: usize) {
    SHARE.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Arrivals the parser keeps per position (1, 2 or 4): more paths give 1-2.5% smaller
/// output for 1.5-2.7x the parse time. Dev override: NZ_ARR.
static ARRIVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);

/// Picks the arrivals for a run of `bytes` on `threads` threads: the most that still
/// finishes well ahead of 7-Zip Ultra. 7-Zip codes 256 MB blocks with one LZMA coder per
/// two threads at about 5 MB/s each, so it is slow on small inputs and uses every core
/// on big ones; nzip runs at about 0.65 / 0.9 / 1.2 MB/s per thread with 4 / 2 / 1
/// arrivals on the CPU alone. The effort never depends on the GPU: a GPU only takes match
/// finding off the CPU, so with one the same archive comes out faster.
pub fn set_effort(threads: usize, bytes: u64) {
    let coders = (threads / 2).max(1).min(bytes.div_ceil(256 << 20).max(1) as usize);
    let rival = coders as f64 * 5.0;
    let ours = |a: usize| threads as f64 * if a == 4 { 0.65 } else if a == 2 { 0.9 } else { 1.2 };
    let a = [4, 2, 1].into_iter().find(|&a| ours(a) >= rival * 1.1).unwrap_or(1);
    ARRIVALS.store(a, std::sync::atomic::Ordering::Relaxed);
}

/// A wait for the GPU counts as 1/GPU_BIAS of the CPU match finder's time: while a chunk
/// waits, its core parses other chunks, whereas CPU match finding takes cores from
/// parsing. (GPU lists also find slightly better matches.)
const GPU_BIAS: u64 = 3;

/// A CPU match finder hands the rest of its sub-block to an idle GPU when at least this
/// much is left.
const STEAL_MIN: usize = 8 << 20;

/// Running average of the CPU match finder's time per byte it handles (inserted or
/// searched), in picoseconds (0 = none yet).
static MF_PS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a chunk's match lists should come from the GPU: when they would arrive (every
/// job reserved ahead included) before the CPU match finder would be done with one
/// sub-block (`cpu_bytes` inserted or searched). Both speeds are measured as the run
/// goes, so the split adapts to any CPU and GPU (which chunks take which path depends on
/// timing). A `true` reserves the GPU: the caller must then run the job.
fn gpu_turn(job_bytes: usize, cpu_bytes: usize) -> bool {
    let per = match MF_PS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => 180_000,
        p => p,
    };
    crate::gpu::mf_reserve(job_bytes, GPU_BIAS * per * cpu_bytes as u64)
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
        return (v << 20).clamp(1 << 20, 256 << 20);
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
    /// Rate 1/32, faster while young: the low 3 bits count the bits seen, and the step is
    /// 1/2^(count + YOUNG_START) until it reaches 1/32, so each sub-block's cold start
    /// costs less.
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

/// Literal model: per bit, counters in three contexts (the previous byte, the match byte,
/// and inside the rel32 target of an x86 branch the operand context) go through a small
/// mixer whose weight set is chosen by the bits seen so far, the matched flag and the
/// operand byte index. The second nibble's counters are also keyed by the first nibble.
#[derive(Clone)]
struct Lit {
    t1: Vec<G>,
    tm: Vec<G>,
    tx: Vec<G>,
    m: MixV<4>,
}

/// Counter rate (shift) of the literal counters.
const LRATE: u32 = 4;
const TX_BITS: u32 = 18;

/// Bytes the model reads before `pos`: the input itself when encoding, the decoder's
/// output (history before the sub-block read as zeros) when decoding.
trait Hist {
    fn at(&self, i: usize) -> u8;
    /// The byte coded at `pos` when encoding (0 when decoding).
    fn lit(&self, pos: usize) -> u32;
    /// The 8 bytes before `pos`, byte pos-1 highest (bytes before 0 read as zeros).
    fn tail(&self, pos: usize) -> u64;
}

#[inline(always)]
fn tail_slow<H: Hist + ?Sized>(buf: &H, pos: usize) -> u64 {
    let mut t = 0u64;
    for k in 1..=8usize {
        t = (t << 8) | if pos >= k { buf.at(pos - k) as u64 } else { 0 };
    }
    t
}

impl Hist for [u8] {
    #[inline(always)]
    fn at(&self, i: usize) -> u8 {
        self[i]
    }
    #[inline(always)]
    fn lit(&self, pos: usize) -> u32 {
        self.get(pos).map_or(0, |&b| b as u32)
    }
    #[inline(always)]
    fn tail(&self, pos: usize) -> u64 {
        if pos >= 8 { u64::from_le_bytes(self[pos - 8..pos].try_into().unwrap()) } else { tail_slow(self, pos) }
    }
}

/// A sub-block being decoded in place: logical byte i >= d0 lives at p[i - d0]; the d0
/// history bytes before it read as zeros (the first pass does not have them yet).
struct DecBuf {
    p: *mut u8,
    d0: usize,
}

impl Hist for DecBuf {
    #[inline(always)]
    fn at(&self, i: usize) -> u8 {
        if i >= self.d0 { unsafe { *self.p.add(i - self.d0) } } else { 0 }
    }
    #[inline(always)]
    fn lit(&self, _pos: usize) -> u32 {
        0
    }
    #[inline(always)]
    fn tail(&self, pos: usize) -> u64 {
        if pos >= self.d0 + 8 { unsafe { (self.p.add(pos - self.d0 - 8) as *const u64).read_unaligned() }.to_le() } else { tail_slow(self, pos) }
    }
}

/// If `pos` is inside the rel32 operand of an x86 CALL/JMP/Jcc, returns (operand byte
/// index 1..=4, context): the filtered operand is a big-endian absolute target, so its
/// high byte tracks the position and each later byte the ones before it.
#[inline(always)]
fn op_ctx<H: Hist + ?Sized>(buf: &H, pos: usize, base: usize) -> (usize, u32) {
    op_ctx_t(buf.tail(pos), pos, base)
}

/// `op_ctx` from the 8 bytes before `pos` (`tail`).
#[inline(always)]
fn op_ctx_t(t: u64, pos: usize, base: usize) -> (usize, u32) {
    let at = |k: usize| ((t >> (64 - 8 * k)) & 0xff) as u8; // byte pos-k
    for k in 1..=4usize {
        if pos < k {
            break;
        }
        let bi = at(k);
        let op = (bi & 0xFE) == 0xE8 || (pos > k && at(k + 1) == 0x0F && (bi & 0xF0) == 0x80);
        if op {
            let abs = base + pos;
            let ctx = match k {
                1 => ((abs >> 16) & 0xff) as u32,
                2 => ((at(1) as u32) << 8) | ((abs >> 8) & 0xff) as u32,
                3 => ((at(2) as u32) << 8) | at(1) as u32,
                _ => at(3) as u32,
            };
            return (k, ctx);
        }
    }
    (0, 0)
}

impl Lit {
    fn new(x86ctx: bool) -> Lit {
        Lit {
            t1: vec![G::INIT; 256 * NG],
            tm: vec![G::INIT; 256 * NG],
            tx: vec![G::INIT; if x86ctx { 1 << TX_BITS } else { NG }],
            // inputs: the three counters and a bias (its weight starts at 0)
            m: MixV::with_init(256 * 2 * 5, 4, [5460, 5460, 5460, 0]),
        }
    }

    /// Hash seed of the x86 operand context (None outside branch targets).
    #[inline(always)]
    fn xc(op: (usize, u32)) -> Option<u32> {
        let (k, octx) = op;
        (k != 0).then(|| (((k as u32) << 16) | octx).wrapping_mul(NG as u32))
    }

    /// Codes (or decodes, per `io`) one literal; `mb` = match byte, `op` = op_ctx.
    #[inline(always)]
    fn code<IO: BitIo>(&mut self, io: &mut IO, byte: u32, prev: u32, mb: u32, matched: bool, op: (usize, u32)) -> u32 {
        // one loop per case, so the bit loop has no x86-context test
        match Self::xc(op) {
            Some(x) => self.code_x::<IO, true>(io, byte, prev, mb, matched, op, x),
            None => self.code_x::<IO, false>(io, byte, prev, mb, matched, op, 0),
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code_x<IO: BitIo, const X: bool>(&mut self, io: &mut IO, byte: u32, prev: u32, mb: u32, matched: bool, op: (usize, u32), xc: u32) -> u32 {
        let Lit { t1, tm, tx, m } = self;
        let row = ((matched as usize) << 8) | (op.0 << 9);
        let mut c0 = 1usize;
        for half in 0..2 {
            let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
            let g1 = &mut t1[prev as usize * NG + sel].0;
            let gm = &mut tm[mb as usize * NG + sel].0;
            let gx = &mut tx[if X { hg(xc + sel as u32, TX_BITS) } else { 0 }].0;
            let mut j = 1usize;
            for q in 0..4 {
                let x = [g1[j].st(), gm[j].st(), if X { gx[j].st() } else { 0 }, 256];
                let p = ((m.predict12(x, c0 | row) << 4) as u32).clamp(48, 65535 - 48);
                let b = io.code((byte >> (7 - (half * 4 + q))) & 1, p);
                m.update(b);
                g1[j].update(b, LRATE);
                gm[j].update(b, LRATE);
                if X {
                    gx[j].update(b, LRATE);
                }
                j = (j << 1) | b as usize;
                c0 = (c0 << 1) | b as usize;
            }
        }
        (c0 & 255) as u32
    }

    /// Cheaper estimate of `cost` for the one-arrival parse (few cores, bulk data): the
    /// context counters' logits averaged with fixed weights instead of mixed. Measured
    /// slightly smaller as well as faster there.
    #[inline(always)]
    fn cost_fast(&self, t: &[u32], byte: u32, prev: u32, mb: u32, op: (usize, u32)) -> u32 {
        let xc = Self::xc(op);
        let mut c0 = 1usize;
        let mut s = 0;
        for half in 0..2 {
            let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
            let g1 = &self.t1[prev as usize * NG + sel].0;
            let gm = &self.tm[mb as usize * NG + sel].0;
            let gx = xc.map(|x| &self.tx[hg(x + sel as u32, TX_BITS)].0);
            let mut j = 1usize;
            for q in 0..4 {
                let b = (byte >> (7 - (half * 4 + q))) & 1;
                let d = match gx {
                    Some(g) => (g1[j].st() + gm[j].st() + g[j].st()) * 11 >> 5,
                    None => (g1[j].st() + gm[j].st()) >> 1,
                };
                let p1 = crate::tables::squash(d).clamp(1, 4095) as u32;
                s += t[(if b == 1 { p1 } else { 4096 - p1 }) as usize];
                j = (j << 1) | b as usize;
                c0 = (c0 << 1) | b as usize;
            }
        }
        s
    }

    /// Cost (1/16 bit) of `byte` under the model as it stands (no updates).
    #[inline(always)]
    fn cost(&self, t: &[u32], byte: u32, prev: u32, mb: u32, matched: bool, op: (usize, u32)) -> u32 {
        let xc = Self::xc(op);
        let row = ((matched as usize) << 8) | (op.0 << 9);
        let mut c0 = 1usize;
        let mut s = 0;
        for half in 0..2 {
            let sel = if half == 0 { 0 } else { 1 + (c0 & 15) };
            let g1 = &self.t1[prev as usize * NG + sel].0;
            let gm = &self.tm[mb as usize * NG + sel].0;
            let gx = xc.map(|x| &self.tx[hg(x + sel as u32, TX_BITS)].0);
            let mut j = 1usize;
            for q in 0..4 {
                let b = (byte >> (7 - (half * 4 + q))) & 1;
                let x = [g1[j].st(), gm[j].st(), gx.map_or(0, |g| g[j].st()), 256];
                let p1 = self.m.peek12(&x, c0 | row).clamp(1, 4095) as u32;
                s += t[(if b == 1 { p1 } else { 4096 - p1 }) as usize];
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
            lit: Lit::new(x86ctx),
            flags: Vec::new(),
            seg_end: 0,
            dense: false,
            is_dense: [Prob::INIT; 2],
        }
    }

    /// Codes one token at `pos`. `buf` holds every byte before `pos` (and, when encoding,
    /// the bytes after it). Returns the token and, for a literal, its byte.
    #[inline(always)]
    fn code<IO: BitIo, H: Hist + ?Sized>(&mut self, io: &mut IO, buf: &H, pos: usize, t: Tok) -> (Tok, u8) {
        let s = self.state;
        let ps = pos & 3;
        let lit_byte = if matches!(t, Tok::Lit) { buf.lit(pos) } else { 0 };
        if pos >= self.seg_end {
            let seg = pos >> SEG_BITS;
            self.seg_end = (seg + 1) << SEG_BITS;
            let want = self.flags.get(seg).copied().unwrap_or(false) as u32;
            self.dense = bit(io, &mut self.is_dense[self.dense as usize], want) == 1;
        }
        let out;
        let mut byte = 0u8;
        if self.dense || bit(io, &mut self.is_match[s][ps], (t != Tok::Lit) as u32) == 0 {
            let tail = buf.tail(pos);
            let prev = (tail >> 56) as u32;
            let r0 = self.reps[0] as usize + 1;
            let mb = if pos >= r0 { buf.at(pos - r0) as u32 } else { 0 };
            let op = if self.x86ctx { op_ctx_t(tail, pos, self.base) } else { (0, 0) };
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
    fn new(n: usize, depth: usize) -> Mf {
        Mf { depth, nice: NICE, head4: vec![0; 1 << H4_BITS], head3: vec![0; 1 << H3_BITS], son: vec![0; 2 * n] }
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
        self.prefetch_ahead(buf, n, p);
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

    /// The tree walk is a chain of dependent loads; warm the start of later walks (callers
    /// go forward through the buffer): the hash bucket 16 positions ahead, and 8 ahead the
    /// bucket's first candidate bytes and tree node. Only a hint: the walk is unchanged.
    #[inline(always)]
    fn prefetch_ahead(&self, buf: &[u8], n: usize, p: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            if p + 20 <= n {
                let (h, _) = Self::hashes(buf, p + 16);
                _mm_prefetch::<_MM_HINT_T0>(self.head4.as_ptr().add(h) as *const i8);
            }
            if p + 12 <= n {
                let (h, _) = Self::hashes(buf, p + 8);
                let c = *self.head4.get_unchecked(h) as usize;
                if c > 0 {
                    _mm_prefetch::<_MM_HINT_T0>(buf.as_ptr().add(c - 1) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(self.son.as_ptr().add(2 * (c - 1)) as *const i8);
                }
            }
        }
    }

    #[inline(always)]
    fn insert(&mut self, buf: &[u8], n: usize, p: usize) {
        self.walk(buf, n, p, None);
    }

    /// Inserts the history bytes [0, d0) (except dense segments).
    fn insert_history(&mut self, buf: &[u8], n: usize, d0: usize, flags: &[bool], step: usize) {
        // every `step`-th position: matches into history start at most step - 1 bytes
        // later, for a fraction of the work
        for p in (0..d0).step_by(step) {
            if !flags[p >> SEG_BITS] {
                self.walk(buf, n, p, None);
            }
        }
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

/// Arrivals per position, plus a compact copy of each position's reject bound (most
/// candidate paths are rejected, and the bound stays in cache where the arrivals do not).
struct Table<const A: usize> {
    /// `A` cheapest arrivals per position, sorted by price
    n: Vec<[Node; A]>,
    /// a path to the position costing this much or more cannot be kept
    lim: Vec<u32>,
}

impl<const A: usize> Table<A> {
    #[inline(always)]
    fn bound(l: &[Node; A]) -> u32 {
        l[A - 1].price.min(l[0].price.saturating_add(TOL + 1))
    }
    #[inline(always)]
    fn put(&mut self, j: usize, nd: Node) {
        insert(&mut self.n[j], nd);
        self.lim[j] = Self::bound(&self.n[j]);
    }
}

struct Parser<const A: usize> {
    nodes: Table<A>,
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
        Parser { nodes: Table { n: vec![[EMPTY; A]; WIN + 2 * MAX_LEN + 4], lim: vec![INF; WIN + 2 * MAX_LEN + 4] }, matches: Vec::with_capacity(64), prices: Prices::new(), since: usize::MAX / 2 }
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
        let last = span.min(self.nodes.n.len() - 1);
        for nd in &mut self.nodes.n[..=last] {
            *nd = [EMPTY; A];
        }
        self.nodes.lim[..=last].fill(INF);
        self.nodes.n[0][0] = Node { price: 0, reps: m.reps, state: m.state as u8, ..EMPTY };
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
            if self.nodes.n[i][0].price == INF {
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
                let nd = self.nodes.n[i][a];
                if nd.price == INF {
                    break;
                }
                let s = nd.state as usize;
                let prices = &self.prices;
                let nodes = &mut self.nodes;
                let relax = |nodes: &mut Table<A>, j: usize, price: u32, tok: Tok| {
                    // far dearer than the best way here: cannot win later
                    if price >= nodes.lim[j] {
                        return;
                    }
                    nodes.put(j, Node { price, from: i as u32, from_a: a as u8, tok, reps: next_reps(nd.reps, tok), state: next_state(s, tok) as u8, chain: false, first: Tok::Lit });
                };
                // `first`, one literal, then rep0 again: LZMA's two-step lookahead
                let lookahead = |nodes: &mut Table<A>, first: Tok, base: u32| {
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
                    let op = if m.x86ctx { op_ctx(buf, at, m.base) } else { (0, 0) };
                    let lit = m.lit.cost(t, buf[at] as u32, buf[at - 1] as u32, buf[at - d] as u32, true, op);
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
                    nodes.put(
                        i + l + 1 + l2,
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
                            let c = if A == 1 { m.lit.cost_fast(t, buf[pos] as u32, prev, mb, op) } else { m.lit.cost(t, buf[pos] as u32, prev, mb, s >= 7, op) };
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
                    if a > 0 {
                        // extra arrivals: the full length only (shorter ones rarely win
                        // from a dearer start)
                        relax(nodes, i + len, base + prices.rlen[ps][len - 2], Tok::Rep(k as u8, len as u16));
                    } else {
                        for l in 2..=len {
                            relax(nodes, i + l, base + prices.rlen[ps][l - 2], Tok::Rep(k as u8, l as u16));
                        }
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
                let nd = self.nodes.n[i][0];
                let j = i + longest;
                let s = nd.state as usize;
                self.nodes.n[j] = [EMPTY; A];
                self.nodes.n[j][0] = Node { price: nd.price, from: i as u32, from_a: 0, tok: long_tok, reps: next_reps(nd.reps, long_tok), state: next_state(s, long_tok) as u8, chain: false, first: Tok::Lit };
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
            let nd = self.nodes.n[j][a];
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

/// The list of position `p` in lists (`idx`, `m`) that start at position `start`.
#[inline(always)]
fn list_at<'a>(idx: &[u32], m: &'a [(u32, u32)], start: usize, p: usize) -> &'a [(u32, u32)] {
    let i = p - start;
    &m[idx[i] as usize..idx[i + 1] as usize]
}

/// From lists of buffer positions counted from `w0`, those of positions [s, e) of a
/// buffer holding `n` coded bytes, as `find_lists` would give them.
fn part_of(idx: &[u32], m: &[(u32, u32)], start: usize, w0: usize, s: usize, e: usize, n: usize) -> Lists {
    let mut oi = Vec::with_capacity(e - s + 1);
    let mut om = Vec::with_capacity((e - s) * 2);
    oi.push(0);
    for p in s..e {
        let maxl = (n - p).min(MAX_LEN) as u32;
        let mut last = 0;
        for &(l, d) in list_at(idx, m, start, p - w0) {
            let l = l.min(maxl);
            if l > last {
                om.push((l, d));
                last = l;
            }
        }
        oi.push(om.len() as u32);
    }
    Lists { start: s, idx: oi, m: om }
}

/// From lists of stream positions counted from `w0`, those of the sub-block
/// [base, base + len) as positions of its buffer, which starts `d0` bytes of history
/// before it: matches that start before the buffer are dropped, lengths stop at its
/// end, and each list keeps strictly increasing lengths.
fn window_of(idx: &[u32], m: &[(u32, u32)], start: usize, w0: usize, base: usize, d0: usize, len: usize) -> Lists {
    let b0 = base - d0;
    let end = base + len;
    let mut oi = Vec::with_capacity(len + 1);
    let mut om = Vec::with_capacity(len * 2);
    oi.push(0);
    for g in base..end {
        let lp = (g - b0) as u32;
        let maxl = (end - g).min(MAX_LEN) as u32;
        let mut last = 0;
        for &(l, d) in list_at(idx, m, start, g - w0) {
            let l = l.min(maxl);
            if d < lp && l > last {
                om.push((l, d));
                last = l;
            }
        }
        oi.push(om.len() as u32);
    }
    Lists { start: d0, idx: oi, m: om }
}

/// Coded bytes per GPU match-list job: the sub-blocks of a chunk take their lists from
/// one job over the chunk and its history, so the GPU indexes each history once per
/// chunk rather than once per sub-block.
const GPU_CHUNK: usize = 32 << 20;

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
unsafe fn encode_mt_avx2(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool, threads: usize, arr: usize, pre: Option<Lists>) -> Vec<u8> {
    encode_sub_mt(buf, n, d0, base, x86ctx, threads, arr, pre)
}


/// Like `encode_sub`, but the parse runs on several threads: match lists are found in
/// order (the match finder sees all earlier data), segments of PSEG bytes are parsed in
/// parallel, and one final pass codes every token with the real model. The decoder is
/// the same; only the encoder's choices can differ slightly at segment starts.
/// `buf[..d0]` is history from earlier sub-blocks (matched into, never coded); bytes
/// [d0, n) are coded. `base` = stream offset of buf[0].
fn encode_sub_mt(buf: &[u8], n: usize, d0: usize, base: usize, x86ctx: bool, threads: usize, arr: usize, pre: Option<Lists>) -> Vec<u8> {
    let flags = dense_segments(&buf[..n]);
    let mut m = Model::new(base, x86ctx);
    m.flags = flags.clone();
    let mut enc = Encoder::new((n - d0) / 3 + 64);
    let mut view = View::new(n, d0);
    let segs: Vec<(usize, usize)> = (0..(n - d0).div_ceil(PSEG)).map(|i| (d0 + i * PSEG, (d0 + (i + 1) * PSEG).min(n))).collect();
    let mut pos = d0;
    // lists from the GPU (`pre`, found for the whole chunk), else the CPU match finder
    // runs ahead on its own thread
    let gpu = pre;
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
            // lower effort also searches less: capped streams much less
            let t = std::time::Instant::now();
            let mut mf = Mf::new(n, if arr != 0 { DEPTH / 8 } else if effort(0) < 4 { DEPTH / 2 } else { DEPTH });
            let step = crate::sys::dev_var("NZ_HSTEP").and_then(|v| v.parse().ok()).unwrap_or(if arr != 0 || effort(0) == 1 { 4 } else { 2 });
            mf.insert_history(buf, n, d0, fl, step);
            let mut busy = t.elapsed();
            let mut done = d0;
            for (k, &(s, e)) in sg.iter().enumerate() {
                // the GPU has nothing to do: it takes over the rest of this sub-block
                if n - s >= STEAL_MIN && crate::gpu::mf_idle() && !crate::sys::dev_flag("NZ_NOSTEAL") {
                    let per = match MF_PS.load(std::sync::atomic::Ordering::Relaxed) {
                        0 => 180_000,
                        p => p,
                    };
                    let w = s - s.min(DICT);
                    if crate::gpu::mf_reserve(n - w, GPU_BIAS * per * (n - s) as u64) {
                        let skip: Vec<bool> = (w >> SEG_BITS..n.div_ceil(1 << SEG_BITS)).map(|g| fl[g] || (g + 1) << SEG_BITS <= s).collect();
                        if let Some(all) = crate::gpu::match_lists_helper(&buf[w..n], SEG_BITS, &skip) {
                            let parts: Vec<Lists> = sg[k..].iter().map(|&(s2, e2)| part_of(all.idx(), all.pairs(), 0, w, s2, e2, n)).collect();
                            drop(all);
                            for l in parts {
                                if tx.send(l).is_err() {
                                    return;
                                }
                            }
                            break;
                        }
                    }
                }
                let t = std::time::Instant::now();
                let l = find_lists(&mut mf, buf, n, fl, s, e);
                busy += t.elapsed();
                done = e;
                if tx.send(l).is_err() {
                    return;
                }
            }
            // time spent working (not waiting for the parser), per byte handled
            let ps = (busy.as_nanos() as u64 * 1000 / (d0 / step + done - d0).max(1) as u64).max(1);
            let old = MF_PS.load(std::sync::atomic::Ordering::Relaxed);
            MF_PS.store(if old == 0 { ps } else { (old * 3 + ps) / 4 }, std::sync::atomic::Ordering::Relaxed);
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
    let mut mf = Mf::new(n, DEPTH);
    let mut parser = Parser::<A>::new();
    let mut enc = Encoder::new((n - d0) / 3 + 64);
    let mut toks = Vec::with_capacity(WIN);
    let mut view = View::new(n, d0);
    m.flags = dense_segments(&buf[..n]);
    mf.insert_history(buf, n, d0, &m.flags, 2);
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

/// First pass for one sub-block of `n` bytes, decoded in place into `out[..n]`, with `d0`
/// bytes of history before it (read as zeros): returns the copies the second pass must
/// redo, in order. A copy is redone when its source may hold a byte the first pass did not
/// know (history, or the target of an earlier redone copy); redoing a correct copy is
/// harmless, so whole copies are tracked, not single bytes.
///
/// Safety: `out` must be valid for writes of `n` bytes that nothing else touches.
unsafe fn decode_sub(src: &[u8], out: *mut u8, n: usize, d0: usize, base: usize, x86ctx: bool) -> Result<Vec<Fixup>, String> {
    let mut m = Model::new(base, x86ctx);
    let mut dec = Decoder::new(src);
    let total = d0 + n;
    let buf = DecBuf { p: out, d0 };
    let mut fix: Vec<Fixup> = Vec::new();
    // bit i set: byte d0 + i may be unknown (the target of a redone copy)
    let mut unk: Vec<u64> = if d0 > 0 { vec![0; n.div_ceil(64)] } else { Vec::new() };
    let mut pos = d0;
    while pos < total {
        let (t, b) = m.code(&mut dec, &buf, pos, Tok::Lit);
        if let Tok::Lit = t {
            unsafe { *out.add(pos - d0) = b };
            pos += 1;
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
            unsafe { copy_match(out, d0, n, pos, d, len) };
            if d0 > 0 {
                if start < d0 {
                    // the history part is unknown; the rest repeats bits from 0 on
                    fix.push(((pos - d0) as u32, d as u32, len as u32));
                    let h = (d0 - start).min(len);
                    bits_set(&mut unk, pos - d0, h);
                    if h < len {
                        bits_copy(&mut unk, 0, pos - d0 + h, len - h, d);
                    }
                } else if bits_any(&unk, start - d0, len) {
                    fix.push(((pos - d0) as u32, d as u32, len as u32));
                    bits_copy(&mut unk, start - d0, pos - d0, len, d);
                }
            }
            pos += len;
        }
        if dec.overrun() {
            return Err("LZ stream truncated".into());
        }
    }
    Ok(fix)
}

/// Whether any of bits a..a+len (len >= 1) is set.
#[inline(always)]
fn bits_any(b: &[u64], a: usize, len: usize) -> bool {
    let (w0, w1) = (a >> 6, (a + len - 1) >> 6);
    let lo = !0u64 << (a & 63);
    let hi = !0u64 >> (63 - ((a + len - 1) & 63));
    if w0 == w1 {
        return b[w0] & lo & hi != 0;
    }
    b[w0] & lo != 0 || b[w0 + 1..w1].iter().any(|&w| w != 0) || b[w1] & hi != 0
}

/// Sets bits a..a+len (len >= 1).
#[inline(always)]
fn bits_set(b: &mut [u64], a: usize, len: usize) {
    let (w0, w1) = (a >> 6, (a + len - 1) >> 6);
    let lo = !0u64 << (a & 63);
    let hi = !0u64 >> (63 - ((a + len - 1) & 63));
    if w0 == w1 {
        b[w0] |= lo & hi;
        return;
    }
    b[w0] |= lo;
    b[w0 + 1..w1].iter_mut().for_each(|w| *w = !0);
    b[w1] |= hi;
}

/// Copies bits src..src+len to dst..dst+len (src < dst), in steps of at most `d` =
/// dst - src bits so an overlapping copy repeats its own output, like the byte copy.
#[inline(always)]
fn bits_copy(b: &mut [u64], src: usize, dst: usize, len: usize, d: usize) {
    let get = |b: &[u64], a: usize, m: usize| -> u64 {
        let (w, o) = (a >> 6, a & 63);
        let mut v = b[w] >> o;
        if o + m > 64 {
            v |= b[w + 1] << (64 - o);
        }
        if m < 64 { v & ((1u64 << m) - 1) } else { v }
    };
    let step = d.min(64);
    let mut k = 0;
    while k < len {
        let m = step.min(len - k);
        let v = get(b, src + k, m);
        let a = dst + k;
        let (w, o) = (a >> 6, a & 63);
        let mask = if m < 64 { (1u64 << m) - 1 } else { !0 };
        b[w] = (b[w] & !(mask << o)) | (v << o);
        if o + m > 64 {
            let r = 64 - o;
            b[w + 1] = (b[w + 1] & !(mask >> r)) | (v >> r);
        }
        k += m;
    }
}

/// Copies `len` bytes from logical `pos - d` to `pos` in a sub-block decoded in place
/// (`out` holds its `n` bytes from logical `d0`; history bytes read as zeros). Writes
/// whole 16-byte chunks when the slack stays inside the sub-block: the extra bytes are
/// overwritten later and never read before that.
#[inline(always)]
unsafe fn copy_match(out: *mut u8, d0: usize, n: usize, pos: usize, d: usize, len: usize) {
    unsafe {
        let at = pos - d0;
        let dst = out.add(at);
        if pos - d >= d0 {
            let src = dst.sub(d);
            if d >= 16 && at + len + 16 <= n {
                let mut k = 0;
                while k < len {
                    (dst.add(k) as *mut u128).write_unaligned((src.add(k) as *const u128).read_unaligned());
                    k += 16;
                }
            } else if d >= len {
                std::ptr::copy_nonoverlapping(src, dst, len);
            } else if d == 1 {
                std::ptr::write_bytes(dst, *src, len);
            } else {
                for k in 0..len {
                    *dst.add(k) = *src.add(k);
                }
            }
        } else {
            // the source starts in the history: those bytes read as zeros
            let h = (d0 - (pos - d)).min(len);
            std::ptr::write_bytes(dst, 0, h);
            for k in h..len {
                *dst.add(k) = *dst.add(k).sub(d);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn encode_avx2(buf: &[u8], n: usize, d0: usize, base: usize, x: bool, arr: usize) -> Vec<u8> {
    encode_sub(buf, n, d0, base, x, arr)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn decode_avx2(src: &[u8], out: *mut u8, n: usize, d0: usize, base: usize, x: bool) -> Result<Vec<Fixup>, String> {
    unsafe { decode_sub(src, out, n, d0, base, x) }
}

fn encode_one(buf: &[u8], n: usize, d0: usize, base: usize, x: bool, arr: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { encode_avx2(buf, n, d0, base, x, arr) };
    }
    encode_sub(buf, n, d0, base, x, arr)
}

/// Safety: as `decode_sub`.
unsafe fn decode_one(src: &[u8], out: *mut u8, n: usize, d0: usize, base: usize, x: bool) -> Result<Vec<Fixup>, String> {
    #[cfg(target_arch = "x86_64")]
    if crate::codec::has_avx2() {
        return unsafe { decode_avx2(src, out, n, d0, base, x) };
    }
    unsafe { decode_sub(src, out, n, d0, base, x) }
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
    let parts: Vec<(usize, &[u8])> = piece_cuts(data, sub, threads).windows(2).map(|w| (w[0], &data[w[0]..w[1]])).collect();
    let code_part = |base: usize, p: &[u8], pre: Option<Lists>| -> Vec<u8> {
        let gpu = pre.is_some();
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let t0 = T0.get_or_init(std::time::Instant::now);
        let ts = t0.elapsed().as_secs_f64();
        let d0 = dict_len(base);
        let n = d0 + p.len();
        crate::par::with_memory(mem_for(n), || {
            // history from the sub-blocks before, then this one, then match finder padding
            let mut buf = Vec::with_capacity(n + NICE + 8);
            buf.extend_from_slice(&data[base - d0..base + p.len()]);
            buf.resize(n + NICE + 8, 0);
            let b0 = base - d0;
            let c = if pre.is_some() || (!crate::sys::dev_flag("NZ_NOMT") && p.len() >= 2 * PSEG) {
                #[cfg(target_arch = "x86_64")]
                let c = if crate::codec::has_avx2() { unsafe { encode_mt_avx2(&buf, n, d0, b0, x86ctx, threads, arr, pre) } } else { encode_sub_mt(&buf, n, d0, b0, x86ctx, threads, arr, pre) };
                #[cfg(not(target_arch = "x86_64"))]
                let c = encode_sub_mt(&buf, n, d0, b0, x86ctx, threads, arr, pre);
                c
            } else {
                encode_one(&buf, n, d0, b0, x86ctx, arr)
            };
            crate::progress::work(p.len());
            if crate::sys::dev_flag("NZ_WTRACE") {
                eprintln!("  enc piece {:>3} MB @{:>4} of {:>4} gpu {} {ts:.2}-{:.2}s", p.len() >> 20, base >> 20, data.len() >> 20, gpu as u8, t0.elapsed().as_secs_f64());
            }
            c
        })
    };
    // chunks of sub-blocks run at once; the GPU finds each chunk's match lists (with its
    // history) in one job, and its sub-blocks take their part. Without a GPU every
    // sub-block runs its own CPU match finder.
    // A chunk takes the GPU or the CPU match finder, whichever is expected to be done first.
    let mut groups: Vec<&[(usize, &[u8])]> = Vec::new();
    let mut g0 = 0;
    for i in 0..parts.len() {
        if i + 1 == parts.len() || parts[i + 1].0 + parts[i + 1].1.len() - parts[g0].0 > GPU_CHUNK {
            groups.push(&parts[g0..=i]);
            g0 = i + 1;
        }
    }
    let coded: Vec<Vec<u8>> = crate::par::map(&groups, threads, |group| {
        let c0 = group[0].0;
        let c1 = group[group.len() - 1].0 + group[group.len() - 1].1.len();
        let w0 = c0 - dict_len(c0);
        let step = crate::sys::dev_var("NZ_HSTEP").and_then(|v| v.parse().ok()).unwrap_or(if arr != 0 || effort(0) == 1 { 4 } else { 2 });
        let cpu_bytes = dict_len(c0) / step + group.iter().map(|g| g.1.len()).max().unwrap_or(0);
        let lists = if crate::sys::dev_flag("NZ_NOGPUMF") || c1 - c0 < 2 * PSEG || !gpu_turn(c1 - w0, cpu_bytes) {
            None
        } else {
            let skip: Vec<bool> = dense_segments(&data[w0..c1]).iter().enumerate().map(|(i, &f)| f || (i + 1) << SEG_BITS <= c0 - w0).collect();
            crate::gpu::match_lists(&data[w0..c1], SEG_BITS, &skip)
        };
        // each sub-block takes its part of the chunk's lists; the buffers then go back
        let parts: Vec<(usize, &[u8], Option<Lists>)> = match lists {
            Some(l) => {
                let w: Vec<Lists> = crate::par::map(group, threads, |&(base, p)| window_of(l.idx(), l.pairs(), 0, w0, base, dict_len(base), p.len()));
                drop(l);
                group.iter().zip(w).map(|(&(b, p), w)| (b, p, Some(w))).collect()
            }
            None => group.iter().map(|&(b, p)| (b, p, None)).collect(),
        };
        let parts: Vec<std::sync::Mutex<Option<Lists>>> = parts.into_iter().map(|t| std::sync::Mutex::new(t.2)).collect();
        let idx: Vec<usize> = (0..group.len()).collect();
        crate::par::map(&idx, threads, |&i| {
            let (base, p) = group[i];
            code_part(base, p, parts[i].lock().unwrap().take())
        })
    })
    .into_iter()
    .flatten()
    .collect();
    let mut out = Vec::with_capacity(coded.iter().map(|c| c.len()).sum::<usize>() + 8 * coded.len());
    crate::container::write_varint(&mut out, coded.len() as u64);
    for (c, p) in coded.iter().zip(&parts) {
        crate::container::write_varint(&mut out, p.1.len() as u64);
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
    let k = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
    if k == 0 || k > n.div_ceil(1 << SEG_BITS).max(1) {
        return Err(e());
    }
    let mut lens = Vec::with_capacity(k);
    for _ in 0..k {
        let raw = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
        let comp = crate::container::read_varint(&mut r).map_err(|_| e())? as usize;
        lens.push((raw, comp));
    }
    if lens.iter().map(|l| l.0).try_fold(0usize, |a, l| a.checked_add(l)) != Some(n) || lens.iter().any(|l| l.0 > SUB_BLOCK) {
        return Err(e());
    }
    if crate::sys::dev_flag("NZ_DTRACE") {
        eprintln!("  lz decode: {} MB in {} pieces", n >> 20, k);
    }
    let mut jobs = Vec::with_capacity(k);
    let (mut off, mut base) = (0usize, 0usize);
    for &(raw, comp) in &lens {
        let s = r.get(off..off.checked_add(comp).ok_or_else(e)?).ok_or_else(e)?;
        off += comp;
        jobs.push((s, raw, base));
        base += raw;
    }
    // pass 1: every sub-block at once, each into its place in `out` (history read as zeros)
    let mut out: Vec<u8> = Vec::with_capacity(n);
    struct Out(*mut u8);
    unsafe impl Sync for Out {}
    impl Out {
        fn at(&self, i: usize) -> *mut u8 {
            unsafe { self.0.add(i) }
        }
    }
    let dst = Out(out.as_mut_ptr());
    let fixes = crate::par::map_ranked(&jobs, threads, |j| j.0.len() as u64 + (j.1 >> 4) as u64, |&(s, len, base)| {
        let d0 = dict_len(base);
        crate::par::with_memory((len as u64 >> 2) + (24 << 20), || {
            let tr = crate::sys::dev_flag("NZ_DTRACE");
            static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let t0 = T0.get_or_init(std::time::Instant::now);
            let a = t0.elapsed().as_secs_f64();
            // jobs cover disjoint ranges of `out`'s capacity
            let d = unsafe { decode_one(s, dst.at(base), len, d0, base - d0, x86ctx) };
            if tr {
                eprintln!("  piece {:>4} MB @{:>4} MB of {:>4}: comp {:>6} KB {a:.2}-{:.2}s {:.3}", len >> 20, base >> 20, n >> 20, s.len() >> 10, t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() - a);
            }
            crate::progress::work(len);
            d
        })
    });
    let fixes = fixes.into_iter().collect::<Result<Vec<_>, _>>()?;
    // every byte was written: each piece fills exactly its range
    unsafe { out.set_len(n) };
    // pass 2, in order: the copies that reached into earlier sub-blocks
    let tp = std::time::Instant::now();
    let mut nfix = 0usize;
    for (fix, &(_, _, at)) in fixes.iter().zip(&jobs) {
        for &(pos, d, len) in fix {
            let dst = at + pos as usize;
            let d = d as usize;
            if d == 0 || d > dst || dst + len as usize > n {
                return Err(e());
            }
            // in steps of at most `d` bytes: an overlapping copy repeats its own output
            let (mut k, end) = (dst, dst + len as usize);
            while k < end {
                let m = (end - k).min(d);
                out.copy_within(k - d..k - d + m, k);
                k += m;
            }
            nfix += len as usize;
        }
    }
    if crate::sys::dev_flag("NZ_DTRACE") {
        eprintln!("  pass 2 of {} MB: {} KB fixed in {:.3}s", n >> 20, nfix >> 10, tp.elapsed().as_secs_f64());
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
        let mut mf = super::Mf::new(buf.len(), super::DEPTH);
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
        let mut lit = super::Lit::new(false);
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
        let mut lit = super::Lit::new(false);
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
