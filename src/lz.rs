//! Fast mode: LZ77 with Huffman-coded symbols, one independent stream per block.
//!
//! Matching is greedy over one hash table (a repeat of the last distance is tried first),
//! with a window as large as the block. Literals go to their own Huffman-coded stream;
//! each match is a sequence (literal run, match length, distance) whose length and
//! distance buckets are Huffman-coded, followed by the bucket's extra bits.
//!
//! Decoding is table driven and bounds-checked, so corrupt input can only produce an
//! error, never undefined behaviour.
//!
//! block: n_seq:varint n_lit:varint raw:u8 lit_bytes:varint x4
//!        tables (literal, literal-run, match-length, distance: 4-bit code lengths)
//!        literal bitstreams (4, or the raw literals), sequence bitstream
//! The literals left after the last sequence end the block.

use crate::container::{read_varint, write_varint};

const MIN_MATCH: usize = 4;
const HASH_BITS: u32 = 17;
/// Bits of the 8-byte (long match) hash table.
const LONG_BITS: u32 = 16;
/// Longest Huffman code (decode tables have 2^MAX_BITS entries).
const MAX_BITS: u32 = 12;
/// Alphabets: bytes; run / length buckets (see `bucket`); distance buckets (0 = repeat).
const LIT_SYMS: usize = 256;
const LEN_SYMS: usize = 16 + 2 * 28;
/// Recent distances a match can repeat for free.
const REPS: usize = 3;
const DIST_SYMS: usize = REPS + 32;

// ---------------------------------------------------------------- buckets

/// Run / length bucket: values below 16 are their own bucket; above, the top two bits
/// pick it and the rest follow as extra bits. Returns (bucket, extra bits, extra).
#[inline(always)]
fn bucket(v: usize) -> (usize, u32, u64) {
    if v < 16 {
        return (v, 0, 0);
    }
    let b = usize::BITS - 1 - v.leading_zeros();
    let m = (v >> (b - 1)) & 1;
    (16 + 2 * (b as usize - 4) + m, b - 1, (v & ((1 << (b - 1)) - 1)) as u64)
}

/// Base value and extra-bit count of a run / length bucket.
#[inline(always)]
fn unbucket(c: usize) -> (usize, u32) {
    if c < 16 {
        return (c, 0);
    }
    let b = (c - 16) / 2 + 4;
    let m = (c - 16) & 1;
    ((2 | m) << (b - 1), b as u32 - 1)
}

/// Distance bucket: below REPS, repeats of the recent distances (0 = the last); else
/// REPS + floor(log2(d)), the bits below the top one following as extra bits.
#[inline(always)]
fn dist_bucket(d: usize) -> (usize, u32, u64) {
    let b = usize::BITS - 1 - d.leading_zeros();
    (REPS + b as usize, b, (d & ((1 << b) - 1)) as u64)
}

/// Recent distances after a match at `d`: a repeat moves to the front, a new one pushes
/// the oldest out. Returns the distance's code (repeat index, or REPS for a new one).
#[inline(always)]
fn use_dist(reps: &mut [usize; REPS], d: usize) -> usize {
    match reps.iter().position(|&r| r == d) {
        Some(k) => {
            reps[..=k].rotate_right(1);
            k
        }
        None => {
            reps.rotate_right(1);
            reps[0] = d;
            REPS
        }
    }
}

// ---------------------------------------------------------------- bits

/// LSB-first bit writer into a buffer sized up front for `max_bits` (a put past that
/// panics, never writes out of bounds).
struct BitW {
    out: Vec<u8>,
    pos: usize,
    acc: u64,
    n: u32,
}

impl BitW {
    fn new(max_bits: usize) -> BitW {
        BitW { out: vec![0u8; max_bits / 8 + 16], pos: 0, acc: 0, n: 0 }
    }
    /// Appends the low `k` (<= 32) bits of `v`, LSB first.
    #[inline(always)]
    fn put(&mut self, v: u64, k: u32) {
        self.acc |= v << self.n;
        self.n += k;
        if self.n >= 32 {
            self.out[self.pos..self.pos + 4].copy_from_slice(&(self.acc as u32).to_le_bytes());
            self.pos += 4;
            self.acc >>= 32;
            self.n -= 32;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        while self.n > 0 {
            self.out[self.pos] = self.acc as u8;
            self.pos += 1;
            self.acc >>= 8;
            self.n = self.n.saturating_sub(8);
        }
        self.out.truncate(self.pos);
        self.out
    }
}

struct BitR<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BitR<'a> {
    /// At least 56 bits from `pos` on (zeros past the end).
    #[inline(always)]
    fn peek(&self) -> u64 {
        let i = self.pos >> 3;
        let w = match self.b.get(i..i + 8) {
            Some(s) => u64::from_le_bytes(s.try_into().unwrap()),
            None => {
                let mut t = [0u8; 8];
                if i < self.b.len() {
                    let r = &self.b[i..];
                    t[..r.len()].copy_from_slice(r);
                }
                u64::from_le_bytes(t)
            }
        };
        w >> (self.pos & 7)
    }
    #[inline(always)]
    fn bits(&mut self, k: u32) -> u64 {
        let v = self.peek() & ((1u64 << k) - 1);
        self.pos += k as usize;
        v
    }
    fn overrun(&self) -> bool {
        self.pos > self.b.len() * 8
    }
}

// ---------------------------------------------------------------- Huffman

/// Code lengths (<= MAX_BITS) for `freq`: Huffman's, then made to fit the limit by
/// lengthening the rarest short codes until the Kraft sum fits.
fn code_lengths(freq: &[u32]) -> Vec<u8> {
    let mut len = vec![0u8; freq.len()];
    let mut syms: Vec<usize> = (0..freq.len()).filter(|&s| freq[s] > 0).collect();
    match syms.len() {
        0 => return len,
        1 => {
            len[syms[0]] = 1;
            return len;
        }
        _ => {}
    }
    syms.sort_by_key(|&s| (freq[s], s));
    // two-queue Huffman: leaves (sorted) and internal nodes (made in order of weight)
    let m = syms.len();
    let mut w: Vec<u64> = syms.iter().map(|&s| freq[s] as u64).collect();
    let mut parent = vec![0usize; 2 * m - 1];
    w.resize(2 * m - 1, 0);
    let (mut li, mut ni) = (0usize, m);
    for k in m..2 * m - 1 {
        let mut take = || {
            if li < m && (ni >= k || w[li] <= w[ni]) {
                li += 1;
                li - 1
            } else {
                ni += 1;
                ni - 1
            }
        };
        let (a, b) = (take(), take());
        w[k] = w[a] + w[b];
        parent[a] = k;
        parent[b] = k;
    }
    let mut depth = vec![0u32; 2 * m - 1];
    for k in (0..2 * m - 2).rev() {
        depth[k] = depth[parent[k]] + 1;
    }
    for (i, &s) in syms.iter().enumerate() {
        len[s] = depth[i].min(MAX_BITS) as u8;
    }
    // over the limit: lengthen the rarest codes shorter than the limit
    let full = 1u64 << MAX_BITS;
    let mut kraft: u64 = syms.iter().map(|&s| full >> len[s]).sum();
    while kraft > full {
        let s = *syms.iter().find(|&&s| (len[s] as u32) < MAX_BITS).unwrap();
        len[s] += 1;
        kraft -= full >> len[s];
    }
    len
}

/// Canonical codes, bit-reversed for an LSB-first stream.
fn codes(len: &[u8]) -> Vec<u16> {
    let mut count = [0u16; MAX_BITS as usize + 1];
    for &l in len {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u16; MAX_BITS as usize + 2];
    let mut c = 0u16;
    for l in 1..=MAX_BITS as usize {
        c = (c + count[l - 1]) << 1;
        next[l] = c;
    }
    len.iter()
        .map(|&l| {
            if l == 0 {
                return 0;
            }
            let v = next[l as usize];
            next[l as usize] += 1;
            v.reverse_bits() >> (16 - l)
        })
        .collect()
}

/// Decode table: entry = symbol << 4 | length (length 0: no such code).
fn table(len: &[u8]) -> Result<Vec<u16>, String> {
    let mut t = vec![0u16; 1 << MAX_BITS];
    let code = codes(len);
    let full = 1u64 << MAX_BITS;
    if len.iter().map(|&l| if l == 0 { 0 } else { full >> l }).sum::<u64>() > full {
        return Err("corrupt Huffman table".into());
    }
    for (s, (&l, &c)) in len.iter().zip(&code).enumerate() {
        if l == 0 {
            continue;
        }
        let e = ((s as u16) << 4) | l as u16;
        let mut k = c as usize;
        while k < t.len() {
            t[k] = e;
            k += 1 << l;
        }
    }
    Ok(t)
}

#[inline(always)]
fn sym(r: &mut BitR, t: &[u16]) -> Result<usize, ()> {
    let e = t[(r.peek() & ((1 << MAX_BITS) - 1)) as usize];
    let l = (e & 15) as usize;
    if l == 0 {
        return Err(());
    }
    r.pos += l;
    Ok((e >> 4) as usize)
}

fn put_lengths(out: &mut Vec<u8>, len: &[u8]) {
    for p in len.chunks(2) {
        out.push(p[0] | p.get(1).map_or(0, |&b| b << 4));
    }
}

fn get_lengths(src: &[u8], i: &mut usize, n: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt LZ block".to_string();
    let bytes = src.get(*i..*i + n.div_ceil(2)).ok_or_else(e)?;
    *i += n.div_ceil(2);
    let mut len = Vec::with_capacity(n + 1);
    for &b in bytes {
        len.push(b & 15);
        len.push(b >> 4);
    }
    len.truncate(n);
    if len.iter().any(|&l| l as u32 > MAX_BITS) {
        return Err(e());
    }
    Ok(len)
}

// ---------------------------------------------------------------- encoder

struct Seq {
    lit: u32,
    len: u32,
    /// below REPS: repeat of a recent distance (0 = the last), else REPS + the distance
    dist: u32,
}

#[inline(always)]
fn read32(s: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(s[i..i + 4].try_into().unwrap())
}

#[inline(always)]
fn read64(s: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(s[i..i + 8].try_into().unwrap())
}

#[inline(always)]
fn extend(s: &[u8], a: usize, b: usize, from: usize) -> usize {
    let mut l = from;
    let n = s.len();
    while b + l + 8 <= n {
        let x = read64(s, a + l) ^ read64(s, b + l);
        if x != 0 {
            return l + (x.trailing_zeros() / 8) as usize;
        }
        l += 8;
    }
    while b + l < n && s[a + l] == s[b + l] {
        l += 1;
    }
    l
}

/// Match finder knobs (dev overrides NZ_FHB, NZ_FWAYS, NZ_FLONG, NZ_FSKIP, NZ_FINS:
/// insert every k-th position inside matches, NZ_FLAZY=2 + NZ_FL2: a second lazy step
/// for matches shorter than that; ~1% smaller, ~3% slower, so off).
fn knob(name: &str, def: usize) -> usize {
    crate::sys::dev_var(name).and_then(|v| v.parse().ok()).unwrap_or(def)
}

/// Hash table of `ways` most recent positions per 5-byte hash.
struct Finder {
    t: Vec<u32>,
    bits: u32,
    ways: usize,
    /// most recent position per 8-byte hash (empty: off): finds long matches the 5-byte
    /// table has lost to newer positions
    tl: Vec<u32>,
    lbits: u32,
}

impl Finder {
    #[inline(always)]
    fn slot(&self, s: &[u8], i: usize) -> usize {
        (((read64(s, i) << 24).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - self.bits)) as usize) * self.ways
    }
    #[inline(always)]
    fn lslot(&self, s: &[u8], i: usize) -> usize {
        (read64(s, i).wrapping_mul(0x9FB2_1C65_1E98_DF25) >> (64 - self.lbits)) as usize
    }
    #[inline(always)]
    fn insert(&mut self, s: &[u8], i: usize) {
        if !self.tl.is_empty() {
            let hl = self.lslot(s, i);
            self.tl[hl] = i as u32;
        }
        let h = self.slot(s, i);
        if self.ways > 1 {
            self.t.copy_within(h..h + self.ways - 1, h + 1);
        }
        self.t[h] = i as u32;
    }
    /// Longest match for `i` among the bucket's positions (then `i` goes in): (pos, len).
    #[inline(always)]
    fn find(&mut self, s: &[u8], i: usize) -> Option<(usize, usize)> {
        let h = self.slot(s, i);
        let v = read32(s, i);
        let mut best: Option<(usize, usize)> = None;
        if !self.tl.is_empty() {
            let hl = self.lslot(s, i);
            let c = self.tl[hl] as usize;
            self.tl[hl] = i as u32;
            if c < i && read64(s, c) == read64(s, i) {
                best = Some((c, extend(s, c, i, 8)));
            }
        }
        for w in 0..self.ways {
            let c = self.t[h + w] as usize;
            if c < i && read32(s, c) == v {
                let l = extend(s, c, i, 4);
                // the longest, but a nearer one wins unless the far one pays for its
                // distance bits
                if best.is_none_or(|b| worth(l, i - c, false) > worth(b.1, i - b.0, false)) {
                    best = Some((c, l));
                }
            }
        }
        if self.ways > 1 {
            self.t.copy_within(h..h + self.ways - 1, h + 1);
        }
        self.t[h] = i as u32;
        best
    }
}

/// Rough worth of a match in quarter bytes: length, less its distance's extra bits (a
/// repeated distance costs nothing).
#[inline(always)]
fn worth(len: usize, d: usize, rep: bool) -> i64 {
    4 * len as i64 - if rep { 0 } else { (usize::BITS - d.leading_zeros()) as i64 }
}

/// Parse into literals and sequences: greedy, optionally one step lazy.
fn parse(s: &[u8]) -> (Vec<u8>, Vec<Seq>) {
    let n = s.len();
    let mut lits = Vec::with_capacity(n / 2);
    let mut seqs = Vec::with_capacity(n / 16);
    let ways = knob("NZ_FWAYS", 1).clamp(1, 8);
    let lbits = knob("NZ_FLONG", LONG_BITS as usize) as u32;
    let mut f = Finder {
        t: vec![0u32; (1 << knob("NZ_FHB", HASH_BITS as usize)) * ways],
        bits: knob("NZ_FHB", HASH_BITS as usize) as u32,
        ways,
        tl: if lbits > 0 { vec![0u32; 1 << lbits] } else { Vec::new() },
        lbits: lbits.max(1),
    };
    let lazy = knob("NZ_FLAZY", 1) != 0;
    let skip = knob("NZ_FSKIP", 7) as u32;
    let ins = knob("NZ_FINS", 0);
    let lazy2 = knob("NZ_FLAZY", 1) >= 2;
    let l2max = knob("NZ_FL2", 16);
    let mut anchor = 0;
    let mut reps = [1usize; REPS];
    let mut i = 1;
    // a repeat of one of the last two distances first (cheap, and its distance costs
    // nothing), else the hash tables: (start, source, length, is a repeat)
    let probe = |f: &mut Finder, i: usize, reps: [usize; REPS]| -> Option<(usize, usize, usize, bool)> {
        let v = read32(s, i);
        for r in reps {
            if i >= r && v == read32(s, i - r) {
                f.insert(s, i);
                return Some((i, i - r, extend(s, i - r, i, 4), true));
            }
        }
        f.find(s, i).map(|(c, l)| (i, c, l, false))
    };
    if n >= 16 {
        let limit = n - 8;
        while i < limit {
            let Some(mut m) = probe(&mut f, i, reps) else {
                // skip faster through data without matches (one more per 2^skip misses)
                i += 1 + ((i - anchor) >> skip);
                continue;
            };
            // one step lazy: a better match at the next position wins over this one
            while lazy && m.2 < 64 && m.0 + 1 < limit {
                match probe(&mut f, m.0 + 1, reps) {
                    Some(m2) if worth(m2.2, m2.0 - m2.1, m2.3) > worth(m.2, m.0 - m.1, m.3) + 4 => m = m2,
                    _ => {
                        if lazy2 && m.2 < l2max && m.0 + 2 < limit {
                            if let Some(m2) = probe(&mut f, m.0 + 2, reps) {
                                if worth(m2.2, m2.0 - m2.1, m2.3) > worth(m.2, m.0 - m.1, m.3) + 8 {
                                    m = m2;
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            let (mut st, mut cs, mut len, _) = m;
            // extend backward into pending literals
            while st > anchor && cs > 0 && s[st - 1] == s[cs - 1] {
                st -= 1;
                cs -= 1;
                len += 1;
            }
            let d = st - cs;
            let code = use_dist(&mut reps, d);
            let dist = if code < REPS { code as u32 } else { (REPS + d) as u32 };
            lits.extend_from_slice(&s[anchor..st]);
            seqs.push(Seq { lit: (st - anchor) as u32, len: (len - MIN_MATCH) as u32, dist });
            i = st + len;
            anchor = i;
            // positions inside the match: a couple into the table
            if i + 8 <= n {
                if ins > 0 {
                    let mut q = st + 1;
                    while q + 2 < i {
                        f.insert(s, q);
                        q += ins;
                    }
                    f.insert(s, i - 2);
                } else {
                    f.insert(s, i - 2);
                    if len > 8 {
                        f.insert(s, st + len / 2);
                    }
                }
            }
        }
    }
    lits.extend_from_slice(&s[anchor..]);
    (lits, seqs)
}

pub fn compress(s: &[u8]) -> Vec<u8> {
    let (lits, seqs) = parse(s);
    let mut fl = vec![0u32; LIT_SYMS];
    for &b in &lits {
        fl[b as usize] += 1;
    }
    let (mut fr, mut fm, mut fd) = (vec![0u32; LEN_SYMS], vec![0u32; LEN_SYMS], vec![0u32; DIST_SYMS]);
    for q in &seqs {
        fr[bucket(q.lit as usize).0] += 1;
        fm[bucket(q.len as usize).0] += 1;
        fd[if (q.dist as usize) < REPS { q.dist as usize } else { dist_bucket(q.dist as usize - REPS).0 }] += 1;
    }
    let (ll, lr, lm, ld) = (code_lengths(&fl), code_lengths(&fr), code_lengths(&fm), code_lengths(&fd));
    let (cl, cr, cm, cd) = (codes(&ll), codes(&lr), codes(&lm), codes(&ld));
    // literals that Huffman would shrink by under 0.5% (packed data) stay raw: copying
    // them decodes much faster
    let huff_bits: u64 = fl.iter().zip(&ll).map(|(&f, &l)| f as u64 * l as u64).sum();
    let raw = huff_bits * 200 >= lits.len() as u64 * 8 * 199;
    // literal k goes to stream k % 4: the decoder runs the four streams side by side
    let lit_bits: Vec<Vec<u8>> = if raw {
        vec![lits.clone(), Vec::new(), Vec::new(), Vec::new()]
    } else {
        // one table entry per byte: code | length << 16
        let cc: Vec<u32> = (0..LIT_SYMS).map(|b| cl[b] as u32 | (ll[b] as u32) << 16).collect();
        let max = (lits.len() / 4 + 1) * MAX_BITS as usize;
        let mut w = [BitW::new(max), BitW::new(max), BitW::new(max), BitW::new(max)];
        let [w0, w1, w2, w3] = &mut w;
        let mut q = lits.chunks_exact(4);
        for c in &mut q {
            let e = [cc[c[0] as usize], cc[c[1] as usize], cc[c[2] as usize], cc[c[3] as usize]];
            w0.put((e[0] & 0xFFFF) as u64, e[0] >> 16);
            w1.put((e[1] & 0xFFFF) as u64, e[1] >> 16);
            w2.put((e[2] & 0xFFFF) as u64, e[2] >> 16);
            w3.put((e[3] & 0xFFFF) as u64, e[3] >> 16);
        }
        for (k, &b) in q.remainder().iter().enumerate() {
            w[k].put(cl[b as usize] as u64, ll[b as usize] as u32);
        }
        w.into_iter().map(|w| w.finish()).collect()
    };
    let ll = if raw { vec![0u8; LIT_SYMS] } else { ll };
    // per sequence at most 3 codes and 23 + 23 + 31 extra bits
    let mut w = BitW::new(seqs.len() * (3 * MAX_BITS as usize + 77) + 64);
    for q in &seqs {
        let (c, k, x) = bucket(q.lit as usize);
        w.put(cr[c] as u64, lr[c] as u32);
        w.put(x, k);
        let (c, k, x) = bucket(q.len as usize);
        w.put(cm[c] as u64, lm[c] as u32);
        w.put(x, k);
        if (q.dist as usize) < REPS {
            w.put(cd[q.dist as usize] as u64, ld[q.dist as usize] as u32);
        } else {
            let (c, k, x) = dist_bucket(q.dist as usize - REPS);
            w.put(cd[c] as u64, ld[c] as u32);
            w.put(x, k);
        }
    }
    let seq_bits = w.finish();
    if crate::sys::dev_flag("NZ_FST") {
        let lb: usize = lit_bits.iter().map(|b| b.len()).sum();
        let bits = |f: &[u32], l: &[u8]| -> u64 { f.iter().zip(l).map(|(&f, &l)| f as u64 * l as u64).sum::<u64>() / 8 };
        let xr: u64 = seqs.iter().map(|q| bucket(q.lit as usize).1 as u64).sum::<u64>() / 8;
        let xm: u64 = seqs.iter().map(|q| bucket(q.len as usize).1 as u64).sum::<u64>() / 8;
        let xd: u64 = seqs.iter().map(|q| if (q.dist as usize) < REPS { 0 } else { dist_bucket(q.dist as usize - REPS).1 as u64 }).sum::<u64>() / 8;
        let reps = seqs.iter().filter(|q| (q.dist as usize) < REPS).count();
        eprintln!(
            "FST n {} lits {} litB {} seqs {} reps {} run {}+{} len {}+{} dist {}+{}",
            s.len(), lits.len(), lb, seqs.len(), reps, bits(&fr, &lr), xr, bits(&fm, &lm), xm, bits(&fd, &ld), xd
        );
    }
    let mut out = Vec::with_capacity(lit_bits.iter().map(|b| b.len()).sum::<usize>() + seq_bits.len() + 256);
    write_varint(&mut out, seqs.len() as u64);
    write_varint(&mut out, lits.len() as u64);
    out.push(raw as u8);
    for b in &lit_bits {
        write_varint(&mut out, b.len() as u64);
    }
    for l in [&ll, &lr, &lm, &ld] {
        put_lengths(&mut out, l);
    }
    for b in &lit_bits {
        out.extend_from_slice(b);
    }
    out.extend_from_slice(&seq_bits);
    out
}

// ---------------------------------------------------------------- decoder

pub fn decompress(src: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt LZ block".to_string();
    let mut r = src;
    let n_seq = read_varint(&mut r).map_err(|_| e())? as usize;
    let n_lit = read_varint(&mut r).map_err(|_| e())? as usize;
    let raw = match r.first() {
        Some(&b) if b <= 1 => b == 1,
        _ => return Err(e()),
    };
    r = &r[1..];
    let mut lit_bytes = [0usize; 4];
    for b in lit_bytes.iter_mut() {
        *b = read_varint(&mut r).map_err(|_| e())? as usize;
    }
    if n_lit > n || n_seq > n / MIN_MATCH + 1 {
        return Err(e());
    }
    let mut i = 0;
    let ll = get_lengths(r, &mut i, LIT_SYMS)?;
    let lr = get_lengths(r, &mut i, LEN_SYMS)?;
    let lm = get_lengths(r, &mut i, LEN_SYMS)?;
    let ld = get_lengths(r, &mut i, DIST_SYMS)?;
    let (tl, tr, tm, td) = (table(&ll)?, table(&lr)?, table(&lm)?, table(&ld)?);
    let mut brs: Vec<BitR> = Vec::with_capacity(4);
    for &b in &lit_bytes {
        brs.push(BitR { b: r.get(i..i.checked_add(b).ok_or_else(e)?).ok_or_else(e)?, pos: 0 });
        i += b;
    }
    let seq_src = &r[i..];
    // literals first, all at once: four independent streams side by side (literal k from
    // stream k % 4), two codes per 64-bit read
    let mut lits = vec![0u8; n_lit + 32];
    let mask = (1u64 << MAX_BITS) - 1;
    let mut bad = false;
    let mut k = 0;
    if raw {
        if brs[0].b.len() != n_lit || brs[1..].iter().any(|b| !b.b.is_empty()) {
            return Err(e());
        }
        lits[..n_lit].copy_from_slice(brs[0].b);
        k = n_lit;
    }
    while k + 8 <= n_lit {
        for (j, br) in brs.iter_mut().enumerate() {
            let w = br.peek();
            let e1 = tl[(w & mask) as usize];
            let l1 = (e1 & 15) as u32;
            let e2 = tl[((w >> l1) & mask) as usize];
            let l2 = (e2 & 15) as u32;
            bad |= l1 == 0 || l2 == 0;
            lits[k + j] = (e1 >> 4) as u8;
            lits[k + 4 + j] = (e2 >> 4) as u8;
            br.pos += (l1 + l2) as usize;
        }
        k += 8;
    }
    while k < n_lit {
        lits[k] = sym(&mut brs[k & 3], &tl).map_err(|_| e())? as u8;
        k += 1;
    }
    if bad || brs.iter().any(|b| b.overrun()) {
        return Err(e());
    }
    // sequences: copies run 16 bytes at a time into slack past the end (written over
    // later, never read before that)
    let cap = n + 32;
    let mut out: Vec<u8> = Vec::with_capacity(cap);
    let o = out.as_mut_ptr();
    let mut ol = 0usize;
    let mut br = BitR { b: seq_src, pos: 0 };
    let mut lp = 0usize;
    let mut reps = [1usize; REPS];
    for _ in 0..n_seq {
        let (base, k) = unbucket(sym(&mut br, &tr).map_err(|_| e())?);
        let lit = base + br.bits(k) as usize;
        let (base, k) = unbucket(sym(&mut br, &tm).map_err(|_| e())?);
        let len = base + br.bits(k) as usize + MIN_MATCH;
        let dc = sym(&mut br, &td).map_err(|_| e())?;
        let d = if dc < REPS {
            reps[..=dc].rotate_right(1);
            reps[0]
        } else {
            let b = (dc - REPS) as u32;
            if b > 31 {
                return Err(e());
            }
            let d = (1usize << b) | br.bits(b) as usize;
            reps.rotate_right(1);
            reps[0] = d;
            d
        };
        if lit > n_lit - lp || lit + len > n - ol || d == 0 || d > ol + lit {
            return Err(e());
        }
        // safety: lp + lit <= n_lit (lits has 32 spare bytes), ol + lit + len <= n (out
        // has 32 spare bytes), and copies only read bytes already written
        unsafe {
            let src = lits.as_ptr().add(lp);
            let dst = o.add(ol);
            let mut j = 0;
            while j < lit {
                (dst.add(j) as *mut u128).write_unaligned((src.add(j) as *const u128).read_unaligned());
                j += 16;
            }
            ol += lit;
            lp += lit;
            let dst = o.add(ol);
            let from = dst.sub(d);
            if d >= 16 {
                let mut j = 0;
                while j < len {
                    (dst.add(j) as *mut u128).write_unaligned((from.add(j) as *const u128).read_unaligned());
                    j += 16;
                }
            } else if d == 1 {
                std::ptr::write_bytes(dst, *from, len);
            } else {
                for j in 0..len {
                    *dst.add(j) = *from.add(j);
                }
            }
            ol += len;
        }
    }
    let rest = n_lit - lp;
    if br.overrun() || ol + rest != n {
        return Err(e());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(lits.as_ptr().add(lp), o.add(ol), rest);
        // every byte of 0..n was written above
        out.set_len(n);
    }
    let _ = cap;
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut v: Vec<u8> = Vec::new();
        for i in 0..100_000u32 {
            v.extend_from_slice(format!("line {} of text {}\n", i % 977, i % 13).as_bytes());
        }
        let mut x = 1u64;
        let noise: Vec<u8> = (0..300_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        for d in [&v[..], b"", b"a", b"abcdabcdabcdabcdabcdabcd", &[0u8; 100_000][..], &noise[..]] {
            let c = super::compress(d);
            assert_eq!(super::decompress(&c, d.len()).unwrap(), d);
        }
        assert!(super::compress(&v).len() < v.len() / 4);
    }

    #[test]
    fn buckets_invert() {
        for v in (0..100_000).chain([1 << 22, (1 << 23) - 1, 1 << 26]) {
            let (c, k, x) = super::bucket(v);
            assert!(c < super::LEN_SYMS);
            let (b, k2) = super::unbucket(c);
            assert_eq!((k, b + x as usize), (k2, v));
        }
    }
}
