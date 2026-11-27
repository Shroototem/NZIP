//! Repeated content across the files of a folder, beyond whole-file duplicates.
//!
//! Files are cut into content-defined chunks (gear rolling hash, ~256 bytes on average, so a
//! boundary depends only on the bytes around it and survives insertions earlier in the
//! file). A chunk whose bytes already appeared earlier in the stream becomes a copy of that
//! spot; neighbouring copies merge into runs, and every run is confirmed byte by byte.
//! Unpacking rebuilds copies from the files already written, so they cost no RAM and the
//! compressor never sees them. LZ windows reach megabytes back; this reaches across the
//! whole folder (rebuilt programs, per-profile copies of the same libraries, ...).

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

const AVG_BITS_DEFAULT: u32 = 8;

/// log2 of the average chunk size (dev override: NZ_CDC).
fn avg_bits() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| crate::sys::dev_var("NZ_CDC").and_then(|v| v.parse().ok()).unwrap_or(AVG_BITS_DEFAULT))
}
/// Smaller files are left whole.
pub const MIN_FILE: u64 = 16 << 10;

/// A piece of a file: bytes carried in the stream, or a copy of `len` bytes from offset
/// `off` of the earlier entry `src` (or of the same file, further back).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seg {
    Lit(u64),
    Copy { src: usize, off: u64, len: u64 },
}

impl Seg {
    pub fn len(&self) -> u64 {
        match *self {
            Seg::Lit(l) => l,
            Seg::Copy { len, .. } => len,
        }
    }
}

const fn gear_table() -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        t[i] = s;
        i += 1;
    }
    t
}

static GEAR: [u64; 256] = gear_table();

/// Content key of a chunk (64-bit; equal keys are only candidates, see `confirm`).
pub fn chunk_key(c: &[u8]) -> u64 {
    let (mut x, mut y) = (0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64);
    // two independent chains, alternate words each: twice the speed of one chain
    let mut w = c.chunks_exact(16);
    for v in &mut w {
        let a = u64::from_le_bytes(v[..8].try_into().unwrap());
        let b = u64::from_le_bytes(v[8..].try_into().unwrap());
        x = (x ^ a).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23);
        y = (y ^ b).wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(31);
    }
    for &b in w.remainder() {
        x = (x ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let k = x ^ y.rotate_left(17) ^ (c.len() as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    k ^ (k >> 29)
}

/// End of the chunk starting at `s` in `b` (`last`: no more bytes follow `b`), or None if
/// more bytes are needed to find it.
#[inline]
pub fn chunk_end(b: &[u8], s: usize, last: bool) -> Option<usize> {
    let bits = avg_bits();
    let (min_chunk, max_chunk) = (1usize << (bits - 2), 1usize << (bits + 3));
    let lim = (s + max_chunk).min(b.len());
    // the hash only depends on the last 64 bytes, so it starts just before MIN_CHUNK
    let mut i = s + min_chunk.saturating_sub(64);
    if i >= lim {
        return if lim - s == max_chunk || last { Some(lim) } else { None };
    }
    let mut h = 0u64;
    while i < lim {
        h = (h << 1).wrapping_add(GEAR[b[i] as usize]);
        i += 1;
        if i - s >= min_chunk && h >> (64 - bits) == 0 {
            return Some(i);
        }
    }
    if lim - s == max_chunk || last { Some(lim) } else { None }
}

/// Mostly printable bytes (text of any kind, including UTF-8): coded by the BWT engine.
pub fn texty(b: &[u8]) -> bool {
    let printable = b.iter().filter(|&&c| matches!(c, 9 | 10 | 13 | 32..=126) || c >= 128).count();
    !b.is_empty() && printable * 100 >= b.len() * 95
}

/// The chunks of a file in order: (key, length), and whether the file starts like text
/// (the plain reference for `chunk_round`).
#[cfg(test)]
fn chunk_file(p: &Path, size: u64) -> io::Result<(Vec<(u64, u32)>, bool)> {
    use std::io::Read;
    let mut f = File::open(p)?.take(size);
    let mut out = Vec::with_capacity((size >> avg_bits()) as usize + 1);
    let mut buf = vec![0u8; 4 << 20];
    let mut have = 0usize;
    let mut eof = false;
    let mut text = None;
    loop {
        while !eof && have < buf.len() {
            let n = f.read(&mut buf[have..])?;
            if n == 0 {
                eof = true;
            }
            have += n;
        }
        let b = &buf[..have];
        let text = *text.get_or_insert_with(|| texty(&b[..have.min(64 << 10)]));
        let mut s = 0;
        while s < have {
            match chunk_end(b, s, eof) {
                Some(e) => {
                    out.push((chunk_key(&b[s..e]), (e - s) as u32));
                    s = e;
                }
                None => break,
            }
        }
        if eof {
            return Ok((out, text));
        }
        buf.copy_within(s..have, 0);
        have -= s;
    }
}

/// Positions (`base` + index + 1) in `b[from..]` after which the gear hash of the bytes so
/// far has its top `bits` bits clear. The hash only remembers the last 64 bytes, so four
/// lanes, each started 64 bytes early, run side by side (the hash is a serial chain of
/// one shift and one add per byte) and give the same positions as one pass from b[0].
fn gear_cands(b: &[u8], from: usize, bits: u32, base: u64, out: &mut Vec<u64>) {
    let g = |h: u64, c: u8| (h << 1).wrapping_add(GEAR[c as usize]);
    let warm = |p: usize| b[p.saturating_sub(64)..p].iter().fold(0u64, |h, &c| g(h, c));
    let q = (b.len() - from) / 4;
    let p = [from, from + q, from + 2 * q, from + 3 * q];
    let (mut h0, mut h1, mut h2, mut h3) = (warm(p[0]), warm(p[1]), warm(p[2]), warm(p[3]));
    let mut v: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::with_capacity((q >> bits) * 2 + 8));
    let sh = 64 - bits;
    let (l0, l1, l2, l3) = (&b[p[0]..p[0] + q], &b[p[1]..p[1] + q], &b[p[2]..p[2] + q], &b[p[3]..p[3] + q]);
    for j in 0..q {
        h0 = g(h0, l0[j]);
        h1 = g(h1, l1[j]);
        h2 = g(h2, l2[j]);
        h3 = g(h3, l3[j]);
        if (h0 >> sh == 0) | (h1 >> sh == 0) | (h2 >> sh == 0) | (h3 >> sh == 0) {
            let at = base + j as u64 + 1;
            if h0 >> sh == 0 {
                out.push(at + p[0] as u64);
            }
            if h1 >> sh == 0 {
                v[0].push(at + p[1] as u64);
            }
            if h2 >> sh == 0 {
                v[1].push(at + p[2] as u64);
            }
            if h3 >> sh == 0 {
                v[2].push(at + p[3] as u64);
            }
        }
    }
    for x in v {
        out.extend_from_slice(&x);
    }
    for (i, &c) in b.iter().enumerate().skip(p[3] + q) {
        h3 = g(h3, c);
        if h3 >> sh == 0 {
            out.push(base + i as u64 + 1);
        }
    }
}

/// Bytes per chunking job: small enough that a slice stays in cache from reading to
/// hashing to keying.
const SLICE: u64 = 4 << 20;

/// One slice's chunks, found as if a chunk began at the slice start: (start, key, len),
/// and the candidate positions (see `chunk_round`).
struct SliceChunks {
    starts: Vec<u64>,
    keys: Vec<(u64, u32)>,
    cands: Vec<u64>,
    text: bool,
}

/// `chunk_file` for every file of a round at once, with the same chunks. A boundary
/// depends only on the 64 bytes before it (the gear hash forgets older bytes) and on the
/// minimum / maximum chunk length since the previous boundary. So every slice of every
/// file is read, hashed and keyed in one pass on its own thread, as if a chunk started
/// at the slice start; joining the slices in order, the real chain soon lands on one of
/// the slice's boundaries, and from there on the slice's chunks are the real ones (the
/// few before are keyed again).
fn chunk_round(round: &[(usize, &Path, u64)], threads: usize) -> Vec<io::Result<(Vec<(u64, u32)>, bool)>> {
    let bits = avg_bits();
    let (min_chunk, max_chunk) = (1u64 << (bits - 2), 1u64 << (bits + 3));
    let mut jobs: Vec<(usize, u64, u64)> = Vec::new(); // (file in round, start, end)
    for (k, &(_, _, size)) in round.iter().enumerate() {
        let mut s = 0;
        loop {
            let e = (s + SLICE).min(size);
            jobs.push((k, s, e));
            if e >= size {
                break;
            }
            s = e;
        }
    }
    let slices: Vec<io::Result<SliceChunks>> = crate::par::map(&jobs, threads, |&(k, s, e)| {
        let (_, p, size) = round[k];
        let f = File::open(p)?;
        let b0 = s.saturating_sub(64);
        let b1 = (e + max_chunk).min(size);
        // one buffer per thread, reused: a fresh one would be faulted in and zeroed page by page
        thread_local!(static BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) });
        let mut b = BUF.with(|x| std::mem::take(&mut *x.borrow_mut()));
        b.resize((b1 - b0) as usize, 0);
        read_at(&f, &mut b, b0)?;
        let text = s == 0 && texty(&b[..b.len().min(64 << 10)]);
        // candidates in (s, b1]
        let mut cands = Vec::with_capacity(((b1 - s) >> bits) as usize * 2 + 8);
        gear_cands(&b, (s - b0) as usize, bits, b0, &mut cands);
        // chunks from s, those that start before e
        let (mut starts, mut keys) = (Vec::new(), Vec::new());
        let mut p0 = s;
        let mut ci = 0;
        while p0 < e {
            while ci < cands.len() && cands[ci] < p0 + min_chunk {
                ci += 1;
            }
            let lim = (p0 + max_chunk).min(size);
            let end = if ci < cands.len() && cands[ci] <= lim { cands[ci] } else { lim };
            starts.push(p0);
            keys.push((chunk_key(&b[(p0 - b0) as usize..(end - b0) as usize]), (end - p0) as u32));
            p0 = end;
        }
        BUF.with(|x| *x.borrow_mut() = b);
        Ok(SliceChunks { starts, keys, cands, text })
    });
    // join each file's slices in order
    let mut out: Vec<io::Result<(Vec<(u64, u32)>, bool)>> = Vec::with_capacity(round.len());
    let mut si = slices.into_iter();
    for (k, &(_, p, size)) in round.iter().enumerate() {
        let n = jobs.iter().filter(|j| j.0 == k).count();
        let mine: Vec<io::Result<SliceChunks>> = si.by_ref().take(n).collect();
        out.push((|| {
            let mut keys: Vec<(u64, u32)> = Vec::with_capacity((size >> bits) as usize + 1);
            let mut text = false;
            let mut at = 0u64; // the real chain: next chunk starts here
            let mut f: Option<File> = None;
            for (j, sl) in mine.into_iter().enumerate() {
                let sl = sl?;
                if j == 0 {
                    text = sl.text;
                }
                // follow the real chain over the slice's candidates until it meets one of
                // the slice's own chunk starts
                let mut ci = 0;
                loop {
                    if let Ok(i) = sl.starts.binary_search(&at) {
                        keys.extend_from_slice(&sl.keys[i..]);
                        let (last_start, last) = (sl.starts[sl.starts.len() - 1], sl.keys[sl.keys.len() - 1]);
                        at = last_start + last.1 as u64;
                        break;
                    }
                    if sl.starts.last().is_none_or(|&l| at > l) {
                        break; // the chain skipped past this slice's chunk starts
                    }
                    while ci < sl.cands.len() && sl.cands[ci] < at + min_chunk {
                        ci += 1;
                    }
                    let lim = (at + max_chunk).min(size);
                    let end = if ci < sl.cands.len() && sl.cands[ci] <= lim { sl.cands[ci] } else { lim };
                    let fh = match &f {
                        Some(x) => x,
                        None => f.insert(File::open(p)?),
                    };
                    let mut buf = vec![0u8; (end - at) as usize];
                    read_at(fh, &mut buf, at)?;
                    keys.push((chunk_key(&buf), (end - at) as u32));
                    at = end;
                }
            }
            Ok((keys, text))
        })());
    }
    out
}

fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    crate::sys::read_at(f, buf, off)
}

/// Chunk index: open addressing over u64 slots holding a 24-bit key fingerprint and the
/// chunk's 40-bit offset (+1) in the files' stream. 8 bytes per chunk, sized from the
/// input and capped by the RAM budget; when full, new chunks are no longer indexed.
/// A fingerprint match is only a candidate (`confirm` compares the bytes).
struct Index {
    slots: Vec<u64>,
    used: usize,
}

impl Index {
    fn new(chunks: u64) -> Index {
        // at most 1/32 of the RAM budget
        let max = ((crate::sys::mem_budget() / 256).max(1 << 17).next_power_of_two() / 2).max(1 << 16);
        let cap = (chunks + chunks / 2).next_power_of_two().clamp(1 << 16, max);
        Index { slots: vec![0; cap as usize], used: 0 }
    }
    #[inline(always)]
    fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let i = key as usize & (self.slots.len() - 1);
            std::arch::x86_64::_mm_prefetch(self.slots.as_ptr().add(i) as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }
    /// The offset of an earlier chunk with this key, else indexes `pos` under it.
    fn find_or_insert(&mut self, key: u64, pos: u64) -> Option<u64> {
        let mask = self.slots.len() - 1;
        let fp = key >> 40;
        let mut i = key as usize & mask;
        loop {
            let v = self.slots[i];
            if v == 0 {
                if self.used < self.slots.len() / 8 * 7 && pos < (1 << 40) - 1 {
                    self.slots[i] = fp << 40 | (pos + 1);
                    self.used += 1;
                }
                return None;
            }
            if v >> 40 == fp {
                return Some((v & ((1 << 40) - 1)) - 1);
            }
            i = (i + 1) & mask;
        }
    }
}

/// In files that are not text (LZ coded), a copy whose source is closer than this in the
/// stream is left to the LZ coder (it reaches that far and codes it as well, without
/// breaking its context), unless it is at least NEAR_RUN long (then copying saves the
/// parse time). Text goes to the BWT engine, which sees its whole block: it keeps every copy.
const NEAR: u64 = 16 << 20;
/// Copies shorter than this are kept only if `worth` it.
const SHORT: u64 = 4 << 10;

/// A short copy costs ~10 bytes in the manifest and breaks the coder's context; it pays
/// only for bytes the coder would spend more on (not padding, zeros or simple tables).
pub fn worth(b: &[u8]) -> bool {
    let bits = crate::codec::entropy_milli(b) as u64 * b.len() as u64 / 1000;
    // the coder usually does ~3x better than order-0 on such data
    bits / 8 / 3 >= 64
}

/// Copies from earlier in the same file must be at least this long: short ones break the
/// coder's context (and an x86 filter's addresses) for little gain. Small chunks pay off
/// across files, where no coder can reach.
const SELF_MIN: u64 = 8 << 10;
const NEAR_RUN: u64 = 64 << 10;

fn push_lit(segs: &mut Vec<Seg>, len: u64) {
    if let Some(Seg::Lit(l)) = segs.last_mut() {
        *l += len;
    } else {
        segs.push(Seg::Lit(len));
    }
}

fn push_copy(segs: &mut Vec<Seg>, src: usize, off: u64, len: u64) {
    if let Some(Seg::Copy { src: s, off: o, len: l }) = segs.last_mut() {
        if *s == src && *o + *l == off {
            *l += len;
            return;
        }
    }
    segs.push(Seg::Copy { src, off, len });
}

/// Files a copy-checking job keeps open at once.
const OPEN_PER_JOB: usize = 4;

/// Index shards (a power of two).
const SHARDS: usize = 64;

/// Files whose chunks are hashed per round (bounds memory; matching runs in stream order).
const ROUND: usize = 512;

/// Finds the repeated chunks of `files` (entry index, path, size; in stream order).
/// Returns, per file, its pieces when it has at least one copy. Copies only point at
/// pieces that stay literal in their file, earlier in the stream.
pub fn find(files: &[(usize, &Path, u64)], threads: usize) -> io::Result<Vec<Option<Vec<Seg>>>> {
    // offsets of the files in their stream
    let mut starts = Vec::with_capacity(files.len() + 1);
    let mut total = 0u64;
    for f in files {
        starts.push(total);
        total += f.2;
    }
    // the index in shards by key: a chunk only meets earlier chunks with its own key, so
    // the shards run in parallel, each in stream order, with the same answers
    let shards: Vec<std::sync::Mutex<Index>> = (0..SHARDS).map(|_| std::sync::Mutex::new(Index::new((total >> avg_bits()) / SHARDS as u64))).collect();
    let shard_of = |key: u64| ((key >> 34) as usize) & (SHARDS - 1);
    let mut out = Vec::with_capacity(files.len());
    let tm = crate::sys::dev_flag("NZ_DDT");
    let t0 = std::time::Instant::now();
    let mut tchunk = 0.0;
    for round in files.chunks(ROUND) {
        let tc = std::time::Instant::now();
        let chunks = chunk_round(round, threads).into_iter().collect::<io::Result<Vec<_>>>()?;
        tchunk += tc.elapsed().as_secs_f64();
        let lapt = |w: &str| {
            if tm {
                eprintln!("    {w} {:.3}", t0.elapsed().as_secs_f64());
            }
        };
        lapt("chunked");
        // every chunk of the round: (key, stream offset), each file's on its own thread
        let o0 = out.len();
        let counts: Vec<usize> = chunks.iter().map(|c| c.0.len()).collect();
        let nk: usize = counts.iter().sum();
        let mut keys: Vec<(u64, u64)> = vec![(0, 0); nk];
        {
            let mut slices = Vec::with_capacity(counts.len());
            let mut rest = &mut keys[..];
            for &c in &counts {
                let (a, b) = rest.split_at_mut(c);
                slices.push(a);
                rest = b;
            }
            crate::par::for_chunks(slices, threads, |j, sl| {
                let mut at = starts[o0 + j];
                for (d, &(key, len)) in sl.iter_mut().zip(&chunks[j].0) {
                    *d = (key, at);
                    at += len as u64;
                }
            });
        }
        lapt("keys");
        // each part of the round's chunks sorted into the shards (in order)
        let part = nk.div_ceil(threads * 2).max(1 << 14);
        let ranges: Vec<(usize, usize)> = (0..nk.div_ceil(part)).map(|p| (p * part, ((p + 1) * part).min(nk))).collect();
        // (key, offset, chunk): read in order by the shard, no lookups back into `keys`
        let parts: Vec<Vec<Vec<(u64, u64, u32)>>> = crate::par::map(&ranges, threads, |&(a, b)| {
            let mut by: Vec<Vec<(u64, u64, u32)>> = vec![Vec::with_capacity((b - a) / SHARDS * 2); SHARDS];
            for (k, &(key, at)) in keys[a..b].iter().enumerate() {
                by[shard_of(key)].push((key, at, (a + k) as u32));
            }
            by
        });
        drop(keys);
        lapt("by");
        // per chunk: the earlier position + 1, or 0 (zeroed memory comes free from the OS)
        let mut found_mem = vec![0u64; nk];
        // safety: AtomicU64 has the size and alignment of u64, and found_mem is not used
        // otherwise while `found` lives
        let found: &[std::sync::atomic::AtomicU64] = unsafe { &*(found_mem.as_mut_slice() as *mut [u64] as *const [std::sync::atomic::AtomicU64]) };
        let ids: Vec<usize> = (0..SHARDS).collect();
        crate::par::map(&ids, threads, |&sh| {
            let mut ix = shards[sh].lock().unwrap();
            for p in &parts {
                let v = &p[sh];
                for (i, &(key, at, k)) in v.iter().enumerate() {
                    if let Some(&(next, ..)) = v.get(i + 8) {
                        ix.prefetch(next);
                    }
                    if let Some(pos) = ix.find_or_insert(key, at) {
                        found[k as usize].store(pos + 1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });
        lapt("hits");
        let mut kbase = Vec::with_capacity(counts.len());
        let mut acc = 0;
        for &c in &counts {
            kbase.push(acc);
            acc += c;
        }
        lapt("found");
        let fis: Vec<usize> = (0..chunks.len()).collect();
        let res = crate::par::map(&fis, threads, |&j| {
            let (ch, text) = (&chunks[j].0, chunks[j].1);
            let fi = o0 + j;
            let base = starts[fi];
            let mut segs = Vec::new();
            for (i, &(_, len)) in ch.iter().enumerate() {
                let len = len as u64;
                match found[kbase[j] + i].load(std::sync::atomic::Ordering::Relaxed) {
                    0 => push_lit(&mut segs, len),
                    pos1 => {
                        let pos = pos1 - 1;
                        let sf = starts.partition_point(|&x| x <= pos) - 1;
                        push_copy(&mut segs, files[sf].0, pos - starts[sf], len);
                    }
                }
            }
            // near, short copies go back to the LZ coder
            let mut kept = Vec::with_capacity(segs.len());
            let mut at = base;
            for g in segs {
                match g {
                    Seg::Copy { src, off, len } => {
                        let sf = files.partition_point(|f| f.0 < src);
                        let near = !text && at - (starts[sf] + off) < NEAR && len < NEAR_RUN;
                        if near || (sf == fi && len < SELF_MIN) {
                            push_lit(&mut kept, len);
                        } else {
                            push_copy(&mut kept, src, off, len);
                        }
                    }
                    Seg::Lit(l) => push_lit(&mut kept, l),
                }
                at += g.len();
            }
            if kept.iter().any(|g| matches!(g, Seg::Copy { .. })) { Some(kept) } else { None }
        });
        out.extend(res);
    }
    if tm {
        eprintln!("  dedup: {} files {} MB, chunking {tchunk:.3}s, chunk+index {:.3}s", files.len(), total >> 20, t0.elapsed().as_secs_f64());
    }
    let path_of: HashMap<usize, &Path> = files.iter().map(|&(i, p, _)| (i, p)).collect();
    // every copy is compared with its source on its own: all files' copies, in groups,
    // on every thread (a few big files would leave most threads idle)
    let mut items: Vec<(usize, u64, usize, u64, u64)> = Vec::new(); // (file, at, src, off, len)
    for (fi, segs) in out.iter().enumerate() {
        let mut at = 0u64;
        for g in segs.iter().flatten() {
            if let Seg::Copy { src, off, len } = *g {
                items.push((fi, at, src, off, len));
            }
            at += g.len();
        }
    }
    if tm {
        eprintln!("  dedup: {} copies, {} MB to confirm", items.len(), items.iter().map(|i| i.4).sum::<u64>() >> 20);
    }
    let mut groups: Vec<&[(usize, u64, usize, u64, u64)]> = Vec::new();
    let (mut g0, mut bytes) = (0usize, 0u64);
    for (k, it) in items.iter().enumerate() {
        bytes += it.4;
        if bytes >= 4 << 20 || k - g0 >= 1024 {
            groups.push(&items[g0..=k]);
            (g0, bytes) = (k + 1, 0);
        }
    }
    if g0 < items.len() {
        groups.push(&items[g0..]);
    }
    let ok: Vec<Result<Vec<bool>, (usize, io::Error)>> = crate::par::map(&groups, threads, |g| {
        let mut open: HashMap<usize, File> = HashMap::new();
        let mut scratch = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
        let mut r = Vec::with_capacity(g.len());
        for &(fi, at, src, off, len) in g.iter() {
            let own = files[fi].0;
            for e in [own, src] {
                if !open.contains_key(&e) {
                    // a few handles per job: jobs run on every thread, and the process may
                    // only hold so many files open
                    if open.len() >= OPEN_PER_JOB {
                        open.retain(|&k, _| k == own || k == src);
                    }
                    open.insert(e, File::open(path_of[&e]).map_err(|x| (fi, x))?);
                }
            }
            let same = same_at(&open[&own], at, &open[&src], off, len, &mut scratch).map_err(|x| (fi, x))?;
            r.push(same && (len >= SHORT || worth(&scratch.0[..len as usize])));
        }
        Ok(r)
    });
    let mut verdict = Vec::with_capacity(items.len());
    for r in ok {
        match r {
            Ok(v) => verdict.extend(v),
            Err((fi, e)) => return Err(io::Error::new(e.kind(), format!("{}: {e}", files[fi].1.display()))),
        }
    }
    let mut verdict = verdict.into_iter();
    let res = out
        .into_iter()
        .map(|segs| {
            let segs = segs?;
            let mut kept = Vec::with_capacity(segs.len());
            for g in segs {
                match g {
                    Seg::Copy { src, off, len } if verdict.next().unwrap_or(false) => push_copy(&mut kept, src, off, len),
                    g => push_lit(&mut kept, g.len()),
                }
            }
            kept.iter().any(|g| matches!(g, Seg::Copy { .. })).then_some(kept)
        })
        .collect();
    if tm {
        eprintln!("  dedup: confirmed at {:.3}s", t0.elapsed().as_secs_f64());
    }
    Ok(res)
}

/// Compares `len` bytes at `a_off` of `a` with `b_off` of `b` (a copy shorter than 1 MB
/// stays in `buf.0`).
fn same_at(a: &File, a_off: u64, b: &File, b_off: u64, len: u64, buf: &mut (Vec<u8>, Vec<u8>)) -> io::Result<bool> {
    let (x, y) = (&mut buf.0, &mut buf.1);
    let mut done = 0u64;
    while done < len {
        let k = (len - done).min(x.len() as u64) as usize;
        read_at(a, &mut x[..k], a_off + done)?;
        read_at(b, &mut y[..k], b_off + done)?;
        if x[..k] != y[..k] {
            return Ok(false);
        }
        done += k as u64;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    #[test]
    fn big_file_chunks_match() {
        let mut x = 5u64;
        let mut v: Vec<u8> = (0..11_000_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 7) as u8 * 31
            })
            .collect();
        let rep = v[100_000..900_000].to_vec();
        v.extend_from_slice(&rep);
        v.extend(std::iter::repeat_n(0u8, 200_000));
        let p = std::env::temp_dir().join(format!("nzip-dedup-test-{}", std::process::id()));
        std::fs::write(&p, &v).unwrap();
        let a = super::chunk_file(&p, v.len() as u64).unwrap();
        // a file cut in several slices, and small files next to it
        let q = p.with_extension("small");
        std::fs::write(&q, &v[..300_000]).unwrap();
        let b = super::chunk_file(&q, 300_000).unwrap();
        let round = [(0, p.as_path(), v.len() as u64), (1, q.as_path(), 300_000), (2, p.as_path(), v.len() as u64)];
        let r = super::chunk_round(&round, 4);
        assert!(r[0].as_ref().unwrap() == &a && r[1].as_ref().unwrap() == &b && r[2].as_ref().unwrap() == &a);
        let _ = std::fs::remove_file(&q);
        let _ = std::fs::remove_file(&p);
    }
}

