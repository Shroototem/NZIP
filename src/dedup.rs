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
use std::io::{self, Read};
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
    let mut w = c.chunks_exact(8);
    for v in &mut w {
        let v = u64::from_le_bytes(v.try_into().unwrap());
        x = (x ^ v).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(23);
        y = (y ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31);
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

/// The chunks of a file in order: (key, length), and whether the file starts like text.
fn chunk_file(p: &Path, size: u64) -> io::Result<(Vec<(u64, u32)>, bool)> {
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

/// Files at least this big are chunked one at a time, each cut into slices hashed on
/// every thread (a few big files would otherwise leave most threads idle).
const BIG_FILE: u64 = 16 << 20;
/// Bytes of a big file in memory at once.
const WINDOW: u64 = 64 << 20;

/// `chunk_file` for big files, with the same chunks: a boundary depends only on the 64
/// bytes before it (the gear hash forgets older bytes) and on the minimum / maximum chunk
/// length since the previous boundary, so slices find candidate positions in parallel and
/// one cheap pass picks the boundaries in order.
fn chunk_big(p: &Path, size: u64, threads: usize, window: u64) -> io::Result<(Vec<(u64, u32)>, bool)> {
    let f = File::open(p)?;
    let bits = avg_bits();
    let (min_chunk, max_chunk) = (1u64 << (bits - 2), 1u64 << (bits + 3));
    let mut out = Vec::with_capacity((size >> bits) as usize + 1);
    let mut text = None;
    let mut start = 0u64; // the last boundary: the window starts there
    let mut buf = Vec::new();
    while start < size {
        let we = (start + window).min(size);
        let last = we == size;
        let b0 = start.saturating_sub(64); // 64 bytes before the window warm the hash up
        buf.resize((we - b0) as usize, 0);
        // filled by every thread (one sequential read would be the slow part)
        let parts: Vec<&mut [u8]> = buf.chunks_mut(4 << 20).collect();
        let errs = std::sync::Mutex::new(None);
        crate::par::for_chunks(parts, threads, |i, part| {
            if let Err(e) = read_at(&f, part, b0 + (i as u64) * (4 << 20)) {
                *errs.lock().unwrap() = Some(e);
            }
        });
        if let Some(e) = errs.into_inner().unwrap() {
            return Err(e);
        }
        text.get_or_insert_with(|| texty(&buf[..buf.len().min(64 << 10)]));
        // candidates: positions i in (start, we] where the hash of the bytes before i fits
        let n = (we - start) as usize;
        let k = threads.clamp(1, 64).min(n.div_ceil(1 << 20));
        let ranges: Vec<(usize, usize)> = (0..k).map(|j| (start as usize - b0 as usize + n * j / k, start as usize - b0 as usize + n * (j + 1) / k)).collect();
        let b = &buf;
        let cands: Vec<Vec<u64>> = crate::par::map(&ranges, threads, |&(r0, r1)| {
            // four quarters side by side: four independent hash chains keep the core busy
            let q = (r1 - r0) / 4;
            let st = [r0, r0 + q, r0 + 2 * q, r0 + 3 * q];
            let mut h = [0u64; 4];
            for k in 0..4 {
                for &c in &b[st[k].saturating_sub(64)..st[k]] {
                    h[k] = (h[k] << 1).wrapping_add(GEAR[c as usize]);
                }
            }
            let mut v: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity((q >> bits) * 2 + 8));
            for j in 0..q {
                for k in 0..4 {
                    let at = st[k] + j;
                    h[k] = (h[k] << 1).wrapping_add(GEAR[b[at] as usize]);
                    if h[k] >> (64 - bits) == 0 {
                        v[k].push(b0 + (at + 1) as u64);
                    }
                }
            }
            for at in r0 + 4 * q..r1 {
                h[3] = (h[3] << 1).wrapping_add(GEAR[b[at] as usize]);
                if h[3] >> (64 - bits) == 0 {
                    v[3].push(b0 + (at + 1) as u64);
                }
            }
            v.concat()
        });
        // boundaries, as `chunk_end` picks them
        let mut bounds = vec![start];
        let mut p0 = start;
        let mut it = cands.iter().flatten().copied().peekable();
        loop {
            while it.peek().is_some_and(|&c| c < p0 + min_chunk) {
                it.next();
            }
            let lim = (p0 + max_chunk).min(we);
            match it.peek() {
                Some(&c) if c <= lim => {
                    bounds.push(c);
                    p0 = c;
                    it.next();
                }
                _ if lim - p0 == max_chunk || (last && lim > p0) => {
                    bounds.push(lim);
                    p0 = lim;
                }
                _ => break,
            }
        }
        // keys of the chunks, in parallel
        let pairs: Vec<(u64, u64)> = bounds.windows(2).map(|w| (w[0], w[1])).collect();
        let groups: Vec<&[(u64, u64)]> = pairs.chunks(4096).collect();
        let keys: Vec<Vec<(u64, u32)>> = crate::par::map(&groups, threads, |g| g.iter().map(|&(a, z)| (chunk_key(&b[(a - b0) as usize..(z - b0) as usize]), (z - a) as u32)).collect());
        out.extend(keys.into_iter().flatten());
        if p0 == start {
            break; // nothing more (only happens at the end)
        }
        start = p0;
    }
    Ok((out, text.unwrap_or(false)))
}

/// Reads `buf.len()` bytes at `off` without moving a shared file position.
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            match f.seek_read(&mut buf[done..], off + done as u64)? {
                0 => return Err(io::ErrorKind::UnexpectedEof.into()),
                k => done += k,
            }
        }
        Ok(())
    }
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
        // small files side by side, big ones one at a time on every thread
        let mut chunks = crate::par::map(round, threads, |&(_, p, size)| if size < BIG_FILE { Some(chunk_file(p, size)) } else { None });
        for (c, &(_, p, size)) in chunks.iter_mut().zip(round) {
            if c.is_none() {
                *c = Some(chunk_big(p, size, threads, WINDOW));
            }
        }
        let chunks = chunks.into_iter().map(|c| c.unwrap()).collect::<io::Result<Vec<_>>>()?;
        tchunk += tc.elapsed().as_secs_f64();
        // every chunk of the round: (key, stream offset)
        let mut keys: Vec<(u64, u64)> = Vec::with_capacity(chunks.iter().map(|c| c.0.len()).sum());
        for (j, (ch, _)) in chunks.iter().enumerate() {
            let mut at = starts[out.len() + j];
            for &(key, len) in ch {
                keys.push((key, at));
                at += len as u64;
            }
        }
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); SHARDS];
        for (k, &(key, _)) in keys.iter().enumerate() {
            by[shard_of(key)].push(k as u32);
        }
        let ids: Vec<usize> = (0..SHARDS).collect();
        let hits: Vec<Vec<(u32, u64)>> = crate::par::map(&ids, threads, |&sh| {
            let mut ix = shards[sh].lock().unwrap();
            by[sh].iter().filter_map(|&k| ix.find_or_insert(keys[k as usize].0, keys[k as usize].1).map(|pos| (k, pos))).collect()
        });
        let mut found: Vec<Option<u64>> = vec![None; keys.len()];
        for (k, pos) in hits.into_iter().flatten() {
            found[k as usize] = Some(pos);
        }
        let mut kf = 0usize;
        for (ch, text) in chunks {
            let fi = out.len();
            let base = starts[fi];
            let mut segs = Vec::new();
            for (_, len) in ch {
                let len = len as u64;
                let hit = found[kf];
                kf += 1;
                match hit {
                    Some(pos) => {
                        let sf = starts.partition_point(|&x| x <= pos) - 1;
                        push_copy(&mut segs, files[sf].0, pos - starts[sf], len);
                    }
                    None => push_lit(&mut segs, len),
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
            out.push(if kept.iter().any(|g| matches!(g, Seg::Copy { .. })) { Some(kept) } else { None });
        }
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
        let mut v: Vec<u8> = (0..3_000_000)
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
        for w in [1 << 20, 3 << 20, 64 << 20] {
            assert!(super::chunk_big(&p, v.len() as u64, 4, w).unwrap() == a, "window {w}");
        }
        let _ = std::fs::remove_file(&p);
    }
}

