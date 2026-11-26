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
use std::io::{self, Read, Seek, SeekFrom};
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
    let mut index = Index::new(total >> avg_bits());
    let mut out = Vec::with_capacity(files.len());
    for round in files.chunks(ROUND) {
        let chunks = crate::par::map(round, threads, |&(_, p, size)| chunk_file(p, size));
        for ch in chunks {
            let (ch, text) = ch?;
            let fi = out.len();
            let base = starts[fi];
            let mut segs = Vec::new();
            let mut off = 0u64;
            for (key, len) in ch {
                let len = len as u64;
                match index.find_or_insert(key, base + off) {
                    Some(pos) => {
                        let sf = starts.partition_point(|&x| x <= pos) - 1;
                        push_copy(&mut segs, files[sf].0, pos - starts[sf], len);
                    }
                    None => push_lit(&mut segs, len),
                }
                off += len;
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
    let path_of: HashMap<usize, &Path> = files.iter().map(|&(i, p, _)| (i, p)).collect();
    let jobs: Vec<(&Path, Option<Vec<Seg>>)> = files.iter().map(|f| f.1).zip(out).collect();
    let checked = crate::par::map(&jobs, threads, |(p, segs)| match segs {
        None => Ok(None),
        Some(s) => confirm(p, s, &path_of).map(|s| Some(s).filter(|s| s.iter().any(|g| matches!(g, Seg::Copy { .. })))),
    });
    files.iter().zip(checked).map(|(f, c)| c.map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", f.1.display())))).collect()
}

/// Compares every copy in the pieces of file `p` with its source; a copy that differs (a
/// key collision) becomes literal.
fn confirm(p: &Path, segs: &[Seg], path_of: &HashMap<usize, &Path>) -> io::Result<Vec<Seg>> {
    let mut own = File::open(p)?;
    let mut src_file: Option<(usize, File)> = None;
    let mut out = Vec::with_capacity(segs.len());
    let mut at = 0u64;
    let mut scratch = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    for &g in segs {
        match g {
            Seg::Lit(l) => push_lit(&mut out, l),
            Seg::Copy { src, off, len } => {
                if src_file.as_ref().is_none_or(|s| s.0 != src) {
                    src_file = Some((src, File::open(path_of[&src])?));
                }
                let f = &mut src_file.as_mut().unwrap().1;
                if same_range(&mut own, at, f, off, len, &mut scratch)? && (len >= SHORT || worth(&scratch.0[..len as usize])) {
                    push_copy(&mut out, src, off, len);
                } else {
                    push_lit(&mut out, len);
                }
            }
        }
        at += g.len();
    }
    Ok(out)
}

/// Compares `len` bytes at `a_off` of `a` with `b_off` of `b`.
fn same_range(a: &mut File, a_off: u64, b: &mut File, b_off: u64, len: u64, buf: &mut (Vec<u8>, Vec<u8>)) -> io::Result<bool> {
    let (x, y) = (&mut buf.0, &mut buf.1);
    a.seek(SeekFrom::Start(a_off))?;
    b.seek(SeekFrom::Start(b_off))?;
    let mut left = len;
    while left > 0 {
        let k = left.min(x.len() as u64) as usize;
        a.read_exact(&mut x[..k])?;
        b.read_exact(&mut y[..k])?;
        if x[..k] != y[..k] {
            return Ok(false);
        }
        left -= k as u64;
    }
    Ok(true)
}
