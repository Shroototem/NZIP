//! Repeated content across the files of a folder, beyond whole-file duplicates.
//!
//! Files are cut into content-defined chunks (gear rolling hash, ~8 KiB on average, so a
//! boundary depends only on the bytes around it and survives insertions earlier in the
//! file). A chunk whose bytes already appeared earlier in the stream becomes a copy of that
//! spot; neighbouring copies merge into runs, and every run is confirmed byte by byte.
//! Unpacking rebuilds copies from the files already written, so they cost no RAM and the
//! compressor never sees them. LZ windows reach megabytes back; this reaches across the
//! whole folder (rebuilt programs, per-profile copies of the same libraries, ...).

use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const AVG_BITS: u32 = 13;
const MIN_CHUNK: usize = 1 << (AVG_BITS - 2);
const MAX_CHUNK: usize = 1 << (AVG_BITS + 3);
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
fn chunk_key(c: &[u8]) -> u64 {
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
fn chunk_end(b: &[u8], s: usize, last: bool) -> Option<usize> {
    let lim = (s + MAX_CHUNK).min(b.len());
    // the hash only depends on the last 64 bytes, so it starts just before MIN_CHUNK
    let mut i = s + MIN_CHUNK - 64;
    if i >= lim {
        return if lim - s == MAX_CHUNK || last { Some(lim) } else { None };
    }
    let mut h = 0u64;
    while i < lim {
        h = (h << 1).wrapping_add(GEAR[b[i] as usize]);
        i += 1;
        if i - s >= MIN_CHUNK && h >> (64 - AVG_BITS) == 0 {
            return Some(i);
        }
    }
    if lim - s == MAX_CHUNK || last { Some(lim) } else { None }
}

/// The chunks of a file in order: (key, length).
fn chunk_file(p: &Path, size: u64) -> io::Result<Vec<(u64, u32)>> {
    let mut f = File::open(p)?.take(size);
    let mut out = Vec::with_capacity((size >> AVG_BITS) as usize + 1);
    let mut buf = vec![0u8; 4 << 20];
    let mut have = 0usize;
    let mut eof = false;
    loop {
        while !eof && have < buf.len() {
            let n = f.read(&mut buf[have..])?;
            if n == 0 {
                eof = true;
            }
            have += n;
        }
        let b = &buf[..have];
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
            return Ok(out);
        }
        buf.copy_within(s..have, 0);
        have -= s;
    }
}

/// Keys are hashes already: the table uses them as they are.
#[derive(Default)]
struct Ident(u64);

impl Hasher for Ident {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

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
    let mut seen: HashMap<u64, (u32, u64), BuildHasherDefault<Ident>> = HashMap::default();
    let mut out = Vec::with_capacity(files.len());
    for round in files.chunks(ROUND) {
        let chunks = crate::par::map(round, threads, |&(_, p, size)| chunk_file(p, size));
        for (k, ch) in chunks.into_iter().enumerate() {
            let ch = ch?;
            let fi = (out.len()) as u32;
            let mut segs = Vec::new();
            let mut off = 0u64;
            let mut copies = false;
            for (key, len) in ch {
                let len = len as u64;
                match seen.get(&key) {
                    Some(&(src, so)) => {
                        push_copy(&mut segs, files[src as usize].0, so, len);
                        copies = true;
                    }
                    None => {
                        seen.insert(key, (fi, off));
                        push_lit(&mut segs, len);
                    }
                }
                off += len;
            }
            debug_assert_eq!(off, round[k].2);
            out.push(if copies { Some(segs) } else { None });
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
                if same_range(&mut own, at, f, off, len, &mut scratch)? {
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
