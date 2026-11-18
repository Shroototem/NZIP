//! Long-range deduplication with content-defined chunking (gear hash, FastCDC-style).
//!
//! Block codecs only see one block at a time, so a repeat that is further away than a
//! block is invisible to them. Dedup finds such repeats across the whole input and
//! replaces them with copy references, leaving a smaller "unique" stream to compress.
//! Repeats closer than `min_distance` are left alone: the block codec handles those
//! better (and cheaper) itself.
//!
//! Recipe: a list of ops. Each op appends `lit` bytes from the unique stream, then
//! copies `copy_len` bytes from earlier output starting at `copy_src`. Copies may
//! overlap their destination (LZ77 semantics) and are applied front to back.

use crate::Error;
use rayon::prelude::*;
use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_128;

const MIN_CHUNK: usize = 4 * 1024;
const MASK: u64 = (1 << 14) - 1; // ~16 KiB average
const MAX_CHUNK: usize = 64 * 1024;
/// Chunking runs independently per segment (in parallel); fixed size keeps output
/// independent of the thread count.
const SEGMENT: usize = 8 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Op {
    pub lit: u64,
    pub copy_len: u64,
    pub copy_src: u64,
}

fn gear_table() -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for v in t.iter_mut() {
        // splitmix64
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *v = z ^ (z >> 31);
    }
    t
}

/// Chunk end offsets (relative to `data`).
fn chunk_ends(data: &[u8], gear: &[u64; 256]) -> Vec<usize> {
    let mut ends = Vec::with_capacity(data.len() / 12_000 + 1);
    let mut start = 0;
    while start < data.len() {
        let limit = (start + MAX_CHUNK).min(data.len());
        let mut i = (start + MIN_CHUNK).min(limit);
        let mut h: u64 = 0;
        let mut cut = limit;
        while i < limit {
            h = (h << 1).wrapping_add(gear[data[i] as usize]);
            if h & MASK == 0 {
                cut = i + 1;
                break;
            }
            i += 1;
        }
        ends.push(cut);
        start = cut;
    }
    ends
}

/// Streaming dedup state, carried across segments: fingerprints of every chunk seen so
/// far (bounded by `max_entries`), so a copy can reference anything earlier in the input.
pub struct Deduper {
    gear: [u64; 256],
    seen: HashMap<u128, u64>,
    min_distance: usize,
    max_entries: usize,
}

/// One segment's result: ops covering exactly the segment's bytes (copy sources are
/// absolute offsets), and the segment's unique (literal) bytes.
pub struct Segment {
    pub ops: Vec<Op>,
    pub unique: Vec<u8>,
    pub saved: u64,
}

impl Deduper {
    /// `max_entries` bounds memory (~48 bytes per entry): once full, new chunks are no
    /// longer remembered, but earlier ones still match.
    pub fn new(min_distance: usize, max_entries: usize) -> Self {
        Self { gear: gear_table(), seen: HashMap::new(), min_distance, max_entries }
    }

    /// Process `input[s..e]` (all earlier bytes of `input` must have been processed).
    pub fn segment(&mut self, input: &[u8], s: usize, e: usize) -> Segment {
        let gear = &self.gear;
        let subs: Vec<(usize, usize)> = (s..e).step_by(SEGMENT).map(|a| (a, (a + SEGMENT).min(e))).collect();
        let chunks: Vec<(usize, usize, u128)> = subs
            .par_iter()
            .flat_map_iter(|&(a, b)| {
                let seg = &input[a..b];
                let mut prev = 0;
                chunk_ends(seg, gear).into_iter().map(move |end| {
                    let c = (a + prev, end - prev, xxh3_128(&seg[prev..end]));
                    prev = end;
                    c
                }).collect::<Vec<_>>()
            })
            .collect();

        let mut ops = Vec::new();
        let mut unique = Vec::new();
        let mut lit_start = s; // start of the pending literal run
        let mut saved = 0u64;
        let mut last_copy: Option<(u64, u64)> = None; // (src_end, dst_end) of the last op's copy
        for &(off, len, h) in &chunks {
            let dup = match self.seen.get(&h) {
                Some(&src) => {
                    let src = src as usize;
                    (off - src > self.min_distance && input[src..src + len] == input[off..off + len]).then_some(src)
                }
                None => {
                    if self.seen.len() < self.max_entries {
                        self.seen.insert(h, off as u64);
                    }
                    None
                }
            };
            let Some(src) = dup else { continue };
            saved += len as u64;
            // Extend the previous copy if this one continues it with no literal between.
            if lit_start == off {
                if let (Some(op), Some((src_end, dst_end))) = (ops.last_mut(), last_copy) {
                    if src_end == src as u64 && dst_end == off as u64 {
                        let op: &mut Op = op;
                        op.copy_len += len as u64;
                        last_copy = Some((src_end + len as u64, dst_end + len as u64));
                        lit_start = off + len;
                        continue;
                    }
                }
            }
            unique.extend_from_slice(&input[lit_start..off]);
            ops.push(Op { lit: (off - lit_start) as u64, copy_len: len as u64, copy_src: src as u64 });
            last_copy = Some(((src + len) as u64, (off + len) as u64));
            lit_start = off + len;
        }
        unique.extend_from_slice(&input[lit_start..e]);
        ops.push(Op { lit: (e - lit_start) as u64, copy_len: 0, copy_src: 0 });
        Segment { ops, unique, saved }
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(b: &[u8], pos: &mut usize) -> Result<u64, Error> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*pos).ok_or(Error::Corrupt("truncated recipe"))?;
        *pos += 1;
        v |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(Error::Corrupt("bad varint in recipe"))
}

pub fn encode_ops(ops: &[Op]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ops.len() * 8 + 8);
    put_varint(&mut out, ops.len() as u64);
    for op in ops {
        put_varint(&mut out, op.lit);
        put_varint(&mut out, op.copy_len);
        if op.copy_len > 0 {
            put_varint(&mut out, op.copy_src);
        }
    }
    out
}

pub fn decode_ops(b: &[u8]) -> Result<Vec<Op>, Error> {
    let mut pos = 0;
    let n = get_varint(b, &mut pos)?;
    // Each op takes at least 2 bytes: bound the allocation by the input.
    if n > b.len() as u64 {
        return Err(Error::Corrupt("recipe op count too large"));
    }
    let mut ops = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let lit = get_varint(b, &mut pos)?;
        let copy_len = get_varint(b, &mut pos)?;
        let copy_src = if copy_len > 0 { get_varint(b, &mut pos)? } else { 0 };
        ops.push(Op { lit, copy_len, copy_src });
    }
    if pos != b.len() {
        return Err(Error::Corrupt("trailing bytes in recipe"));
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n).map(|_| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            x as u8
        }).collect()
    }

    /// Reference decoder for tests: rebuild the original from per-segment ops.
    fn rebuild(segs: &[Segment]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for sg in segs {
            let mut u = 0usize;
            for op in &sg.ops {
                out.extend_from_slice(&sg.unique[u..u + op.lit as usize]);
                u += op.lit as usize;
                for k in 0..op.copy_len as usize {
                    let b = out[op.copy_src as usize + k];
                    out.push(b);
                }
            }
            assert_eq!(u, sg.unique.len());
        }
        out
    }

    #[test]
    fn finds_far_repeats_across_segments() {
        let a = noise(3 << 20, 1);
        let b = noise(2 << 20, 2);
        let mut input = a.clone();
        input.extend_from_slice(&b);
        input.extend_from_slice(&a); // 3 MiB repeat, 5 MiB away
        for seg in [input.len(), 4 << 20, 1 << 20] {
            let mut d = Deduper::new(1 << 20, usize::MAX);
            let segs: Vec<Segment> = (0..input.len()).step_by(seg).map(|s| d.segment(&input, s, (s + seg).min(input.len()))).collect();
            let saved: u64 = segs.iter().map(|s| s.saved).sum();
            assert!(saved > 2 << 20, "segment {seg}: saved {saved}");
            for sg in &segs {
                assert_eq!(decode_ops(&encode_ops(&sg.ops)).unwrap(), sg.ops);
            }
            assert_eq!(rebuild(&segs), input);
        }
    }

    #[test]
    fn ignores_near_repeats() {
        let a = noise(1 << 20, 3);
        let mut input = a.clone();
        input.extend_from_slice(&a);
        let sg = Deduper::new(4 << 20, usize::MAX).segment(&input, 0, input.len());
        assert_eq!(sg.saved, 0);
        assert_eq!(sg.unique, input);
    }

    #[test]
    fn entry_cap_bounds_memory() {
        let a = noise(2 << 20, 4);
        let mut input = a.clone();
        input.extend_from_slice(&noise(2 << 20, 5));
        input.extend_from_slice(&a);
        let mut d = Deduper::new(1 << 20, 10);
        let sg = d.segment(&input, 0, input.len());
        assert!(d.seen.len() <= 10);
        assert_eq!(rebuild(&[sg]), input);
    }
}
