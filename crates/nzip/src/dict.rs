//! Word dictionary transform for text (max mode, before LZP + BWT).
//!
//! Frequent words (maximal runs of ASCII letters, case-folded) are replaced by 1-3 byte
//! codes from the high byte range; capitalised and ALL-CAPS words get a flag byte before
//! the code. Bytes that would be ambiguous (the escape/flag bytes and all bytes >= 0x80)
//! are escaped. The dictionary is chosen per block from the block itself and stored at
//! the front of the transformed block, so the BWT also compresses it.
//!
//! Measured (Python prototype, `train/dict/wrt.py`): enwik8 -1.45%, enwik9 -1.6%, and
//! enwik9 shrinks to 62% before sorting, small enough for a 16 GB card to sort in one go.
//!
//! Layout and code assignment depend only on the block's bytes (chunking is by size, not
//! thread count), so the output is identical on every machine; decoding is a table walk
//! that runs in parallel over the stored chunk table.

use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const ESC: u8 = 1;
const CAP: u8 = 2;
const UP: u8 = 3;
const MAGIC: u8 = b'W';
/// One-byte codes, two-byte prefixes; the remaining high values are three-byte prefixes.
const C1: usize = 32;
const C2: usize = 64;
const C3: usize = 128 - C1 - C2;
const MAX_WORDS: usize = 60_000;
const MAX_WORD_LEN: usize = 32;
const MIN_FREQ: u32 = 4;
/// Input bytes per independently coded chunk (fixed, so output never depends on threads).
const CHUNK: usize = 4 << 20;
/// Blocks smaller than this are left alone (dictionary overhead, little to gain).
pub const MIN_BLOCK: usize = 1 << 20;

#[inline]
fn is_letter(b: u8) -> bool {
    b.is_ascii_alphabetic()
}
#[inline]
fn special(b: u8) -> bool {
    b >= 0x80 || b == ESC || b == CAP || b == UP
}

/// FxHash (rustc's hasher): fast and good enough for short words.
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut v = [0u8; 8];
            v[..c.len()].copy_from_slice(c);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(v)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Chunk boundaries: every CHUNK bytes, moved forward so no letter run is split.
fn chunks(data: &[u8]) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut s = 0;
    while s < data.len() {
        let mut e = (s + CHUNK).min(data.len());
        while e < data.len() && is_letter(data[e - 1]) && is_letter(data[e]) {
            e += 1;
        }
        v.push((s, e));
        s = e;
    }
    v
}

/// Calls `f(word)` for each maximal letter run in `d`.
fn for_words(d: &[u8], mut f: impl FnMut(usize, usize)) {
    let mut i = 0;
    while i < d.len() {
        if is_letter(d[i]) {
            let s = i;
            while i < d.len() && is_letter(d[i]) {
                i += 1;
            }
            f(s, i);
        } else {
            i += 1;
        }
    }
}

fn count(data: &[u8], ch: &[(usize, usize)]) -> FxMap<Box<[u8]>, u32> {
    ch.par_iter()
        .map(|&(s, e)| {
            let d = &data[s..e];
            let mut m: FxMap<Box<[u8]>, u32> = FxMap::default();
            let mut buf = [0u8; MAX_WORD_LEN];
            for_words(d, |a, b| {
                let n = b - a;
                if !(2..=MAX_WORD_LEN).contains(&n) {
                    return;
                }
                for (o, &c) in buf.iter_mut().zip(&d[a..b]) {
                    *o = c.to_ascii_lowercase();
                }
                if let Some(v) = m.get_mut(&buf[..n]) {
                    *v += 1;
                } else {
                    m.insert(buf[..n].into(), 1);
                }
            });
            m
        })
        .reduce(FxMap::default, |mut a, b| {
            if a.len() < b.len() {
                return merge(b, a);
            }
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        })
}
fn merge(mut a: FxMap<Box<[u8]>, u32>, b: FxMap<Box<[u8]>, u32>) -> FxMap<Box<[u8]>, u32> {
    for (k, v) in b {
        *a.entry(k).or_insert(0) += v;
    }
    a
}

/// Code of the `i`-th dictionary word (in code order).
fn code(i: usize) -> ([u8; 3], usize) {
    if i < C1 {
        return ([0x80 + i as u8, 0, 0], 1);
    }
    let i = i - C1;
    if i < C2 * 128 {
        return ([0x80 + (C1 + i / 128) as u8, 0x80 + (i % 128) as u8, 0], 2);
    }
    let i = i - C2 * 128;
    let (a, r) = (i / (128 * 128), i % (128 * 128));
    ([0x80 + (C1 + C2 + a) as u8, 0x80 + (r / 128) as u8, 0x80 + (r % 128) as u8], 3)
}
const CAPACITY: usize = C1 + C2 * 128 + C3 * 128 * 128;

/// Picks the dictionary (in code order) from word counts.
fn select(counts: FxMap<Box<[u8]>, u32>) -> Vec<Box<[u8]>> {
    let mut cand: Vec<(Box<[u8]>, u32)> = counts.into_iter().filter(|(_, f)| *f >= MIN_FREQ).collect();
    // Value ~ bytes saved; ties broken by the word so the choice is deterministic.
    let score = |w: &[u8], f: u32| f as f64 * (w.len() as f64 - 1.5);
    cand.sort_unstable_by(|a, b| score(&b.0, b.1).total_cmp(&score(&a.0, a.1)).then_with(|| a.0.cmp(&b.0)));
    cand.truncate(MAX_WORDS.min(CAPACITY));
    // Most frequent words get the shortest codes...
    cand.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut words: Vec<Box<[u8]>> = cand.into_iter().map(|(w, _)| w).collect();
    // ...and within each code length, words sorted by their reversed spelling, so words
    // with the same ending get neighbouring codes (helps the BWT; measured -0.4%).
    let rev = |a: &Box<[u8]>, b: &Box<[u8]>| a.iter().rev().cmp(b.iter().rev());
    let n1 = C1.min(words.len());
    let n2 = (C1 + C2 * 128).min(words.len());
    words[..n1].sort_unstable_by(rev);
    words[n1..n2].sort_unstable_by(rev);
    words[n2..].sort_unstable_by(rev);
    words
}

fn put_u32(v: &mut Vec<u8>, x: usize) {
    v.extend_from_slice(&(x as u32).to_le_bytes());
}

/// Encodes one chunk.
fn encode_chunk(d: &[u8], table: &FxMap<&[u8], ([u8; 3], usize)>, out: &mut Vec<u8>) {
    let mut buf = [0u8; MAX_WORD_LEN];
    let mut i = 0;
    while i < d.len() {
        let c = d[i];
        if !is_letter(c) {
            if special(c) {
                out.push(ESC);
            }
            out.push(c);
            i += 1;
            continue;
        }
        let s = i;
        while i < d.len() && is_letter(d[i]) {
            i += 1;
        }
        let w = &d[s..i];
        let n = w.len();
        let mut hit = None;
        if (2..=MAX_WORD_LEN).contains(&n) {
            let mut upper = 0;
            for (o, &b) in buf.iter_mut().zip(w) {
                upper += b.is_ascii_uppercase() as usize;
                *o = b.to_ascii_lowercase();
            }
            let flag = if upper == 0 {
                Some(None)
            } else if upper == 1 && w[0].is_ascii_uppercase() {
                Some(Some(CAP))
            } else if upper == n {
                Some(Some(UP))
            } else {
                None
            };
            if let Some(flag) = flag {
                if let Some(&(cb, cl)) = table.get(&buf[..n]) {
                    hit = Some((flag, cb, cl));
                }
            }
        }
        match hit {
            Some((flag, cb, cl)) => {
                if let Some(f) = flag {
                    out.push(f);
                }
                out.extend_from_slice(&cb[..cl]);
            }
            None => out.extend_from_slice(w),
        }
    }
}

/// Transforms `data`, or returns None if it isn't worth it (not text-like, or it would
/// shrink by less than 15% before compression).
pub fn encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < MIN_BLOCK || !looks_textual(data) {
        return None;
    }
    let tr = std::env::var_os("NZIP_TRACE").is_some();
    let t0 = std::time::Instant::now();
    let ch = chunks(data);
    let counts = count(data, &ch);
    let t1 = t0.elapsed().as_secs_f64();
    let nuniq = counts.len();
    let words = select(counts);
    if tr {
        eprintln!("dict: count {t1:.2}s ({nuniq} distinct), select {:.2}s", t0.elapsed().as_secs_f64() - t1);
    }
    if words.is_empty() {
        return None;
    }
    let table: FxMap<&[u8], ([u8; 3], usize)> = words.iter().enumerate().map(|(i, w)| (&w[..], code(i))).collect();
    let bodies: Vec<Vec<u8>> = ch
        .par_iter()
        .map(|&(s, e)| {
            let mut o = Vec::with_capacity((e - s) * 3 / 4);
            encode_chunk(&data[s..e], &table, &mut o);
            o
        })
        .collect();
    if tr {
        eprintln!("dict: encoded at {:.2}s", t0.elapsed().as_secs_f64());
    }
    let dict_len: usize = words.iter().map(|w| w.len() + 1).sum();
    let body_len: usize = bodies.iter().map(|b| b.len()).sum();
    let total = 1 + 4 + 4 + dict_len + 4 + 8 * ch.len() + body_len;
    if total > data.len() / 20 * 17 {
        return None;
    }
    let mut out = Vec::with_capacity(total);
    out.push(MAGIC);
    put_u32(&mut out, words.len());
    put_u32(&mut out, dict_len);
    for w in &words {
        out.extend_from_slice(w);
        out.push(b'\n');
    }
    put_u32(&mut out, ch.len());
    for (b, &(s, e)) in bodies.iter().zip(&ch) {
        put_u32(&mut out, b.len());
        put_u32(&mut out, e - s);
    }
    for b in bodies {
        out.extend_from_slice(&b);
    }
    debug_assert_eq!(out.len(), total);
    Some(out)
}

/// Cheap check on a sample: mostly ASCII letters/spaces, few high bytes.
fn looks_textual(data: &[u8]) -> bool {
    let step = (data.len() / (1 << 20)).max(1);
    let (mut letters, mut high, mut n) = (0usize, 0usize, 0usize);
    for blk in data.chunks(4096).step_by(step).take(256) {
        for &b in blk {
            letters += (is_letter(b) || b == b' ') as usize;
            high += (b >= 0x80) as usize;
        }
        n += blk.len();
    }
    n > 0 && letters * 2 >= n && high * 10 <= n
}

/// Inverse of [`encode`]: writes exactly `out.len()` bytes or reports corruption.
pub fn decode(src: &[u8], out: &mut [u8]) -> Result<(), &'static str> {
    let mut p = 0usize;
    let mut take = |n: usize| -> Result<&[u8], &'static str> {
        let s = src.get(p..p + n).ok_or("dictionary block truncated")?;
        p += n;
        Ok(s)
    };
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
    if take(1)?[0] != MAGIC {
        return Err("bad dictionary block");
    }
    let n_words = u32_at(take(4)?);
    let dict_len = u32_at(take(4)?);
    if n_words > CAPACITY {
        return Err("bad dictionary size");
    }
    let dict = take(dict_len)?;
    let mut words: Vec<&[u8]> = Vec::with_capacity(n_words);
    for w in dict.split(|&b| b == b'\n').take(n_words) {
        words.push(w);
    }
    if words.len() != n_words || dict.last() != Some(&b'\n') && n_words > 0 {
        return Err("bad dictionary");
    }
    let n_chunks = u32_at(take(4)?);
    let tab = take(n_chunks.checked_mul(8).ok_or("bad chunk table")?)?;
    let mut parts = Vec::with_capacity(n_chunks);
    let (mut es, mut ds) = (0usize, 0usize);
    let body_start = p;
    for c in tab.chunks(8) {
        let (el, dl) = (u32_at(&c[..4]), u32_at(&c[4..]));
        parts.push((body_start + es, el, dl));
        es += el;
        ds += dl;
    }
    if body_start + es != src.len() || ds != out.len() {
        return Err("dictionary chunk table mismatch");
    }
    // Split `out` into the chunks' destination slices and decode them in parallel.
    let mut dsts = Vec::with_capacity(n_chunks);
    let mut rest = out;
    for &(_, _, dl) in &parts {
        let (a, b) = rest.split_at_mut(dl);
        dsts.push(a);
        rest = b;
    }
    parts.par_iter().zip(dsts.into_par_iter()).try_for_each(|(&(s, el, _), dst)| decode_chunk(&src[s..s + el], &words, dst))
}

fn decode_chunk(src: &[u8], words: &[&[u8]], dst: &mut [u8]) -> Result<(), &'static str> {
    let (mut i, mut o) = (0usize, 0usize);
    let n = words.len();
    let bad = "corrupt dictionary chunk";
    while i < src.len() {
        let b = src[i];
        i += 1;
        let mut flag = 0u8;
        let mut c = b;
        match b {
            ESC => {
                let v = *src.get(i).ok_or(bad)?;
                i += 1;
                *dst.get_mut(o).ok_or(bad)? = v;
                o += 1;
                continue;
            }
            CAP | UP => {
                flag = b;
                c = *src.get(i).ok_or(bad)?;
                i += 1;
                if c < 0x80 {
                    return Err(bad);
                }
            }
            0x00..=0x7f => {
                *dst.get_mut(o).ok_or(bad)? = b;
                o += 1;
                continue;
            }
            _ => {}
        }
        let hi = (c - 0x80) as usize;
        let idx = if hi < C1 {
            hi
        } else if hi < C1 + C2 {
            let x = *src.get(i).ok_or(bad)?;
            i += 1;
            C1 + (hi - C1) * 128 + (x.wrapping_sub(0x80) as usize & 0x7f)
        } else {
            let (x, y) = (*src.get(i).ok_or(bad)?, *src.get(i + 1).ok_or(bad)?);
            i += 2;
            C1 + C2 * 128 + (hi - C1 - C2) * 16384 + ((x.wrapping_sub(0x80) as usize & 0x7f) << 7) + (y.wrapping_sub(0x80) as usize & 0x7f)
        };
        if idx >= n {
            return Err(bad);
        }
        let w = words[idx];
        let d = dst.get_mut(o..o + w.len()).ok_or(bad)?;
        d.copy_from_slice(w);
        match flag {
            CAP => d[0] = d[0].to_ascii_uppercase(),
            UP => d.make_ascii_uppercase(),
            _ => {}
        }
        o += w.len();
    }
    if o != dst.len() { Err(bad) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(n: usize) -> Vec<u8> {
        let mut words: Vec<String> = ["the", "The", "THE", "of", "and", "Compression", "x", "naïve", "über", "WoRd"].iter().map(|s| s.to_string()).collect();
        for i in 0..20_000u32 {
            let w: String = (0..3 + i % 7).map(|k| (b'a' + ((i * 7 + k * 13) % 26) as u8) as char).collect();
            words.push(if i % 5 == 0 { w.to_uppercase() } else { w });
        }
        let mut v = Vec::new();
        let mut s = 12345u64;
        while v.len() < n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (s >> 33) as usize;
            // Zipf-ish: small indexes much more often.
            let idx = (r % words.len()) >> (r % 9);
            v.extend_from_slice(words[idx].as_bytes());
            v.push([b' ', b' ', b' ', b'\n', b',', 1, 2, 3, 0xff][(s >> 20) as usize % 9]);
        }
        v.truncate(n);
        v
    }

    #[test]
    fn roundtrip_text() {
        let d = text(3 << 20);
        let e = encode(&d).expect("text should transform");
        assert!(e.len() < d.len());
        let mut out = vec![0u8; d.len()];
        decode(&e, &mut out).unwrap();
        assert_eq!(out, d);
    }

    #[test]
    fn corrupt_is_rejected() {
        let d = text(2 << 20);
        let mut e = encode(&d).unwrap();
        let mut out = vec![0u8; d.len()];
        let n = e.len();
        e.truncate(n - 1);
        assert!(decode(&e, &mut out).is_err());
    }

    #[test]
    fn binary_is_skipped() {
        let d: Vec<u8> = (0..(2u32 << 20)).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        assert!(encode(&d).is_none());
    }
}
