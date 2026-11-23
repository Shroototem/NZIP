//! Mixed blocks: a block whose parts suit different engines is cut into pieces of three
//! kinds, each kind gathered into one stream:
//!   * text pieces (zip entries of JavaScript, JSON, XML...) go to the text engine,
//!   * binary pieces go to the LZ engine,
//!   * raw pieces (random-looking chunks with no repeats: video, audio, other archives)
//!     are stored as they are, so they cost memcpy time instead of a full LZ pass.
//! A piece table puts them back in order.
//!
//! Text cuts come from zip local file headers; raw cuts from a per-chunk test. The split
//! is lossless whatever the data says (the tests only choose where to cut).

use crate::container::{read_varint, write_varint};

pub const TEXT: u8 = 0;
pub const BIN: u8 = 1;
pub const RAW: u8 = 2;

/// A piece of the block: `len` bytes of kind TEXT, BIN or RAW.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub len: usize,
    pub kind: u8,
}

fn push(out: &mut Vec<Piece>, len: usize, kind: u8) {
    if len == 0 {
        return;
    }
    match out.last_mut() {
        Some(l) if l.kind == kind => l.len += len,
        _ => out.push(Piece { len, kind }),
    }
}

/// Text test for one piece: mostly letters, almost no control bytes.
pub fn is_text(b: &[u8]) -> bool {
    if b.len() < 64 {
        return false;
    }
    let mut letters = 0usize;
    let mut ctrl = 0usize;
    for &c in b {
        if c.is_ascii_alphabetic() {
            letters += 1;
        } else if c < 9 || (c > 13 && c < 32) || c == 127 {
            ctrl += 1;
        }
    }
    letters * 100 >= b.len() * 25 && ctrl * 200 <= b.len()
}

fn u16_at(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

/// Text/binary pieces of a block that contains zip entries (stored or deflated), or None
/// when it does not or when splitting would not separate text from binary.
pub fn zip_pieces(b: &[u8]) -> Option<Vec<Piece>> {
    let n = b.len();
    let mut out: Vec<Piece> = Vec::new();
    let mut entries = 0usize;
    let mut gap_start = 0usize;
    let mut i = 0usize;
    while i + 30 <= n {
        // find the next local file header signature
        match b[i..n - 29].windows(4).position(|w| w == b"PK\x03\x04") {
            Some(k) => i += k,
            None => break,
        }
        let flags = u16_at(b, i + 6);
        let method = u16_at(b, i + 8);
        let csize = u32_at(b, i + 18);
        let usize_ = u32_at(b, i + 22);
        let nlen = u16_at(b, i + 26);
        let xlen = u16_at(b, i + 28);
        let data = i + 30 + nlen + xlen;
        let ok = (method == 0 || method == 8)
            && flags & 8 == 0
            && nlen > 0
            && nlen < 1024
            && data <= n
            && csize <= n - data
            && (method != 0 || csize == usize_);
        if !ok {
            i += 1;
            continue;
        }
        push(&mut out, i - gap_start, BIN);
        push(&mut out, data - i, BIN);
        let body = &b[data..data + csize];
        push(&mut out, csize, if method == 0 && is_text(body) { TEXT } else { BIN });
        entries += 1;
        i = data + csize;
        gap_start = i;
    }
    push(&mut out, n - gap_start, BIN);
    let text: usize = out.iter().filter(|p| p.kind == TEXT).map(|p| p.len).sum();
    // worth it only when both streams are substantial
    (entries >= 8 && text >= n / 5 && n - text >= n / 50).then_some(out)
}

/// Raw-chunk test granularity.
const CHUNK: usize = 64 << 10;

/// Gear table for content-defined anchors (fixed, from splitmix64).
const GEAR: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < 256 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        t[i] = z ^ (z >> 31);
        i += 1;
    }
    t
};

/// Content-defined anchors of `b` (about one per 64 bytes): the rolling hash at each
/// anchor depends only on the 64 bytes before it, so repeated content gives equal keys
/// wherever it sits.
fn anchors(b: &[u8]) -> Vec<u64> {
    let mut out = Vec::with_capacity(b.len() / 48);
    let mut h = 0u64;
    for (i, &c) in b.iter().enumerate() {
        h = (h << 1).wrapping_add(GEAR[c as usize]);
        if h >> 58 == 0 && i >= 64 {
            out.push(h);
        }
    }
    out
}

/// Turns binary chunks that look random and share no content with the rest of the block
/// into raw pieces. A chunk stays binary when its byte entropy is below the limit or when
/// enough of its anchors also occur elsewhere in the block (either copy of a repeat keeps
/// the LZ engine's view of it).
pub fn raw_split(b: &[u8], pieces: &[Piece], threads: usize) -> Vec<Piece> {
    // chunk boundaries at block offsets that are multiples of CHUNK, inside BIN pieces
    let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // (piece index, start, end)
    let mut at = 0;
    for (k, p) in pieces.iter().enumerate() {
        if p.kind == BIN {
            let mut s = at;
            while s < at + p.len {
                let e = ((s / CHUNK + 1) * CHUNK).min(at + p.len);
                spans.push((k, s, e));
                s = e;
            }
        }
        at += p.len;
    }
    let info: Vec<(u32, Vec<u64>)> = crate::par::map(&spans, threads, |&(_, s, e)| {
        let c = &b[s..e];
        (crate::codec::entropy_milli(c), anchors(c))
    });
    let mut count: std::collections::HashMap<u64, u32> = std::collections::HashMap::with_capacity(b.len() / 48);
    for (_, a) in &info {
        for &h in a {
            *count.entry(h).or_insert(0) += 1;
        }
    }
    let raw: Vec<bool> = info
        .iter()
        .zip(&spans)
        .map(|((ent, a), &(_, s, e))| {
            let shared = a.iter().filter(|h| count[h] > 1).count();
            // ~64 bytes per anchor: raw when < 2% of the chunk is shared content
            e - s >= 4096 && *ent >= RAW_ENTROPY && shared * 64 * 50 < e - s
        })
        .collect();
    // every cut costs the binary stream (x86 targets and matches that span it), so only
    // long raw runs are cut out, and only when raw data is a real share of the block
    let mut raw = raw;
    let mut k = 0;
    while k < raw.len() {
        if !raw[k] {
            k += 1;
            continue;
        }
        let mut j = k;
        while j < raw.len() && raw[j] {
            j += 1;
        }
        let run: usize = spans[k..j].iter().map(|&(_, s, e)| e - s).sum();
        if run < MIN_RAW_RUN {
            raw[k..j].iter_mut().for_each(|r| *r = false);
        }
        k = j;
    }
    let total: usize = spans.iter().zip(&raw).filter(|(_, r)| **r).map(|(&(_, s, e), _)| e - s).sum();
    if total * 8 < b.len() {
        return pieces.to_vec();
    }
    let mut out = Vec::with_capacity(pieces.len() + 16);
    let mut si = 0;
    for (k, p) in pieces.iter().enumerate() {
        if p.kind != BIN {
            push(&mut out, p.len, p.kind);
            continue;
        }
        while si < spans.len() && spans[si].0 == k {
            let (_, s, e) = spans[si];
            push(&mut out, e - s, if raw[si] { RAW } else { BIN });
            si += 1;
        }
    }
    out
}

/// Byte entropy (milli-bits) at or above which a repeat-free chunk is stored raw.
const RAW_ENTROPY: u32 = 7990;
/// Shortest run of raw chunks worth cutting out of the binary stream.
const MIN_RAW_RUN: usize = 256 << 10;

/// Gathers the pieces into one stream per kind (TEXT, BIN, RAW).
pub fn gather(b: &[u8], pieces: &[Piece]) -> [Vec<u8>; 3] {
    let mut out: [Vec<u8>; 3] = Default::default();
    let mut at = 0;
    for p in pieces {
        out[p.kind as usize].extend_from_slice(&b[at..at + p.len]);
        at += p.len;
    }
    out
}

/// Inverse of `gather`.
pub fn scatter(parts: &[Vec<u8>], pieces: &[Piece]) -> Result<Vec<u8>, String> {
    let e = || "corrupt piece table".to_string();
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    let mut pos = [0usize; 3];
    for p in pieces {
        let k = p.kind as usize;
        let src = parts.get(k).ok_or_else(e)?;
        out.extend_from_slice(src.get(pos[k]..pos[k] + p.len).ok_or_else(e)?);
        pos[k] += p.len;
    }
    if pos.iter().zip(parts).any(|(&n, p)| n != p.len()) {
        return Err(e());
    }
    Ok(out)
}

/// Piece table: count, then each piece's kind and length.
pub fn write_table(pieces: &[Piece]) -> Vec<u8> {
    let mut w = Vec::new();
    write_varint(&mut w, pieces.len() as u64);
    for p in pieces {
        w.push(p.kind);
        write_varint(&mut w, p.len as u64);
    }
    w
}

pub fn read_table(mut r: &[u8], total: usize) -> Result<Vec<Piece>, String> {
    let e = || "corrupt piece table".to_string();
    let k = read_varint(&mut r).map_err(|_| e())? as usize;
    if k > total + 1 {
        return Err(e());
    }
    let mut out = Vec::with_capacity(k);
    let mut sum = 0usize;
    for _ in 0..k {
        let kind = *r.first().ok_or_else(e)?;
        r = &r[1..];
        if kind > RAW {
            return Err(e());
        }
        let len = read_varint(&mut r).map_err(|_| e())? as usize;
        sum = sum.checked_add(len).ok_or_else(e)?;
        out.push(Piece { len, kind });
    }
    if sum != total {
        return Err(e());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_split_round_trip() {
        // random bytes, a repeat of part of them, then text-like bytes
        let mut x = 1u64;
        let mut b: Vec<u8> = (0..2_000_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let rep = b[10_000..80_000].to_vec();
        b.extend_from_slice(&rep);
        b.extend((0..100_000).map(|i| b"some words here "[i % 16]));
        let pieces = raw_split(&b, &[Piece { len: b.len(), kind: BIN }], 4);
        assert!(pieces.iter().any(|p| p.kind == RAW));
        // the repeated region (both copies) stays binary
        let mut at = 0;
        for p in &pieces {
            if p.kind == RAW {
                assert!(at + p.len <= 10_000 || at >= 80_000);
            }
            at += p.len;
        }
        let parts = gather(&b, &pieces);
        assert_eq!(scatter(&parts, &pieces).unwrap(), b);
        let t = write_table(&pieces);
        assert_eq!(read_table(&t, b.len()).unwrap(), pieces);
    }
}
