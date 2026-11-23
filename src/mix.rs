//! Mixed blocks: a block whose parts suit different engines (a .jar / omni.ja / zip of
//! JavaScript next to images and binary tables) is cut into pieces, text pieces are
//! gathered into one stream for the text engine and the rest into one for the binary
//! engine. A piece table puts them back in order.
//!
//! Piece boundaries come from the zip local file headers when the block holds a zip;
//! the split is lossless whatever the headers say (they only choose where to cut).

use crate::container::{read_varint, write_varint};

/// A piece of the block: `len` bytes that go to the text stream when `text` is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub len: usize,
    pub text: bool,
}

/// Text test for one piece: mostly letters, almost no control bytes.
fn is_text(b: &[u8]) -> bool {
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

/// Pieces of a block that contains zip entries (stored or deflated), or None when it
/// does not or when splitting would not separate text from binary.
pub fn zip_pieces(b: &[u8]) -> Option<Vec<Piece>> {
    let n = b.len();
    let mut out: Vec<Piece> = Vec::new();
    let push = |out: &mut Vec<Piece>, len: usize, text: bool| {
        if len == 0 {
            return;
        }
        match out.last_mut() {
            Some(l) if l.text == text => l.len += len,
            _ => out.push(Piece { len, text }),
        }
    };
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
        push(&mut out, i - gap_start, false);
        push(&mut out, data - i, false);
        let body = &b[data..data + csize];
        push(&mut out, csize, method == 0 && is_text(body));
        entries += 1;
        i = data + csize;
        gap_start = i;
    }
    push(&mut out, n - gap_start, false);
    let text: usize = out.iter().filter(|p| p.text).map(|p| p.len).sum();
    // worth it only when both streams are substantial
    (entries >= 8 && text >= n / 5 && n - text >= n / 50).then_some(out)
}

/// Gathers the text pieces and the binary pieces into two streams.
pub fn gather(b: &[u8], pieces: &[Piece]) -> (Vec<u8>, Vec<u8>) {
    let mut t = Vec::new();
    let mut o = Vec::new();
    let mut at = 0;
    for p in pieces {
        let s = &b[at..at + p.len];
        if p.text { t.extend_from_slice(s) } else { o.extend_from_slice(s) }
        at += p.len;
    }
    (t, o)
}

/// Inverse of `gather`.
pub fn scatter(t: &[u8], o: &[u8], pieces: &[Piece]) -> Result<Vec<u8>, String> {
    let e = || "corrupt piece table".to_string();
    let mut out = Vec::with_capacity(t.len() + o.len());
    let (mut ti, mut oi) = (0usize, 0usize);
    for p in pieces {
        if p.text {
            out.extend_from_slice(t.get(ti..ti + p.len).ok_or_else(e)?);
            ti += p.len;
        } else {
            out.extend_from_slice(o.get(oi..oi + p.len).ok_or_else(e)?);
            oi += p.len;
        }
    }
    if ti != t.len() || oi != o.len() {
        return Err(e());
    }
    Ok(out)
}

/// Piece table: count, first piece's kind, then lengths (kinds alternate).
pub fn write_table(pieces: &[Piece]) -> Vec<u8> {
    let mut w = Vec::new();
    write_varint(&mut w, pieces.len() as u64);
    w.push(pieces.first().map_or(0, |p| p.text as u8));
    for p in pieces {
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
    let mut text = *r.first().ok_or_else(e)? == 1;
    r = &r[1..];
    let mut out = Vec::with_capacity(k);
    let mut sum = 0usize;
    for _ in 0..k {
        let len = read_varint(&mut r).map_err(|_| e())? as usize;
        sum = sum.checked_add(len).ok_or_else(e)?;
        out.push(Piece { len, text });
        text = !text;
    }
    if sum != total {
        return Err(e());
    }
    Ok(out)
}
