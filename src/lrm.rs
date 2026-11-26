//! Long-range copies inside one block.
//!
//! The LZ engine matches at most `lzx::DICT` (16 MB) back, but a block can be 256 MB (or
//! more once zip entries are inflated) and often repeats itself from much further back:
//! the same library classes in many jars, sections of a big program built twice, ...
//! Repeats further back than that become copies: the block is cut into content-defined
//! chunks (the folder dedup's chunker), a chunk seen before becomes a copy, neighbouring
//! copies merge and are extended byte by byte at both ends. The LZ engine then codes only
//! the remaining bytes, and the decoder rebuilds the copies from the block's own output.
//!
//! ```text
//! table : count:varint { lit:varint len:varint dist:varint }*   (the rest is literal)
//! ```

use crate::container::{read_varint, write_varint};
use std::collections::HashMap;

/// Copies shorter than this must be `worth` it (see dedup.rs).
const SHORT: usize = 4 << 10;

struct Copy {
    at: usize,
    len: usize,
    dist: usize,
}

/// Splits `data` into a copy table and the literal bytes, if repeats further back than
/// `far` save enough to matter.
pub fn split(data: &[u8], far: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    if data.len() <= far + (1 << 20) {
        return None;
    }
    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(data.len() >> 8);
    let mut copies: Vec<Copy> = Vec::new();
    let mut s = 0;
    while s < data.len() {
        let e = crate::dedup::chunk_end(data, s, true).unwrap_or(data.len());
        let key = crate::dedup::chunk_key(&data[s..e]);
        // the latest earlier occurrence: if it is within reach, the LZ engine finds it
        if let Some(src) = seen.insert(key, s) {
            if s - src > far && data[src..src + (e - s)] == data[s..e] {
                let dist = s - src;
                match copies.last_mut() {
                    Some(c) if c.dist == dist && c.at + c.len == s => c.len += e - s,
                    _ => copies.push(Copy { at: s, len: e - s, dist }),
                }
            }
        }
        s = e;
    }
    drop(seen);
    // extend every copy over the bytes around it that still match, then keep those worth it
    let mut kept: Vec<Copy> = Vec::with_capacity(copies.len());
    for i in 0..copies.len() {
        let floor = kept.last().map_or(0, |k: &Copy| k.at + k.len);
        let ceil = copies.get(i + 1).map_or(data.len(), |n| n.at);
        let c = &copies[i];
        let (mut a, mut b) = (c.at, c.at + c.len);
        while a > floor && a > c.dist && data[a - 1] == data[a - 1 - c.dist] {
            a -= 1;
        }
        while b < ceil && data[b] == data[b - c.dist] {
            b += 1;
        }
        let len = b - a;
        if len >= SHORT || crate::dedup::worth(&data[a..b]) {
            kept.push(Copy { at: a, len, dist: c.dist });
        }
    }
    let saved: usize = kept.iter().map(|c| c.len).sum();
    if crate::sys::dev_flag("NZIP_TRACE") {
        eprintln!("  long-range copies: {} of {} MB in {} copies", saved >> 20, data.len() >> 20, kept.len());
    }
    if saved < 1 << 20 || crate::sys::dev_flag("NZ_NOLRM") {
        return None;
    }
    let mut table = Vec::with_capacity(kept.len() * 8 + 8);
    let mut lits = Vec::with_capacity(data.len() - saved);
    write_varint(&mut table, kept.len() as u64);
    let mut at = 0;
    for c in &kept {
        write_varint(&mut table, (c.at - at) as u64);
        write_varint(&mut table, c.len as u64);
        write_varint(&mut table, c.dist as u64);
        lits.extend_from_slice(&data[at..c.at]);
        at = c.at + c.len;
    }
    lits.extend_from_slice(&data[at..]);
    Some((table, lits))
}

/// Rebuilds a block of `raw_len` bytes from its copy table and literal bytes.
pub fn join(mut table: &[u8], lits: &[u8], raw_len: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt long-range table".to_string();
    let n = read_varint(&mut table).map_err(|_| e())? as usize;
    if n > raw_len {
        return Err(e());
    }
    let mut out = Vec::with_capacity(raw_len);
    let mut lp = 0usize;
    for _ in 0..n {
        let lit = read_varint(&mut table).map_err(|_| e())? as usize;
        let len = read_varint(&mut table).map_err(|_| e())? as usize;
        let dist = read_varint(&mut table).map_err(|_| e())? as usize;
        let l = lits.get(lp..lp.checked_add(lit).ok_or_else(e)?).ok_or_else(e)?;
        out.extend_from_slice(l);
        lp += lit;
        if dist == 0 || dist > out.len() || out.len() + len > raw_len {
            return Err(e());
        }
        // a copy may overlap its own output: in steps of at most `dist` bytes
        let mut left = len;
        while left > 0 {
            let k = left.min(dist);
            let from = out.len() - dist;
            out.extend_from_within(from..from + k);
            left -= k;
        }
    }
    out.extend_from_slice(lits.get(lp..).ok_or_else(e)?);
    if out.len() != raw_len {
        return Err(e());
    }
    Ok(out)
}
