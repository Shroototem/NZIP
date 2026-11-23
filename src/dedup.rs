//! Long-range dedup: removes long repeats (>= `min` bytes, any distance) before the main
//! engine, so BWT does not pay for duplicated files byte by byte.
//!
//! Output: the bytes that are not part of a repeat, plus records
//! `lit_len:varint (len - min):varint dist:varint` (one per repeat; trailing literals implied).

use crate::container::{read_varint, write_varint};

const S: usize = 16;

#[inline(always)]
fn hash_at(b: &[u8], p: usize, min: usize) -> u64 {
    let mut h = 0u64;
    for &c in &b[p..p + min] {
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(c as u64 + 1);
    }
    h
}

/// Splits `b` into (literals, records).
pub fn split(b: &[u8], min: usize) -> (Vec<u8>, Vec<u8>) {
    let n = b.len();
    let mut lits = Vec::with_capacity(n);
    let mut recs = Vec::new();
    if n < min + S {
        lits.extend_from_slice(b);
        return (lits, recs);
    }
    let bits = ((n / S).max(1 << 16).next_power_of_two().trailing_zeros() + 1).min(28);
    let mask = (1usize << bits) - 1;
    let mut tab = vec![0u32; 1 << bits];
    let slot = |h: u64| (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize & mask;
    // multiplier^min for the rolling update
    let mul = 0x100_0000_01B3u64;
    let mut pw = 1u64;
    for _ in 0..min {
        pw = pw.wrapping_mul(mul);
    }
    let mut lit_start = 0usize;
    let mut next_anchor = 0usize; // next multiple of S to insert
    let mut p = 0usize;
    let mut h = hash_at(b, 0, min);
    while p + min <= n {
        let c = tab[slot(h)] as usize;
        let mut matched = false;
        if c > 0 {
            let q = c - 1;
            if q < p && b[q..q + min] == b[p..p + min] {
                // extend forward
                let mut e = min;
                while p + e < n && b[q + e] == b[p + e] {
                    e += 1;
                }
                // extend backward into pending literals
                let mut s = 0;
                while p - s > lit_start && q > s && b[q - s - 1] == b[p - s - 1] {
                    s += 1;
                }
                let (mp, mq, ml) = (p - s, q - s, e + s);
                lits.extend_from_slice(&b[lit_start..mp]);
                write_varint(&mut recs, (mp - lit_start) as u64);
                write_varint(&mut recs, (ml - min) as u64);
                write_varint(&mut recs, (mp - mq) as u64);
                // anchors inside the repeat
                while next_anchor < mp + ml {
                    if next_anchor + min <= n {
                        tab[slot(hash_at(b, next_anchor, min))] = next_anchor as u32 + 1;
                    }
                    next_anchor += S;
                }
                p = mp + ml;
                lit_start = p;
                if p + min <= n {
                    h = hash_at(b, p, min);
                }
                matched = true;
            }
        }
        if !matched {
            if p == next_anchor {
                tab[slot(h)] = p as u32 + 1;
                next_anchor += S;
            }
            if p + min < n {
                // same value as hash_at(b, p + 1, min)
                h = h.wrapping_mul(mul).wrapping_sub(pw.wrapping_mul(b[p] as u64 + 1)).wrapping_add(b[p + min] as u64 + 1);
            }
            p += 1;
        }
    }
    lits.extend_from_slice(&b[lit_start..]);
    (lits, recs)
}

/// Rebuilds the input from `split`'s output.
pub fn join(lits: &[u8], mut recs: &[u8], n: usize, min: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt dedup records".to_string();
    let mut out = Vec::with_capacity(n);
    let mut li = 0usize;
    while !recs.is_empty() {
        let ll = read_varint(&mut recs).map_err(|_| e())? as usize;
        let ml = read_varint(&mut recs).map_err(|_| e())? as usize + min;
        let d = read_varint(&mut recs).map_err(|_| e())? as usize;
        let l = lits.get(li..li + ll).ok_or_else(e)?;
        out.extend_from_slice(l);
        li += ll;
        let p = out.len();
        if d == 0 || d > p || p + ml > n {
            return Err(e());
        }
        for i in 0..ml {
            let c = out[p - d + i];
            out.push(c);
        }
    }
    out.extend_from_slice(lits.get(li..).ok_or_else(e)?);
    if out.len() != n {
        return Err(e());
    }
    Ok(out)
}
