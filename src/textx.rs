//! Reversible word transform for text, built from the block's own data (no fixed
//! dictionary, so it adapts to any language that uses Latin letters, to source code, logs, XML…).
//!
//! Frequent words become 1- or 2-byte codes; capitalised / upper-case forms reuse the same
//! code after a flag byte. Code bytes are chosen among the byte values the block rarely
//! uses; their literal occurrences are escaped. The CM model then sees fewer, more
//! meaningful symbols: higher-order contexts span more text and there is less to code.
//!
//! ```text
//! 'W' m:u8 code_bytes[m] m1:u8 nwords:varint words(a-z, ' '-terminated)... body
//! code_bytes = [ESC, CAP, UPPER, one-byte codes (m1)..., two-byte prefixes...]
//! ```

use crate::container::{read_varint, write_varint};
use std::collections::HashMap;

const MAGIC: u8 = b'W';

/// Cheap test: most bytes are letters, digits, spaces or common punctuation.
pub fn looks_like_text(data: &[u8]) -> bool {
    if data.len() < 4096 {
        return false;
    }
    let sample = &data[..data.len().min(1 << 20)];
    let mut letters = 0usize;
    let mut ctrl = 0usize;
    for &b in sample {
        if b.is_ascii_alphabetic() {
            letters += 1;
        } else if b < 9 || (b > 13 && b < 32) {
            ctrl += 1;
        }
    }
    letters * 100 >= sample.len() * 45 && ctrl * 100 <= sample.len()
}

#[derive(PartialEq, Clone, Copy)]
enum Case {
    Lower,
    Cap,
    Upper,
    Mixed,
}

fn case_of(w: &[u8]) -> Case {
    let up = w.iter().filter(|b| b.is_ascii_uppercase()).count();
    if up == 0 {
        Case::Lower
    } else if up == w.len() && w.len() >= 2 {
        Case::Upper
    } else if up == 1 && w[0].is_ascii_uppercase() {
        Case::Cap
    } else {
        Case::Mixed
    }
}

fn words(data: &[u8], mut f: impl FnMut(usize, usize)) {
    let mut i = 0;
    let n = data.len();
    while i < n {
        if data[i].is_ascii_alphabetic() {
            let s = i;
            while i < n && data[i].is_ascii_alphabetic() {
                i += 1;
            }
            f(s, i);
        } else {
            i += 1;
        }
    }
}

/// Input is processed in ~1 MiB chunks cut at word boundaries. The cut points depend only
/// on the data, so output never depends on the thread count.
const CHUNK: usize = 1 << 20;

fn chunk_ranges(data: &[u8]) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut s = 0;
    while s < data.len() {
        let mut e = (s + CHUNK).min(data.len());
        while e < data.len() && data[e].is_ascii_alphabetic() {
            e += 1;
        }
        v.push((s, e));
        s = e;
    }
    v
}

fn count_words(data: &[u8]) -> HashMap<Vec<u8>, u32> {
    let mut freq: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut lw: Vec<u8> = Vec::with_capacity(64);
    words(data, |s, e| {
        let w = &data[s..e];
        if e - s < 2 || e - s > 40 || case_of(w) == Case::Mixed {
            return;
        }
        lw.clear();
        lw.extend(w.iter().map(|b| b.to_ascii_lowercase()));
        match freq.get_mut(&lw[..]) {
            Some(v) => *v += 1,
            None => {
                freq.insert(lw.clone(), 1);
            }
        }
    });
    freq
}

struct Codes<'a> {
    code_bytes: &'a [u8],
    is_code: [bool; 256],
    m1: usize,
    n_one: usize,
    index: HashMap<&'a [u8], u32>,
}

fn encode_body(data: &[u8], c: &Codes) -> Vec<u8> {
    let (esc, cap, upper) = (c.code_bytes[0], c.code_bytes[1], c.code_bytes[2]);
    let mut out = Vec::with_capacity(data.len() * 2 / 3 + 16);
    let n = data.len();
    let mut i = 0;
    let mut lw: Vec<u8> = Vec::with_capacity(64);
    while i < n {
        let b = data[i];
        if b.is_ascii_alphabetic() {
            let s = i;
            while i < n && data[i].is_ascii_alphabetic() {
                i += 1;
            }
            let w = &data[s..i];
            let case = case_of(w);
            let hit = if case == Case::Mixed || w.len() > 40 {
                None
            } else {
                lw.clear();
                lw.extend(w.iter().map(|b| b.to_ascii_lowercase()));
                c.index.get(&lw[..]).copied()
            };
            match hit {
                Some(code) => {
                    match case {
                        Case::Cap => out.push(cap),
                        Case::Upper => out.push(upper),
                        _ => {}
                    }
                    let code = code as usize;
                    if code < c.n_one {
                        out.push(c.code_bytes[3 + code]);
                    } else {
                        let j = code - c.n_one;
                        out.push(c.code_bytes[3 + c.m1 + j / 256]);
                        out.push((j % 256) as u8);
                    }
                }
                None => out.extend_from_slice(w),
            }
        } else {
            if c.is_code[b as usize] {
                out.push(esc);
            }
            out.push(b);
            i += 1;
        }
    }
    out
}

pub fn encode(data: &[u8], threads: usize) -> Option<Vec<u8>> {
    let ranges = chunk_ranges(data);
    // 1. byte usage -> code bytes = rarest non-letter values
    let hists = crate::par::map(&ranges, threads, |&(s, e)| {
        let mut c = [0u64; 256];
        for &b in &data[s..e] {
            c[b as usize] += 1;
        }
        c
    });
    let mut cnt = [0u64; 256];
    for h in &hists {
        for i in 0..256 {
            cnt[i] += h[i];
        }
    }
    let mut order: Vec<u8> = (0..=255u8).filter(|b| !b.is_ascii_alphabetic()).collect();
    order.sort_by_key(|&b| (cnt[b as usize], b));
    let budget = data.len() as u64 / 300;
    let mut used = 0u64;
    let mut code_bytes = Vec::new();
    for &b in &order {
        if code_bytes.len() >= 3 + 192 || used + cnt[b as usize] > budget {
            break;
        }
        used += cnt[b as usize];
        code_bytes.push(b);
    }
    if code_bytes.len() < 3 + 8 {
        return None;
    }
    let ncodes = code_bytes.len() - 3;
    let m1 = (ncodes / 2).min(96);
    let m2 = ncodes - m1;

    // 2. word statistics (lower-cased; mixed-case words are left alone), counted per chunk on
    //    an evenly spread sample of at most ~48 chunks: only the top few thousand words matter,
    //    and the sample ranks them the same as the whole block would.
    let step = ranges.len().div_ceil(48).max(1);
    let sample: Vec<(usize, usize)> = ranges.iter().copied().step_by(step).collect();
    let maps = crate::par::map(&sample, threads, |&(s, e)| count_words(&data[s..e]));
    let mut freq: HashMap<Vec<u8>, u32> = HashMap::new();
    for mp in maps {
        if freq.is_empty() {
            freq = mp;
            continue;
        }
        for (w, f) in mp {
            *freq.entry(w).or_insert(0) += f;
        }
    }
    let mut cand: Vec<(Vec<u8>, u32)> = freq.into_iter().filter(|(_, f)| *f >= 3).collect();
    // by frequency (ties: alphabetical) - deterministic
    cand.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut one: Vec<Vec<u8>> = Vec::new();
    let mut two: Vec<Vec<u8>> = Vec::new();
    for (w, _) in cand {
        if one.len() < m1 {
            one.push(w);
        } else if two.len() < m2 * 256 {
            if w.len() >= 3 {
                two.push(w);
            }
        } else {
            break;
        }
    }
    if one.is_empty() {
        return None;
    }
    // alphabetical inside each tier: neighbouring codes = similar words
    one.sort();
    two.sort();
    let dict: Vec<Vec<u8>> = one.iter().chain(two.iter()).cloned().collect();
    let mut is_code = [false; 256];
    for &b in &code_bytes {
        is_code[b as usize] = true;
    }
    let codes = Codes {
        code_bytes: &code_bytes,
        is_code,
        m1,
        n_one: one.len(),
        index: dict.iter().enumerate().map(|(i, w)| (&w[..], i as u32)).collect(),
    };

    // 3. bodies, in parallel
    let bodies = crate::par::map(&ranges, threads, |&(s, e)| encode_body(&data[s..e], &codes));

    // 4. header
    let mut out = Vec::with_capacity(bodies.iter().map(|b| b.len()).sum::<usize>() + dict.len() * 8 + 1024);
    out.push(MAGIC);
    out.push(code_bytes.len() as u8);
    out.extend_from_slice(&code_bytes);
    out.push(one.len() as u8);
    write_varint(&mut out, dict.len() as u64);
    for w in &dict {
        out.extend_from_slice(w);
        out.push(b' ');
    }
    write_varint(&mut out, bodies.len() as u64);
    for b in &bodies {
        write_varint(&mut out, b.len() as u64);
    }
    for b in bodies {
        out.extend_from_slice(&b);
    }
    Some(out)
}

struct Roles<'a> {
    role: [u8; 256],
    val: [u32; 256],
    m1: usize,
    dict: Vec<&'a [u8]>,
}

fn decode_body(r: &[u8], t: &Roles, out: &mut Vec<u8>) -> Result<(), String> {
    let err = || "corrupt text transform".to_string();
    let mut i = 0;
    let n = r.len();
    let mut case = 0u8;
    while i < n {
        let b = r[i];
        i += 1;
        let word = match t.role[b as usize] {
            0 => {
                out.push(b);
                continue;
            }
            1 => {
                out.push(*r.get(i).ok_or_else(err)?);
                i += 1;
                continue;
            }
            2 | 3 => {
                case = t.role[b as usize];
                continue;
            }
            4 => t.val[b as usize] as usize,
            _ => {
                let lo = *r.get(i).ok_or_else(err)? as usize;
                i += 1;
                t.m1 + t.val[b as usize] as usize * 256 + lo
            }
        };
        let w = t.dict.get(word).ok_or_else(err)?;
        let start = out.len();
        out.extend_from_slice(w);
        if case == 2 {
            out[start] = out[start].to_ascii_uppercase();
        } else if case == 3 {
            out[start..].make_ascii_uppercase();
        }
        case = 0;
    }
    Ok(())
}

pub fn decode(src: &[u8], threads: usize) -> Result<Vec<u8>, String> {
    let err = || "corrupt text transform".to_string();
    let mut r = src;
    let mut take = |n: usize| -> Result<&[u8], String> {
        if r.len() < n {
            return Err(err());
        }
        let (a, b) = r.split_at(n);
        r = b;
        Ok(a)
    };
    if take(1)?[0] != MAGIC {
        return Err(err());
    }
    let m = take(1)?[0] as usize;
    let code_bytes = take(m)?.to_vec();
    if m < 4 {
        return Err(err());
    }
    let m1 = take(1)?[0] as usize;
    let mut rr: &[u8] = r;
    let nwords = read_varint(&mut rr).map_err(|_| err())? as usize;
    r = rr;
    let mut dict: Vec<&[u8]> = Vec::with_capacity(nwords.min(1 << 20));
    for _ in 0..nwords {
        let e = r.iter().position(|&b| b == b' ').ok_or_else(err)?;
        dict.push(&r[..e]);
        r = &r[e + 1..];
    }
    // role of each byte: 0 literal, 1 esc, 2 cap, 3 upper, 4 one-byte code, 5 prefix
    let mut role = [0u8; 256];
    let mut val = [0u32; 256];
    for (i, &b) in code_bytes.iter().enumerate() {
        role[b as usize] = match i {
            0 => 1,
            1 => 2,
            2 => 3,
            _ if i - 3 < m1 => 4,
            _ => 5,
        };
        val[b as usize] = if i >= 3 + m1 { (i - 3 - m1) as u32 } else { i.saturating_sub(3) as u32 };
    }
    let roles = Roles { role, val, m1, dict };
    let mut rr: &[u8] = r;
    let nb = read_varint(&mut rr).map_err(|_| err())? as usize;
    let mut lens = Vec::with_capacity(nb.min(1 << 20));
    for _ in 0..nb {
        lens.push(read_varint(&mut rr).map_err(|_| err())? as usize);
    }
    let mut parts = Vec::with_capacity(nb);
    let mut off = 0usize;
    for l in lens {
        parts.push(rr.get(off..off + l).ok_or_else(err)?);
        off += l;
    }
    let outs = crate::par::map(&parts, threads, |p| {
        let mut o = Vec::with_capacity(p.len() * 3 / 2);
        decode_body(p, &roles, &mut o).map(|_| o)
    });
    let mut out = Vec::with_capacity(outs.iter().map(|o| o.as_ref().map_or(0, |v| v.len())).sum());
    for o in outs {
        out.extend_from_slice(&o?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let mut t = Vec::new();
        for i in 0..3000 {
            t.extend_from_slice(b"The quick brown fox jumps over the LAZY dog; THE end. ");
            t.extend_from_slice(&[format!("item{i} \x01\x02 mIxEd Words and more words\n").as_bytes(), &[0xff, b' ']].concat());
        }
        let e = encode(&t, 4).expect("transform");
        assert!(e.len() < t.len());
        assert_eq!(decode(&e, 3).unwrap(), t);
        assert_eq!(encode(&t, 1).unwrap(), e, "output must not depend on threads");
    }
}

#[cfg(test)]
mod dbg {
    #[test]
    #[ignore]
    fn stats() {
        let p = std::env::var("NZ_FILE").unwrap();
        let d = std::fs::read(p).unwrap();
        let e = super::encode(&d, 8).unwrap();
        eprintln!("in {} out {} m {} m1 {} nwords {}", d.len(), e.len(), e[1], e[2 + e[1] as usize], {
            let mut r = &e[3 + e[1] as usize..];
            crate::container::read_varint(&mut r).unwrap()
        });
    }
}
