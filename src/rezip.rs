//! Deflate recompression. Zip entries (jar, apk, docx, nupkg...) and gzip members hold
//! data that zlib already squeezed with a weak 32 KB-window coder, and that the engines
//! cannot see through. When our zlib port rebuilds a stream bit for bit from its inflated
//! bytes, the block carries the inflated bytes instead (plus the level and memLevel that
//! rebuild it), and the decoder deflates them again. Streams no level reproduces stay as
//! they are, so the result is lossless whatever the data.

use crate::container::{read_varint, write_varint};
use crate::mix::{BIN, Piece, TEXT};

/// One rebuilt stream: `gap` unchanged bytes since the previous one, then a deflate
/// stream of `comp` bytes that inflates to `raw` bytes and deflates back with `params`
/// (level | memLevel << 4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rec {
    pub gap: usize,
    pub raw: usize,
    pub comp: usize,
    pub params: u8,
}

/// Levels / memLevels to try, most common first (Java and most zip tools use 6 / 8).
const TRIES: [(u8, u8); 10] = [(6, 8), (9, 8), (1, 8), (5, 8), (4, 8), (7, 8), (8, 8), (3, 8), (2, 8), (6, 9)];

/// Smallest stream worth the bookkeeping.
const MIN_COMP: usize = 64;

/// Candidate stream: where the deflate data starts, the header's compressed size if it
/// gives one, and whether the content (once inflated) belongs to a zip entry.
struct Cand {
    start: usize,
    csize: Option<usize>,
}

fn u16_at(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

/// Deflate streams named by zip local headers (method 8) and gzip member headers.
fn candidates(b: &[u8]) -> Vec<Cand> {
    let mut out = Vec::new();
    let n = b.len();
    let mut i = 0;
    while i + 30 <= n {
        let Some(k) = b[i..n - 9].windows(2).position(|w| w == b"PK" || w == [0x1F, 0x8B]) else { break };
        i += k;
        if b[i] == b'P' {
            if i + 30 <= n && b[i + 2] == 3 && b[i + 3] == 4 && u16_at(b, i + 8) == 8 {
                let flags = u16_at(b, i + 6);
                let nlen = u16_at(b, i + 26);
                let xlen = u16_at(b, i + 28);
                let start = i + 30 + nlen + xlen;
                let csize = u32_at(b, i + 18);
                if flags & 1 == 0 && nlen > 0 && nlen < 1024 && start < n {
                    let csize = if flags & 8 == 0 { Some(csize) } else { None };
                    if csize.is_none_or(|c| c >= MIN_COMP && c <= n - start) {
                        out.push(Cand { start, csize });
                    }
                }
            }
        } else if b[i + 2] == 8 && b[i + 3] & 0xE0 == 0 {
            // gzip: fixed 10-byte header, then optional extra / name / comment / crc
            let flg = b[i + 3];
            let mut p = i + 10;
            let mut ok = true;
            if flg & 4 != 0 {
                ok = p + 2 <= n;
                if ok {
                    p += 2 + u16_at(b, p);
                }
            }
            for bit in [8u8, 16] {
                if ok && flg & bit != 0 {
                    match b.get(p..).and_then(|s| s.iter().position(|&c| c == 0)) {
                        Some(z) => p += z + 1,
                        None => ok = false,
                    }
                }
            }
            if flg & 2 != 0 {
                p += 2;
            }
            if ok && p < n {
                out.push(Cand { start: p, csize: None });
            }
        }
        i += 2;
    }
    out
}

/// The levels that rebuild `comp` from `raw`, or None.
fn find_params(raw: &[u8], comp: &[u8]) -> Option<u8> {
    TRIES.iter().find(|&&(l, m)| crate::zlib::deflate_eq(raw, l, m, comp)).map(|&(l, m)| l | (m << 4))
}

/// Text test for an inflated entry (same rule as the zip split).
fn is_text(b: &[u8]) -> bool {
    crate::mix::is_text(b)
}

pub struct Expanded {
    pub data: Vec<u8>,
    pub recs: Vec<Rec>,
    /// text / binary pieces of `data` (inflated entries classified by content)
    pub pieces: Vec<Piece>,
}

/// Replaces every deflate stream in `b` that our zlib port rebuilds exactly with its
/// inflated bytes. None when there is nothing to gain. The output stays under `limit`.
pub fn expand(b: &[u8], threads: usize, limit: usize) -> Option<Expanded> {
    if crate::sys::dev_flag("NZ_NOREZIP") {
        return None;
    }
    let tm = crate::sys::dev_flag("NZ_RZT");
    let t0 = std::time::Instant::now();
    let lap = |what: &str| {
        if tm {
            eprintln!("  rezip {what}: {:.2}s", t0.elapsed().as_secs_f64());
        }
    };
    let cands = candidates(b);
    lap("scan");
    if cands.is_empty() {
        return None;
    }
    // inflate every candidate in parallel (false positives fail fast)
    let cap = limit.min(1 << 30);
    let found: Vec<Option<(usize, usize, Vec<u8>)>> = crate::par::map(&cands, threads, |c| {
        let src = &b[c.start..];
        let src = match c.csize {
            Some(cs) => &src[..cs],
            None => src,
        };
        let (raw, used) = crate::zlib::inflate(src, cap)?;
        if c.csize.is_some_and(|cs| cs != used) || used < MIN_COMP {
            return None;
        }
        Some((c.start, used, raw))
    });
    lap("inflate");
    // keep non-overlapping streams in order
    let mut picks: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    let mut end = 0;
    for (s, used, raw) in found.into_iter().flatten() {
        if s >= end {
            end = s + used;
            picks.push((s, used, raw));
        }
    }
    if picks.is_empty() {
        return None;
    }
    // which ones zlib rebuilds, and how
    let params: Vec<Option<(u8, bool)>> =
        crate::par::map(&picks, threads, |(s, used, raw)| find_params(raw, &b[*s..*s + *used]).map(|p| (p, is_text(raw))));
    lap("verify");
    let mut out = Vec::with_capacity(b.len() * 2);
    let mut recs = Vec::new();
    let mut pieces: Vec<Piece> = Vec::new();
    let push = |pieces: &mut Vec<Piece>, len: usize, kind: u8| {
        if len == 0 {
            return;
        }
        match pieces.last_mut() {
            Some(l) if l.kind == kind => l.len += len,
            _ => pieces.push(Piece { len, kind }),
        }
    };
    let mut at = 0;
    let mut gained = 0usize;
    for ((s, used, raw), p) in picks.iter().zip(params) {
        let Some((p, text)) = p else { continue };
        if out.len() + (s - at) + raw.len() + (b.len() - s) > limit {
            break;
        }
        out.extend_from_slice(&b[at..*s]);
        push(&mut pieces, s - at, BIN);
        out.extend_from_slice(raw);
        push(&mut pieces, raw.len(), if text { TEXT } else { BIN });
        recs.push(Rec { gap: s - at, raw: raw.len(), comp: *used, params: p });
        gained += used;
        at = s + used;
    }
    // only worth it when a real share of the block was deflated
    if recs.is_empty() || gained * 50 < b.len() {
        return None;
    }
    out.extend_from_slice(&b[at..]);
    push(&mut pieces, b.len() - at, BIN);
    lap("assemble");
    Some(Expanded { data: out, recs, pieces })
}

pub fn write_recs(recs: &[Rec]) -> Vec<u8> {
    let mut w = Vec::with_capacity(recs.len() * 8 + 8);
    write_varint(&mut w, recs.len() as u64);
    for r in recs {
        write_varint(&mut w, r.gap as u64);
        write_varint(&mut w, r.raw as u64);
        write_varint(&mut w, r.comp as u64);
        w.push(r.params);
    }
    w
}

pub fn read_recs(r: &mut &[u8], raw_len: usize) -> Result<Vec<Rec>, String> {
    let e = || "corrupt deflate table".to_string();
    let k = read_varint(r).map_err(|_| e())? as usize;
    if k > raw_len / MIN_COMP + 1 {
        return Err(e());
    }
    let mut out = Vec::with_capacity(k);
    let mut total = 0usize;
    for _ in 0..k {
        let gap = read_varint(r).map_err(|_| e())? as usize;
        let raw = read_varint(r).map_err(|_| e())? as usize;
        let comp = read_varint(r).map_err(|_| e())? as usize;
        let params = *r.first().ok_or_else(e)?;
        *r = &r[1..];
        let (l, m) = (params & 15, params >> 4);
        total = total.checked_add(gap).and_then(|t| t.checked_add(comp)).ok_or_else(e)?;
        if total > raw_len || !(1..=9).contains(&l) || !(1..=9).contains(&m) {
            return Err(e());
        }
        out.push(Rec { gap, raw, comp, params });
    }
    Ok(out)
}

/// Inverse of `expand`: deflates each recorded stream again (in parallel).
pub fn restore(data: &[u8], recs: &[Rec], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt deflate block".to_string();
    // where each stream's inflated bytes sit in `data`
    let mut spans = Vec::with_capacity(recs.len());
    let mut at = 0usize;
    for r in recs {
        at = at.checked_add(r.gap).ok_or_else(e)?;
        let end = at.checked_add(r.raw).ok_or_else(e)?;
        if end > data.len() {
            return Err(e());
        }
        spans.push((at, end, r.params));
        at = end;
    }
    // biggest streams first (they share one queue with the other blocks' work)
    let comp: Vec<Vec<u8>> = crate::par::map_ranked(&spans, threads, |&(s, t, _)| (t - s) as u64, |&(s, t, p)| crate::zlib::deflate(&data[s..t], p & 15, p >> 4));
    let mut out = Vec::with_capacity(raw_len);
    let mut at = 0;
    for ((r, &(s, t, _)), c) in recs.iter().zip(&spans).zip(&comp) {
        if c.len() != r.comp {
            return Err(e());
        }
        out.extend_from_slice(&data[at..s]);
        out.extend_from_slice(c);
        at = t;
    }
    out.extend_from_slice(&data[at..]);
    if out.len() != raw_len {
        return Err(e());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_restore_gzip_and_zip() {
        let text: Vec<u8> = (0..200_000u32).flat_map(|i| format!("line {} of some text\n", i % 977).into_bytes()).collect();
        let c = crate::zlib::deflate(&text, 6, 8);
        let mut b = b"junk before ".to_vec();
        // gzip member: header, stream, crc + size (not checked here)
        b.extend_from_slice(&[0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 3]);
        b.extend_from_slice(&c);
        b.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // zip entry with a data descriptor
        let c9 = crate::zlib::deflate(&text[..100_000], 9, 8);
        b.extend_from_slice(b"PK\x03\x04\x14\x00\x08\x00\x08\x00");
        b.extend_from_slice(&[0; 16]);
        b.extend_from_slice(&[5, 0, 0, 0]);
        b.extend_from_slice(b"a.txt");
        b.extend_from_slice(&c9);
        b.extend_from_slice(b" tail");
        let x = expand(&b, 4, usize::MAX).unwrap();
        assert_eq!(x.recs.len(), 2);
        assert_eq!(x.recs[0].params, 6 | 8 << 4);
        assert_eq!(x.recs[1].params, 9 | 8 << 4);
        let t = write_recs(&x.recs);
        let recs = read_recs(&mut &t[..], b.len()).unwrap();
        assert_eq!(restore(&x.data, &recs, b.len(), 4).unwrap(), b);
        assert_eq!(x.pieces.iter().map(|p| p.len).sum::<usize>(), x.data.len());
    }
}
