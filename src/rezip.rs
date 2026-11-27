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
/// (level | memLevel << 4). `splits`: clean block boundaries (stream offset, output bit
/// offset) where the decoder cuts the rebuild into pieces that run in parallel.
#[derive(Clone, Debug, PartialEq)]
pub struct Rec {
    pub gap: usize,
    pub raw: usize,
    pub comp: usize,
    pub params: u8,
    pub splits: Vec<(usize, u64)>,
}

/// Streams at least this big (inflated) get split points.
const SPLIT_MIN: usize = 2 << 20;

/// Inflated bytes per rebuild piece: about the same rebuild time at every level (level 9
/// walks hash chains up to 4096 long).
fn seg_len(level: u8) -> usize {
    match level {
        8.. => 1 << 20,
        7 => 2 << 20,
        _ => 3 << 20,
    }
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
    /// zlib stream: the inflated bytes must match the Adler-32 trailer
    zlib: bool,
}

fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for c in d.chunks(5552) {
        for &x in c {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    b << 16 | a
}

fn u16_at(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

/// zlib header with a 32 KB window and no preset dictionary (levels 0-1, 2-5, 6, 7-9).
fn zlib_head(a: u8, b: u8) -> bool {
    a == 0x78 && matches!(b, 0x01 | 0x5E | 0x9C | 0xDA)
}

/// Deflate streams named by zip local headers (method 8), gzip member headers and zlib
/// headers (Minecraft region chunks, PDF and git objects, ...; the Adler-32 trailer weeds
/// out false hits).
fn candidates(b: &[u8]) -> Vec<Cand> {
    let mut out = Vec::new();
    let n = b.len();
    let mut i = 0;
    while i + 30 <= n {
        let Some(k) = b[i..n - 9].windows(2).position(|w| w == b"PK" || w == [0x1F, 0x8B] || zlib_head(w[0], w[1])) else { break };
        i += k;
        if b[i] == 0x78 {
            out.push(Cand { start: i + 2, csize: None, zlib: true });
        } else if b[i] == b'P' {
            if i + 30 <= n && b[i + 2] == 3 && b[i + 3] == 4 && u16_at(b, i + 8) == 8 {
                let flags = u16_at(b, i + 6);
                let nlen = u16_at(b, i + 26);
                let xlen = u16_at(b, i + 28);
                let start = i + 30 + nlen + xlen;
                let csize = u32_at(b, i + 18);
                if flags & 1 == 0 && nlen > 0 && nlen < 1024 && start < n {
                    let csize = if flags & 8 == 0 { Some(csize) } else { None };
                    if csize.is_none_or(|c| c >= MIN_COMP && c <= n - start) {
                        out.push(Cand { start, csize, zlib: false });
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
                out.push(Cand { start: p, csize: None, zlib: false });
            }
        }
        i += 2;
    }
    out
}

/// The levels that rebuild `comp` from `raw` and the split points of the rebuild, or None.
fn find_params(raw: &[u8], comp: &[u8], threads: usize) -> Option<(u8, Vec<(usize, u64)>)> {
    if raw.len() < SPLIT_MIN || crate::sys::dev_flag("NZ_NOSPLIT") {
        return TRIES.iter().find(|&&(l, m)| crate::zlib::deflate_eq(raw, l, m, comp)).map(|&(l, m)| (l | (m << 4), Vec::new()));
    }
    // big streams: the run that matches also lists the clean block boundaries
    for &(l, m) in &TRIES {
        let pts = if l >= 4 {
            crate::zlib::deflate_points(raw, l, m, comp)
        } else {
            crate::zlib::deflate_eq(raw, l, m, comp).then(Vec::new)
        };
        if let Some(pts) = pts {
            if crate::sys::dev_flag("NZ_RZT") && pts.is_empty() {
                eprintln!("  splits: {} KB level {l}: no clean boundary", raw.len() >> 10);
            }
            return Some((l | (m << 4), splits_of(raw, comp, l, m, &pts, threads)));
        }
    }
    None
}

/// Split points every `seg_len` bytes or so, kept only if the pieces rebuilt apart give
/// `comp` back exactly.
fn splits_of(raw: &[u8], comp: &[u8], l: u8, m: u8, pts: &[(usize, u64)], threads: usize) -> Vec<(usize, u64)> {
    let step = seg_len(l);
    let mut cuts = Vec::new();
    let mut last = 0usize;
    for &(at, bit) in pts {
        if at >= last + step && raw.len() - at >= step / 2 {
            cuts.push((at, bit));
            last = at;
        }
    }
    if cuts.is_empty() {
        return cuts;
    }
    let rec = Rec { gap: 0, raw: raw.len(), comp: comp.len(), params: l | (m << 4), splits: cuts };
    let ok = rebuild(raw, &rec, threads).is_some_and(|c| c == comp);
    if crate::sys::dev_flag("NZ_RZT") {
        eprintln!("  splits: {} KB level {l}: {} boundaries, {} cuts, verified {ok}", raw.len() >> 10, pts.len(), rec.splits.len());
    }
    if ok { rec.splits } else { Vec::new() }
}

/// Pieces of one stream's rebuild: (start, stop, lead bits).
fn pieces_of(r: &Rec) -> Vec<(usize, Option<usize>, u32)> {
    let mut v = Vec::with_capacity(r.splits.len() + 1);
    let (mut start, mut lead) = (0usize, 0u32);
    for &(at, bit) in &r.splits {
        v.push((start, Some(at), lead));
        start = at;
        lead = (bit & 7) as u32;
    }
    v.push((start, None, lead));
    v
}

/// Joins the pieces of one stream (in order).
fn join(parts: impl Iterator<Item = Option<(Vec<u8>, u64)>>, cap: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(cap);
    let mut len = 0u64;
    for p in parts {
        let (seg, bits) = p?;
        if bits < len & 7 || bits.div_ceil(8) as usize > seg.len() {
            return None;
        }
        crate::zlib::append_bits(&mut out, &mut len, &seg, bits);
    }
    Some(out)
}

/// Deflates one stream again from its inflated bytes, its pieces in parallel.
fn rebuild(raw: &[u8], r: &Rec, threads: usize) -> Option<Vec<u8>> {
    let (l, m) = (r.params & 15, r.params >> 4);
    if r.splits.is_empty() {
        return Some(crate::zlib::deflate(raw, l, m));
    }
    let pieces = pieces_of(r);
    let parts = crate::par::map(&pieces, threads, |&(s, t, lead)| crate::zlib::deflate_seg(raw, l, m, s, t, lead));
    join(parts.into_iter(), r.comp)
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
        if c.zlib && b.get(c.start + used..c.start + used + 4) != Some(&adler32(&raw).to_be_bytes()[..]) {
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
    let params: Vec<Option<(u8, Vec<(usize, u64)>, bool)>> =
        crate::par::map(&picks, threads, |(s, used, raw)| find_params(raw, &b[*s..*s + *used], threads).map(|(p, sp)| (p, sp, is_text(raw))));
    lap("verify");
    if tm {
        let mut h = [0usize; 256];
        let mut miss = 0;
        for p in &params {
            match p {
                Some((p, ..)) => h[*p as usize] += 1,
                None => miss += 1,
            }
        }
        let lv: Vec<String> = (0..256).filter(|&i| h[i] > 0).map(|i| format!("l{}m{}:{}", i & 15, i >> 4, h[i])).collect();
        eprintln!("  rezip {} cands, {} streams, {} not rebuilt, {}", cands.len(), picks.len(), miss, lv.join(" "));
    }
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
        let Some((p, splits, text)) = p else { continue };
        if out.len() + (s - at) + raw.len() + (b.len() - s) > limit {
            break;
        }
        out.extend_from_slice(&b[at..*s]);
        push(&mut pieces, s - at, BIN);
        out.extend_from_slice(raw);
        push(&mut pieces, raw.len(), if text { TEXT } else { BIN });
        recs.push(Rec { gap: s - at, raw: raw.len(), comp: *used, params: p, splits });
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
        write_varint(&mut w, r.splits.len() as u64);
        let (mut at, mut bit) = (0usize, 0u64);
        for &(a, b) in &r.splits {
            write_varint(&mut w, (a - at) as u64);
            write_varint(&mut w, b - bit);
            (at, bit) = (a, b);
        }
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
        // split points: strictly inside the stream, in order, levels 4..=9 only
        let ns = read_varint(r).map_err(|_| e())? as usize;
        if ns > raw / (1 << 16) || (ns > 0 && l < 4) {
            return Err(e());
        }
        let mut splits = Vec::with_capacity(ns);
        let (mut at, mut bit) = (0usize, 0u64);
        for _ in 0..ns {
            let da = read_varint(r).map_err(|_| e())? as usize;
            let db = read_varint(r).map_err(|_| e())?;
            at = at.checked_add(da).ok_or_else(e)?;
            bit = bit.checked_add(db).ok_or_else(e)?;
            if da == 0 || db == 0 || at >= raw || bit >= comp as u64 * 8 {
                return Err(e());
            }
            splits.push((at, bit));
        }
        out.push(Rec { gap, raw, comp, params, splits });
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
    // every stream's rebuild pieces, costliest first (they share one queue with the other
    // blocks' work); level 7+ walks long hash chains
    let mut jobs = Vec::new();
    for (k, (r, &(s, t, _))) in recs.iter().zip(&spans).enumerate() {
        for (a, b, lead) in pieces_of(r) {
            jobs.push((k, s, t, a, b, lead));
        }
    }
    let cost = |j: &(usize, usize, usize, usize, Option<usize>, u32)| {
        let (s, t) = (j.1, j.2);
        let len = j.4.unwrap_or(t - s) - j.3;
        len as u64 * if recs[j.0].params & 15 >= 7 { 4 } else { 1 }
    };
    let parts = crate::par::map_ranked(&jobs, threads, cost, |&(k, s, t, a, b, lead)| {
        let r = &recs[k];
        let (l, m) = (r.params & 15, r.params >> 4);
        if r.splits.is_empty() { Some((crate::zlib::deflate(&data[s..t], l, m), r.comp as u64 * 8)) } else { crate::zlib::deflate_seg(&data[s..t], l, m, a, b, lead) }
    });
    let mut parts = parts.into_iter();
    let mut comp: Vec<Vec<u8>> = Vec::with_capacity(recs.len());
    for r in recs {
        let n = r.splits.len() + 1;
        comp.push(join(parts.by_ref().take(n), r.comp).ok_or_else(e)?);
    }
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

    #[test]
    fn expand_restore_zlib() {
        // region-file style: length, type byte, zlib header, stream, Adler-32, padding
        let text: Vec<u8> = (0..50_000u32).flat_map(|i| format!("tag {} value {}\n", i % 311, i % 7).into_bytes()).collect();
        let mut b = vec![0u8; 4096];
        for part in text.chunks(120_000) {
            let c = crate::zlib::deflate(part, 6, 8);
            b.extend_from_slice(&((c.len() + 7) as u32).to_be_bytes());
            b.extend_from_slice(&[2, 0x78, 0x9C]);
            b.extend_from_slice(&c);
            b.extend_from_slice(&super::adler32(part).to_be_bytes());
            b.resize(b.len().next_multiple_of(4096), 0);
        }
        let x = expand(&b, 4, usize::MAX).unwrap();
        assert_eq!(x.recs.len(), text.len().div_ceil(120_000));
        let t = write_recs(&x.recs);
        let recs = read_recs(&mut &t[..], b.len()).unwrap();
        assert_eq!(restore(&x.data, &recs, b.len(), 4).unwrap(), b);
        // a wrong trailer: not a zlib stream, left alone
        let mut bad = b.clone();
        let i = 4096 + 5 + recs[0].comp;
        bad[i] ^= 1;
        let n = expand(&bad, 4, usize::MAX).map_or(0, |x| x.recs.len());
        assert_eq!(n, recs.len() - 1);
    }
}
