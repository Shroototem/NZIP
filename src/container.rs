//! Archive format and the parallel block pipeline.
//!
//! ```text
//! header : "NZIP" ver:u8 flags:u8 mode:u8 total_len:varint block_size:varint
//! block  : kind:u8 raw_len:varint comp_len:varint crc32:u32le payload[comp_len]
//! end    : kind = 0
//! BWT payload : flags:u8 (1 = word transform, 2 = x86 filter) bwt_len:varint idx[k]:u32le coder-stream
//! LZ payload  : lz-stream (fast mode)
//! LZX payload : flags:u8 (1 = x86 split) [4 raw lens, 4 comp lens] lzx stream(s) (max mode, binary)
//! LRM payload : table_len:varint packed_len:varint long-range copy table (lrm.rs, LZX
//!               coded), inner kind:u8
//!               inner raw_len:varint, inner payload (the bytes left after the copies)
//! MIX payload : table_len:varint piece-table, then 2 parts (text, binary), each
//!               kind:u8 raw_len:varint comp_len:varint payload
//! ```
//! Input is read in waves of blocks; each wave is compressed in parallel and written in
//! order, so memory stays bounded for any input size. Blocks are independent: the decoder
//! picks its own thread count from its free RAM and always gets identical bytes.

use crate::codec::entropy_milli;
use crate::par;
use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"NZIP";
pub const VERSION: u8 = 10;
pub const FLAG_DIR: u8 = 1;

const KIND_END: u8 = 0;
const KIND_STORED: u8 = 1;
const KIND_BWT: u8 = 4;
const KIND_LZ: u8 = 5;
const KIND_LZX: u8 = 6;
const KIND_MIX: u8 = 7;
const KIND_REZIP: u8 = 8;
const KIND_LRM: u8 = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Fast,
    Max,
}

pub struct Options {
    pub mode: Mode,
    pub threads: usize,
    pub block_size: Option<usize>,
    pub verbose: bool,
}

pub fn write_varint(w: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 127) as u8;
        v >>= 7;
        if v == 0 {
            w.push(b);
            return;
        }
        w.push(b | 128);
    }
}

pub fn read_varint<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let mut b = [0u8];
        r.read_exact(&mut b)?;
        v |= ((b[0] & 127) as u64) << shift;
        if b[0] < 128 {
            return Ok(v);
        }
    }
    Err(bad("bad varint"))
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

/// Peak bytes one worker needs for a block of `len` (compress or decompress).
pub fn block_mem(mode: Mode, len: usize) -> u64 {
    match mode {
        // input + transformed copy + suffix array / inverse table (4n) + output, + coder
        Mode::Max => len as u64 * 7 + crate::bwtc::MEM,
        Mode::Fast => len as u64 * 3 + (1 << 20),
    }
}

/// Largest Max-mode block by default. Bigger blocks compress text better, but every
/// machine that decodes the archive needs ~7x the block size in RAM for one worker; 256 MiB
/// keeps that under 2 GB.
pub const MAX_BLOCK: usize = 256 << 20;

/// Picks the block size from the input size, the hardware and free RAM / VRAM.
/// * Max + usable GPU: blocks as large as free VRAM, free RAM and MAX_BLOCK allow (the GPU
///   sorts them in turn while every CPU thread codes); bigger blocks compress better.
/// * Max, CPU only: the same size cap from free RAM. Both engines run every core inside
///   one block (coder segments, LZ pieces), and small blocks cost 2-25% (executables lose
///   cross-section matches, text loses context); a 256 MB CPU sort still beats 7-Zip's time.
/// Blocks are split evenly. The archive records the size, so decoding adapts to any machine.
pub fn choose_block_size(total: u64, mode: Mode) -> usize {
    let total = total.max(1);
    let even = |cap: u64| -> usize {
        let cap = cap.max(1 << 20);
        let nb = total.div_ceil(cap);
        total.div_ceil(nb).max(1) as usize
    };
    match mode {
        Mode::Fast => 4 << 20,
        Mode::Max => {
            let ram = crate::sys::mem_budget();
            // small inputs never wait for the GPU driver to start (~0.2 s)
            let vram = if total >= 32 << 20 { crate::gpu::free_vram() } else { None };
            match vram {
                Some(v) => {
                    let by_vram = v.saturating_sub(256 << 20) / 34;
                    even(by_vram.min(ram / 7).min(MAX_BLOCK as u64))
                }
                _ => even((ram / 7).max(8 << 20).min(MAX_BLOCK as u64)),
            }
        }
    }
}

/// Blocks that can be in flight within the memory budget (at least 1).
pub fn fit_threads(want: usize, per_block: u64) -> usize {
    let fit = (crate::sys::mem_budget() / per_block.max(1)) as usize;
    want.min(fit).max(1)
}

struct Packed {
    kind: u8,
    raw_len: usize,
    crc: u32,
    data: Vec<u8>,
    note: &'static str,
}

fn stored(data: &[u8], crc: u32, note: &'static str) -> Packed {
    Packed { kind: KIND_STORED, raw_len: data.len(), crc, data: data.to_vec(), note }
}

static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn trace(what: &str, len: usize) {
    if crate::sys::dev_flag("NZIP_TRACE") {
        let t = T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64();
        eprintln!("  [{t:7.2}s] {what} ({} MB)", len >> 20);
    }
}

fn pack_block(data: &[u8], mode: Mode, threads: usize) -> Packed {
    trace("block start", data.len());
    let crc = crate::codec::crc32_par(data, threads);
    if data.is_empty() {
        return stored(data, crc, "");
    }
    // zip / gzip content first: deflated data looks random to the test below
    if mode == Mode::Max {
        if let Some((kind, out, note)) = pack_rezip(data, threads) {
            if out.len() < data.len() {
                return Packed { kind, raw_len: data.len(), crc, data: out, note };
            }
        }
    }
    // Already-compressed data (JPEG, MP4, zip, ...) looks random: store it at memcpy speed.
    let sample = &data[..data.len().min(1 << 20)];
    if entropy_milli(sample) > 7990 && entropy_milli(data) > 7990 {
        return stored(data, crc, "incompressible");
    }
    let (kind, out, note) = match mode {
        Mode::Fast => (KIND_LZ, crate::lz::compress(data), "lz"),
        Mode::Max => pack_max(data, threads),
    };
    if out.len() >= data.len() {
        return stored(data, crc, "expanded");
    }
    Packed { kind, raw_len: data.len(), crc, data: out, note }
}

/// BWT payload: flags, length, chain indices, coder stream.
fn bwt_payload(src: &[u8], flags: u8, threads: usize) -> Vec<u8> {
    trace("transform done", src.len());
    let (b, idx) = crate::gpu::bwt(src);
    trace("bwt done", src.len());
    let mut out = Vec::with_capacity(src.len() / 3 + 64);
    out.push(flags);
    write_varint(&mut out, src.len() as u64);
    for i in idx {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out.extend_from_slice(&crate::bwtc::encode(&b, threads));
    trace("coded", src.len());
    out
}

/// LZ payload: flags (1 = x86 split) then either one LZ stream, or the four split streams
/// (chunk count, per-chunk part lengths, raw lengths, compressed lengths, data).
fn lz_payload(data: &[u8], x86: bool, threads: usize, arr: usize) -> Vec<u8> {
    let mut out = Vec::new();
    if x86 {
        let (sp, lens) = crate::x86::split_chunks(data, threads);
        let streams = [&sp.main, &sp.flags, &sp.addr, &sp.jump];
        // all four streams at once: the small ones are single sub-blocks
        let coded: Vec<Vec<u8>> = par::map(&streams, 4, |s| crate::lzx::encode(s, threads, false, true, arr));
        out.push(1);
        write_varint(&mut out, lens.len() as u64);
        for l in &lens {
            for &v in l {
                write_varint(&mut out, v as u64);
            }
        }
        for s in streams {
            write_varint(&mut out, s.len() as u64);
        }
        for c in &coded {
            write_varint(&mut out, c.len() as u64);
        }
        for c in coded {
            out.extend_from_slice(&c);
        }
    } else {
        out.push(0);
        out.extend_from_slice(&crate::lzx::encode(data, threads, false, false, arr));
    }
    trace("lz done", data.len());
    out
}

/// Blocks up to this size are coded by both engines (the smaller kept) instead of
/// picking one from a sample.
const FULL_TRY: usize = 16 << 20;

/// Evenly spread slices of a big block, used to pick an engine cheaply.
fn sample_of(data: &[u8]) -> Vec<u8> {
    const SLICES: usize = 4;
    const SLICE: usize = 256 << 10;
    let step = data.len() / SLICES;
    let mut s = Vec::with_capacity(SLICES * SLICE);
    for i in 0..SLICES {
        s.extend_from_slice(&data[i * step..(i * step + SLICE).min(data.len())]);
    }
    s
}

/// Max mode: text -> word transform + BWT. Anything else -> the BWT engine or the LZ engine,
/// whichever codes a sample of the block (or the whole block, if small) smaller.
fn pack_max(data: &[u8], threads: usize) -> (u8, Vec<u8>, &'static str) {
    use crate::mix::{BIN, Piece};
    let zip = crate::mix::zip_pieces(data);
    if zip.is_none() && crate::textx::looks_like_text(data) {
        return pack_plain(data, threads, 0);
    }
    pack_pieces(data, zip.unwrap_or_else(|| vec![Piece { len: data.len(), kind: BIN }]), threads, 0)
}

/// Packs a block cut into text / binary pieces: raw chunks are split off, then either one
/// engine codes it all or a mixed block keeps the kinds apart.
fn pack_pieces(data: &[u8], base: Vec<crate::mix::Piece>, threads: usize, arr: usize) -> (u8, Vec<u8>, &'static str) {
    use crate::mix::{BIN, RAW, TEXT};
    let pieces = crate::mix::raw_split(data, &base, threads);
    let has = |k: u8| pieces.iter().any(|p| p.kind == k);
    if !has(RAW) && !has(TEXT) {
        return pack_plain(data, threads, arr);
    }
    if !has(BIN) && !has(TEXT) {
        return (KIND_STORED, data.to_vec(), "incompressible");
    }
    (KIND_MIX, mix_payload(data, &pieces, threads, arr), if has(TEXT) { "zip split" } else { "raw split" })
}

/// The pieces of `p` that fall in `s..e`, cut at both ends.
fn slice_pieces(p: &[crate::mix::Piece], s: usize, e: usize) -> Vec<crate::mix::Piece> {
    let mut out = Vec::new();
    let mut at = 0;
    for q in p {
        let (a, b) = (at.max(s), (at + q.len).min(e));
        if a < b {
            out.push(crate::mix::Piece { len: b - a, kind: q.kind });
        }
        at += q.len;
        if at >= e {
            break;
        }
    }
    out
}

/// Deflate recompression (see rezip.rs): the table of rebuilt streams, then the inflated
/// block in chunks no bigger than the block (each packed like a block of its own, all at
/// once).
fn pack_rezip(data: &[u8], threads: usize) -> Option<(u8, Vec<u8>, &'static str)> {
    let x = crate::rezip::expand(data, threads, data.len() * REZIP_GROWTH + (16 << 20))?;
    trace("deflate streams inflated", x.data.len());
    let chunk = data.len().max(16 << 20);
    let tab = crate::rezip::write_recs(&x.recs);
    let mut out = Vec::with_capacity(data.len() / 2);
    write_varint(&mut out, tab.len() as u64);
    out.extend_from_slice(&tab);
    let n = x.data.len().div_ceil(chunk);
    write_varint(&mut out, n as u64);
    let spans: Vec<(usize, usize)> = (0..n).map(|c| (c * chunk, ((c + 1) * chunk).min(x.data.len()))).collect();
    let packed = par::map(&spans, n, |&(s, e)| {
        let part = &x.data[s..e];
        let (kind, payload, _) = pack_pieces(part, slice_pieces(&x.pieces, s, e), threads, REZIP_EFFORT);
        if payload.len() >= part.len() { (KIND_STORED, part.to_vec()) } else { (kind, payload) }
    });
    for (&(s, e), (kind, payload)) in spans.iter().zip(packed) {
        out.push(kind);
        write_varint(&mut out, (e - s) as u64);
        write_varint(&mut out, payload.len() as u64);
        out.extend_from_slice(&payload);
    }
    Some((KIND_REZIP, out, "deflate rebuilt"))
}

/// Largest inflated size, as a multiple of the block (bounds decoder memory).
const REZIP_GROWTH: usize = 3;

/// LZ effort for inflated deflate data: 7-Zip passes over deflated data quickly (it is
/// incompressible to LZMA), while nzip codes about twice the bytes once they are inflated.
const REZIP_EFFORT: usize = 1;

fn unpack_rezip(p: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt deflate block".to_string();
    let mut r = p;
    let tl = read_varint(&mut r).map_err(|_| e())? as usize;
    let recs = crate::rezip::read_recs(&mut r.get(..tl).ok_or_else(e)?, raw_len)?;
    r = &r[tl..];
    let n = read_varint(&mut r).map_err(|_| e())? as usize;
    let cap = raw_len * REZIP_GROWTH + (16 << 20);
    if n > cap / (16 << 20) + 2 {
        return Err(e());
    }
    let mut parts = Vec::with_capacity(n);
    let mut total = 0usize;
    for _ in 0..n {
        let kind = read_u8(&mut r).map_err(|_| e())?;
        let len = read_varint(&mut r).map_err(|_| e())? as usize;
        let c = read_varint(&mut r).map_err(|_| e())? as usize;
        total += len;
        if kind == KIND_REZIP || total > cap {
            return Err(e());
        }
        parts.push((kind, len, r.get(..c).ok_or_else(e)?));
        r = &r[c..];
    }
    let out = par::fan(&parts, |&(kind, len, c)| unpack_payload(kind, c, len, threads));
    let mut data = Vec::with_capacity(total);
    for (p, &(_, len, _)) in out.into_iter().zip(&parts) {
        let p = p?;
        if p.len() != len {
            return Err(e());
        }
        data.extend_from_slice(&p);
    }
    trace("inflated data decoded", data.len());
    let out = crate::rezip::restore(&data, &recs, raw_len, threads);
    trace("deflate rebuilt", raw_len);
    out
}

/// Mixed block: text, binary and raw pieces packed apart (see mix.rs).
fn mix_payload(data: &[u8], pieces: &[crate::mix::Piece], threads: usize, arr: usize) -> Vec<u8> {
    let parts = crate::mix::gather(data, pieces);
    let packed = par::map(&parts, 2, |p| if p.is_empty() { (KIND_STORED, Vec::new(), "") } else { pack_plain(p, threads, arr) });
    let tab = crate::mix::write_table(pieces);
    let mut out = Vec::with_capacity(tab.len() + packed.iter().map(|p| p.1.len() + 16).sum::<usize>());
    write_varint(&mut out, tab.len() as u64);
    out.extend_from_slice(&tab);
    for (i, (raw, (kind, payload, _))) in parts.iter().zip(packed).enumerate() {
        let (kind, payload) =
            if i == crate::mix::RAW as usize || payload.len() >= raw.len() { (KIND_STORED, raw.clone()) } else { (kind, payload) };
        out.push(kind);
        write_varint(&mut out, raw.len() as u64);
        write_varint(&mut out, payload.len() as u64);
        out.extend_from_slice(&payload);
    }
    out
}

fn unpack_mix(p: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt mixed block".to_string();
    let mut r = p;
    let tl = read_varint(&mut r).map_err(|_| e())? as usize;
    let pieces = crate::mix::read_table(r.get(..tl).ok_or_else(e)?, raw_len)?;
    r = &r[tl..];
    let mut parts = Vec::with_capacity(3);
    for _ in 0..3 {
        let kind = read_u8(&mut r).map_err(|_| e())?;
        let n = read_varint(&mut r).map_err(|_| e())? as usize;
        let c = read_varint(&mut r).map_err(|_| e())? as usize;
        if n > raw_len || kind == KIND_MIX {
            return Err(e());
        }
        parts.push((kind, n, r.get(..c).ok_or_else(e)?));
        r = &r[c..];
    }
    let out = par::fan(&parts, |&(kind, n, c)| unpack_payload(kind, c, n, threads)).into_iter().collect::<Result<Vec<_>, _>>()?;
    crate::mix::scatter(&out, &pieces)
}

/// One engine for the whole block.
/// `arr`: cap on the LZ parser's effort (0 = the machine's).
fn pack_plain(data: &[u8], threads: usize, arr: usize) -> (u8, Vec<u8>, &'static str) {
    if crate::textx::looks_like_text(data) {
        if let Some(t) = crate::textx::encode(data, threads) {
            if t.len() * 100 < data.len() * 95 {
                return (KIND_BWT, bwt_payload(&t, 1, threads), "text+bwt");
            }
        }
        return (KIND_BWT, bwt_payload(data, 0, threads), "bwt");
    }
    // repeats further back than the LZ engine reaches become copies first
    if let Some((table, lits)) = crate::lrm::split(data, crate::lzx::DICT) {
        let (kind, payload, note) = pack_bin(&lits, threads, arr);
        let tz = crate::lzx::encode(&table, threads, false, false, 0);
        let mut out = Vec::with_capacity(tz.len() + payload.len() + 16);
        write_varint(&mut out, table.len() as u64);
        write_varint(&mut out, tz.len() as u64);
        out.extend_from_slice(&tz);
        out.push(kind);
        write_varint(&mut out, lits.len() as u64);
        out.extend_from_slice(&payload);
        return (KIND_LRM, out, note);
    }
    pack_bin(data, threads, arr)
}

/// Binary data: x86 filter if it is machine code, then the LZ or the BWT engine.
fn pack_bin(data: &[u8], threads: usize, arr: usize) -> (u8, Vec<u8>, &'static str) {
    let x86 = crate::x86::looks_like_x86(data);
    let bwt_full = |d: &[u8]| -> Vec<u8> {
        if x86 {
            let mut t = d.to_vec();
            crate::x86::encode(&mut t);
            bwt_payload(&t, 2, threads)
        } else {
            bwt_payload(d, 0, threads)
        }
    };
    // small blocks: both engines code all of it at once and the smaller wins (no sample)
    if data.len() <= FULL_TRY {
        let mut both = par::map(&[false, true], 2, |&lz| if lz { lz_payload(data, x86, threads, arr) } else { bwt_full(data) });
        let (l, b) = (both.pop().unwrap(), both.pop().unwrap());
        return if l.len() < b.len() { (KIND_LZX, l, if x86 { "x86+lz" } else { "lz" }) } else { (KIND_BWT, b, if x86 { "x86+bwt" } else { "bwt" }) };
    }
    // both engines try the sample at once
    let s = sample_of(data);
    let tried = par::map(&[false, true], 2, |&lz| if lz { lz_payload(&s, x86, threads, arr).len() } else { bwt_full(&s).len() });
    let (sb, sl) = (tried[0], tried[1]);
    trace(&format!("engine trial: bwt {sb} lz {sl}"), data.len());
    // a small sample cuts LZ off from its long-range matches (BWT barely needs them), so
    // on binary data BWT must win clearly
    let margin = if crate::dedup::texty(&s) { 100 } else { 85 };
    if sb * 100 >= sl * margin {
        (KIND_LZX, lz_payload(data, x86, threads, arr), if x86 { "x86+lz" } else { "lz" })
    } else {
        (KIND_BWT, bwt_full(data), if x86 { "x86+bwt" } else { "bwt" })
    }
}

fn unpack_lrm(p: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt long-range block".to_string();
    let mut r = p;
    let tl = read_varint(&mut r).map_err(|_| e())? as usize;
    let tz = read_varint(&mut r).map_err(|_| e())? as usize;
    if tl > raw_len {
        return Err(e());
    }
    let table = crate::lzx::decode(r.get(..tz).ok_or_else(e)?, tl, threads, false)?;
    r = &r[tz..];
    let kind = read_u8(&mut r).map_err(|_| e())?;
    let n = read_varint(&mut r).map_err(|_| e())? as usize;
    if n > raw_len || kind == KIND_LRM {
        return Err(e());
    }
    let lits = unpack_payload(kind, r, n, threads)?;
    if lits.len() != n {
        return Err(e());
    }
    let t = std::time::Instant::now();
    let r = crate::lrm::join(&table, &lits, raw_len);
    trace(&format!("lrm join {:.3}s", t.elapsed().as_secs_f64()), raw_len);
    r
}

fn unpack_lz(p: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt LZ block".to_string();
    let flags = *p.first().ok_or_else(e)?;
    let mut r = &p[1..];
    if flags & 1 == 0 {
        return crate::lzx::decode(r, raw_len, threads, false);
    }
    let nchunks = read_varint(&mut r).map_err(|_| e())? as usize;
    if nchunks > raw_len / (1 << 20) + 2 {
        return Err(e());
    }
    let mut lens = vec![[0usize; 4]; nchunks];
    for l in lens.iter_mut() {
        for v in l.iter_mut() {
            *v = read_varint(&mut r).map_err(|_| e())? as usize;
        }
    }
    let mut raw = [0usize; 4];
    let mut comp = [0usize; 4];
    for v in raw.iter_mut().chain(comp.iter_mut()) {
        *v = read_varint(&mut r).map_err(|_| e())? as usize;
    }
    if raw.iter().any(|&l| l > raw_len.max(1 << 16) * 2) {
        return Err(e());
    }
    let mut jobs = Vec::with_capacity(4);
    let mut off = 0usize;
    for i in 0..4 {
        jobs.push((r.get(off..off + comp[i]).ok_or_else(e)?, raw[i]));
        off += comp[i];
    }
    trace("lz streams start", raw_len);
    let streams = par::fan(&jobs, |&(c, n)| crate::lzx::decode(c, n, threads, false)).into_iter().collect::<Result<Vec<_>, _>>()?;
    trace("lz streams done", raw_len);
    let j = crate::x86::join_chunks([&streams[0], &streams[1], &streams[2], &streams[3]], &lens, raw_len, threads);
    trace("x86 join done", raw_len);
    j
}

fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

pub struct Stats {
    pub in_bytes: u64,
    pub out_bytes: u64,
    pub blocks: usize,
    pub threads: usize,
    pub block_size: usize,
    /// largest block actually written
    pub largest: u64,
}

/// `cuts`: sorted stream offsets where a block boundary is preferred: (offset, type change),
/// at every file start.
pub fn compress<R: Read, W: Write>(mut input: R, total: u64, flags: u8, mut out: W, o: &Options, cuts: &[(u64, bool)]) -> io::Result<Stats> {
    // the GPU driver starts in the background while the first blocks are read and tried
    if o.mode == Mode::Max && total >= 4 << 20 {
        crate::gpu::warm();
    }
    let bs = o.block_size.unwrap_or_else(|| choose_block_size(total, o.mode)).max(1);
    let threads = o.threads.max(1);
    // blocks in flight: limited by free RAM; spare threads work inside each block
    let workers = fit_threads(threads, block_mem(o.mode, bs));
    let mut hdr = Vec::new();
    hdr.extend_from_slice(MAGIC);
    hdr.push(VERSION);
    hdr.push(flags);
    hdr.push(o.mode as u8);
    write_varint(&mut hdr, total);
    write_varint(&mut hdr, bs as u64);
    out.write_all(&hdr)?;
    let mut st = Stats { in_bytes: 0, out_bytes: hdr.len() as u64, blocks: 0, threads: workers, block_size: bs, largest: 0 };
    // a preferred cut is taken once the block holds at least this much
    let min_fill = (bs as u64 / 4).min(4 << 20).max(1);
    let mut pos = 0u64;
    let mut ci = 0usize;

    // LZ sub-blocks are sized from the work in flight per thread (fixed for the run, so
    // the output does not depend on timing)
    crate::lzx::set_share(((workers * bs) as u64).min(total.max(1)).div_ceil(threads as u64) as usize);
    crate::lzx::set_effort(threads, total);
    let mut eof = false;
    let mut nblock = 0usize;
    let verbose = o.verbose;
    par::pipeline(
        workers,
        workers,
        true,
        || {
            if eof {
                return Ok(None);
            }
            let end = pos + bs as u64;
            while ci < cuts.len() && cuts[ci].0 <= pos {
                ci += 1;
            }
            let ahead = cuts[ci..].iter().take_while(|c| c.0 < end);
            // a change of file type ends the block (once it holds min_fill); otherwise the
            // block takes as many whole files as fit, cutting a file only if it must
            let want = match ahead.clone().find(|c| c.1 && c.0 >= pos + min_fill) {
                Some(c) => c.0 - pos,
                None if end >= total => bs as u64,
                None => ahead.filter(|c| c.0 >= pos + bs as u64 / 2).last().map_or(bs as u64, |c| c.0 - pos),
            };
            let want = want as usize;
            let mut b = vec![0u8; want];
            let n = read_full(&mut input, &mut b)?;
            b.truncate(n);
            pos += n as u64;
            if n < want {
                eof = true;
            }
            if n == 0 {
                return Ok(None);
            }
            nblock += 1;
            Ok(Some(b))
        },
        |b: Vec<u8>| {
            let p = pack_block(&b, o.mode, threads);
            crate::progress::block(b.len() as u64);
            p
        },
        |p: Packed| {
            let mut h = Vec::with_capacity(24);
            h.push(p.kind);
            write_varint(&mut h, p.raw_len as u64);
            write_varint(&mut h, p.data.len() as u64);
            h.extend_from_slice(&p.crc.to_le_bytes());
            out.write_all(&h)?;
            out.write_all(&p.data)?;
            st.in_bytes += p.raw_len as u64;
            st.out_bytes += (h.len() + p.data.len()) as u64;
            st.blocks += 1;
            st.largest = st.largest.max(p.raw_len as u64);
            if verbose {
                eprintln!("  block {:>4}: {:>11} -> {:>11} {}", st.blocks, p.raw_len, p.data.len(), p.note);
            }
            Ok(())
        },
    )?;
    out.write_all(&[KIND_END])?;
    st.out_bytes += 1;
    out.flush()?;
    Ok(st)
}

pub struct Header {
    pub flags: u8,
    pub mode: Mode,
    pub total: u64,
    pub block_size: u64,
}

pub fn read_header<R: Read>(r: &mut R) -> io::Result<Header> {
    let mut m = [0u8; 4];
    r.read_exact(&mut m)?;
    if &m != MAGIC {
        return Err(bad("not an nzip archive"));
    }
    let ver = read_u8(r)?;
    if ver != VERSION {
        return Err(bad(&format!("archive version {ver} is not supported (this build reads {VERSION})")));
    }
    let flags = read_u8(r)?;
    let mode = match read_u8(r)? {
        0 => Mode::Fast,
        1 => Mode::Max,
        _ => return Err(bad("unknown mode")),
    };
    let total = read_varint(r)?;
    let block_size = read_varint(r)?;
    Ok(Header { flags, mode, total, block_size })
}

struct Raw {
    kind: u8,
    raw_len: usize,
    crc: u32,
    data: Vec<u8>,
}

fn unpack_bwt(p: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt BWT block".to_string();
    let flags = *p.first().ok_or_else(e)?;
    let mut r = &p[1..];
    let n = read_varint(&mut r).map_err(|_| e())? as usize;
    let k = crate::sais::chains_for(n);
    if n > raw_len.max(1 << 16) * 2 || r.len() < 4 * k {
        return Err(e());
    }
    let idx: Vec<u32> = (0..k).map(|i| u32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap())).collect();
    trace("decode start", n);
    let b = crate::bwtc::decode(&r[4 * k..], n, threads)?;
    trace("coder decoded", n);
    let t = crate::sais::unbwt(&b, &idx, threads)?;
    trace("unbwt done", n);
    drop(b);
    if flags & 1 != 0 {
        crate::textx::decode(&t, threads)
    } else if flags & 2 != 0 {
        let mut t = t;
        crate::x86::decode(&mut t);
        Ok(t)
    } else {
        Ok(t)
    }
}

fn unpack_payload(kind: u8, data: &[u8], raw_len: usize, threads: usize) -> Result<Vec<u8>, String> {
    match kind {
        KIND_STORED => Ok(data.to_vec()),
        KIND_BWT => unpack_bwt(data, raw_len, threads),
        KIND_LZ => crate::lz::decompress(data, raw_len),
        KIND_LZX => unpack_lz(data, raw_len, threads),
        KIND_MIX => unpack_mix(data, raw_len, threads),
        KIND_REZIP => unpack_rezip(data, raw_len, threads),
        KIND_LRM => unpack_lrm(data, raw_len, threads),
        _ => Err("unknown block kind".into()),
    }
}

fn unpack_block(b: &Raw, threads: usize) -> io::Result<Vec<u8>> {
    let out = unpack_payload(b.kind, &b.data, b.raw_len, threads).map_err(|e| bad(&e))?;
    trace("payload done", b.raw_len);
    if out.len() != b.raw_len || crate::codec::crc32_par(&out, threads) != b.crc {
        return Err(bad("CRC mismatch: archive is corrupt"));
    }
    Ok(out)
}

/// Decompress the block stream after the header into `out`. Returns bytes written.
pub fn decompress<R: Read, W: Write>(mut input: R, h: &Header, mut out: W, threads: usize) -> io::Result<u64> {
    let bs = h.block_size as usize;
    let threads = threads.max(1);
    // block workers only queue work (CPU tokens limit the threads that run), so more blocks
    // than threads may start, as memory allows: every block's pieces are known early
    let workers = fit_threads(4 * threads, block_mem(h.mode, bs));
    let mut written = 0u64;
    let mut done = false;
    par::pipeline(
        workers,
        workers * 2,
        false,
        || {
            if done {
                return Ok(None);
            }
            let kind = read_u8(&mut input)?;
            if kind == KIND_END {
                done = true;
                return Ok(None);
            }
            let raw_len = read_varint(&mut input)? as usize;
            let comp_len = read_varint(&mut input)? as usize;
            if raw_len > bs.max(1) || comp_len > raw_len + (1 << 16) {
                return Err(bad("corrupt block header"));
            }
            let mut c = [0u8; 4];
            input.read_exact(&mut c)?;
            let crc = u32::from_le_bytes(c);
            let mut data = vec![0u8; comp_len];
            input.read_exact(&mut data)?;
            Ok(Some(Raw { kind, raw_len, crc, data }))
        },
        |b: Raw| {
            trace("block decode start", b.raw_len);
            let r = unpack_block(&b, threads);
            trace("block decoded", b.raw_len);
            crate::progress::block(b.raw_len as u64);
            r
        },
        |r: io::Result<Vec<u8>>| {
            let d = r?;
            out.write_all(&d)?;
            written += d.len() as u64;
            Ok(())
        },
    )?;
    out.flush()?;
    Ok(written)
}
