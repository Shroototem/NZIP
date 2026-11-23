//! Archive format and the parallel block pipeline.
//!
//! ```text
//! header : "NZIP" ver:u8 flags:u8 mode:u8 total_len:varint block_size:varint
//! block  : kind:u8 raw_len:varint comp_len:varint crc32:u32le payload[comp_len]
//! end    : kind = 0
//! BWT payload : flags:u8 (1 = word transform, 2 = x86 filter) bwt_len:varint idx[k]:u32le coder-stream
//! LZ payload  : lz-stream (fast mode)
//! LZX payload : flags:u8 (1 = x86 split) [4 raw lens, 4 comp lens] lzx stream(s) (max mode, binary)
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
pub const VERSION: u8 = 3;
pub const FLAG_DIR: u8 = 1;

const KIND_END: u8 = 0;
const KIND_STORED: u8 = 1;
const KIND_BWT: u8 = 4;
const KIND_LZ: u8 = 5;
const KIND_LZX: u8 = 6;
const KIND_MIX: u8 = 7;

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
/// * Max, CPU only: about one block per thread, since each suffix sort runs on one thread.
/// Blocks are split evenly. The archive records the size, so decoding adapts to any machine.
pub fn choose_block_size(total: u64, mode: Mode, threads: usize) -> usize {
    let threads = threads.max(1) as u64;
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
                _ => {
                    // both engines run every core inside one block (coder segments, LZ
                    // pieces), so a block is never cut below 32 MB just to feed threads:
                    // small blocks cost 10-25% on mid-size files
                    let per = total.div_ceil(threads).max(32 << 20);
                    let by_ram = (ram / 7).max(8 << 20);
                    even(per.min(by_ram).min(CPU_BLOCK as u64))
                }
            }
        }
    }
}

/// Largest block when the CPU sorts (one sort per thread).
pub const CPU_BLOCK: usize = 64 << 20;

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
    if std::env::var_os("NZIP_TRACE").is_some() {
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
fn lz_payload(data: &[u8], x86: bool, threads: usize) -> Vec<u8> {
    let mut out = Vec::new();
    if x86 {
        let (sp, lens) = crate::x86::split_chunks(data, threads);
        let streams = [&sp.main, &sp.flags, &sp.addr, &sp.jump];
        // all four streams at once: the small ones are single sub-blocks
        let coded: Vec<Vec<u8>> = par::map(&streams, 4, |s| crate::lzx::encode(s, threads, false, true));
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
        out.extend_from_slice(&crate::lzx::encode(data, threads, false, false));
    }
    out
}

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
    use crate::mix::{BIN, Piece, RAW, TEXT};
    let zip = crate::mix::zip_pieces(data);
    if zip.is_none() && crate::textx::looks_like_text(data) {
        return pack_plain(data, threads);
    }
    let base = zip.unwrap_or_else(|| vec![Piece { len: data.len(), kind: BIN }]);
    let pieces = crate::mix::raw_split(data, &base, threads);
    let has = |k: u8| pieces.iter().any(|p| p.kind == k);
    if !has(RAW) && !has(TEXT) {
        return pack_plain(data, threads);
    }
    if !has(BIN) && !has(TEXT) {
        return (KIND_STORED, data.to_vec(), "incompressible");
    }
    (KIND_MIX, mix_payload(data, &pieces, threads), if has(TEXT) { "zip split" } else { "raw split" })
}

/// Mixed block: text, binary and raw pieces packed apart (see mix.rs).
fn mix_payload(data: &[u8], pieces: &[crate::mix::Piece], threads: usize) -> Vec<u8> {
    let parts = crate::mix::gather(data, pieces);
    let packed = par::map(&parts, 2, |p| if p.is_empty() { (KIND_STORED, Vec::new(), "") } else { pack_plain(p, threads) });
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
    let out = par::map(&parts, 3, |&(kind, n, c)| unpack_payload(kind, c, n, threads)).into_iter().collect::<Result<Vec<_>, _>>()?;
    crate::mix::scatter(&out, &pieces)
}

/// One engine for the whole block.
fn pack_plain(data: &[u8], threads: usize) -> (u8, Vec<u8>, &'static str) {
    if crate::textx::looks_like_text(data) {
        if let Some(t) = crate::textx::encode(data, threads) {
            if t.len() * 100 < data.len() * 95 {
                return (KIND_BWT, bwt_payload(&t, 1, threads), "text+bwt");
            }
        }
        return (KIND_BWT, bwt_payload(data, 0, threads), "bwt");
    }
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
    if data.len() <= 2 << 20 {
        let b = bwt_full(data);
        let l = lz_payload(data, x86, threads);
        return if l.len() < b.len() { (KIND_LZX, l, if x86 { "x86+lz" } else { "lz" }) } else { (KIND_BWT, b, if x86 { "x86+bwt" } else { "bwt" }) };
    }
    let s = sample_of(data);
    let sb = bwt_full(&s).len();
    let sl = lz_payload(&s, x86, threads).len();
    trace(if sl <= sb { "engine: lz" } else { "engine: bwt" }, data.len());
    if sl <= sb {
        (KIND_LZX, lz_payload(data, x86, threads), if x86 { "x86+lz" } else { "lz" })
    } else {
        (KIND_BWT, bwt_full(data), if x86 { "x86+bwt" } else { "bwt" })
    }
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
    let streams = par::map(&jobs, 4, |&(c, n)| crate::lzx::decode(c, n, threads, false)).into_iter().collect::<Result<Vec<_>, _>>()?;
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

/// `cuts`: sorted stream offsets where a block boundary is preferred (file / type changes).
pub fn compress<R: Read, W: Write>(mut input: R, total: u64, flags: u8, mut out: W, o: &Options, cuts: &[u64]) -> io::Result<Stats> {
    let bs = o.block_size.unwrap_or_else(|| choose_block_size(total, o.mode, o.threads)).max(1);
    let threads = o.threads.max(1);
    // blocks in flight: limited by free RAM; spare threads work inside each block
    let workers = fit_threads(threads, block_mem(o.mode, bs));
    let wave_len = if o.mode == Mode::Fast { workers * 4 } else { workers };
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

    loop {
        let mut wave: Vec<Vec<u8>> = Vec::new();
        let mut eof = false;
        while wave.len() < wave_len {
            let mut want = bs as u64;
            while ci < cuts.len() && cuts[ci] < pos + min_fill {
                ci += 1;
            }
            if ci < cuts.len() && cuts[ci] < pos + want {
                want = cuts[ci] - pos;
            }
            let want = want as usize;
            let mut b = vec![0u8; want];
            let n = read_full(&mut input, &mut b)?;
            b.truncate(n);
            pos += n as u64;
            if n > 0 {
                wave.push(b);
            }
            if n < want {
                eof = true;
                break;
            }
        }
        // Every block may use all threads inside: while blocks overlap the OS shares the
        // cores, and the last block running gets the whole machine.
        if let Some(d) = std::env::var_os("NZ_DUMPBLOCKS") {
            for (i, b) in wave.iter().enumerate() {
                std::fs::write(std::path::Path::new(&d).join(format!("b{}", st.blocks + i + 1)), b)?;
            }
        }
        let wave_bytes: usize = wave.iter().map(|b| b.len()).sum();
        crate::lzx::set_share(wave_bytes.div_ceil(threads));
        let packed = par::map(&wave, workers, |b| {
            let p = pack_block(b, o.mode, threads);
            crate::progress::block(b.len() as u64);
            p
        });
        drop(wave);
        for p in packed {
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
            if o.verbose {
                eprintln!("  block {:>4}: {:>11} -> {:>11} {}", st.blocks, p.raw_len, p.data.len(), p.note);
            }
        }
        if eof {
            break;
        }
    }
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
        _ => Err("unknown block kind".into()),
    }
}

fn unpack_block(b: &Raw, threads: usize) -> io::Result<Vec<u8>> {
    let out = unpack_payload(b.kind, &b.data, b.raw_len, threads).map_err(|e| bad(&e))?;
    if out.len() != b.raw_len || crate::codec::crc32_par(&out, threads) != b.crc {
        return Err(bad("CRC mismatch: archive is corrupt"));
    }
    Ok(out)
}

/// Decompress the block stream after the header into `out`. Returns bytes written.
pub fn decompress<R: Read, W: Write>(mut input: R, h: &Header, mut out: W, threads: usize) -> io::Result<u64> {
    let bs = h.block_size as usize;
    let threads = threads.max(1);
    let workers = fit_threads(threads, block_mem(h.mode, bs));
    let wave_len = if h.mode == Mode::Fast { workers * 4 } else { workers };
    let mut written = 0u64;
    let mut done = false;
    while !done {
        let mut wave = Vec::new();
        while wave.len() < wave_len {
            let kind = read_u8(&mut input)?;
            if kind == KIND_END {
                done = true;
                break;
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
            wave.push(Raw { kind, raw_len, crc, data });
        }
        let unpacked = par::map(&wave, workers, |b| {
            trace("block decode start", b.raw_len);
            let r = unpack_block(b, threads);
            trace("block decoded", b.raw_len);
            crate::progress::block(b.raw_len as u64);
            r
        });
        trace("wave decoded", 0);
        for r in unpacked {
            let d = r?;
            out.write_all(&d)?;
            written += d.len() as u64;
        }
    }
    out.flush()?;
    Ok(written)
}
