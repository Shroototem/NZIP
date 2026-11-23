//! Archive format and the parallel block pipeline.
//!
//! ```text
//! header : "NZIP" ver:u8 flags:u8 mode:u8 total_len:varint block_size:varint
//! block  : kind:u8 raw_len:varint comp_len:varint crc32:u32le payload[comp_len]
//! end    : kind = 0
//! BWT payload : flags:u8 (1 = word transform, 2 = x86 filter) bwt_len:varint idx[k]:u32le coder-stream
//! LZ payload  : lz-stream
//! ```
//! Input is read in waves of blocks; each wave is compressed in parallel and written in
//! order, so memory stays bounded for any input size. Blocks are independent: the decoder
//! picks its own thread count from its free RAM and always gets identical bytes.

use crate::codec::{crc32, entropy_milli};
use crate::par;
use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"NZIP";
pub const VERSION: u8 = 1;
pub const FLAG_DIR: u8 = 1;

const KIND_END: u8 = 0;
const KIND_STORED: u8 = 1;
const KIND_BWT: u8 = 4;
const KIND_LZ: u8 = 5;

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
            match crate::gpu::free_vram() {
                Some(v) if total >= 4 << 20 => {
                    let by_vram = v.saturating_sub(256 << 20) / 34;
                    even(by_vram.min(ram / 7).min(MAX_BLOCK as u64))
                }
                _ => {
                    let per = total.div_ceil(threads);
                    let by_ram = (ram / (7 * threads)).max(8 << 20);
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
    let crc = crc32(data);
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
        Mode::Max => {
            let mut flags = 0u8;
            let mut note = "bwt";
            let t = if crate::textx::looks_like_text(data) {
                match crate::textx::encode(data, threads) {
                    Some(t) if t.len() * 100 < data.len() * 95 => {
                        flags |= 1;
                        note = "text+bwt";
                        Some(t)
                    }
                    _ => None,
                }
            } else if crate::x86::looks_like_x86(data) {
                let mut t = data.to_vec();
                crate::x86::encode(&mut t);
                flags |= 2;
                note = "x86+bwt";
                Some(t)
            } else {
                None
            };
            let src: &[u8] = t.as_deref().unwrap_or(data);
            trace("transform done", src.len());
            let (b, idx) = crate::gpu::bwt(src);
            let src_len = src.len();
            trace("bwt done", src_len);
            let mut out = Vec::with_capacity(src.len() / 3 + 64);
            out.push(flags);
            write_varint(&mut out, src.len() as u64);
            for i in idx {
                out.extend_from_slice(&i.to_le_bytes());
            }
            drop(t);
            out.extend_from_slice(&crate::bwtc::encode(&b, threads));
            trace("coded", src_len);
            (KIND_BWT, out, note)
        }
    };
    if out.len() >= data.len() {
        return stored(data, crc, "expanded");
    }
    Packed { kind, raw_len: data.len(), crc, data: out, note }
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
    let mut st = Stats { in_bytes: 0, out_bytes: hdr.len() as u64, blocks: 0, threads: workers, block_size: bs };
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
        let packed = par::map(&wave, workers, |b| pack_block(b, o.mode, threads));
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

fn unpack_block(b: &Raw, threads: usize) -> io::Result<Vec<u8>> {
    let out = match b.kind {
        KIND_STORED => b.data.clone(),
        KIND_BWT => unpack_bwt(&b.data, b.raw_len, threads).map_err(|e| bad(&e))?,
        KIND_LZ => crate::lz::decompress(&b.data, b.raw_len).map_err(|e| bad(&e))?,
        _ => return Err(bad("unknown block kind")),
    };
    if out.len() != b.raw_len || crc32(&out) != b.crc {
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
        for r in par::map(&wave, workers, |b| unpack_block(b, threads)) {
            let d = r?;
            out.write_all(&d)?;
            written += d.len() as u64;
        }
    }
    out.flush()?;
    Ok(written)
}
