//! On-disk container format, version 2. All integers little-endian.
//!
//! ```text
//! FileHeader (40 bytes)
//!   magic "NZIP" | version u16 | flags u16 | orig_len u64 | orig_xxh3 u64
//!   n_frames u32 | reserved u32 | header_check u64 (xxh3 of the 32 bytes before it)
//! Frame * n_frames
//!   FrameHeader (40 bytes)
//!     kind u8 | codec u8 | transform u8 | reserved u8 | param u32
//!     raw_len u32 | comp_len u32 | raw_xxh3 u64 | comp_xxh3 u64
//!     header_check u64 (xxh3 of the 32 bytes before it)
//!   payload [comp_len]
//! ```
//!
//! With `FLAG_DEDUP` the frames come in segments: one recipe frame (zstd-compressed
//! dedup ops; `param` = number of data frames that follow for this segment) then those
//! data frames. Ops take literals from the segment's data in order and copy from any
//! earlier output (absolute offsets). Without the flag, all frames are data frames.
//! The file header is written last by the streaming compressor.
//!
//! `comp_xxh3` is verified before a payload reaches any decoder, so corrupt or
//! hostile input never gets parsed by the C codecs. `raw_xxh3` checks each block
//! after decoding and `orig_xxh3` checks the whole file.

use crate::Error;
use xxhash_rust::xxh3::xxh3_64;

pub const MAGIC: [u8; 4] = *b"NZIP";
pub const VERSION: u16 = 2;
pub const FILE_HEADER_LEN: usize = 40;
pub const FRAME_HEADER_LEN: usize = 40;

pub const FLAG_DEDUP: u16 = 1;
/// The original data is a folder packed as a tar stream (the CLI packs and extracts it).
pub const FLAG_DIR: u16 = 2;
/// Deflate streams were unpacked (`expand.rs`): the body is a sequence of waves.
pub const FLAG_UNPACK: u16 = 4;

pub const KIND_DATA: u8 = 0;
pub const KIND_RECIPE: u8 = 1;

pub const CODEC_STORE: u8 = 0;
pub const CODEC_ZSTD: u8 = 1;
pub const CODEC_BSC: u8 = 2;
/// ncore GPU entropy core (Phase 2, experimental test model). CPU-decodable.
pub const CODEC_NCORE: u8 = 3;

pub const TF_E8E9: u8 = 1;
pub const TF_REORDER: u8 = 2;
/// Word dictionary transform (`dict.rs`); `param` holds the transformed length.
pub const TF_DICT: u8 = 4;

/// Largest raw block the format allows (also bounded by libbsc's int sizes).
pub const MAX_RAW_BLOCK: usize = bsc_sys::MAX_BLOCK;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    pub flags: u16,
    pub orig_len: u64,
    pub orig_xxh3: u64,
    pub n_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub kind: u8,
    pub codec: u8,
    pub transform: u8,
    pub param: u32,
    pub raw_len: u32,
    pub comp_len: u32,
    pub raw_xxh3: u64,
    pub comp_xxh3: u64,
}

fn u16_at(b: &[u8], o: usize) -> u16 { u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) }
fn u32_at(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64_at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

impl FileHeader {
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.orig_len.to_le_bytes());
        out.extend_from_slice(&self.orig_xxh3.to_le_bytes());
        out.extend_from_slice(&self.n_frames.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let check = xxh3_64(&out[start..]);
        out.extend_from_slice(&check.to_le_bytes());
    }

    pub fn read(b: &[u8]) -> Result<Self, Error> {
        if b.len() < FILE_HEADER_LEN || b[..4] != MAGIC {
            return Err(Error::Format("not an nzip file"));
        }
        if xxh3_64(&b[..32]) != u64_at(b, 32) {
            return Err(Error::Corrupt("file header checksum mismatch"));
        }
        let version = u16_at(b, 4);
        if version != VERSION {
            return Err(Error::Format("unsupported nzip format version"));
        }
        let h = FileHeader { flags: u16_at(b, 6), orig_len: u64_at(b, 8), orig_xxh3: u64_at(b, 16), n_frames: u32_at(b, 24) };
        if h.flags & !(FLAG_DEDUP | FLAG_DIR | FLAG_UNPACK) != 0 {
            return Err(Error::Format("unknown file flags"));
        }
        Ok(h)
    }
}

impl FrameHeader {
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&[self.kind, self.codec, self.transform, 0]);
        out.extend_from_slice(&self.param.to_le_bytes());
        out.extend_from_slice(&self.raw_len.to_le_bytes());
        out.extend_from_slice(&self.comp_len.to_le_bytes());
        out.extend_from_slice(&self.raw_xxh3.to_le_bytes());
        out.extend_from_slice(&self.comp_xxh3.to_le_bytes());
        let check = xxh3_64(&out[start..]);
        out.extend_from_slice(&check.to_le_bytes());
    }

    pub fn read(b: &[u8]) -> Result<Self, Error> {
        if b.len() < FRAME_HEADER_LEN {
            return Err(Error::Corrupt("truncated frame header"));
        }
        if xxh3_64(&b[..32]) != u64_at(b, 32) {
            return Err(Error::Corrupt("frame header checksum mismatch"));
        }
        let h = FrameHeader {
            kind: b[0], codec: b[1], transform: b[2], param: u32_at(b, 4), raw_len: u32_at(b, 8),
            comp_len: u32_at(b, 12), raw_xxh3: u64_at(b, 16), comp_xxh3: u64_at(b, 24),
        };
        if h.kind > KIND_RECIPE || h.codec > CODEC_NCORE || h.transform & !(TF_E8E9 | TF_REORDER | TF_DICT) != 0 {
            return Err(Error::Format("unknown frame kind, codec or transform"));
        }
        if h.raw_len as usize > MAX_RAW_BLOCK {
            return Err(Error::Corrupt("frame larger than the format allows"));
        }
        if h.codec == CODEC_STORE && h.comp_len != h.raw_len {
            return Err(Error::Corrupt("stored frame length mismatch"));
        }
        Ok(h)
    }
}

/// A parsed frame: header plus a borrowed payload.
pub struct Frame<'a> {
    pub header: FrameHeader,
    pub payload: &'a [u8],
}

/// Parse and validate the whole container structure (no decoding yet).
pub fn parse(data: &[u8]) -> Result<(FileHeader, Vec<Frame<'_>>), Error> {
    let fh = FileHeader::read(data)?;
    let mut pos = FILE_HEADER_LEN;
    // Every frame needs at least its header, so this bounds the allocation.
    let max_frames = (data.len() - FILE_HEADER_LEN) / FRAME_HEADER_LEN;
    if fh.n_frames as usize > max_frames {
        return Err(Error::Corrupt("frame count exceeds file size"));
    }
    let mut frames = Vec::with_capacity(fh.n_frames as usize);
    for _ in 0..fh.n_frames {
        let header = FrameHeader::read(&data[pos..])?;
        pos += FRAME_HEADER_LEN;
        let end = pos.checked_add(header.comp_len as usize).ok_or(Error::Corrupt("frame length overflow"))?;
        if end > data.len() {
            return Err(Error::Corrupt("truncated frame payload"));
        }
        frames.push(Frame { header, payload: &data[pos..end] });
        pos = end;
    }
    if pos != data.len() {
        return Err(Error::Corrupt("trailing bytes after last frame"));
    }
    Ok((fh, frames))
}
