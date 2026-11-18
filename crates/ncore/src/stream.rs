//! Lane layout and serialisation of an ncore stream.
//!
//! ```text
//! u32 n_lanes | u32 lane_len (bytes per lane; last lane may be shorter) | u64 total_len
//! u32 words[n_lanes]   (u16 word count of each lane's rANS stream)
//! u16 lane streams, concatenated (each already in decoder order)
//! ```

use crate::Error;

pub struct Layout {
    pub n_lanes: usize,
    pub lane_len: usize,
    pub total: usize,
}

impl Layout {
    /// Lanes of `lane_len` bytes (the last one shorter).
    pub fn new(total: usize, lane_len: usize) -> Self {
        let lane_len = lane_len.max(1);
        Self { n_lanes: total.div_ceil(lane_len), lane_len, total }
    }
    pub fn lane_range(&self, i: usize) -> std::ops::Range<usize> {
        let s = i * self.lane_len;
        s..(s + self.lane_len).min(self.total)
    }
}

/// Header plus per-lane word counts (everything before the packed words).
pub fn write_header(layout: &Layout, counts: impl Iterator<Item = u32>, out: &mut Vec<u8>) {
    out.extend_from_slice(&(layout.n_lanes as u32).to_le_bytes());
    out.extend_from_slice(&(layout.lane_len as u32).to_le_bytes());
    out.extend_from_slice(&(layout.total as u64).to_le_bytes());
    for c in counts {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

pub fn header_len(n_lanes: usize) -> usize {
    16 + 4 * n_lanes
}

pub fn write(layout: &Layout, lanes: &[Vec<u16>]) -> Vec<u8> {
    let words: usize = lanes.iter().map(|l| l.len()).sum();
    let mut out = Vec::with_capacity(header_len(lanes.len()) + 2 * words);
    write_header(layout, lanes.iter().map(|l| l.len() as u32), &mut out);
    for l in lanes {
        for w in l {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// Parse a stream without copying: layout, per-lane word offsets (prefix sums), and
/// the packed little-endian word area.
pub fn parse(b: &[u8]) -> Result<(Layout, Vec<usize>, &[u8]), Error> {
    if b.len() < 16 {
        return Err(Error::Corrupt("short header"));
    }
    let n_lanes = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let lane_len = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let total = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let total = usize::try_from(total).map_err(|_| Error::Corrupt("total too large"))?;
    if lane_len == 0 || n_lanes != total.div_ceil(lane_len) || n_lanes > (b.len() - 16) / 4 {
        return Err(Error::Corrupt("bad lane layout"));
    }
    let mut offs = Vec::with_capacity(n_lanes + 1);
    let mut acc = 0usize;
    offs.push(0);
    for i in 0..n_lanes {
        let o = 16 + 4 * i;
        acc = acc.checked_add(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize).ok_or(Error::Corrupt("overflow"))?;
        offs.push(acc);
    }
    let body = &b[header_len(n_lanes)..];
    if body.len() != 2 * acc {
        return Err(Error::Corrupt("word count mismatch"));
    }
    Ok((Layout { n_lanes, lane_len, total }, offs, body))
}

/// Parse a stream: layout, per-lane word offsets (prefix sums), and all words.
pub fn read(b: &[u8]) -> Result<(Layout, Vec<usize>, Vec<u16>), Error> {
    let (layout, offs, body) = parse(b)?;
    let words = body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Ok((layout, offs, words))
}
