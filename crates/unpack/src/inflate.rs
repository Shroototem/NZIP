//! Deflate decoder that also records the stream's exact structure (blocks, their header
//! bits, every match), so the stream can be rebuilt bit for bit from its plain text.

/// LSB-first bit reader with a 64-bit refill buffer.
pub struct BitReader<'a> {
    d: &'a [u8],
    byte: usize, // next byte to load
    buf: u64,
    n: u32, // valid bits in buf
}

impl<'a> BitReader<'a> {
    pub fn new(d: &'a [u8]) -> Self {
        Self { d, byte: 0, buf: 0, n: 0 }
    }
    #[inline(always)]
    fn refill(&mut self) {
        if self.byte + 8 <= self.d.len() {
            let v = u64::from_le_bytes(self.d[self.byte..self.byte + 8].try_into().unwrap());
            self.buf |= v << self.n;
            let take = (63 - self.n) >> 3;
            self.byte += take as usize;
            self.n += take * 8;
        } else {
            while self.n <= 56 && self.byte < self.d.len() {
                self.buf |= (self.d[self.byte] as u64) << self.n;
                self.byte += 1;
                self.n += 8;
            }
        }
    }
    /// Up to 32 bits without consuming; None if the input ends first.
    #[inline(always)]
    pub fn peek(&mut self, n: u32) -> Option<u32> {
        if self.n < n {
            self.refill();
            if self.n < n {
                return None;
            }
        }
        Some((self.buf & ((1u64 << n) - 1)) as u32)
    }
    #[inline(always)]
    pub fn bits(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        let v = self.peek(n)?;
        self.consume(n);
        Some(v)
    }
    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.n -= n;
    }
    /// Bit position from the start.
    pub fn pos(&self) -> usize {
        self.byte * 8 - self.n as usize
    }
    /// Skips `n` bits (used for stored blocks; may cross the buffer).
    pub fn skip(&mut self, n: usize) {
        let p = self.pos() + n;
        self.byte = p >> 3;
        self.buf = 0;
        self.n = 0;
        let r = (p & 7) as u32;
        if r != 0 {
            self.refill();
            self.consume(r);
        }
    }
}

/// Canonical Huffman decoder: 10-bit lookup plus a slow path for longer codes.
pub struct Huff {
    fast: Vec<u32>, // (sym << 4) | len, len 0 = slow path
    counts: [u16; 16],
    syms: Vec<u16>,
}

const FAST: u32 = 10;

impl Huff {
    pub fn new(lens: &[u8]) -> Option<Self> {
        let mut counts = [0u16; 16];
        for &l in lens {
            if l > 15 {
                return None;
            }
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        // Reject over-subscribed sets, and incomplete ones unless a single code (as zlib).
        let mut left: i32 = 1;
        let mut max = 0;
        for (l, &c) in counts.iter().enumerate().skip(1) {
            left = (left << 1) - c as i32;
            if left < 0 {
                return None;
            }
            if c > 0 {
                max = l;
            }
        }
        if left > 0 && max > 1 {
            return None;
        }
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + counts[i - 1];
        }
        let mut syms = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l != 0 {
                syms[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        let mut fast = vec![0u32; 1 << FAST];
        let mut code = 0u32;
        let mut next = [0u32; 16];
        for l in 1..16 {
            code = (code + counts[l - 1] as u32) << 1;
            next[l] = code;
        }
        for (s, &l) in lens.iter().enumerate() {
            let l = l as u32;
            if l == 0 || l > FAST {
                continue;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            let mut rev = 0u32;
            for k in 0..l {
                rev |= ((c >> k) & 1) << (l - 1 - k);
            }
            let mut r = rev;
            while r < 1 << FAST {
                fast[r as usize] = ((s as u32) << 4) | l;
                r += 1 << l;
            }
        }
        Some(Self { fast, counts, syms })
    }

    #[inline]
    pub fn decode(&self, br: &mut BitReader) -> Option<u16> {
        if let Some(v) = br.peek(FAST) {
            let e = self.fast[v as usize];
            if e & 15 != 0 {
                br.consume(e & 15);
                return Some((e >> 4) as u16);
            }
        }
        // Slow canonical decode, bit by bit.
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return self.syms.get((index + (code - first)) as usize).copied();
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

pub const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
pub const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
pub const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
pub const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
pub const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

pub fn fixed_lens() -> ([u8; 288], [u8; 32]) {
    let mut l = [0u8; 288];
    l[..144].fill(8);
    l[144..256].fill(9);
    l[256..280].fill(7);
    l[280..].fill(8);
    (l, [5u8; 32])
}

/// Reads a dynamic block header; returns literal/length and distance code lengths.
pub fn read_dynamic_header(br: &mut BitReader) -> Option<(Vec<u8>, Vec<u8>)> {
    let hlit = br.bits(5)? as usize + 257;
    let hdist = br.bits(5)? as usize + 1;
    let hclen = br.bits(4)? as usize + 4;
    let mut cl = [0u8; 19];
    for &i in &CL_ORDER[..hclen] {
        cl[i] = br.bits(3)? as u8;
    }
    let ch = Huff::new(&cl)?;
    let mut lens = Vec::with_capacity(hlit + hdist);
    while lens.len() < hlit + hdist {
        let s = ch.decode(br)?;
        match s {
            0..=15 => lens.push(s as u8),
            16 => {
                let p = *lens.last()?;
                for _ in 0..3 + br.bits(2)? {
                    lens.push(p);
                }
            }
            17 => lens.extend(std::iter::repeat_n(0, 3 + br.bits(3)? as usize)),
            _ => lens.extend(std::iter::repeat_n(0, 11 + br.bits(7)? as usize)),
        }
    }
    if lens.len() != hlit + hdist || hlit > 286 || hdist > 30 {
        return None;
    }
    let d = lens.split_off(hlit);
    Some((lens, d))
}

#[derive(Debug, Clone)]
pub enum Kind {
    Stored { pad: u8, pad_bits: u8 },
    Fixed,
    /// Header bits after BTYPE, verbatim (LSB-first packed) and their count.
    Dynamic { header: Vec<u8>, header_bits: usize },
}

#[derive(Debug, Clone)]
pub struct Block {
    pub last: bool,
    pub kind: Kind,
    /// Plain text range this block produces.
    pub start: usize,
    pub end: usize,
}

/// A match token: plain-text position, length, distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub pos: u32,
    pub len: u16,
    pub dist: u16,
}

pub struct Parsed {
    pub plain: Vec<u8>,
    pub blocks: Vec<Block>,
    pub matches: Vec<Match>,
    /// Bytes of the input the deflate stream occupies.
    pub consumed: usize,
    /// Value of the unused bits after the last block (to the byte boundary) and their count.
    pub tail: u8,
    pub tail_bits: u8,
}

pub fn extract_bits(d: &[u8], from: usize, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n.div_ceil(8)];
    let mut br = BitReader::new(d);
    br.skip(from);
    let mut i = 0;
    while i < n {
        let k = (n - i).min(16) as u32;
        let v = br.bits(k).unwrap_or(0);
        for b in 0..k as usize {
            if (v >> b) & 1 != 0 {
                out[(i + b) >> 3] |= 1 << ((i + b) & 7);
            }
        }
        i += k as usize;
    }
    out
}

/// Parses a raw deflate stream. `limit` caps the plain-text size.
pub fn parse(d: &[u8], limit: usize) -> Option<Parsed> {
    let mut br = BitReader::new(d);
    let mut plain: Vec<u8> = Vec::new();
    let mut blocks = Vec::new();
    let mut matches = Vec::new();
    loop {
        let last = br.bits(1)? == 1;
        let bt = br.bits(2)?;
        let start = plain.len();
        let kind;
        match bt {
            0 => {
                let pad_bits = ((8 - (br.pos() & 7)) & 7) as u8;
                let pad = br.bits(pad_bits as u32)? as u8;
                let len = br.bits(16)? as usize;
                let nlen = br.bits(16)? as usize;
                if len != !nlen & 0xffff || plain.len() + len > limit {
                    return None;
                }
                let byte = br.pos() >> 3;
                plain.extend_from_slice(d.get(byte..byte + len)?);
                br.skip(len * 8);
                kind = Kind::Stored { pad, pad_bits };
            }
            1 | 2 => {
                let (ll, dl, k) = if bt == 1 {
                    let (a, b) = fixed_lens();
                    (a.to_vec(), b.to_vec(), Kind::Fixed)
                } else {
                    let h0 = br.pos();
                    let (a, b) = read_dynamic_header(&mut br)?;
                    let n = br.pos() - h0;
                    (a, b, Kind::Dynamic { header: extract_bits(d, h0, n), header_bits: n })
                };
                kind = k;
                if ll.get(256).copied().unwrap_or(0) == 0 {
                    return None; // no end-of-block code
                }
                let lh = Huff::new(&ll)?;
                let dh = Huff::new(&dl)?;
                loop {
                    let s = lh.decode(&mut br)?;
                    if s < 256 {
                        plain.push(s as u8);
                    } else if s == 256 {
                        break;
                    } else {
                        let i = (s - 257) as usize;
                        if i >= 29 {
                            return None;
                        }
                        let len = LEN_BASE[i] as usize + br.bits(LEN_EXTRA[i] as u32)? as usize;
                        let ds = dh.decode(&mut br)? as usize;
                        if ds >= 30 {
                            return None;
                        }
                        let dist = DIST_BASE[ds] as usize + br.bits(DIST_EXTRA[ds] as u32)? as usize;
                        if dist > plain.len() || plain.len() + len > limit {
                            return None;
                        }
                        matches.push(Match { pos: plain.len() as u32, len: len as u16, dist: dist as u16 });
                        let from = plain.len() - dist;
                        if dist >= len {
                            plain.extend_from_within(from..from + len);
                        } else {
                            // Overlapping copy: the source grows as it is copied.
                            let (mut src, mut left) = (from, len);
                            while left > 0 {
                                let k = left.min(plain.len() - src);
                                plain.extend_from_within(src..src + k);
                                src += k;
                                left -= k;
                            }
                        }
                    }
                }
            }
            _ => return None,
        }
        blocks.push(Block { last, kind, start, end: plain.len() });
        if last {
            break;
        }
    }
    let tail_bits = ((8 - (br.pos() & 7)) & 7) as u8;
    let tail = br.bits(tail_bits as u32)? as u8;
    Some(Parsed { plain, blocks, matches, consumed: br.pos() >> 3, tail, tail_bits })
}

/// What a deflate stream is, without keeping its output: bytes used, output size, token
/// bytes (literals + ~4 per copy) and, if asked, the Adler-32 of the output (computed
/// through a 32 KiB window, so memory stays small however large the output).
pub struct Measure {
    pub consumed: usize,
    pub out_len: usize,
    pub token_bytes: usize,
    pub adler: u32,
}

pub fn measure(d: &[u8], limit: usize, want_adler: bool) -> Option<Measure> {
    const W: usize = 1 << 15;
    let mut win = if want_adler { vec![0u8; W] } else { Vec::new() };
    let (mut a, mut b, mut pending) = (1u32, 0u32, 0u32);
    let mut out = 0usize;
    let mut token_bytes = 0usize;
    let mut br = BitReader::new(d);
    macro_rules! emit {
        ($v:expr) => {{
            let v = $v;
            if want_adler {
                win[out & (W - 1)] = v;
                a += v as u32;
                b += a;
                pending += 1;
                if pending == 5552 {
                    a %= 65521;
                    b %= 65521;
                    pending = 0;
                }
            }
            out += 1;
        }};
    }
    loop {
        let last = br.bits(1)? == 1;
        match br.bits(2)? {
            0 => {
                let pad = ((8 - (br.pos() & 7)) & 7) as u32;
                br.bits(pad)?;
                let len = br.bits(16)? as usize;
                if len != !(br.bits(16)? as usize) & 0xffff || out + len > limit {
                    return None;
                }
                let byte = br.pos() >> 3;
                for &v in d.get(byte..byte + len)? {
                    emit!(v);
                }
                br.skip(len * 8);
                token_bytes += len;
            }
            bt @ (1 | 2) => {
                let (ll, dl) = if bt == 1 {
                    let (x, y) = fixed_lens();
                    (x.to_vec(), y.to_vec())
                } else {
                    read_dynamic_header(&mut br)?
                };
                if ll.get(256).copied().unwrap_or(0) == 0 {
                    return None;
                }
                let lh = Huff::new(&ll)?;
                let dh = Huff::new(&dl)?;
                loop {
                    let s = lh.decode(&mut br)?;
                    if s < 256 {
                        emit!(s as u8);
                        token_bytes += 1;
                    } else if s == 256 {
                        break;
                    } else {
                        let i = (s - 257) as usize;
                        if i >= 29 {
                            return None;
                        }
                        let len = LEN_BASE[i] as usize + br.bits(LEN_EXTRA[i] as u32)? as usize;
                        let ds = dh.decode(&mut br)? as usize;
                        if ds >= 30 {
                            return None;
                        }
                        let dist = DIST_BASE[ds] as usize + br.bits(DIST_EXTRA[ds] as u32)? as usize;
                        if dist > out || out + len > limit {
                            return None;
                        }
                        if want_adler {
                            for _ in 0..len {
                                let v = win[(out - dist) & (W - 1)];
                                emit!(v);
                            }
                        } else {
                            out += len;
                        }
                        token_bytes += 4;
                    }
                }
            }
            _ => return None,
        }
        if last {
            break;
        }
    }
    let tail = ((8 - (br.pos() & 7)) & 7) as u32;
    br.bits(tail)?;
    Some(Measure { consumed: br.pos() >> 3, out_len: out, token_bytes, adler: ((b % 65521) << 16) | (a % 65521) })
}
