//! Binary arithmetic coder (32-bit, carry-less). Probabilities are 16-bit integers, so the
//! coder works the same on every machine.

use crate::tables::stretch;

pub struct Encoder {
    x1: u32,
    x2: u32,
    pub out: Vec<u8>,
}

impl Encoder {
    pub fn new(cap: usize) -> Self {
        Encoder { x1: 0, x2: u32::MAX, out: Vec::with_capacity(cap) }
    }

    /// Code `bit` where `p1` = P(bit == 1) scaled by 65536 (1..=65535).
    #[inline(always)]
    pub fn encode(&mut self, bit: u32, p1: u32) {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + ((range as u64 * p1 as u64) >> 16) as u32;
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    /// Codes the low `k` bits of `v` (k <= 16) as equally likely, in one step when the
    /// range is wide enough, else bit by bit (the decoder makes the same choice).
    #[inline(always)]
    pub fn encode_direct(&mut self, v: u32, k: u32) {
        let range = self.x2 - self.x1;
        if range >> (k + 8) == 0 {
            for i in (0..k).rev() {
                self.encode((v >> i) & 1, 32768);
            }
            return;
        }
        let step = range >> k;
        self.x1 += v * step;
        self.x2 = self.x1 + step - 1;
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Enough bytes for the decoder to land inside [x1, x2].
        let b = self.x1.to_be_bytes();
        self.out.extend_from_slice(&b);
        self.out
    }
}

pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    src: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: u32::MAX, x: 0, src, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.byte();
        }
        d
    }

    #[inline(always)]
    fn byte(&mut self) -> u32 {
        let b = if self.pos < self.src.len() { self.src[self.pos] } else { 0 };
        self.pos += 1;
        b as u32
    }

    #[inline(always)]
    pub fn decode(&mut self, p1: u32) -> u32 {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + ((range as u64 * p1 as u64) >> 16) as u32;
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.byte();
        }
        bit
    }

    /// Inverse of `Encoder::encode_direct`.
    #[inline(always)]
    pub fn decode_direct(&mut self, k: u32) -> u32 {
        let range = self.x2 - self.x1;
        if range >> (k + 8) == 0 {
            let mut v = 0;
            for _ in 0..k {
                v = (v << 1) | self.decode(32768);
            }
            return v;
        }
        let step = range >> k;
        let v = ((self.x - self.x1) / step).min((1 << k) - 1);
        self.x1 += v * step;
        self.x2 = self.x1 + step - 1;
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.byte();
        }
        v
    }

    /// True if the decoder read past the end of its input (corrupt or truncated data).
    pub fn overrun(&self) -> bool {
        self.pos > self.src.len() + 4
    }
}

/// Encoder or decoder behind one interface.
pub trait BitIo {
    /// Codes `bit` (ignored when decoding) with P(1) = p/65536; returns the actual bit.
    fn code(&mut self, bit: u32, p: u32) -> u32;
    /// Codes `k` (<= 16) equally likely bits of `v` (ignored when decoding).
    fn direct(&mut self, v: u32, k: u32) -> u32;
}

impl BitIo for Encoder {
    #[inline(always)]
    fn code(&mut self, bit: u32, p: u32) -> u32 {
        self.encode(bit, p);
        bit
    }
    #[inline(always)]
    fn direct(&mut self, v: u32, k: u32) -> u32 {
        self.encode_direct(v, k);
        v
    }
}

impl BitIo for Decoder<'_> {
    #[inline(always)]
    fn code(&mut self, _bit: u32, p: u32) -> u32 {
        self.decode(p)
    }
    #[inline(always)]
    fn direct(&mut self, _v: u32, k: u32) -> u32 {
        self.decode_direct(k)
    }
}

/// Two probabilities on one context: one adapts fast (rate 1/8), one slowly (1/128).
/// Pure shifts: cheap, and identical on every CPU.
#[derive(Clone, Copy)]
pub struct Dual(pub [u16; 2]);

impl Dual {
    pub const INIT: Dual = Dual([32768; 2]);
    /// Both logits (scaled by 256).
    #[inline(always)]
    pub fn st2(self) -> (i32, i32) {
        (stretch(self.0[0] as i32 >> 4), stretch(self.0[1] as i32 >> 4))
    }
    #[inline(always)]
    pub fn update(&mut self, y: u32, fast: u32, slow: u32) {
        let t = ((y << 16) - y) as i32;
        let a = self.0[0] as i32;
        let b = self.0[1] as i32;
        self.0[0] = (a + ((t - a) >> fast)) as u16;
        self.0[1] = (b + ((t - b) >> slow)) as u16;
    }
}

