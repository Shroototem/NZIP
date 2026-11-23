//! Binary arithmetic coder (32-bit, carry-less). Probabilities are 16-bit integers, so the
//! coder works the same on every machine.

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
        let b = self.src.get(self.pos).copied().unwrap_or(0);
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

    /// True if the decoder read past the end of its input (corrupt or truncated data).
    pub fn overrun(&self) -> bool {
        self.pos > self.src.len() + 4
    }
}
