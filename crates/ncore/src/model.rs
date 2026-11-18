//! Test-consumer model for the core: an adaptive order-1 + hashed order-2 frequency model.
//!
//! It exists to exercise the core end to end (distribution -> quantise -> rANS) with
//! realistic GPU memory traffic; it is **not** a competitive model (that is the neural
//! tier's job). The GPU kernel mirrors this code exactly.

pub const O2_BITS: u32 = 10;
pub const INC: u32 = 32;
pub const LIMIT: u32 = 65000;
pub const W2: u32 = 4;

pub struct O2Model {
    pub cnt1: Vec<u16>, // [256 * 256]
    pub cnt2: Vec<u16>, // [(1 << O2_BITS) * 256]
    pub c1: u32,
    pub c2: u32,
}

#[inline]
pub fn o2_hash(c2: u32, c1: u32) -> u32 {
    (((c2 << 8) | c1).wrapping_mul(2654435761)) >> (32 - O2_BITS)
}

impl O2Model {
    pub fn new() -> Self {
        Self { cnt1: vec![0; 256 * 256], cnt2: vec![0; (1 << O2_BITS) * 256], c1: 0, c2: 0 }
    }

    /// Raw weights for the next byte.
    #[inline]
    pub fn weights(&self, r: &mut [u32; 256]) {
        let a = (self.c1 * 256) as usize;
        let b = (o2_hash(self.c2, self.c1) * 256) as usize;
        for s in 0..256 {
            r[s] = self.cnt1[a + s] as u32 + W2 * self.cnt2[b + s] as u32;
        }
    }

    #[inline]
    pub fn update(&mut self, byte: u32) {
        let a = (self.c1 * 256) as usize;
        let b = (o2_hash(self.c2, self.c1) * 256) as usize;
        bump(&mut self.cnt1[a..a + 256], byte);
        bump(&mut self.cnt2[b..b + 256], byte);
        self.c2 = self.c1;
        self.c1 = byte;
    }
}

impl Default for O2Model {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn bump(row: &mut [u16], s: u32) {
    let v = row[s as usize] as u32 + INC;
    if v > LIMIT {
        for c in row.iter_mut() {
            *c >>= 1;
        }
        row[s as usize] = ((row[s as usize] as u32) + INC / 2) as u16;
    } else {
        row[s as usize] = v as u16;
    }
}
