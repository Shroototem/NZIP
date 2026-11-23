//! Reversible E8/E9 filter for x86 / x86-64 machine code (executables, DLLs, .so, .o).
//!
//! CALL/JMP rel32 targets are rewritten as block-relative absolute addresses, so repeated
//! calls to the same function become repeated byte strings. Only displacements whose top
//! byte is 0x00 or 0xFF are touched (24-bit arithmetic), and the encoder runs backwards
//! while the decoder runs forwards, so each sees exactly the bytes the other saw.

#[inline(always)]
fn hit(b: &[u8], i: usize) -> bool {
    (b[i] & 0xFE) == 0xE8 && (b[i + 4].wrapping_add(1) & 0xFE) == 0
}

/// True if the block looks like x86 code: many CALL/JMP opcodes with short displacements.
pub fn looks_like_x86(b: &[u8]) -> bool {
    if b.len() < 64 << 10 {
        return false;
    }
    let s = &b[..b.len().min(4 << 20)];
    let mut n = 0usize;
    for i in 0..s.len() - 4 {
        n += hit(s, i) as usize;
    }
    // x86 code: ~0.5-1.5% of bytes; text and most other data: far less
    n * 1000 >= s.len() * 3
}

pub fn encode(b: &mut [u8]) {
    if b.len() < 5 {
        return;
    }
    for i in (0..=b.len() - 5).rev() {
        if hit(b, i) {
            let a = (b[i + 1] as u32 | (b[i + 2] as u32) << 8 | (b[i + 3] as u32) << 16).wrapping_add(i as u32);
            b[i + 1] = a as u8;
            b[i + 2] = (a >> 8) as u8;
            b[i + 3] = (a >> 16) as u8;
        }
    }
}

pub fn decode(b: &mut [u8]) {
    if b.len() < 5 {
        return;
    }
    for i in 0..=b.len() - 5 {
        if hit(b, i) {
            let a = (b[i + 1] as u32 | (b[i + 2] as u32) << 8 | (b[i + 3] as u32) << 16).wrapping_sub(i as u32);
            b[i + 1] = a as u8;
            b[i + 2] = (a >> 8) as u8;
            b[i + 3] = (a >> 16) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut r = 5u32;
        let mut v: Vec<u8> = (0..300_000)
            .map(|i| {
                r = r.wrapping_mul(1103515245).wrapping_add(12345);
                if i % 7 == 0 { 0xE8 } else if i % 11 == 0 { 0xFF } else if i % 13 == 0 { 0 } else { (r >> 16) as u8 }
            })
            .collect();
        v.extend_from_slice(&[0xE8, 0, 0, 0, 0, 0xE9, 0xFF]);
        let orig = v.clone();
        super::encode(&mut v);
        assert_ne!(v, orig);
        super::decode(&mut v);
        assert_eq!(v, orig);
    }
}
