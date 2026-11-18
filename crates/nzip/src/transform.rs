//! Reversible pre-transforms applied to a block before its codec.
//!
//! * E8E9: converts x86 CALL/JMP relative targets to absolute positions so repeated
//!   calls to the same function become identical byte strings. Every E8/E9 opcode is
//!   transformed and its 4 operand bytes skipped, on both sides, so the scan stays in
//!   lockstep and the transform is always exactly invertible (wrapping arithmetic).
//! * Record reorder (bsc filter): for fixed-size records (tables, multi-byte samples),
//!   regroups bytes by field. Detected and applied through libbsc.

pub fn e8e9_encode(buf: &mut [u8]) {
    e8e9(buf, true)
}

pub fn e8e9_decode(buf: &mut [u8]) {
    e8e9(buf, false)
}

fn e8e9(buf: &mut [u8], forward: bool) {
    let n = buf.len();
    let mut i = 0usize;
    while i + 5 <= n {
        if buf[i] & 0xFE == 0xE8 {
            let a = u32::from_le_bytes(buf[i + 1..i + 5].try_into().unwrap());
            let pos = i as u32;
            let v = if forward { a.wrapping_add(pos) } else { a.wrapping_sub(pos) };
            buf[i + 1..i + 5].copy_from_slice(&v.to_le_bytes());
            i += 5;
        } else {
            i += 1;
        }
    }
}

/// Heuristic gate, then an empirical check: does E8E9 make a sample compress better?
pub fn should_e8e9(data: &[u8]) -> bool {
    if data.len() < 64 * 1024 {
        return false;
    }
    // Plausible CALL/JMP: opcode followed by a small signed displacement (high byte 00/FF).
    // Counted on up to 4 MiB from the middle of the block.
    let m = data.len().min(4 << 20);
    let mid = &data[(data.len() - m) / 2..][..m];
    let hits = mid.windows(5).filter(|w| w[0] & 0xFE == 0xE8 && (w[4] == 0 || w[4] == 0xFF)).count();
    if hits * 1000 < m * 2 {
        return false; // fewer than 0.2% of bytes: not x86 code
    }
    let n = data.len().min(1 << 20);
    let start = (data.len() - n) / 2;
    let plain = &data[start..start + n];
    let mut t = plain.to_vec();
    e8e9_encode(&mut t);
    let (a, b) = (zstd::bulk::compress(plain, 1), zstd::bulk::compress(&t, 1));
    matches!((a, b), (Ok(a), Ok(b)) if (b.len() as f64) < 0.99 * a.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn e8e9_roundtrip(mut v in proptest::collection::vec(any::<u8>(), 0..5000), salt in any::<u8>()) {
            // Sprinkle opcodes so the transform actually triggers.
            for i in (0..v.len()).step_by(7 + (salt as usize % 5)) { v[i] = 0xE8 | (i as u8 & 1); }
            let orig = v.clone();
            e8e9_encode(&mut v);
            e8e9_decode(&mut v);
            prop_assert_eq!(v, orig);
        }
    }
}
