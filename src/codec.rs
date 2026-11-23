//! Small shared helpers: CPU feature detection, entropy estimate, CRC-32.


pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::env::var_os("NZIP_NO_AVX2").is_none()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("lzcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Order-0 entropy in bits per byte, scaled by 1000 (8000 = random).
pub fn entropy_milli(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let mut h = [0u32; 256];
    for &b in data {
        h[b as usize] += 1;
    }
    // integer log2 approximation in 1/1024 units, deterministic
    let n = data.len() as u64;
    let mut bits: u64 = 0;
    for &c in h.iter() {
        if c > 0 {
            bits += c as u64 * (log2_fixed(n) - log2_fixed(c as u64));
        }
    }
    (bits * 1000 / 1024 / n) as u32
}

/// log2(x) * 1024, integer.
fn log2_fixed(x: u64) -> u64 {
    let e = 63 - x.leading_zeros() as u64;
    // linear interpolation of the mantissa is accurate to ~0.09 bits, enough for a heuristic
    let frac = if e >= 10 { (x >> (e - 10)) & 1023 } else { (x << (10 - e)) & 1023 };
    e * 1024 + frac
}

// ---------------------------------------------------------------- CRC-32 (slicing by 8)

static CRC: std::sync::OnceLock<[[u32; 256]; 8]> = std::sync::OnceLock::new();

fn crc_tables() -> &'static [[u32; 256]; 8] {
    CRC.get_or_init(|| {
        let mut t = [[0u32; 256]; 8];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            t[0][i as usize] = c;
        }
        for i in 0..256 {
            for k in 1..8 {
                let p = t[k - 1][i];
                t[k][i] = (p >> 8) ^ t[0][(p & 0xff) as usize];
            }
        }
        t
    })
}

pub fn crc32(data: &[u8]) -> u32 {
    let t = crc_tables();
    let mut c = !0u32;
    let mut chunks = data.chunks_exact(8);
    for ch in &mut chunks {
        let a = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) ^ c;
        let b = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]);
        c = t[7][(a & 0xff) as usize]
            ^ t[6][((a >> 8) & 0xff) as usize]
            ^ t[5][((a >> 16) & 0xff) as usize]
            ^ t[4][(a >> 24) as usize]
            ^ t[3][(b & 0xff) as usize]
            ^ t[2][((b >> 8) & 0xff) as usize]
            ^ t[1][((b >> 16) & 0xff) as usize]
            ^ t[0][(b >> 24) as usize];
    }
    for &b in chunks.remainder() {
        c = t[0][((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

}
