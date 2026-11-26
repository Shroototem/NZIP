//! Small shared helpers: CPU feature detection, entropy estimate, CRC-32.


pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !crate::sys::dev_flag("NZIP_NO_AVX2")
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
    !crc_update(!0, data)
}

/// Runs the CRC register `c` over `data`: carry-less multiply folding where the CPU
/// has it (many times faster), else slicing by 8. Both give the same value.
fn crc_update(c: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if data.len() >= 128 && has_clmul() {
        return unsafe { crc_clmul(c, data) };
    }
    crc_sw(c, data)
}

fn crc_sw(mut c: u32, data: &[u8]) -> u32 {
    let t = crc_tables();
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
    c
}

#[cfg(target_arch = "x86_64")]
fn has_clmul() -> bool {
    static H: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *H.get_or_init(|| std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("sse4.1"))
}

/// CRC-32 by folding 128-bit lanes with carry-less multiplies (Intel's "Fast CRC
/// Computation for Generic Polynomials Using PCLMULQDQ", bit-reflected form), then a
/// Barrett reduction; the tail (< 16 bytes) goes through `crc_sw`. Needs >= 64 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
unsafe fn crc_clmul(c: u32, mut data: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    const K1: i64 = 0x1_5444_2bd4;
    const K2: i64 = 0x1_c6e4_1596;
    const K3: i64 = 0x1_7519_97d0;
    const K4: i64 = 0x0_ccaa_009e;
    const K5: i64 = 0x1_63cd_6124;
    const P: i64 = 0x1_DB71_0641;
    const U: i64 = 0x1_F701_1641;
    #[inline(always)]
    unsafe fn get(d: &mut &[u8]) -> __m128i {
        let r = unsafe { _mm_loadu_si128(d.as_ptr() as *const __m128i) };
        *d = &d[16..];
        r
    }
    #[inline(always)]
    unsafe fn fold(a: __m128i, b: __m128i, k: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(_mm_xor_si128(b, _mm_clmulepi64_si128(a, k, 0x00)), _mm_clmulepi64_si128(a, k, 0x11)) }
    }
    unsafe {
        let mut x3 = get(&mut data);
        let mut x2 = get(&mut data);
        let mut x1 = get(&mut data);
        let mut x0 = get(&mut data);
        x3 = _mm_xor_si128(x3, _mm_cvtsi32_si128(c as i32));
        let k12 = _mm_set_epi64x(K2, K1);
        while data.len() >= 64 {
            x3 = fold(x3, get(&mut data), k12);
            x2 = fold(x2, get(&mut data), k12);
            x1 = fold(x1, get(&mut data), k12);
            x0 = fold(x0, get(&mut data), k12);
        }
        let k34 = _mm_set_epi64x(K4, K3);
        let mut x = fold(x3, x2, k34);
        x = fold(x, x1, k34);
        x = fold(x, x0, k34);
        while data.len() >= 16 {
            x = fold(x, get(&mut data), k34);
        }
        // 128 -> 64 bits
        let lo32 = _mm_set_epi32(0, 0, 0, !0);
        let x = _mm_xor_si128(_mm_clmulepi64_si128(x, k34, 0x10), _mm_srli_si128(x, 8));
        let x = _mm_xor_si128(_mm_clmulepi64_si128(_mm_and_si128(x, lo32), _mm_set_epi64x(0, K5), 0x00), _mm_srli_si128(x, 4));
        // Barrett reduction 64 -> 32 bits
        let pu = _mm_set_epi64x(U, P);
        let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, lo32), pu, 0x10);
        let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, lo32), pu, 0x00);
        let r = _mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32;
        crc_sw(r, data)
    }
}

/// CRC-32 of A||B from crc(A), crc(B) and len(B) (zlib's crc32_combine: multiply by
/// x^(8*len_b) in GF(2), by repeated squaring of the "one zero byte" operator).
pub fn crc32_combine(crc_a: u32, crc_b: u32, len_b: u64) -> u32 {
    fn times(m: &[u32; 32], mut v: u32) -> u32 {
        let mut s = 0;
        let mut i = 0;
        while v != 0 {
            if v & 1 != 0 {
                s ^= m[i];
            }
            v >>= 1;
            i += 1;
        }
        s
    }
    fn square(m: &[u32; 32]) -> [u32; 32] {
        let mut r = [0u32; 32];
        for (i, x) in r.iter_mut().enumerate() {
            *x = times(m, m[i]);
        }
        r
    }
    if len_b == 0 {
        return crc_a;
    }
    // operator for one zero bit
    let mut odd = [0u32; 32];
    odd[0] = 0xEDB8_8320;
    for (i, x) in odd.iter_mut().enumerate().skip(1) {
        *x = 1 << (i - 1);
    }
    let mut even = square(&odd); // 2 bits
    odd = square(&even); // 4 bits
    let mut c = crc_a;
    let mut len = len_b;
    loop {
        even = square(&odd);
        if len & 1 != 0 {
            c = times(&even, c);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
        odd = square(&even);
        if len & 1 != 0 {
            c = times(&odd, c);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
    }
    c ^ crc_b
}

/// CRC-32 computed over 4 MiB pieces in parallel (same value as `crc32`).
pub fn crc32_par(data: &[u8], threads: usize) -> u32 {
    const PIECE: usize = 4 << 20;
    if data.len() <= PIECE || threads <= 1 {
        return crc32(data);
    }
    let pieces: Vec<&[u8]> = data.chunks(PIECE).collect();
    let crcs = crate::par::map(&pieces, threads, |p| crc32(p));
    let mut c = crcs[0];
    for (p, &x) in pieces.iter().zip(&crcs).skip(1) {
        c = crc32_combine(c, x, p.len() as u64);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_combine_matches() {
        let v: Vec<u8> = (0..20_000_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        assert_eq!(crc32_par(&v, 8), crc32(&v));
        let (a, b) = v.split_at(12345);
        assert_eq!(crc32_combine(crc32(a), crc32(b), b.len() as u64), crc32(&v));
    }

    #[test]
    fn crc_hw_matches_table() {
        let v: Vec<u8> = (0..100_000u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect();
        for len in (0..300).chain([1000, 4095, 4096, 4097, 65_537, 100_000]) {
            assert_eq!(!crc_update(!0, &v[..len]), !crc_sw(!0, &v[..len]), "len {len}");
        }
    }

    #[test]
    fn crc_known() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

}
