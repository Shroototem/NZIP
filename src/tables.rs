//! Integer-only logistic tables (squash / stretch), built at compile time.
//!
//! Everything here is built with integer arithmetic, so the tables are bit-identical on every
//! CPU, OS and compiler. No floating point is used anywhere on the coding path; that is what
//! makes an archive decode correctly on any machine.


const KNOTS: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348,
    3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
];

/// Logistic function in fixed point (interpolated over 33 knots): `d` is a logit scaled by
/// 256, the result a probability scaled by 4096 (1..=4095).
const fn squash_calc(d: i32) -> i32 {
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 1;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (KNOTS[i] * (128 - w) + KNOTS[i + 1] * w + 64) >> 7
}

const SQUASH: [u16; 4096] = {
    let mut t = [0u16; 4096];
    let mut i = 0;
    while i < 4096 {
        t[i] = squash_calc(i as i32 - 2048) as u16;
        i += 1;
    }
    t
};

/// Inverse of squash: probability (0..4096) -> logit (-2047..=2047).
const STRETCH: [i16; 4096] = {
    let mut t = [0i16; 4096];
    let mut pi = 0usize;
    let mut d = -2047;
    while d <= 2047 {
        let v = squash_calc(d) as usize;
        while pi <= v {
            t[pi] = d as i16;
            pi += 1;
        }
        d += 1;
    }
    while pi < 4096 {
        t[pi] = 2047;
        pi += 1;
    }
    t
};

#[inline(always)]
pub fn squash(d: i32) -> i32 {
    SQUASH[(d.clamp(-2047, 2047) + 2048) as usize] as i32
}

#[inline(always)]
pub fn stretch(p: i32) -> i32 {
    STRETCH[p as usize & 4095] as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stretch_inverts_squash() {
        for p in 1..4095 {
            let d = stretch(p);
            assert!((squash(d) - p as i32).abs() <= 40, "p={p} d={d}");
        }
    }
}
