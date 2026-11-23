//! Online neural mixer: a gated linear network in 16-bit fixed point.
//!
//! Inputs are logits (scaled by 256); one weight row is selected per prediction by a small
//! context ("gating"), and the row learns online from every coded bit. Integer-only, so it
//! produces the same weights and outputs on every CPU (with or without AVX2).

use crate::tables::squash;

/// Model constants. They are part of the archive format: encoder and decoder must agree.
/// With `--features tune`, `NZ_K="i=v,..."` overrides them for experiments only.
pub fn knob(i: usize) -> i32 {
    const DEFAULTS: [i32; 4] = [
        2,   // 0: mixer learning rate
        7,   // 1: slow counter rate (shift)
        3,   // 2: fast counter rate (shift)
        0,
    ];
    #[cfg(feature = "tune")]
    {
        static K: std::sync::OnceLock<[i32; 4]> = std::sync::OnceLock::new();
        K.get_or_init(|| {
            let mut k = DEFAULTS;
            if let Ok(v) = std::env::var("NZ_K") {
                for kv in v.split(',') {
                    if let Some((a, b)) = kv.split_once('=') {
                        if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<i32>()) {
                            k[a] = b;
                        }
                    }
                }
            }
            k
        })[i]
    }
    #[cfg(not(feature = "tune"))]
    DEFAULTS[i]
}

const WSHIFT: i32 = 13; // weight 1.0 = 8192

/// Input lanes: one AVX2 register of i16. Unused lanes stay zero.
pub const N: usize = 16;

pub struct Mixer {
    x: [i16; N],
    nx: usize,
    w: Vec<[i16; N]>,
    sel: usize,
    pr: i32,
    pub lr: i32,
}

impl Mixer {
    /// Up to `N` inputs, `rows` selectable weight rows.
    pub fn new(n: usize, rows: usize) -> Mixer {
        assert!(n <= N);
        Mixer { x: [0; N], nx: 0, w: vec![[2048; N]; rows], sel: 0, pr: 2048, lr: knob(0) }
    }

    #[inline(always)]
    pub fn add(&mut self, v: i32) {
        self.x[self.nx & (N - 1)] = v as i16;
        self.nx += 1;
    }

    #[inline(always)]
    pub fn select(&mut self, row: usize) {
        self.sel = row;
    }

    /// P(1) as a 12-bit probability.
    #[inline(always)]
    pub fn predict(&mut self) -> i32 {
        let w = &self.w[self.sel];
        let mut s: i32 = 0;
        for i in 0..N {
            s = s.wrapping_add(self.x[i] as i32 * w[i] as i32);
        }
        self.pr = squash((s >> WSHIFT).clamp(-2047, 2047));
        self.pr
    }

    #[inline(always)]
    pub fn update(&mut self, y: u32) {
        let e = ((((y as i32) << 12) - self.pr) * self.lr).clamp(-32767, 32767);
        let w = &mut self.w[self.sel];
        for i in 0..N {
            let d = ((self.x[i] as i32 * e) + 0x8000) >> 16;
            w[i] = (w[i] as i32 + d).clamp(-32768, 32767) as i16;
        }
        self.nx = 0;
    }
}
