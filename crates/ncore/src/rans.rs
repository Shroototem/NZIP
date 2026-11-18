//! 32-bit rANS with 16-bit renormalisation and a 2^16 probability scale.
//!
//! State `x` lives in [L, 2^32) with L = 2^16. Words are emitted in encode order; the
//! decoder consumes them in reverse, so streams are stored reversed (see [`finish`]).

use crate::SCALE_BITS;

pub const L: u32 = 1 << 16;

/// Encode one symbol (backwards). Pushes 0 or 1 words.
#[inline]
pub fn put(x: &mut u32, cum: u32, freq: u32, out: &mut Vec<u16>) {
    let x_max = freq << 16; // ((L >> SCALE_BITS) << 16) * freq with L = 2^16, SCALE = 16
    if *x >= x_max {
        out.push(*x as u16);
        *x >>= 16;
    }
    *x = ((*x / freq) << SCALE_BITS) + (*x % freq) + cum;
}

/// Flush the final state (2 words).
#[inline]
pub fn flush(x: u32, out: &mut Vec<u16>) {
    out.push(x as u16);
    out.push((x >> 16) as u16);
}

/// Decoder: the slot (cumulative-frequency position) for the next symbol.
#[inline]
pub fn slot(x: u32) -> u32 {
    x & ((1 << SCALE_BITS) - 1)
}

/// Decoder: advance past a symbol with (cum, freq), refilling from `words[*pos..]`.
#[inline]
pub fn advance(x: &mut u32, cum: u32, freq: u32, words: &[u16], pos: &mut usize) -> bool {
    *x = freq * (*x >> SCALE_BITS) + slot(*x) - cum;
    if *x < L {
        let Some(&w) = words.get(*pos) else { return false };
        *x = (*x << 16) | w as u32;
        *pos += 1;
    }
    true
}

/// Decoder: initial state from the stream head.
#[inline]
pub fn init(words: &[u16], pos: &mut usize) -> Option<u32> {
    let hi = *words.get(*pos)? as u32;
    let lo = *words.get(*pos + 1)? as u32;
    *pos += 2;
    Some((hi << 16) | lo)
}
