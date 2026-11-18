//! Universal deflate unpacking: any deflate stream <-> its tokens (literals, copies, block
//! layout), bit exact (see `tokens`).

pub mod inflate;
pub mod tokens;

/// zlib's Adler-32.
pub fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for c in d.chunks(5552) {
        for &x in c {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}
