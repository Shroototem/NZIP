//! Deflate <-> tokens, bit exact.
//!
//! A deflate stream is split into what it actually says: its literal bytes, its copy
//! records (literals before the copy, copy length, distance) and its block layout (verbatim
//! dynamic headers). Those compress far better than the Huffman-coded original and need no
//! expansion of the copies (image data would grow ~14x), and they rebuild the original
//! bits exactly: same tokens + same code lengths = same bitstream. Works for any deflate
//! encoder; [`split`] verifies the rebuild before returning.

use crate::inflate::{self, BitReader, Huff};

pub struct Tokens {
    /// Literal bytes (stored-block bytes included), in order.
    pub lits: Vec<u8>,
    /// Copy records: varint literals-before, length - 3 (one byte), varint distance.
    /// Distance 0 (impossible otherwise) marks length 258 written with the non-canonical
    /// code 284 + 31; the real distance follows as another varint.
    pub copies: Vec<u8>,
    /// Block layout and stream tail.
    pub recipe: Vec<u8>,
}

fn put_var(v: &mut Vec<u8>, mut x: u32) {
    while x >= 0x80 {
        v.push(x as u8 | 0x80);
        x >>= 7;
    }
    v.push(x as u8);
}
#[inline]
fn get_var(d: &[u8], p: &mut usize) -> Option<u32> {
    let mut x = 0u32;
    for s in (0..35).step_by(7) {
        let b = *d.get(*p)?;
        *p += 1;
        x |= ((b & 0x7f) as u32) << s;
        if b < 0x80 {
            return Some(x);
        }
    }
    None
}

/// Decodes the deflate stream at the start of `d` into tokens. Returns the tokens and the
/// number of input bytes the stream used, or None if it isn't valid deflate, produces more
/// than `limit` literal + copy bytes of output, or doesn't rebuild exactly.
pub fn split(d: &[u8], limit: usize) -> Option<(Tokens, usize)> {
    let mut br = BitReader::new(d);
    let (mut lits, mut copies, mut recipe) = (Vec::new(), Vec::new(), Vec::new());
    let mut out_len = 0usize; // plain bytes produced (bounds distances)
    loop {
        let last = br.bits(1)? == 1;
        let bt = br.bits(2)?;
        match bt {
            0 => {
                let pad_bits = ((8 - (br.pos() & 7)) & 7) as u8;
                let pad = br.bits(pad_bits as u32)? as u8;
                let len = br.bits(16)? as usize;
                let nlen = br.bits(16)? as usize;
                if len != !nlen & 0xffff || out_len + len > limit {
                    return None;
                }
                let byte = br.pos() >> 3;
                lits.extend_from_slice(d.get(byte..byte + len)?);
                br.skip(len * 8);
                out_len += len;
                recipe.push((last as u8) << 2);
                recipe.extend_from_slice(&[pad, pad_bits]);
                put_var(&mut recipe, len as u32);
            }
            1 | 2 => {
                let (ll, dl) = if bt == 1 {
                    recipe.push(1 | (last as u8) << 2);
                    let (a, b) = inflate::fixed_lens();
                    (a.to_vec(), b.to_vec())
                } else {
                    let h0 = br.pos();
                    let (a, b) = inflate::read_dynamic_header(&mut br)?;
                    let n = br.pos() - h0;
                    recipe.push(2 | (last as u8) << 2);
                    put_var(&mut recipe, n as u32);
                    recipe.extend_from_slice(&inflate::extract_bits(d, h0, n));
                    (a, b)
                };
                if ll.get(256).copied().unwrap_or(0) == 0 {
                    return None;
                }
                let lh = Huff::new(&ll)?;
                let dh = Huff::new(&dl)?;
                let (mut n_copies, mut run) = (0u32, 0u32);
                loop {
                    let s = lh.decode(&mut br)?;
                    if s < 256 {
                        lits.push(s as u8);
                        run += 1;
                        out_len += 1;
                    } else if s == 256 {
                        break;
                    } else {
                        let i = (s - 257) as usize;
                        if i >= 29 {
                            return None;
                        }
                        let len = inflate::LEN_BASE[i] as usize + br.bits(inflate::LEN_EXTRA[i] as u32)? as usize;
                        let ds = dh.decode(&mut br)? as usize;
                        if ds >= 30 {
                            return None;
                        }
                        let dist = inflate::DIST_BASE[ds] as usize + br.bits(inflate::DIST_EXTRA[ds] as u32)? as usize;
                        if dist > out_len || out_len + len > limit {
                            return None;
                        }
                        put_var(&mut copies, run);
                        copies.push((len - 3) as u8);
                        // Length 258 also has a non-canonical code (284 + 31, which zlib
                        // decodes); mark it with distance 0 so the rebuild uses it too.
                        if len == 258 && i == 27 {
                            put_var(&mut copies, 0);
                        }
                        put_var(&mut copies, dist as u32);
                        run = 0;
                        n_copies += 1;
                        out_len += len;
                    }
                }
                put_var(&mut recipe, n_copies);
                put_var(&mut recipe, run);
            }
            _ => return None,
        }
        if last {
            break;
        }
    }
    let tail_bits = ((8 - (br.pos() & 7)) & 7) as u8;
    let tail = br.bits(tail_bits as u32)? as u8;
    recipe.extend_from_slice(&[tail, tail_bits]);
    let used = br.pos() >> 3;
    let t = Tokens { lits, copies, recipe };
    // Verify: the tokens must rebuild the exact bytes.
    if join(&t.lits, &t.copies, &t.recipe)?[..] != d[..used] {
        return None;
    }
    Some((t, used))
}

struct BitWriter {
    out: Vec<u8>,
    buf: u64,
    n: u32,
}
impl BitWriter {
    #[inline(always)]
    fn put(&mut self, v: u32, n: u32) {
        self.buf |= (v as u64) << self.n;
        self.n += n;
        if self.n >= 32 {
            self.out.extend_from_slice(&(self.buf as u32).to_le_bytes());
            self.buf >>= 32;
            self.n -= 32;
        }
    }
    fn align_flush(&mut self) {
        while self.n >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }
}

/// Canonical codes (bit-reversed for LSB-first output) from code lengths.
fn codes(lens: &[u8]) -> Vec<(u16, u8)> {
    let mut count = [0u32; 16];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut c = 0;
    for l in 1..16 {
        c = (c + count[l - 1]) << 1;
        next[l] = c;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                return (0, 0);
            }
            let v = next[l as usize];
            next[l as usize] += 1;
            ((v.reverse_bits() >> (32 - l as u32)) as u16, l)
        })
        .collect()
}

/// Length -> (code index 0..29, extra bits value) and distance -> (code, extra) tables.
fn len_code(len: usize) -> usize {
    if len == 258 { 28 } else { inflate::LEN_BASE.iter().rposition(|&b| b as usize <= len).unwrap() }
}
fn dist_code(d: usize) -> usize {
    inflate::DIST_BASE.iter().rposition(|&b| b as usize <= d).unwrap()
}

/// Rebuilds the deflate stream from its tokens. None if the tokens are inconsistent.
pub fn join(lits: &[u8], copies: &[u8], recipe: &[u8]) -> Option<Vec<u8>> {
    let mut w = BitWriter { out: Vec::with_capacity(lits.len() + copies.len() / 2), buf: 0, n: 0 };
    let (mut lp, mut cp, mut rp) = (0usize, 0usize, 0usize);
    // Distance code lookup for small distances (the common case), slow path beyond.
    let mut dcode_small = [0u8; 513];
    for (d, c) in dcode_small.iter_mut().enumerate().skip(1) {
        *c = dist_code(d) as u8;
    }
    let mut lcode = [0u8; 259];
    for (l, c) in lcode.iter_mut().enumerate().skip(3) {
        *c = len_code(l) as u8;
    }
    loop {
        let f = *recipe.get(rp)?;
        rp += 1;
        let last = f & 4 != 0;
        w.put(last as u32, 1);
        match f & 3 {
            0 => {
                w.put(0, 2);
                let (pad, pad_bits) = (*recipe.get(rp)?, *recipe.get(rp + 1)?);
                rp += 2;
                let len = get_var(recipe, &mut rp)? as usize;
                w.put(pad as u32, pad_bits as u32);
                w.align_flush();
                if w.n != 0 {
                    return None;
                }
                w.out.extend_from_slice(&(len as u16).to_le_bytes());
                w.out.extend_from_slice(&(!(len as u16)).to_le_bytes());
                w.out.extend_from_slice(lits.get(lp..lp + len)?);
                lp += len;
            }
            k @ (1 | 2) => {
                w.put(k as u32, 2);
                let (ll, dl) = if k == 1 {
                    let (a, b) = inflate::fixed_lens();
                    (a.to_vec(), b.to_vec())
                } else {
                    let hb = get_var(recipe, &mut rp)? as usize;
                    let hdr = recipe.get(rp..rp + hb.div_ceil(8))?;
                    rp += hb.div_ceil(8);
                    let mut br = BitReader::new(hdr);
                    let lens = inflate::read_dynamic_header(&mut br)?;
                    if br.pos() != hb {
                        return None;
                    }
                    let mut r = BitReader::new(hdr);
                    let mut left = hb;
                    while left > 0 {
                        let k = left.min(16) as u32;
                        w.put(r.bits(k)?, k);
                        left -= k as usize;
                    }
                    lens
                };
                let lc = codes(&ll);
                let dc = codes(&dl);
                let n_copies = get_var(recipe, &mut rp)?;
                let tail_run = get_var(recipe, &mut rp)? as usize;
                let lit = |w: &mut BitWriter, b: u8| -> Option<()> {
                    let (c, l) = lc[b as usize];
                    if l == 0 {
                        return None;
                    }
                    w.put(c as u32, l as u32);
                    Some(())
                };
                for _ in 0..n_copies {
                    let run = get_var(copies, &mut cp)? as usize;
                    for &b in lits.get(lp..lp + run)? {
                        lit(&mut w, b)?;
                    }
                    lp += run;
                    let len = *copies.get(cp)? as usize + 3;
                    cp += 1;
                    let mut dist = get_var(copies, &mut cp)? as usize;
                    let mut c = lcode[len] as usize;
                    if dist == 0 && len == 258 {
                        c = 27; // non-canonical 284 + 31
                        dist = get_var(copies, &mut cp)? as usize;
                    }
                    if dist == 0 || dist > 32768 {
                        return None;
                    }
                    let (code, l) = *lc.get(257 + c)?;
                    if l == 0 {
                        return None;
                    }
                    w.put(code as u32, l as u32);
                    w.put((len - inflate::LEN_BASE[c] as usize) as u32, inflate::LEN_EXTRA[c] as u32);
                    let dcx = if dist <= 512 { dcode_small[dist] as usize } else { dist_code(dist) };
                    let (code, l) = *dc.get(dcx)?;
                    if l == 0 {
                        return None;
                    }
                    w.put(code as u32, l as u32);
                    w.put((dist - inflate::DIST_BASE[dcx] as usize) as u32, inflate::DIST_EXTRA[dcx] as u32);
                }
                for &b in lits.get(lp..lp + tail_run)? {
                    lit(&mut w, b)?;
                }
                lp += tail_run;
                let (c, l) = lc[256];
                if l == 0 {
                    return None;
                }
                w.put(c as u32, l as u32);
            }
            _ => return None,
        }
        if last {
            break;
        }
    }
    let (tail, tail_bits) = (*recipe.get(rp)?, *recipe.get(rp + 1)?);
    w.put(tail as u32, tail_bits as u32);
    w.align_flush();
    if w.n != 0 || lp != lits.len() || cp != copies.len() || rp + 2 != recipe.len() {
        return None;
    }
    Some(w.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zstd_free_streams() {
        // A hand-made stream: fixed block "abcabcabc" with a copy, then a stored block.
        // Build it with our own writer by splitting a known-good stream instead: use a
        // tiny deflate produced by the stored-block path.
        let mut d = vec![0b0000_0001u8]; // last, stored (after 3 header bits, pad 5)
        d.extend_from_slice(&5u16.to_le_bytes());
        d.extend_from_slice(&(!5u16).to_le_bytes());
        d.extend_from_slice(b"hello");
        let (t, used) = split(&d, 1 << 20).unwrap();
        assert_eq!(used, d.len());
        assert_eq!(t.lits, b"hello");
        assert_eq!(join(&t.lits, &t.copies, &t.recipe).unwrap(), d);
    }

    /// Length 258 written as code 284 + 31 (accepted by zlib) must survive split/join.
    #[test]
    fn noncanonical_258() {
        // Fixed block: literal 'a', copy 258 at distance 1 via the non-canonical code.
        let mut copies = Vec::new();
        put_var(&mut copies, 1);
        copies.push(255);
        put_var(&mut copies, 0);
        put_var(&mut copies, 1);
        let mut recipe = vec![1 | 4];
        put_var(&mut recipe, 1);
        put_var(&mut recipe, 0);
        let d = (0..8u8)
            .find_map(|tb| {
                let mut r = recipe.clone();
                r.extend_from_slice(&[0, tb]);
                join(b"a", &copies, &r)
            })
            .unwrap();
        let m = crate::inflate::measure(&d, 1 << 20, false).unwrap();
        assert_eq!(m.out_len, 259);
        let (t, used) = split(&d, 1 << 20).unwrap();
        assert_eq!(used, d.len());
        assert_eq!(t.copies, copies);
        assert_eq!(join(&t.lits, &t.copies, &t.recipe).unwrap(), d);
    }
}
