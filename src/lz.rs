//! Fast mode: byte-aligned LZ77 (greedy, hashed 4-byte matches, 64 KiB window).
//! Built for speed on every core; decoding is bounds-checked, so corrupt input can only
//! produce an error, never undefined behaviour.
//!
//! sequence: token(lit:4 | match-4:4) [lit ext] literals [offset:u16le [match ext]]
//! The last sequence carries literals only.

const MIN_MATCH: usize = 4;
const HASH_BITS: u32 = 16;
const WINDOW: usize = 65535;

#[inline(always)]
fn read32(s: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]])
}

#[inline(always)]
fn hash(v: u32) -> usize {
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
}

fn put_len(out: &mut Vec<u8>, mut n: usize) {
    while n >= 255 {
        out.push(255);
        n -= 255;
    }
    out.push(n as u8);
}

fn emit(out: &mut Vec<u8>, lits: &[u8], m: Option<(usize, usize)>) {
    let ll = lits.len();
    let ml = m.map_or(0, |(_, l)| l - MIN_MATCH);
    out.push(((ll.min(15) as u8) << 4) | ml.min(15) as u8);
    if ll >= 15 {
        put_len(out, ll - 15);
    }
    out.extend_from_slice(lits);
    if let Some((off, _)) = m {
        out.extend_from_slice(&(off as u16).to_le_bytes());
        if ml >= 15 {
            put_len(out, ml - 15);
        }
    }
}

pub fn compress(s: &[u8]) -> Vec<u8> {
    let n = s.len();
    let mut out = Vec::with_capacity(n / 2 + 16);
    let mut table = vec![0u32; 1 << HASH_BITS];
    let mut anchor = 0;
    let mut i = 0;
    if n >= 16 {
        let limit = n - 8;
        while i < limit {
            let v = read32(s, i);
            let h = hash(v);
            let cand = table[h] as usize;
            table[h] = i as u32;
            if cand < i && i - cand <= WINDOW && read32(s, cand) == v {
                // extend forward
                let mut len = MIN_MATCH;
                while i + len < n && s[cand + len] == s[i + len] {
                    len += 1;
                }
                // extend backward into pending literals
                let (mut st, mut cs) = (i, cand);
                while st > anchor && cs > 0 && s[st - 1] == s[cs - 1] {
                    st -= 1;
                    cs -= 1;
                    len += 1;
                }
                emit(&mut out, &s[anchor..st], Some((st - cs, len)));
                i = st + len;
                anchor = i;
                if i >= 2 && i + 4 <= n {
                    table[hash(read32(s, i - 2))] = (i - 2) as u32;
                }
            } else {
                // skip faster through data without matches
                i += 1 + ((i - anchor) >> 6);
            }
        }
    }
    emit(&mut out, &s[anchor..], None);
    out
}

pub fn decompress(src: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt LZ block".to_string();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0;
    let len_ext = |i: &mut usize, mut v: usize| -> Result<usize, String> {
        loop {
            let b = *src.get(*i).ok_or_else(e)? as usize;
            *i += 1;
            v += b;
            if b != 255 {
                return Ok(v);
            }
        }
    };
    loop {
        let t = *src.get(i).ok_or_else(e)? as usize;
        i += 1;
        let mut ll = t >> 4;
        if ll == 15 {
            ll = len_ext(&mut i, 15)?;
        }
        let lits = src.get(i..i + ll).ok_or_else(e)?;
        out.extend_from_slice(lits);
        i += ll;
        if i == src.len() {
            break;
        }
        let off = u16::from_le_bytes([*src.get(i).ok_or_else(e)?, *src.get(i + 1).ok_or_else(e)?]) as usize;
        i += 2;
        let mut ml = t & 15;
        if ml == 15 {
            ml = len_ext(&mut i, 15)?;
        }
        ml += MIN_MATCH;
        if off == 0 || off > out.len() || out.len() + ml > n {
            return Err(e());
        }
        let start = out.len() - off;
        if off >= ml {
            out.extend_from_within(start..start + ml);
        } else {
            for k in 0..ml {
                let b = out[start + k];
                out.push(b);
            }
        }
    }
    if out.len() != n {
        return Err(e());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut v: Vec<u8> = Vec::new();
        for i in 0..100_000u32 {
            v.extend_from_slice(format!("line {} of text {}\n", i % 977, i % 13).as_bytes());
        }
        for d in [&v[..], b"", b"a", b"abcdabcdabcdabcdabcdabcd", &[0u8; 100_000][..]] {
            let c = super::compress(d);
            assert_eq!(super::decompress(&c, d.len()).unwrap(), d);
        }
        assert!(super::compress(&v).len() < v.len() / 2);
    }
}
