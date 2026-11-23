//! Reversible branch filter for x86 / x86-64 machine code (executables, DLLs, .so, .o).
//!
//! CALL/JMP rel32 (E8/E9) and Jcc rel32 (0F 80..8F) targets are rewritten as block-relative
//! absolute addresses, stored big-endian, so repeated calls to the same function become
//! repeated byte strings whose stable high bytes come first. Only displacements whose top
//! byte is 0x00 or 0xFF are touched (24-bit arithmetic; the top byte is never changed).
//! The decision at each position reads only opcode bytes and the top byte, which that
//! position's own rewrite never changes; the encoder runs backwards and the decoder
//! forwards, so the decoder undoes the steps in exact reverse order.

/// Offset of the rel32 operand if an E8/E9/Jcc with a short displacement starts at `i`.
#[inline(always)]
fn hit_at(b: &[u8], i: usize) -> Option<usize> {
    let o = if (b[i] & 0xFE) == 0xE8 {
        i + 1
    } else if b[i] == 0x0F && (b[i + 1] & 0xF0) == 0x80 {
        i + 2
    } else {
        return None;
    };
    if o + 4 <= b.len() && (b[o + 3].wrapping_add(1) & 0xFE) == 0 { Some(o) } else { None }
}

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
    if b.len() < 6 {
        return;
    }
    for i in (0..=b.len() - 6).rev() {
        if let Some(o) = hit_at(b, i) {
            let a = (b[o] as u32 | (b[o + 1] as u32) << 8 | (b[o + 2] as u32) << 16).wrapping_add(o as u32);
            b[o] = (a >> 16) as u8;
            b[o + 1] = (a >> 8) as u8;
            b[o + 2] = a as u8;
        }
    }
}

pub fn decode(b: &mut [u8]) {
    if b.len() < 6 {
        return;
    }
    for i in 0..=b.len() - 6 {
        if let Some(o) = hit_at(b, i) {
            let a = ((b[o] as u32) << 16 | (b[o + 1] as u32) << 8 | b[o + 2] as u32).wrapping_sub(o as u32);
            b[o] = a as u8;
            b[o + 1] = (a >> 8) as u8;
            b[o + 2] = (a >> 16) as u8;
        }
    }
}

// ---------------------------------------------------------------- split filter (BCJ2-like)

/// A byte that ends a CALL/JMP (E8/E9) or Jcc (0F 8x) opcode, given the byte before it.
#[inline(always)]
fn opcode_end(prev: u8, b: u8) -> bool {
    (b & 0xFE) == 0xE8 || (prev == 0x0F && (b & 0xF0) == 0x80)
}

/// Split x86 code into three streams:
/// * `main`: the code with every converted rel32 operand removed, so instruction
///   sequences that differ only in their branch targets become long matches;
/// * `flags`: one bit per opcode candidate in `main` (1 = its operand moved to `addr`);
/// * `addr`: converted targets as 32-bit big-endian absolute addresses.
pub struct Split {
    pub main: Vec<u8>,
    pub flags: Vec<u8>,
    /// CALL targets
    pub addr: Vec<u8>,
    /// JMP / Jcc targets
    pub jump: Vec<u8>,
}

pub fn split(b: &[u8]) -> Split {
    let n = b.len();
    let mut main = Vec::with_capacity(n);
    let mut addr = Vec::with_capacity(n / 32);
    let mut jump = Vec::with_capacity(n / 64);
    let mut flags = Vec::with_capacity(n / 64);
    let (mut acc, mut nb) = (0u8, 0u32);
    let mut push_flag = |f: bool, flags: &mut Vec<u8>| {
        acc |= (f as u8) << nb;
        nb += 1;
        if nb == 8 {
            flags.push(acc);
            acc = 0;
            nb = 0;
        }
    };
    let mut prev = 0u8;
    let mut j = 0;
    while j < n {
        let c = b[j];
        main.push(c);
        if opcode_end(prev, c) {
            let conv = j + 4 < n && (b[j + 4].wrapping_add(1) & 0xFE) == 0;
            push_flag(conv, &mut flags);
            if conv {
                let rel = u32::from_le_bytes([b[j + 1], b[j + 2], b[j + 3], b[j + 4]]);
                let abs = rel.wrapping_add((j + 5) as u32);
                if c == 0xE8 { &mut addr } else { &mut jump }.extend_from_slice(&abs.to_be_bytes());
                prev = b[j + 4];
                j += 5;
                continue;
            }
        }
        prev = c;
        j += 1;
    }
    if nb > 0 {
        flags.push(acc);
    }
    Split { main, flags, addr, jump }
}

/// Inverse of `split`; `n` = original length.
pub fn join(main: &[u8], flags: &[u8], addr: &[u8], jump: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let e = || "corrupt x86 streams".to_string();
    let mut out = Vec::with_capacity(n);
    let (mut fi, mut ai, mut ji) = (0usize, 0usize, 0usize);
    let mut prev = 0u8;
    let mut mi = 0;
    while out.len() < n {
        let c = *main.get(mi).ok_or_else(e)?;
        mi += 1;
        out.push(c);
        if opcode_end(prev, c) {
            let f = (flags.get(fi >> 3).ok_or_else(e)? >> (fi & 7)) & 1;
            fi += 1;
            if f == 1 {
                let a = if c == 0xE8 {
                    ai += 4;
                    addr.get(ai - 4..ai).ok_or_else(e)?
                } else {
                    ji += 4;
                    jump.get(ji - 4..ji).ok_or_else(e)?
                };
                let abs = u32::from_be_bytes([a[0], a[1], a[2], a[3]]);
                let rel = abs.wrapping_sub(out.len() as u32 + 4);
                out.extend_from_slice(&rel.to_le_bytes());
                prev = out[out.len() - 1];
                continue;
            }
        }
        prev = c;
    }
    if out.len() != n || mi != main.len() {
        return Err(e());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let mut r = 5u32;
        let mut v: Vec<u8> = (0..300_000)
            .map(|i| {
                r = r.wrapping_mul(1103515245).wrapping_add(12345);
                if i % 7 == 0 { 0xE8 } else if i % 17 == 0 { 0x0F } else if i % 19 == 0 { 0x85 } else if i % 11 == 0 { 0xFF } else if i % 13 == 0 { 0 } else { (r >> 16) as u8 }
            })
            .collect();
        v.extend_from_slice(&[0xE8, 0, 0, 0, 0, 0xE9, 0xFF]);
        let orig = v.clone();
        super::encode(&mut v);
        assert_ne!(v, orig);
        super::decode(&mut v);
        assert_eq!(v, orig);
        let sp = super::split(&orig);
        assert!(sp.main.len() < orig.len());
        assert_eq!(super::join(&sp.main, &sp.flags, &sp.addr, &sp.jump, orig.len()).unwrap(), orig);
        let tail = [0xE8u8, 1, 2];
        let sp = super::split(&tail);
        assert_eq!(super::join(&sp.main, &sp.flags, &sp.addr, &sp.jump, 3).unwrap(), tail);
    }
}
