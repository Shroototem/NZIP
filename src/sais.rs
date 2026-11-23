//! Suffix array construction (SA-IS, Nong–Zhang–Chan; linear time) and the
//! Burrows–Wheeler transform built on it. Pure integer code: identical results everywhere.

const EMPTY: u32 = u32::MAX;

trait Sym: Copy {
    fn idx(self) -> usize;
}
impl Sym for u8 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}
impl Sym for u32 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

#[inline(always)]
fn is_s(t: &[u64], i: usize) -> bool {
    (t[i >> 6] >> (i & 63)) & 1 == 1
}

#[inline(always)]
fn is_lms(t: &[u64], i: usize) -> bool {
    i > 0 && is_s(t, i) && !is_s(t, i - 1)
}

fn heads(cnt: &[u32], out: &mut [u32]) {
    let mut sum = 0u32;
    for (o, &c) in out.iter_mut().zip(cnt) {
        *o = sum;
        sum += c;
    }
}

fn tails(cnt: &[u32], out: &mut [u32]) {
    let mut sum = 0u32;
    for (o, &c) in out.iter_mut().zip(cnt) {
        sum += c;
        *o = sum;
    }
}

fn induce<T: Sym>(s: &[T], sa: &mut [u32], t: &[u64], cnt: &[u32], bkt: &mut [u32]) {
    let n = s.len();
    // L-type suffixes, left to right. The virtual sentinel's predecessor (n-1) goes first.
    heads(cnt, bkt);
    let c = s[n - 1].idx();
    sa[bkt[c] as usize] = (n - 1) as u32;
    bkt[c] += 1;
    for i in 0..n {
        let j = sa[i];
        if j != EMPTY && j > 0 {
            let j = j as usize - 1;
            if !is_s(t, j) {
                let c = s[j].idx();
                sa[bkt[c] as usize] = j as u32;
                bkt[c] += 1;
            }
        }
    }
    // S-type suffixes, right to left.
    tails(cnt, bkt);
    for i in (0..n).rev() {
        let j = sa[i];
        if j != EMPTY && j > 0 {
            let j = j as usize - 1;
            if is_s(t, j) {
                let c = s[j].idx();
                bkt[c] -= 1;
                sa[bkt[c] as usize] = j as u32;
            }
        }
    }
}

fn lms_equal<T: Sym>(s: &[T], t: &[u64], a: usize, b: usize) -> bool {
    let n = s.len();
    let mut i = 0;
    loop {
        let (x, y) = (a + i, b + i);
        if x >= n || y >= n {
            return false; // reached the (unique) sentinel
        }
        if s[x].idx() != s[y].idx() || is_s(t, x) != is_s(t, y) {
            return false;
        }
        if i > 0 && (is_lms(t, x) || is_lms(t, y)) {
            return is_lms(t, x) && is_lms(t, y);
        }
        i += 1;
    }
}

fn sais<T: Sym>(s: &[T], sa: &mut [u32], k: usize) {
    let n = s.len();
    debug_assert_eq!(sa.len(), n);
    if n == 1 {
        sa[0] = 0;
        return;
    }
    // types (bit set = S-type); s[n-1] is L-type because the sentinel is smaller
    let mut t = vec![0u64; n / 64 + 1];
    for i in (0..n - 1).rev() {
        let (a, b) = (s[i].idx(), s[i + 1].idx());
        if a < b || (a == b && is_s(&t, i + 1)) {
            t[i >> 6] |= 1 << (i & 63);
        }
    }
    let mut cnt = vec![0u32; k];
    for &c in s {
        cnt[c.idx()] += 1;
    }
    let mut bkt = vec![0u32; k];

    // Stage 1: sort LMS substrings by induction.
    sa.fill(EMPTY);
    tails(&cnt, &mut bkt);
    for i in 1..n {
        if is_lms(&t, i) {
            let c = s[i].idx();
            bkt[c] -= 1;
            sa[bkt[c] as usize] = i as u32;
        }
    }
    induce(s, sa, &t, &cnt, &mut bkt);

    // Compact the sorted LMS positions to the front.
    let mut m = 0;
    for i in 0..n {
        let j = sa[i];
        if j != EMPTY && is_lms(&t, j as usize) {
            sa[m] = j;
            m += 1;
        }
    }
    // Name LMS substrings.
    sa[m..].fill(EMPTY);
    let mut names = 0u32;
    let mut prev = usize::MAX;
    for i in 0..m {
        let p = sa[i] as usize;
        if prev == usize::MAX || !lms_equal(s, &t, prev, p) {
            names += 1;
            prev = p;
        }
        sa[m + p / 2] = names - 1;
    }
    let mut j = n;
    for i in (m..n).rev() {
        if sa[i] != EMPTY {
            j -= 1;
            sa[j] = sa[i];
        }
    }

    // Stage 2: sort LMS suffixes (recursing if names are not unique).
    {
        let (head, s1) = sa.split_at_mut(n - m);
        let sa1 = &mut head[..m];
        if (names as usize) < m {
            sais(&*s1, sa1, names as usize);
        } else {
            for (i, &c) in s1.iter().enumerate() {
                sa1[c as usize] = i as u32;
            }
        }
    }
    // Map ranks back to text positions.
    let mut j = n - m;
    for i in 1..n {
        if is_lms(&t, i) {
            sa[j] = i as u32;
            j += 1;
        }
    }
    for i in 0..m {
        sa[i] = sa[n - m + sa[i] as usize];
    }
    sa[m..].fill(EMPTY);

    // Stage 3: place sorted LMS suffixes at bucket tails, induce the rest.
    tails(&cnt, &mut bkt);
    for i in (0..m).rev() {
        let p = sa[i];
        sa[i] = EMPTY;
        let c = s[p as usize].idx();
        bkt[c] -= 1;
        sa[bkt[c] as usize] = p;
    }
    induce(s, sa, &t, &cnt, &mut bkt);
}

pub fn suffix_array(s: &[u8]) -> Vec<u32> {
    assert!(s.len() < u32::MAX as usize - 1, "block too large for 32-bit suffix array");
    let mut sa = vec![0u32; s.len()];
    if !s.is_empty() {
        sais(s, &mut sa, 256);
    }
    sa
}

/// Number of independent decoding chains for a block of `n` bytes. The inverse transform
/// follows them in parallel (across threads, and 8 at a time within a thread to hide memory
/// latency). Stored in the archive, so decoding never depends on the encoder's thread count.
pub fn chains_for(n: usize) -> usize {
    // one chain per 16 KiB: inverse BWT walks 8 chains at a time per CPU thread (to overlap
    // cache misses) and one per GPU thread, and both need many to keep every core busy
    (n >> 14).clamp(8, 16384)
}

/// BWT with a virtual end-of-block sentinel. Returns the transform and the matrix rows of
/// text positions 0, seg, 2*seg, ... (idx[0] is the classic primary index).
pub fn bwt_from_sa(s: &[u8], sa: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let n = s.len();
    let k = chains_for(n);
    let mut idx = vec![0u32; k];
    if n == 0 {
        return (Vec::new(), idx);
    }
    let seg = n.div_ceil(k);
    let mut out = vec![0u8; n];
    for (r, &p) in sa.iter().enumerate() {
        let p = p as usize;
        out[r] = if p == 0 { s[n - 1] } else { s[p - 1] };
        if p % seg == 0 {
            idx[p / seg] = (r + 1) as u32;
        }
    }
    (out, idx)
}

pub fn bwt(s: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let sa = suffix_array(s);
    bwt_from_sa(s, &sa)
}

struct SendPtr(*mut u32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Inverse BWT using up to `threads` threads.
pub fn unbwt(b: &[u8], idx: &[u32], threads: usize) -> Result<Vec<u8>, String> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let k = chains_for(n);
    if idx.len() != k {
        return Err("bad BWT index count".into());
    }
    let p1 = idx[0] as usize; // row of suffix 0 = position of '$' in L'
    let seg = n.div_ceil(k);
    let used = n.div_ceil(seg); // chains with a non-empty segment
    if p1 == 0 || p1 > n || idx[..used].iter().any(|&r| r == 0 || r as usize > n) {
        return Err("bad BWT index".into());
    }
    // L' (n+1 rows): L'[0] = b[p1-1], L'[p1] = '$', L'[r] = b[r-1] otherwise.
    if let Some(out) = crate::gpu::unbwt(b, &idx[..used], seg) {
        return Ok(out);
    }
    let lch = |r: usize| -> u8 { if r == 0 { b[p1 - 1] } else { b[r - 1] } };
    let threads = threads.clamp(1, 64).min((n >> 16).max(1));
    let rows = n + 1;
    let part = rows.div_ceil(threads);
    // per-part character counts
    let ranges: Vec<(usize, usize)> = (0..threads).map(|t| (t * part, ((t + 1) * part).min(rows))).collect();
    let counts: Vec<[u32; 256]> = crate::par::map(&ranges, threads, |&(a, z)| {
        let mut c = [0u32; 256];
        for r in a..z {
            if r != p1 {
                c[lch(r) as usize] += 1;
            }
        }
        c
    });
    let mut c1 = [0u32; 257];
    let mut sum = 1u32;
    for c in 0..256 {
        c1[c] = sum;
        sum += counts.iter().map(|x| x[c]).sum::<u32>();
    }
    c1[256] = sum;
    // starting F-row of each part for each character
    let mut starts = vec![[0u32; 256]; threads];
    for c in 0..256 {
        let mut o = c1[c];
        for t in 0..threads {
            starts[t][c] = o;
            o += counts[t][c];
        }
    }
    // psi[F-row] = L-row of the same character occurrence (parallel, disjoint writes)
    let mut psi = vec![0u32; rows];
    psi[0] = p1 as u32;
    let ptr = SendPtr(psi.as_mut_ptr());
    let work: Vec<usize> = (0..threads).collect();
    crate::par::map(&work, threads, |&t| {
        let ptr = &ptr;
        let mut next = starts[t];
        let (a, z) = ranges[t];
        for r in a..z {
            if r == p1 {
                continue;
            }
            let c = lch(r) as usize;
            // SAFETY: every F-row index is produced exactly once across all parts.
            unsafe { *ptr.0.add(next[c] as usize) = r as u32 };
            next[c] += 1;
        }
    });
    drop(ptr);
    // F[r] = largest c with c1[c] <= r
    let fch = |r: u32| -> u8 {
        let mut lo = 0usize;
        let mut step = 128;
        while step > 0 {
            if c1[lo + step] <= r {
                lo += step;
            }
            step >>= 1;
        }
        lo as u8
    };
    let mut out = vec![0u8; n];
    const G: usize = 8;
    let groups: Vec<&mut [u8]> = out.chunks_mut(seg * G).collect();
    let psi = &psi;
    crate::par::for_chunks(groups, threads, |g, chunk| {
        let mut segs: Vec<&mut [u8]> = chunk.chunks_mut(seg).collect();
        let mut rows = [0u32; G];
        for (j, r) in rows.iter_mut().enumerate().take(segs.len()) {
            *r = idx[g * G + j];
        }
        let len0 = segs[0].len();
        for i in 0..len0 {
            for j in 0..segs.len() {
                if i < segs[j].len() {
                    let r = rows[j];
                    segs[j][i] = fch(r);
                    rows[j] = psi[r as usize];
                }
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sa(s: &[u8]) -> Vec<u32> {
        let mut v: Vec<u32> = (0..s.len() as u32).collect();
        v.sort_by(|&a, &b| s[a as usize..].cmp(&s[b as usize..]));
        v
    }

    #[test]
    fn sa_matches_naive() {
        let mut rng = 7u64;
        for len in [1usize, 2, 3, 5, 10, 17, 64, 100, 1000, 5000] {
            for alpha in [1u64, 2, 4, 26, 256] {
                let s: Vec<u8> = (0..len)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((rng >> 33) % alpha) as u8 + b'a' * (alpha < 26) as u8
                    })
                    .collect();
                assert_eq!(suffix_array(&s), naive_sa(&s), "len {len} alpha {alpha}");
                let (b, idx) = bwt(&s);
                assert_eq!(unbwt(&b, &idx, 3).unwrap(), s, "bwt len {len} alpha {alpha}");
            }
        }
        let big: Vec<u8> = (0..3_000_000u32).map(|i| (i.wrapping_mul(i) >> 7) as u8 % 5 + b'a').collect();
        let (b, idx) = bwt(&big);
        assert_eq!(unbwt(&b, &idx, 8).unwrap(), big);
        for s in [&b"banana"[..], b"mississippi", b"aaaaaaaaaa", b"abababababab"] {
            assert_eq!(suffix_array(s), naive_sa(s));
            let (b, idx) = bwt(s);
            assert_eq!(unbwt(&b, &idx, 2).unwrap(), s);
        }
    }
}
