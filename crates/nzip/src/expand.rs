//! Universal deflate unpacking around the core compressor (`FLAG_UNPACK` archives).
//!
//! Deflate streams found anywhere in the input (after zlib headers, in gzip members, in
//! zip entries; PNG, docx, jar, pdf, ... all reduce to these) are replaced by their tokens:
//! literal bytes, copy records and block layout (`unpack::tokens`), which rebuild the
//! original bytes exactly (verified while compressing) and compress far better than the
//! Huffman-coded original, without expanding the copies.
//!
//! Layout after the file header (`FLAG_UNPACK`, n_frames = number of waves):
//! * per wave: `u64` stream count, `u64 len` + literals archive, `u64 len` + copies archive, for consecutive
//!   streams totalling at most [`WAVE`] token bytes (a constant, so any machine can decode
//!   it); waves compress in parallel and stream out in order;
//! * main archive: every original byte outside the streams, compressed as one ordinary
//!   nzip stream (folder-wide dedup and dictionary still apply);
//! * stream table (zstd): per stream its position, sizes and recipe;
//! * `u64` main length, `u64` raw table length, `u64` table length.
//!
//! Decoding writes the main bytes straight to their final places, then rebuilds each
//! stream into its gap; the whole-file checksum is verified at the end.

use rayon::prelude::*;
use std::io::{Seek, SeekFrom, Write};
use xxhash_rust::xxh3::{xxh3_64, Xxh3Default};

use crate::format::*;
use crate::{Device, Error, Options, Progress, Sink};

/// Token bytes per wave (constant: bounds decoder memory on every machine).
pub const WAVE: usize = 512 << 20;
/// Largest output one stream may decode to.
const ITEM_LIMIT: usize = 1 << 30;
/// Streams shorter than this aren't worth a table entry.
const MIN_STREAM: usize = 64;

/// Candidate deflate starts: after zlib headers (78 01/5E/9C/DA), gzip headers, zip local
/// headers with method 8.
fn candidates(d: &[u8]) -> Vec<(usize, bool)> {
    const CH: usize = 16 << 20;
    let n = d.len();
    let mut v: Vec<(usize, bool)> = (0..n.div_ceil(CH))
        .into_par_iter()
        .flat_map_iter(|c| {
            let (s, e) = (c * CH, ((c + 1) * CH).min(n.saturating_sub(2)));
            let mut out = Vec::new();
            let mut i = s;
            while i < e {
                match d[i] {
                    0x78 if matches!(d[i + 1], 0x01 | 0x5e | 0x9c | 0xda) => out.push((i + 2, true)),
                    0x1f if d[i + 1] == 0x8b && i + 10 < n && d[i + 2] == 8 => {
                        if let Some(p) = gzip_body(&d[i..]) {
                            out.push((i + p, false));
                        }
                    }
                    b'P' if i + 30 < n && d[i + 1..i + 4] == *b"K\x03\x04" && d[i + 8] == 8 && d[i + 9] == 0 => {
                        let nl = u16::from_le_bytes([d[i + 26], d[i + 27]]) as usize;
                        let el = u16::from_le_bytes([d[i + 28], d[i + 29]]) as usize;
                        out.push((i + 30 + nl + el, false));
                    }
                    _ => {}
                }
                i += 1;
            }
            out
        })
        .collect();
    v.sort_unstable();

    v.dedup_by_key(|c| c.0);
    v.retain(|c| c.0 < n);
    v
}

/// Offset of the deflate data in a gzip member, if the header parses.
fn gzip_body(d: &[u8]) -> Option<usize> {
    let flg = *d.get(3)?;
    let mut p = 10;
    if flg & 4 != 0 {
        let xl = u16::from_le_bytes([*d.get(p)?, *d.get(p + 1)?]) as usize;
        p += 2 + xl;
    }
    for bit in [8u8, 16] {
        if flg & bit != 0 {
            p += d.get(p..)?.iter().position(|&b| b == 0)? + 1;
        }
    }
    if flg & 2 != 0 {
        p += 2;
    }
    (p < d.len()).then_some(p)
}

struct Stream {
    start: usize,
    end: usize,
    /// Estimated token bytes (literals + ~4 per copy), for planning waves.
    size: usize,
}

fn put_var(v: &mut Vec<u8>, mut x: u64) {
    while x >= 0x80 {
        v.push(x as u8 | 0x80);
        x >>= 7;
    }
    v.push(x as u8);
}
fn get_var(d: &[u8], p: &mut usize) -> Result<u64, Error> {
    let mut x = 0u64;
    for s in (0..64).step_by(7) {
        let b = *d.get(*p).ok_or(Error::Corrupt("truncated stream table"))?;
        *p += 1;
        x |= ((b & 0x7f) as u64) << s;
        if b < 0x80 {
            return Ok(x);
        }
    }
    Err(Error::Corrupt("bad varint"))
}

fn trace() -> bool {
    std::env::var_os("NZIP_TRACE").is_some()
}

/// Pass 1: checks every candidate in parallel and keeps the non-overlapping streams in
/// order. A bare zlib header is weak evidence (random bytes match it often), so such a
/// stream must be followed by the zlib checksum of its output.
fn find_streams(input: &[u8], pool: &rayon::ThreadPool) -> Vec<Stream> {
    let cand = candidates(input);
    // Fakes cost only effort, never correctness: every accepted stream is rebuilt and
    // compared byte for byte before use. So the checksum (which needs the whole output) is
    // computed only for suspicious candidates: short ones or ones that barely expand.
    let check = |&(c, zlib): &(usize, bool)| -> Option<Stream> {
        let m = unpack::inflate::measure(&input[c..], ITEM_LIMIT, false)?;
        let used = m.consumed;
        if used < MIN_STREAM || m.out_len <= used {
            return None;
        }
        if zlib && (used < 4096 || m.out_len < 2 * used) {
            let m = unpack::inflate::measure(&input[c..], ITEM_LIMIT, true)?;
            if input.get(c + used..c + used + 4)? != m.adler.to_be_bytes() {
                return None;
            }
        }
        Some(Stream { start: c, end: c + used, size: m.token_bytes })
    };
    let found: Vec<Option<Stream>> = pool.install(|| cand.par_iter().map(check).collect());
    let mut out = Vec::new();
    let mut pos = 0;
    for f in found.into_iter().flatten() {
        if f.start >= pos {
            pos = f.end;
            out.push(f);
        }
    }
    out
}

/// Compresses with unpacking. Returns None (nothing written) if no stream was found.
pub fn compress<W: Write + Seek>(input: &[u8], opt: &Options, w: &mut W, progress: Progress<'_>) -> Result<Option<u64>, Error> {
    let t0 = std::time::Instant::now();
    let threads = crate::thread_count(opt.threads);
    let pool = crate::pool(threads);
    let streams = find_streams(input, &pool);
    if trace() {
        eprintln!("unpack: {} streams found in {:.2}s", streams.len(), t0.elapsed().as_secs_f64());
    }
    if streams.is_empty() {
        return Ok(None);
    }
    let inner = Options { unpack: false, dir_archive: false, ..opt.clone() };
    let start = w.stream_position()?;
    w.write_all(&[0u8; FILE_HEADER_LEN])?;
    let mut written = FILE_HEADER_LEN as u64;

    let mut waves: Vec<std::ops::Range<usize>> = Vec::new();
    let (mut first, mut size) = (0, 0usize);
    for (k, s) in streams.iter().enumerate() {
        if size + s.size > WAVE && k > first {
            waves.push(first..k);
            first = k;
            size = 0;
        }
        size += s.size;
    }
    waves.push(first..streams.len());
    let budget = crate::memory_budget(opt.mem_limit);
    // Waves stream through `par` slots (each ~WAVE x work factor); no barrier between them.
    let par = (budget / (WAVE * 8)).clamp(1, 8);
    let main_len = input.len() - streams.iter().map(|s| s.end - s.start).sum::<usize>();
    let build_main = || {
        let mut main = Vec::with_capacity(main_len);
        let mut pos = 0;
        for s in &streams {
            main.extend_from_slice(&input[pos..s.start]);
            pos = s.end;
        }
        main.extend_from_slice(&input[pos..]);
        main
    };
    type Meta = Vec<(usize, usize, Vec<u8>)>; // per stream: lits len, copies len, recipe
    // One wave: split its streams into tokens, compress literals and copies.
    let job = |r: &std::ops::Range<usize>| -> Result<(Vec<u8>, Vec<u8>, Meta), Error> {
        let toks: Vec<unpack::tokens::Tokens> = pool.install(|| streams[r.clone()].par_iter().map(|s| {
            let (t, used) = unpack::tokens::split(&input[s.start..], ITEM_LIMIT)
                .ok_or(Error::Codec("deflate stream failed to split on second pass".into()))?;
            if s.start + used != s.end {
                return Err(Error::Codec("deflate stream length changed".into()));
            }
            Ok(t)
        }).collect::<Result<_, Error>>())?;
        let (mut lits, mut copies, mut meta) = (Vec::new(), Vec::new(), Vec::with_capacity(toks.len()));
        for t in toks {
            lits.extend_from_slice(&t.lits);
            copies.extend_from_slice(&t.copies);
            meta.push((t.lits.len(), t.copies.len(), t.recipe));
        }
        let in_len: usize = streams[r.clone()].iter().map(|s| s.end - s.start).sum();
        let total = lits.len() + copies.len();
        let scale = |x: u64| progress((x as u128 * in_len as u128 / total.max(1) as u128) as u64);
        let mut a = std::io::Cursor::new(Vec::new());
        crate::compress_to_progress(&lits, &inner, &mut a, &scale)?;
        let mut b = std::io::Cursor::new(Vec::new());
        crate::compress_to_progress(&copies, &inner, &mut b, &scale)?;
        Ok((a.into_inner(), b.into_inner(), meta))
    };
    let mut table = Vec::new();
    put_var(&mut table, streams.len() as u64);
    let mut prev_end = 0;
    // The main bytes compress alongside the waves when they fit comfortably in memory.
    let main_alongside = main_len <= budget / 4;
    let mut main_arch: Option<Result<Vec<u8>, Error>> = None;
    std::thread::scope(|sc| -> Result<(), Error> {
        let main_h = main_alongside.then(|| sc.spawn(|| -> Result<Vec<u8>, Error> {
            let main = build_main();
            let mut c = std::io::Cursor::new(Vec::new());
            crate::compress_to_progress(&main, &inner, &mut c, progress)?;
            Ok(c.into_inner())
        }));
        let (tx, rx) = std::sync::mpsc::channel();
        let waves_ref = &waves;
        let job = &job;
        let workers = sc.spawn(move || {
            let slots = crate::pool(par);
            slots.install(|| waves_ref.par_iter().enumerate().with_max_len(1).for_each_with(tx, |tx, (i, r)| {
                let _ = tx.send((i, job(r)));
            }));
        });
        // Write waves in order as they finish.
        let mut held = std::collections::BTreeMap::new();
        let mut next = 0;
        for (i, res) in rx {
            held.insert(i, res);
            while let Some(res) = held.remove(&next) {
                let (a, b, meta) = res?;
                w.write_all(&(meta.len() as u64).to_le_bytes())?;
                written += 8;
                for arch in [&a, &b] {
                    w.write_all(&(arch.len() as u64).to_le_bytes())?;
                    w.write_all(arch)?;
                    written += 8 + arch.len() as u64;
                }
                for (s, (ll, cl, recipe)) in streams[waves[next].clone()].iter().zip(meta) {
                    for v in [s.start - prev_end, s.end - s.start, ll, cl, recipe.len()] {
                        put_var(&mut table, v as u64);
                    }
                    table.extend_from_slice(&recipe);
                    prev_end = s.end;
                }
                next += 1;
            }
        }
        workers.join().expect("wave workers");
        if next != waves.len() {
            return Err(Error::Codec("wave missing".into()));
        }
        if let Some(h) = main_h {
            main_arch = Some(h.join().expect("main thread"));
        }
        Ok(())
    })?;
    if trace() {
        eprintln!("unpack: {} waves done at {:.2}s", waves.len(), t0.elapsed().as_secs_f64());
    }
    let main_written = match main_arch {
        Some(a) => {
            let a = a?;
            w.write_all(&a)?;
            a.len() as u64
        }
        None => crate::compress_to_progress(&build_main(), &inner, w, progress)?,
    };
    written += main_written;
    if trace() {
        eprintln!("unpack: main {} -> {} bytes at {:.2}s", main_len, main_written, t0.elapsed().as_secs_f64());
    }
    let t = zstd::bulk::compress(&table, 19).map_err(|e| Error::Codec(format!("zstd: {e}")))?;
    w.write_all(&t)?;
    for v in [main_written, table.len() as u64, t.len() as u64] {
        w.write_all(&v.to_le_bytes())?;
    }
    written += t.len() as u64 + 24;
    let mut hdr = Vec::with_capacity(FILE_HEADER_LEN);
    let flags = FLAG_UNPACK | if opt.dir_archive { FLAG_DIR } else { 0 };
    FileHeader { flags, orig_len: input.len() as u64, orig_xxh3: xxh3_64(input), n_frames: waves.len() as u32 }.write(&mut hdr);
    w.seek(SeekFrom::Start(start))?;
    w.write_all(&hdr)?;
    w.seek(SeekFrom::Start(start + written))?;
    w.flush()?;
    Ok(Some(written))
}

/// A stream table entry.
struct Entry {
    out: u64,
    clen: usize,
    lits: usize,
    copies: usize,
    recipe: std::ops::Range<usize>,
}

/// Parsed `FLAG_UNPACK` file: wave archives (literals, copies), main archive, table.
pub struct Parts<'a> {
    pub main: &'a [u8],
    /// (stream count, literals archive, copies archive) per wave.
    pub waves: Vec<(usize, &'a [u8], &'a [u8])>,
    table: Vec<u8>,
    entries: Vec<Entry>,
}

pub fn parts<'a>(data: &'a [u8], fh: &FileHeader) -> Result<Parts<'a>, Error> {
    let bad = |m| Error::Corrupt(m);
    let u64_at = |p: usize| -> Result<u64, Error> { Ok(u64::from_le_bytes(data.get(p..p + 8).ok_or(bad("truncated unpack file"))?.try_into().unwrap())) };
    let n = data.len();
    let tlen = u64_at(n.checked_sub(8).ok_or(bad("truncated unpack file"))?)? as usize;
    let traw = u64_at(n - 16)? as usize;
    let mlen = u64_at(n - 24)? as usize;
    let tstart = (n - 24).checked_sub(tlen).ok_or(bad("bad table size"))?;
    let mstart = tstart.checked_sub(mlen).ok_or(bad("bad main size"))?;
    let table = zstd::bulk::decompress(&data[tstart..n - 24], traw.min(1 << 31)).map_err(|_| bad("bad stream table"))?;
    let main = &data[mstart..tstart];
    let mut p = FILE_HEADER_LEN;
    let mut waves = Vec::new();
    for _ in 0..fh.n_frames {
        let count = u64_at(p)? as usize;
        p += 8;
        let mut arch = [&data[..0]; 2];
        for a in &mut arch {
            let l = u64_at(p)? as usize;
            *a = data.get(p + 8..p + 8 + l).ok_or(bad("truncated wave"))?;
            p += 8 + l;
        }
        waves.push((count, arch[0], arch[1]));
    }
    if p != mstart {
        return Err(bad("unpack file layout mismatch"));
    }
    let mut q = 0;
    let count = get_var(&table, &mut q)? as usize;
    let mut entries = Vec::with_capacity(count.min(1 << 24));
    let mut out = 0u64;
    for _ in 0..count {
        out += get_var(&table, &mut q)?;
        let clen = get_var(&table, &mut q)? as usize;
        let lits = get_var(&table, &mut q)? as usize;
        let copies = get_var(&table, &mut q)? as usize;
        let rlen = get_var(&table, &mut q)? as usize;
        if q + rlen > table.len() || lits > ITEM_LIMIT || copies > ITEM_LIMIT {
            return Err(bad("bad stream entry"));
        }
        entries.push(Entry { out, clen, lits, copies, recipe: q..q + rlen });
        q += rlen;
        out += clen as u64;
    }
    if out > fh.orig_len {
        return Err(bad("streams exceed file size"));
    }
    Ok(Parts { main, waves, table, entries })
}

/// Writes main-archive offsets to their places in the output (around the streams).
struct Scatter<'a> {
    sink: &'a mut dyn Sink,
    /// (main offset, output offset) of each run of main bytes, sorted.
    runs: Vec<(u64, u64)>,
}
impl Scatter<'_> {
    fn each(&self, off: u64, len: usize, mut f: impl FnMut(u64, std::ops::Range<usize>) -> Result<(), Error>) -> Result<(), Error> {
        let mut k = self.runs.partition_point(|r| r.0 <= off).saturating_sub(1);
        let mut done = 0usize;
        while done < len {
            let (m, o) = self.runs[k];
            let run_end = self.runs.get(k + 1).map_or(u64::MAX, |r| r.0);
            let at = off + done as u64;
            let n = ((run_end - at) as usize).min(len - done);
            f(o + (at - m), done..done + n)?;
            done += n;
            k += 1;
        }
        Ok(())
    }
}
impl Sink for Scatter<'_> {
    fn write_at(&mut self, off: u64, data: &[u8]) -> Result<(), Error> {
        let mut jobs = Vec::new();
        self.each(off, data.len(), |o, r| {
            jobs.push((o, r));
            Ok(())
        })?;
        for (o, r) in jobs {
            self.sink.write_at(o, &data[r])?;
        }
        Ok(())
    }
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let mut jobs = Vec::new();
        self.each(off, buf.len(), |o, r| {
            jobs.push((o, r));
            Ok(())
        })?;
        for (o, r) in jobs {
            self.sink.read_at(o, &mut buf[r])?;
        }
        Ok(())
    }
}

pub fn decompress(data: &[u8], fh: &FileHeader, sink: &mut dyn Sink, threads: usize, device: Device, mem_limit: usize, progress: Progress<'_>) -> Result<u64, Error> {
    let t0 = std::time::Instant::now();
    let threads = crate::thread_count(threads);
    let pool = crate::pool(threads);
    let parts = parts(data, fh)?;
    // Main bytes go straight to their final positions.
    let mut runs = Vec::with_capacity(parts.entries.len() + 1);
    let (mut m, mut o) = (0u64, 0u64);
    for e in &parts.entries {
        runs.push((m, o));
        m += e.out - o;
        o = e.out + e.clen as u64;
    }
    runs.push((m, o));
    let main_len = m + (fh.orig_len - o);
    {
        let mut sc = Scatter { sink, runs };
        let got = crate::decompress_to_progress(parts.main, &mut sc, threads, device, mem_limit, progress)?;
        if got != main_len {
            return Err(Error::Corrupt("main length mismatch"));
        }
    }
    if trace() {
        eprintln!("unpack: main decoded at {:.2}s", t0.elapsed().as_secs_f64());
    }
    // Streams, wave by wave (several waves at once if memory allows).
    let mut jobs = Vec::with_capacity(parts.waves.len());
    let mut k = 0usize;
    for &(count, la, ca) in &parts.waves {
        let end = k.checked_add(count).filter(|&e| e <= parts.entries.len()).ok_or(Error::Corrupt("streams don't fit the waves"))?;
        jobs.push(((la, ca), k..end));
        k = end;
    }
    if k != parts.entries.len() {
        return Err(Error::Corrupt("streams don't fit the waves"));
    }
    let par = (crate::memory_budget(mem_limit) / (WAVE * 3)).clamp(1, 8);
    let shared = std::sync::Mutex::new(&mut *sink);
    let slots = crate::pool(par);
    slots.install(|| jobs.par_iter().with_max_len(1).try_for_each(|((la, ca), r)| -> Result<(), Error> {
        let lits = crate::decompress_with(la, threads, device)?;
        let copies = crate::decompress_with(ca, threads, device)?;
        let ents = &parts.entries[r.clone()];
        if lits.len() != ents.iter().map(|e| e.lits).sum::<usize>() || copies.len() != ents.iter().map(|e| e.copies).sum::<usize>() {
            return Err(Error::Corrupt("wave size mismatch"));
        }
        let mut sl = Vec::with_capacity(ents.len());
        let (mut lp, mut cp) = (0, 0);
        for e in ents {
            sl.push((&lits[lp..lp + e.lits], &copies[cp..cp + e.copies]));
            lp += e.lits;
            cp += e.copies;
        }
        let built: Vec<(u64, Vec<u8>)> = pool.install(|| ents.par_iter().zip(sl).map(|(e, (l, c))| {
            let s = unpack::tokens::join(l, c, &parts.table[e.recipe.clone()]).ok_or(Error::Corrupt("deflate rebuild failed"))?;
            if s.len() != e.clen {
                return Err(Error::Corrupt("deflate rebuild length mismatch"));
            }
            Ok((e.out, s))
        }).collect::<Result<_, Error>>())?;
        let mut sk = shared.lock().unwrap_or_else(|e| e.into_inner());
        for (at, s) in built {
            sk.write_at(at, &s)?;
            progress(s.len() as u64);
        }
        Ok(())
    }))?;
    if trace() {
        eprintln!("unpack: streams rebuilt at {:.2}s", t0.elapsed().as_secs_f64());
    }
    // Whole-file checksum over the finished output.
    let mut h = Xxh3Default::new();
    let mut buf = vec![0u8; 16 << 20];
    let mut off = 0u64;
    while off < fh.orig_len {
        let n = ((fh.orig_len - off) as usize).min(buf.len());
        sink.read_at(off, &mut buf[..n])?;
        h.update(&buf[..n]);
        off += n as u64;
    }
    if h.digest() != fh.orig_xxh3 {
        return Err(Error::Corrupt("whole-file checksum mismatch"));
    }
    Ok(fh.orig_len)
}
