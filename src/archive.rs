//! Folder support: a folder becomes one stream (manifest + file contents), so similar files
//! share one model and repeats across files are found by the match model.
//!
//! ```text
//! manifest_len:u64le  manifest  contents...
//! manifest = count:varint { kind:u8 path_len:varint path(utf-8, '/') extra }
//!   kind 0 = file, extra = size:varint (its bytes follow in the stream, in entry order)
//!   kind 1 = folder, no extra
//!   kind 2 = duplicate file, extra = index:varint of an earlier kind-0/3 entry with the
//!            same bytes (no stream bytes; unpacking copies the earlier file)
//!   kind 3 = file in pieces (see dedup.rs), extra = size:varint count:varint pieces:
//!            literal = (len << 1):varint, its bytes follow in the stream;
//!            copy    = (len << 1 | 1):varint back:varint off:varint, bytes [off, off + len)
//!                      of entry (this index - back), an earlier kind-0/3 entry or this one
//! ```

use crate::container::{read_varint, write_varint};
use crate::dedup::Seg;
use std::fs::{self, File};
use std::io::{self, Read, Seek, Write};
use std::path::{Component, Path, PathBuf};

pub struct Entry {
    pub rel: String,
    pub dir: bool,
    pub size: u64,
    pub path: PathBuf,
    /// Index of an earlier entry with identical bytes (set by `find_dups`).
    pub dup: Option<usize>,
    /// The file's pieces when parts of it repeat earlier content (set by `find_repeats`).
    pub segs: Option<Vec<Seg>>,
}

/// Lists a folder recursively. Files are grouped by extension and name so that similar data
/// sits together in the stream (better models, better matches).
/// `skip` (the archive being written) is left out even when it lies inside `root`.
pub fn scan(root: &Path, skip: Option<&Path>) -> io::Result<Vec<Entry>> {
    let skip = skip.and_then(|p| Some((p.file_name()?.to_os_string(), fs::canonicalize(p).ok()?)));
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let mut items: Vec<_> = fs::read_dir(&d)?.collect::<Result<_, _>>()?;
        items.sort_by_key(|e| e.file_name());
        for e in items {
            let p = e.path();
            let ft = e.file_type()?;
            let rel = p.strip_prefix(root).unwrap().components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            if ft.is_dir() {
                out.push(Entry { rel, dir: true, size: 0, path: p.clone(), dup: None, segs: None });
                stack.push(p);
            } else if ft.is_file() {
                if let Some((name, canon)) = &skip {
                    if e.file_name() == *name && fs::canonicalize(&p).ok().as_ref() == Some(canon) {
                        continue;
                    }
                }
                let size = e.metadata()?.len();
                out.push(Entry { rel, dir: false, size, path: p, dup: None, segs: None });
            }
            // symlinks and special files are skipped
        }
    }
    // folders first; files by extension, then by name (copies of one file in different
    // folders sit side by side), then by path
    let name = |e: &Entry| e.rel.rsplit_once('/').map_or(e.rel.as_str(), |(_, n)| n).to_string();
    let ext = |e: &Entry| e.rel.rsplit_once('.').map(|(_, x)| x.to_ascii_lowercase()).unwrap_or_default();
    out.sort_by_cached_key(|e| (!e.dir, ext(e), name(e), e.rel.clone()));
    Ok(out)
}

pub fn manifest(entries: &[Entry]) -> Vec<u8> {
    let mut m = Vec::new();
    write_varint(&mut m, entries.len() as u64);
    for (i, e) in entries.iter().enumerate() {
        m.push(if e.dir { 1 } else if e.dup.is_some() { 2 } else if e.segs.is_some() { 3 } else { 0 });
        write_varint(&mut m, e.rel.len() as u64);
        m.extend_from_slice(e.rel.as_bytes());
        match (e.dir, e.dup) {
            (true, _) => {}
            (false, Some(k)) => write_varint(&mut m, k as u64),
            (false, None) => {
                write_varint(&mut m, e.size);
                if let Some(segs) = &e.segs {
                    write_varint(&mut m, segs.len() as u64);
                    for g in segs {
                        match *g {
                            Seg::Lit(l) => write_varint(&mut m, l << 1),
                            Seg::Copy { src, off, len } => {
                                write_varint(&mut m, len << 1 | 1);
                                write_varint(&mut m, (i - src) as u64);
                                write_varint(&mut m, off);
                            }
                        }
                    }
                }
            }
        }
    }
    m
}

/// Content hash used only to find candidate duplicates (they are then compared byte by
/// byte, so a collision can never merge different files).
fn file_hash(p: &Path) -> io::Result<u64> {
    let mut f = File::open(p)?;
    let mut buf = vec![0u8; 1 << 20];
    let (mut a, mut b) = (0x243F_6A88_85A3_08D3u64, 0x1319_8A2E_0370_7344u64);
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let mut c = buf[..n].chunks_exact(16);
        for w in &mut c {
            let x = u64::from_le_bytes(w[..8].try_into().unwrap());
            let y = u64::from_le_bytes(w[8..].try_into().unwrap());
            a = (a ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
            b = (b ^ y).wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(31);
        }
        for &v in c.remainder() {
            a = (a ^ v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    Ok(a ^ b.rotate_left(17))
}

fn same_bytes(p: &Path, q: &Path) -> io::Result<bool> {
    let (mut f, mut g) = (File::open(p)?, File::open(q)?);
    let (mut x, mut y) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    loop {
        let n = read_full(&mut f, &mut x)?;
        let m = read_full(&mut g, &mut y)?;
        if n != m || x[..n] != y[..m] {
            return Ok(false);
        }
        if n == 0 {
            return Ok(true);
        }
    }
}

fn read_full(f: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut k = 0;
    while k < buf.len() {
        let n = f.read(&mut buf[k..])?;
        if n == 0 {
            break;
        }
        k += n;
    }
    Ok(k)
}

/// Marks files whose bytes equal an earlier file's: they cost nothing in the stream and
/// are copied when unpacking. Only files that share a size are read, and every match is
/// confirmed byte by byte. Returns the bytes saved.
pub fn find_dups(entries: &mut [Entry], threads: usize) -> io::Result<u64> {
    use std::collections::HashMap;
    let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        if !e.dir && e.size > 0 {
            by_size.entry(e.size).or_default().push(i);
        }
    }
    let cand: Vec<usize> = by_size.into_values().filter(|v| v.len() > 1).flatten().collect();
    if cand.is_empty() {
        return Ok(0);
    }
    let hashes = crate::par::map(&cand, threads, |&i| file_hash(&entries[i].path));
    // first entry (in stream order) per (size, hash) is the original
    let mut groups: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    for (&i, h) in cand.iter().zip(hashes) {
        groups.entry((entries[i].size, h?)).or_default().push(i);
    }
    let pairs: Vec<(usize, usize)> = groups
        .into_values()
        .filter(|v| v.len() > 1)
        .flat_map(|mut v| {
            v.sort_unstable();
            let first = v[0];
            v.into_iter().skip(1).map(move |j| (first, j))
        })
        .collect();
    let same = crate::par::map(&pairs, threads, |&(i, j)| same_bytes(&entries[i].path, &entries[j].path));
    let mut saved = 0;
    for (&(i, j), ok) in pairs.iter().zip(same) {
        if ok? {
            entries[j].dup = Some(i);
            saved += entries[j].size;
        }
    }
    Ok(saved)
}

/// Marks repeated pieces of files (see dedup.rs): they are copied from earlier files when
/// unpacking instead of being compressed. Returns the bytes saved.
pub fn find_repeats(entries: &mut [Entry], threads: usize) -> io::Result<u64> {
    let files: Vec<(usize, &Path, u64)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.dir && e.dup.is_none() && e.size >= crate::dedup::MIN_FILE)
        .map(|(i, e)| (i, e.path.as_path(), e.size))
        .collect();
    let found = crate::dedup::find(&files, threads)?;
    let idx: Vec<usize> = files.iter().map(|f| f.0).collect();
    let mut saved = 0;
    for (i, segs) in idx.into_iter().zip(found) {
        if let Some(s) = &segs {
            saved += s.iter().map(|g| if matches!(g, Seg::Copy { .. }) { g.len() } else { 0 }).sum::<u64>();
        }
        entries[i].segs = segs;
    }
    Ok(saved)
}

/// Bytes of a file that go into the stream: (offset, length) ranges.
fn stream_ranges(e: &Entry) -> Vec<(u64, u64)> {
    match &e.segs {
        None => vec![(0, e.size)],
        Some(segs) => {
            let mut at = 0;
            let mut r = Vec::new();
            for g in segs {
                if let Seg::Lit(l) = *g {
                    r.push((at, l));
                }
                at += g.len();
            }
            r
        }
    }
}

/// Reader over manifest + every file's bytes.
pub struct PackReader {
    head: io::Cursor<Vec<u8>>,
    files: std::vec::IntoIter<(PathBuf, Vec<(u64, u64)>)>,
    /// open file and its ranges still to read
    cur: Option<(File, std::vec::IntoIter<(u64, u64)>)>,
    /// bytes left in the range being read
    left: u64,
}

/// Inside a big ELF file (program, library, object), the offsets in its stream bytes
/// where the kind of section changes (code / strings / other data): blocks may end there,
/// so string tables and code get the engine that suits them. `ranges` = the file's
/// stream ranges (offset, length).
fn elf_cuts(p: &Path, ranges: &[(u64, u64)]) -> Vec<u64> {
    let mut f = match File::open(p) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut h = [0u8; 64];
    if read_full(&mut f, &mut h).unwrap_or(0) < 64 || &h[..4] != b"\x7fELF" || h[4] != 2 || h[5] != 1 {
        return Vec::new();
    }
    let u16_at = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as u64;
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let (shoff, shentsize, shnum) = (u64_at(&h, 0x28), u16_at(&h, 0x3a), u16_at(&h, 0x3c));
    if shentsize < 64 || shnum == 0 || shnum > 1 << 16 {
        return Vec::new();
    }
    let mut t = vec![0u8; (shentsize * shnum) as usize];
    if f.seek(io::SeekFrom::Start(shoff)).is_err() || read_full(&mut f, &mut t).unwrap_or(0) < t.len() {
        return Vec::new();
    }
    // (file offset, kind): 0 code, 1 strings, 2 other
    let mut secs: Vec<(u64, u8)> = (0..shnum as usize)
        .filter_map(|i| {
            let b = &t[i * shentsize as usize..];
            let typ = u32::from_le_bytes(b[4..8].try_into().unwrap());
            let (flags, off, size) = (u64_at(b, 8), u64_at(b, 24), u64_at(b, 32));
            if typ == 8 || size < 1 << 20 {
                return None; // no bytes in the file, or too small to matter
            }
            Some((off, if flags & 4 != 0 { 0 } else if typ == 3 || flags & 0x20 != 0 { 1 } else { 2 }))
        })
        .collect();
    secs.sort_unstable();
    let mut out = Vec::new();
    let mut prev = u8::MAX;
    for (off, kind) in secs {
        if kind != prev && prev != u8::MAX {
            // file offset -> offset in the file's stream bytes (only inside literal ranges)
            let mut at = 0;
            for &(ro, rl) in ranges {
                if off >= ro && off < ro + rl {
                    if off > 0 {
                        out.push(at + off - ro);
                    }
                    break;
                }
                at += rl;
            }
        }
        prev = kind;
    }
    out
}

/// ELF files at least this big get block cuts where their section kind changes.
const ELF_SPLIT: u64 = 64 << 20;

/// Files at least this big and without an extension get blocks of their own.
const BIG_FILE: u64 = 4 << 20;

impl PackReader {
    /// Returns the reader, the stream length, and preferred block cut points: every file
    /// start, flagged where the extension group changes (and around big files without an
    /// extension).
    pub fn new(entries: &[Entry]) -> (PackReader, u64, Vec<(u64, bool)>) {
        let m = manifest(entries);
        let mut head = (m.len() as u64).to_le_bytes().to_vec();
        head.extend_from_slice(&m);
        let files: Vec<&Entry> = entries.iter().filter(|e| !e.dir && e.dup.is_none()).collect();
        let ext = |e: &Entry| e.rel.rsplit_once('.').map(|(_, x)| x.to_ascii_lowercase()).unwrap_or_default();
        let mut cuts = Vec::new();
        let mut off = head.len() as u64;
        let mut prev_ext: Option<String> = None;
        let mut prev_big = false;
        let mut list = Vec::with_capacity(files.len());
        for e in &files {
            let x = ext(e);
            let r = stream_ranges(e);
            let size: u64 = r.iter().map(|q| q.1).sum();
            if size > 0 {
                // big files without an extension can be anything (text, images, programs):
                // they keep blocks of their own, so each gets its own engine
                let big = size >= BIG_FILE;
                let alone = x.is_empty() && (big || prev_big);
                cuts.push((off, alone || prev_ext.as_ref().is_some_and(|p| *p != x)));
                // huge programs: blocks follow their sections (smaller files stay whole, so
                // similar files share blocks)
                if size >= ELF_SPLIT {
                    for c in elf_cuts(&e.path, &r) {
                        cuts.push((off + c, true));
                    }
                }
                prev_ext = Some(x);
                prev_big = big;
            }
            off += size;
            list.push((e.path.clone(), r));
        }
        let total = off;
        (PackReader { head: io::Cursor::new(head), files: list.into_iter(), cur: None, left: 0 }, total, cuts)
    }
}

impl Read for PackReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.head.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        loop {
            if self.left > 0 {
                let f = &mut self.cur.as_mut().unwrap().0;
                let k = (buf.len() as u64).min(self.left) as usize;
                let n = f.read(&mut buf[..k])?;
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "a file shrank while being read"));
                }
                self.left -= n as u64;
                return Ok(n);
            }
            if let Some((f, ranges)) = &mut self.cur {
                if let Some((off, len)) = ranges.next() {
                    f.seek(io::SeekFrom::Start(off))?;
                    self.left = len;
                    continue;
                }
                self.cur = None;
            }
            match self.files.next() {
                Some((p, ranges)) => self.cur = Some((File::open(&p)?, ranges.into_iter())),
                None => return Ok(0),
            }
        }
    }
}

fn safe_join(root: &Path, rel: &str) -> io::Result<PathBuf> {
    let p = Path::new(rel);
    if rel.is_empty() || p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsafe path in archive: {rel}")));
    }
    Ok(root.join(p))
}

/// How a file is rebuilt, in order.
enum Part {
    /// the next bytes of the stream
    Stream(u64),
    /// bytes [off, off + len) of an earlier file (or of this one)
    Copy(PathBuf, u64, u64),
}

/// Writer that rebuilds the folder from the decompressed stream.
pub struct Unpacker {
    root: PathBuf,
    head: Vec<u8>,
    need: Option<usize>,
    /// Files in stream order: path, and either its parts or the earlier file it duplicates.
    files: std::collections::VecDeque<(PathBuf, Result<std::collections::VecDeque<Part>, PathBuf>)>,
    cur: Option<(File, std::collections::VecDeque<Part>)>,
    /// Bytes written by copying (duplicate files and repeated pieces; not in the stream).
    copied: u64,
}

impl Unpacker {
    pub fn new(root: &Path) -> Unpacker {
        Unpacker { root: root.to_path_buf(), head: Vec::new(), need: None, files: Default::default(), cur: None, copied: 0 }
    }

    fn parse_manifest(&mut self) -> io::Result<()> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        fs::create_dir_all(&self.root)?;
        let mut r = io::Cursor::new(&self.head[8..]);
        let n = read_varint(&mut r)?;
        // path and kind of every entry so far (duplicates and copies point back at these)
        let mut seen: Vec<(PathBuf, u8)> = Vec::new();
        for _ in 0..n {
            let mut k = [0u8];
            r.read_exact(&mut k)?;
            let len = read_varint(&mut r)? as usize;
            let mut name = vec![0u8; len];
            r.read_exact(&mut name)?;
            let rel = String::from_utf8(name).map_err(|_| bad("bad path"))?;
            let p = safe_join(&self.root, &rel)?;
            let src = |seen: &Vec<(PathBuf, u8)>, i: usize| match seen.get(i) {
                Some((q, 0 | 3)) => Ok(q.clone()),
                _ => Err(bad("bad copy source")),
            };
            match k[0] {
                1 => fs::create_dir_all(&p)?,
                0 => {
                    let size = read_varint(&mut r)?;
                    self.files.push_back((p.clone(), Ok([Part::Stream(size)].into_iter().filter(|q| !matches!(q, Part::Stream(0))).collect())));
                }
                2 => {
                    let i = read_varint(&mut r)? as usize;
                    self.files.push_back((p.clone(), Err(src(&seen, i)?)));
                }
                3 => {
                    let size = read_varint(&mut r)?;
                    let count = read_varint(&mut r)?;
                    let mut parts = std::collections::VecDeque::new();
                    let mut at = 0u64;
                    let me = seen.len();
                    for _ in 0..count {
                        let v = read_varint(&mut r)?;
                        let l = v >> 1;
                        if v & 1 == 0 {
                            parts.push_back(Part::Stream(l));
                        } else {
                            let back = read_varint(&mut r)? as usize;
                            let off = read_varint(&mut r)?;
                            let from = if back == 0 {
                                // an earlier part of this same file
                                if off.checked_add(l).is_none_or(|e| e > at) {
                                    return Err(bad("bad copy source"));
                                }
                                p.clone()
                            } else {
                                src(&seen, me.checked_sub(back).ok_or_else(|| bad("bad copy source"))?)?
                            };
                            parts.push_back(Part::Copy(from, off, l));
                        }
                        at = at.checked_add(l).ok_or_else(|| bad("bad piece"))?;
                    }
                    if at != size {
                        return Err(bad("pieces do not add up to the file size"));
                    }
                    self.files.push_back((p.clone(), Ok(parts)));
                }
                _ => return Err(bad("bad entry kind")),
            }
            seen.push((p, k[0]));
        }
        Ok(())
    }

    /// Runs the copies at the front of the current file; closes it once it is complete.
    fn run_copies(&mut self) -> io::Result<()> {
        let Some((f, parts)) = &mut self.cur else { return Ok(()) };
        while let Some(Part::Copy(src, off, len)) = parts.front() {
            let mut g = File::open(src)?;
            g.seek(io::SeekFrom::Start(*off))?;
            let n = io::copy(&mut g.take(*len), f)?;
            if n != *len {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "copy source is too short"));
            }
            self.copied += n;
            parts.pop_front();
        }
        if parts.is_empty() {
            self.cur = None;
        }
        Ok(())
    }

    fn open_next(&mut self) -> io::Result<()> {
        while self.cur.is_none() {
            match self.files.pop_front() {
                Some((p, how)) => {
                    if let Some(d) = p.parent() {
                        fs::create_dir_all(d)?;
                    }
                    match how {
                        // the original came earlier in the stream, so it is complete
                        Err(src) => self.copied += fs::copy(&src, &p)?,
                        Ok(parts) => {
                            let f = File::create(&p)?;
                            if !parts.is_empty() {
                                self.cur = Some((f, parts));
                                self.run_copies()?;
                            }
                        }
                    }
                }
                None => return Ok(()),
            }
        }
        Ok(())
    }

    /// Ends unpacking; returns the bytes written by copying (not decoded from the stream).
    pub fn finish(mut self) -> io::Result<u64> {
        if self.need.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "missing folder manifest"));
        }
        self.open_next()?; // creates any remaining empty or copied files
        if self.cur.is_some() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "archive ended before all files were written"));
        }
        Ok(self.copied)
    }
}

impl Write for Unpacker {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let total = buf.len();
        while !buf.is_empty() {
            if self.need.map_or(true, |n| self.head.len() < n) {
                let want = match self.need {
                    None => 8 - self.head.len(),
                    Some(n) => n - self.head.len(),
                };
                let k = want.min(buf.len());
                self.head.extend_from_slice(&buf[..k]);
                buf = &buf[k..];
                if self.need.is_none() && self.head.len() == 8 {
                    let ml = u64::from_le_bytes(self.head[..8].try_into().unwrap()) as usize;
                    self.need = Some(8 + ml);
                }
                if let Some(n) = self.need {
                    if self.head.len() == n {
                        self.parse_manifest()?;
                    }
                }
                continue;
            }
            self.open_next()?;
            let Some((f, parts)) = &mut self.cur else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "extra data after last file"));
            };
            let Some(Part::Stream(left)) = parts.front_mut() else { unreachable!() };
            let k = (*left).min(buf.len() as u64) as usize;
            f.write_all(&buf[..k])?;
            *left -= k as u64;
            buf = &buf[k..];
            if *left == 0 {
                parts.pop_front();
                self.run_copies()?;
            }
        }
        Ok(total)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
