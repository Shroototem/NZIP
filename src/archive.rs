//! Folder support: a folder becomes one stream (manifest + file contents), so similar files
//! share one model and repeats across files are found by the match model.
//!
//! ```text
//! manifest_len:u64le  manifest  contents...
//! manifest = count:varint { kind:u8 path_len:varint path(utf-8, '/') extra }
//!   kind 0 = file, extra = size:varint (its bytes follow in the stream, in entry order)
//!   kind 1 = folder, no extra
//!   kind 2 = duplicate file, extra = index:varint of an earlier kind-0 entry with the same
//!            bytes (no stream bytes; unpacking copies the earlier file)
//! ```

use crate::container::{read_varint, write_varint};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

pub struct Entry {
    pub rel: String,
    pub dir: bool,
    pub size: u64,
    pub path: PathBuf,
    /// Index of an earlier entry with identical bytes (set by `find_dups`).
    pub dup: Option<usize>,
}

/// Lists a folder recursively. Files are grouped by extension so that similar data sits
/// together in the stream (better models, better matches).
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
                out.push(Entry { rel, dir: true, size: 0, path: p.clone(), dup: None });
                stack.push(p);
            } else if ft.is_file() {
                if let Some((name, canon)) = &skip {
                    if e.file_name() == *name && fs::canonicalize(&p).ok().as_ref() == Some(canon) {
                        continue;
                    }
                }
                let size = e.metadata()?.len();
                out.push(Entry { rel, dir: false, size, path: p, dup: None });
            }
            // symlinks and special files are skipped
        }
    }
    let ext = |e: &Entry| e.rel.rsplit_once('.').map(|(_, x)| x.to_ascii_lowercase()).unwrap_or_default();
    out.sort_by(|a, b| b.dir.cmp(&a.dir).then_with(|| ext(a).cmp(&ext(b))).then_with(|| a.rel.cmp(&b.rel)));
    Ok(out)
}

pub fn manifest(entries: &[Entry]) -> Vec<u8> {
    let mut m = Vec::new();
    write_varint(&mut m, entries.len() as u64);
    for e in entries {
        m.push(if e.dir { 1 } else if e.dup.is_some() { 2 } else { 0 });
        write_varint(&mut m, e.rel.len() as u64);
        m.extend_from_slice(e.rel.as_bytes());
        match (e.dir, e.dup) {
            (true, _) => {}
            (false, Some(k)) => write_varint(&mut m, k as u64),
            (false, None) => write_varint(&mut m, e.size),
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

/// Reader over manifest + every file's bytes.
pub struct PackReader {
    head: io::Cursor<Vec<u8>>,
    files: std::vec::IntoIter<(PathBuf, u64)>,
    cur: Option<io::Take<File>>,
}

/// Files at least this big get blocks of their own.
const BIG_FILE: u64 = 4 << 20;

impl PackReader {
    /// Returns the reader, the stream length, and preferred block cut points (stream
    /// offsets): around every big file, and wherever the extension group changes.
    pub fn new(entries: &[Entry]) -> (PackReader, u64, Vec<u64>) {
        let m = manifest(entries);
        let mut head = (m.len() as u64).to_le_bytes().to_vec();
        head.extend_from_slice(&m);
        let files: Vec<&Entry> = entries.iter().filter(|e| !e.dir && e.dup.is_none()).collect();
        let ext = |e: &Entry| e.rel.rsplit_once('.').map(|(_, x)| x.to_ascii_lowercase()).unwrap_or_default();
        let mut cuts = Vec::new();
        let mut off = head.len() as u64;
        let mut prev_ext: Option<String> = None;
        let mut prev_big = false;
        for e in &files {
            let x = ext(e);
            let big = e.size >= BIG_FILE;
            if big || prev_big || prev_ext.as_ref().is_some_and(|p| *p != x) {
                cuts.push(off);
            }
            off += e.size;
            prev_ext = Some(x);
            prev_big = big;
        }
        let total = off;
        let list = files.iter().map(|e| (e.path.clone(), e.size)).collect::<Vec<_>>();
        (PackReader { head: io::Cursor::new(head), files: list.into_iter(), cur: None }, total, cuts)
    }
}

impl Read for PackReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.head.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        loop {
            if let Some(f) = &mut self.cur {
                let n = f.read(buf)?;
                if n > 0 {
                    return Ok(n);
                }
                if f.limit() > 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "a file shrank while being read"));
                }
                self.cur = None;
            }
            match self.files.next() {
                Some((p, size)) => self.cur = Some(File::open(&p)?.take(size)),
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

/// Writer that rebuilds the folder from the decompressed stream.
pub struct Unpacker {
    root: PathBuf,
    head: Vec<u8>,
    need: Option<usize>,
    /// Files in stream order: path, size, and the earlier file to copy for duplicates.
    files: std::collections::VecDeque<(PathBuf, u64, Option<PathBuf>)>,
    cur: Option<(File, u64)>,
    /// Bytes written by copying duplicates (not in the stream).
    copied: u64,
}

impl Unpacker {
    pub fn new(root: &Path) -> Unpacker {
        Unpacker { root: root.to_path_buf(), head: Vec::new(), need: None, files: Default::default(), cur: None, copied: 0 }
    }

    fn parse_manifest(&mut self) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let mut r = io::Cursor::new(&self.head[8..]);
        let n = read_varint(&mut r)?;
        // path and kind of every entry so far (duplicates point back at these)
        let mut seen: Vec<(PathBuf, u8)> = Vec::new();
        for _ in 0..n {
            let mut k = [0u8];
            r.read_exact(&mut k)?;
            let len = read_varint(&mut r)? as usize;
            let mut name = vec![0u8; len];
            r.read_exact(&mut name)?;
            let rel = String::from_utf8(name).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad path"))?;
            let p = safe_join(&self.root, &rel)?;
            match k[0] {
                1 => fs::create_dir_all(&p)?,
                0 => {
                    let size = read_varint(&mut r)?;
                    self.files.push_back((p.clone(), size, None));
                }
                2 => {
                    let i = read_varint(&mut r)? as usize;
                    let src = match seen.get(i) {
                        Some((q, 0)) => q.clone(),
                        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad duplicate entry")),
                    };
                    self.files.push_back((p.clone(), 0, Some(src)));
                }
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad entry kind")),
            }
            seen.push((p, k[0]));
        }
        Ok(())
    }

    fn open_next(&mut self) -> io::Result<()> {
        while self.cur.is_none() {
            match self.files.pop_front() {
                Some((p, size, src)) => {
                    if let Some(d) = p.parent() {
                        fs::create_dir_all(d)?;
                    }
                    if let Some(src) = src {
                        // the original came earlier in the stream, so it is complete
                        self.copied += fs::copy(&src, &p)?;
                        continue;
                    }
                    let f = File::create(&p)?;
                    if size > 0 {
                        self.cur = Some((f, size));
                    }
                }
                None => return Ok(()),
            }
        }
        Ok(())
    }

    /// Ends unpacking; returns the bytes written by copying duplicate files.
    pub fn finish(mut self) -> io::Result<u64> {
        if self.need.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "missing folder manifest"));
        }
        self.open_next()?; // creates any remaining empty files
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
            let Some((f, left)) = &mut self.cur else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "extra data after last file"));
            };
            let k = (*left).min(buf.len() as u64) as usize;
            f.write_all(&buf[..k])?;
            *left -= k as u64;
            buf = &buf[k..];
            if *left == 0 {
                self.cur = None;
            }
        }
        Ok(total)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
