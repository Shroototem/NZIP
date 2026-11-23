//! Folder support: a folder becomes one stream (manifest + file contents), so similar files
//! share one model and repeats across files are found by the match model.
//!
//! ```text
//! manifest_len:u64le  manifest  contents...
//! manifest = count:varint { kind:u8 (0=file 1=dir) path_len:varint path(utf-8, '/') [size:varint] }
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
}

/// Lists a folder recursively. Files are grouped by extension so that similar data sits
/// together in the stream (better models, better matches).
pub fn scan(root: &Path) -> io::Result<Vec<Entry>> {
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
                out.push(Entry { rel, dir: true, size: 0, path: p.clone() });
                stack.push(p);
            } else if ft.is_file() {
                let size = e.metadata()?.len();
                out.push(Entry { rel, dir: false, size, path: p });
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
        m.push(e.dir as u8);
        write_varint(&mut m, e.rel.len() as u64);
        m.extend_from_slice(e.rel.as_bytes());
        if !e.dir {
            write_varint(&mut m, e.size);
        }
    }
    m
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
        let files: Vec<&Entry> = entries.iter().filter(|e| !e.dir).collect();
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
    files: std::collections::VecDeque<(PathBuf, u64)>,
    cur: Option<(File, u64)>,
}

impl Unpacker {
    pub fn new(root: &Path) -> Unpacker {
        Unpacker { root: root.to_path_buf(), head: Vec::new(), need: None, files: Default::default(), cur: None }
    }

    fn parse_manifest(&mut self) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let mut r = io::Cursor::new(&self.head[8..]);
        let n = read_varint(&mut r)?;
        for _ in 0..n {
            let mut k = [0u8];
            r.read_exact(&mut k)?;
            let len = read_varint(&mut r)? as usize;
            let mut name = vec![0u8; len];
            r.read_exact(&mut name)?;
            let rel = String::from_utf8(name).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad path"))?;
            let p = safe_join(&self.root, &rel)?;
            if k[0] == 1 {
                fs::create_dir_all(&p)?;
            } else {
                let size = read_varint(&mut r)?;
                self.files.push_back((p, size));
            }
        }
        Ok(())
    }

    fn open_next(&mut self) -> io::Result<()> {
        while self.cur.is_none() {
            match self.files.pop_front() {
                Some((p, size)) => {
                    if let Some(d) = p.parent() {
                        fs::create_dir_all(d)?;
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

    pub fn finish(mut self) -> io::Result<()> {
        if self.need.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "missing folder manifest"));
        }
        self.open_next()?; // creates any remaining empty files
        if self.cur.is_some() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "archive ended before all files were written"));
        }
        Ok(())
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
