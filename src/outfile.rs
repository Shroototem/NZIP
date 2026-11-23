//! Safe output: every file or folder nzip writes is first created under a fresh temporary
//! name next to its destination (created exclusively, so it can never be a file the user
//! already had), and only renamed into place once it is complete. On an error, Ctrl-C,
//! SIGTERM or SIGHUP the temporary output is removed, so no partial archive or half
//! unpacked folder is ever left behind. Cleanup only ever touches paths nzip created.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Temporary outputs that exist right now (removed on interruption).
static LIVE: Mutex<Vec<(PathBuf, bool)>> = Mutex::new(Vec::new());

fn register(p: &Path, dir: bool) {
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).push((p.to_path_buf(), dir));
}

fn unregister(p: &Path) {
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).retain(|(q, _)| q != p);
}

/// Removes every live temporary output (called on interruption).
fn cleanup_all() {
    let live = std::mem::take(&mut *LIVE.lock().unwrap_or_else(|e| e.into_inner()));
    for (p, dir) in live {
        let _ = if dir { fs::remove_dir_all(&p) } else { fs::remove_file(&p) };
    }
}

/// Temporary sibling name for `dest`: `.<name>.nzpart-<pid>-<k>` in the same folder, so the
/// final rename never crosses file systems.
fn temp_name(dest: &Path, k: u32) -> PathBuf {
    let name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_else(|| OsString::from("out"));
    let mut t = OsString::from(".");
    t.push(name);
    t.push(format!(".nzpart-{}-{k}", std::process::id()));
    dest.with_file_name(t)
}

fn exists_err(p: &Path) -> io::Error {
    io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists (use -f to overwrite)", p.display()))
}

/// An output file being written. Call `commit` once it is complete; dropping it without
/// `commit` (error, panic) deletes the temporary file.
pub struct OutFile {
    tmp: PathBuf,
    dest: PathBuf,
    force: bool,
    done: bool,
}

impl OutFile {
    pub fn create(dest: &Path, force: bool) -> io::Result<(OutFile, File)> {
        if !force && dest.exists() {
            return Err(exists_err(dest));
        }
        for k in 0..1000 {
            let tmp = temp_name(dest, k);
            match OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(f) => {
                    register(&tmp, false);
                    return Ok((OutFile { tmp, dest: dest.to_path_buf(), force, done: false }, f));
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::other("no free temporary name"))
    }

    /// Path being written (the folder scanner skips it).
    pub fn temp_path(&self) -> &Path {
        &self.tmp
    }

    /// Moves the finished file into place (the file must be flushed and closed).
    pub fn commit(mut self) -> io::Result<()> {
        if !self.force && self.dest.exists() {
            return Err(exists_err(&self.dest));
        }
        fs::rename(&self.tmp, &self.dest)?;
        unregister(&self.tmp);
        self.done = true;
        Ok(())
    }
}

impl Drop for OutFile {
    fn drop(&mut self) {
        if !self.done {
            let _ = fs::remove_file(&self.tmp);
            unregister(&self.tmp);
        }
    }
}

/// An output folder being unpacked. Everything goes into a fresh temporary folder that is
/// renamed into place (or, with -f onto an existing folder, merged into it) on `commit`.
pub struct OutDir {
    tmp: PathBuf,
    dest: PathBuf,
    force: bool,
    done: bool,
}

impl OutDir {
    pub fn create(dest: &Path, force: bool) -> io::Result<OutDir> {
        if !force && dest.exists() {
            return Err(exists_err(dest));
        }
        if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        for k in 0..1000 {
            let tmp = temp_name(dest, k);
            match fs::create_dir(&tmp) {
                Ok(()) => {
                    register(&tmp, true);
                    return Ok(OutDir { tmp, dest: dest.to_path_buf(), force, done: false });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::other("no free temporary name"))
    }

    pub fn temp_path(&self) -> &Path {
        &self.tmp
    }

    pub fn commit(mut self) -> io::Result<()> {
        if self.dest.exists() {
            if !self.force {
                return Err(exists_err(&self.dest));
            }
            if !self.dest.is_dir() {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists and is not a folder", self.dest.display())));
            }
            merge(&self.tmp, &self.dest)?;
            fs::remove_dir_all(&self.tmp)?;
        } else {
            fs::rename(&self.tmp, &self.dest)?;
        }
        unregister(&self.tmp);
        self.done = true;
        Ok(())
    }
}

impl Drop for OutDir {
    fn drop(&mut self) {
        if !self.done {
            let _ = fs::remove_dir_all(&self.tmp);
            unregister(&self.tmp);
        }
    }
}

/// Moves the contents of `src` into the existing folder `dst` (-f): files replace files,
/// folders merge; a file never replaces a folder or the other way round.
fn merge(src: &Path, dst: &Path) -> io::Result<()> {
    for e in fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            if to.is_dir() {
                merge(&e.path(), &to)?;
                continue;
            }
            if to.exists() {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists and is not a folder", to.display())));
            }
        } else if to.is_dir() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} is a folder", to.display())));
        }
        fs::rename(e.path(), &to)?;
    }
    Ok(())
}

/// Default archive name for `input`: `<name>.nz` next to it, never inside it, also for
/// `.`, `..` and paths ending in a separator.
pub fn archive_name(input: &Path) -> io::Result<PathBuf> {
    let abs = fs::canonicalize(input)?;
    let name = abs.file_name().map(|n| n.to_os_string()).unwrap_or_else(|| OsString::from("root"));
    let mut n = name;
    n.push(".nz");
    Ok(match abs.parent() {
        Some(p) => p.join(n),
        None => PathBuf::from(n),
    })
}

/// Default output for `archive`: its name without `.nz` (or with `.out` added), in the same
/// folder; never the archive's own folder.
pub fn unpack_name(archive: &Path) -> PathBuf {
    let name = archive.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let out = match name.strip_suffix(".nz") {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => format!("{name}.out"),
    };
    archive.with_file_name(out)
}

// ------------------------------------------------------------------ interruption

/// Installs the Ctrl-C / termination handler that removes temporary outputs, then exits
/// with status 130.
pub fn install_handler() {
    imp::install();
}

#[cfg(unix)]
mod imp {
    use std::sync::atomic::{AtomicI32, Ordering};

    unsafe extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
        fn pipe(fds: *mut i32) -> i32;
        fn write(fd: i32, buf: *const std::ffi::c_void, n: usize) -> isize;
        fn read(fd: i32, buf: *mut std::ffi::c_void, n: usize) -> isize;
    }

    static WFD: AtomicI32 = AtomicI32::new(-1);

    // Only async-signal-safe work here: wake the cleanup thread.
    extern "C" fn on_signal(_sig: i32) {
        let fd = WFD.load(Ordering::Relaxed);
        if fd >= 0 {
            let b = 1u8;
            unsafe { write(fd, (&b as *const u8).cast(), 1) };
        }
    }

    pub fn install() {
        let mut fds = [0i32; 2];
        if unsafe { pipe(fds.as_mut_ptr()) } != 0 {
            return;
        }
        WFD.store(fds[1], Ordering::Relaxed);
        let rfd = fds[0];
        let spawned = std::thread::Builder::new().name("nzip-signal".into()).spawn(move || {
            let mut b = 0u8;
            loop {
                let r = unsafe { read(rfd, (&mut b as *mut u8).cast(), 1) };
                if r == 1 {
                    break;
                }
                if r == 0 {
                    return;
                }
            }
            super::cleanup_all();
            eprintln!("\nnzip: interrupted, partial output removed");
            std::process::exit(130);
        });
        if spawned.is_ok() {
            // SIGHUP, SIGINT, SIGTERM
            for sig in [1, 2, 15] {
                unsafe { signal(sig, on_signal as *const () as usize) };
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }

    // Runs on its own thread, so ordinary cleanup is fine. Files are opened with
    // FILE_SHARE_DELETE by std, so they can be deleted while still open.
    unsafe extern "system" fn on_ctrl(_kind: u32) -> i32 {
        super::cleanup_all();
        eprintln!("\nnzip: interrupted, partial output removed");
        std::process::exit(130);
    }

    pub fn install() {
        unsafe { SetConsoleCtrlHandler(Some(on_ctrl), 1) };
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    pub fn install() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let d = std::env::temp_dir().join(format!("nzip-names-{}", std::process::id()));
        fs::create_dir_all(d.join("sub")).unwrap();
        let a = archive_name(&d.join("sub/")).unwrap();
        assert_eq!(a, fs::canonicalize(&d).unwrap().join("sub.nz"));
        let a = archive_name(&d.join("sub/.")).unwrap();
        assert_eq!(a, fs::canonicalize(&d).unwrap().join("sub.nz"));
        assert_eq!(unpack_name(Path::new("x/a.nz")), Path::new("x/a"));
        assert_eq!(unpack_name(Path::new("x/.nz")), Path::new("x/.nz.out"));
        assert_eq!(unpack_name(Path::new("x/a.bin")), Path::new("x/a.bin.out"));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn temp_never_reuses_existing() {
        let d = std::env::temp_dir().join(format!("nzip-tmp-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        let dest = d.join("a.nz");
        // a user file that happens to have our first temporary name is never touched
        fs::write(temp_name(&dest, 0), b"user").unwrap();
        {
            let (o, f) = OutFile::create(&dest, false).unwrap();
            assert_ne!(o.temp_path(), temp_name(&dest, 0));
            drop(f);
            // dropped without commit: its own temporary file is removed
        }
        assert_eq!(fs::read(temp_name(&dest, 0)).unwrap(), b"user");
        assert!(!dest.exists());
        let (o, f) = OutFile::create(&dest, false).unwrap();
        drop(f);
        o.commit().unwrap();
        assert!(dest.exists());
        assert_eq!(fs::read_dir(&d).unwrap().count(), 2);
        fs::remove_dir_all(&d).unwrap();
    }
}
