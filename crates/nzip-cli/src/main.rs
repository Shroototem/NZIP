use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use nzip::{Mode, Options};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const EXAMPLES: &str = "Examples:
  nzip -c <input> <output>            compress with max (best) compression
  nzip -c -m fast <input> <output>    compress fast
  nzip -c -m store <input> <output>   store only (checksummed)
  nzip -d <input.nz> <output>         decompress
  nzip -t <input.nz>                  verify an archive without writing anything
  nzip -i <input.nz>                  show an archive's blocks

Output names are optional: compress writes <input>.nz, decompress strips .nz.";

#[derive(Parser)]
#[command(name = "nzip", version, about = "Fast lossless compressor using the GPU and all CPU cores",
          after_help = EXAMPLES)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    /// Best compression (default): large-block BWT, GPU when it fits, CPU otherwise
    #[value(alias = "balanced")]
    Max,
    /// Fastest: zstd on all CPU cores
    Fast,
    /// No compression, checksummed only
    Store,
    /// Developer test of the GPU entropy core (weak test model; hidden from --help)
    #[value(hide = true)]
    CoreTest,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    /// Fastest available hardware per stage (GPU if present)
    Auto,
    /// Require the GPU (error if none)
    Gpu,
    /// CPU only
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a file or folder (default output: <input>.nz)
    #[command(alias = "c", short_flag = 'c', long_flag = "compress", after_help = "Examples:
  nzip -c <input> <output>            max (best) compression, the default
  nzip -c -m fast <input> <output>    fast
  nzip -c -m store <input> <output>   store only
  nzip -c -D cpu <input> <output>     CPU only")]
    Compress {
        /// File or folder to compress
        input: PathBuf,
        /// Archive to write (default: <input>.nz)
        output: Option<PathBuf>,
        #[arg(short, long, value_enum, default_value = "max")]
        mode: ModeArg,
        /// zstd level for fast mode (1..22)
        #[arg(short, long, default_value_t = 3)]
        level: i32,
        /// Block size, e.g. 16M, 128M, 512K (default depends on mode)
        #[arg(short, long, value_parser = parse_size)]
        block: Option<usize>,
        /// Worker threads (default: all CPUs)
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
        /// Hardware for compute-heavy stages
        #[arg(short = 'D', long, value_enum, default_value = "auto")]
        device: DeviceArg,
        /// Memory budget for buffers in flight, e.g. 4G (default: half of available RAM,
        /// respecting container/cgroup limits)
        #[arg(short = 'M', long, value_parser = parse_size)]
        memory: Option<usize>,
        #[arg(long)]
        no_dedup: bool,
        #[arg(long)]
        no_transforms: bool,
        /// Overwrite the output if it exists
        #[arg(short, long)]
        force: bool,
        #[arg(short, long)]
        quiet: bool,
    },
    /// Decompress a file or folder (default output: <input> without .nz)
    #[command(alias = "d", short_flag = 'd', long_flag = "decompress", after_help = "Examples:
  nzip -d <input.nz> <output>
  nzip -d <input.nz>                  writes <input> without .nz")]
    Decompress {
        /// Archive to decompress
        input: PathBuf,
        /// File to write (default: <input> without .nz)
        output: Option<PathBuf>,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
        /// Hardware for compute-heavy stages
        #[arg(short = 'D', long, value_enum, default_value = "auto")]
        device: DeviceArg,
        /// Memory budget for buffers in flight, e.g. 4G (default: half of available RAM,
        /// respecting container/cgroup limits)
        #[arg(short = 'M', long, value_parser = parse_size)]
        memory: Option<usize>,
        #[arg(short, long)]
        force: bool,
        #[arg(short, long)]
        quiet: bool,
    },
    /// Decompress and verify all checksums without keeping the output
    #[command(alias = "t", short_flag = 't', long_flag = "test")]
    Test {
        input: PathBuf,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
        /// Memory budget for buffers in flight, e.g. 4G (default: half of available RAM,
        /// respecting container/cgroup limits)
        #[arg(short = 'M', long, value_parser = parse_size)]
        memory: Option<usize>,
    },
    /// Show the container layout
    #[command(alias = "i", short_flag = 'i', long_flag = "info")]
    Info { input: PathBuf },
    /// Show whether this build and machine can use the GPU, and its free memory
    #[command(long_flag = "gpu")]
    Gpu,
}

fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    let (num, mul) = match s.chars().last() {
        Some('K' | 'k') => (&s[..s.len() - 1], 1usize << 10),
        Some('M' | 'm') => (&s[..s.len() - 1], 1 << 20),
        Some('G' | 'g') => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<usize>().map(|n| n * mul).map_err(|e| format!("bad size '{s}': {e}"))
}

/// Removes a temporary file or folder when dropped (error paths included).
struct TempPath(PathBuf);
impl Drop for TempPath {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// `<path>` with `suffix` appended to its last component ("dir/" -> "dir.nz").
fn with_suffix(path: &Path, suffix: &str) -> Result<PathBuf> {
    let name = match path.file_name() {
        Some(n) => n.to_os_string(),
        None => std::fs::canonicalize(path)?.file_name().map(|n| n.to_os_string()).context("can't name the output; give one explicitly")?,
    };
    let mut n = name;
    n.push(suffix);
    Ok(path.parent().map(|p| p.join(&n)).unwrap_or_else(|| PathBuf::from(&n)))
}

/// Live progress line on stderr, shown only on a terminal and without --quiet.
struct Meter {
    label: &'static str,
    total: u64,
    done: std::sync::atomic::AtomicU64,
    start: Instant,
    last: std::sync::Mutex<Instant>,
    on: bool,
}

impl Meter {
    fn new(label: &'static str, total: u64, quiet: bool) -> Self {
        use std::io::IsTerminal;
        let on = !quiet && std::io::stderr().is_terminal();
        Self { label, total, done: 0.into(), start: Instant::now(), last: std::sync::Mutex::new(Instant::now()), on }
    }
    fn add(&self, n: u64) {
        let d = self.done.fetch_add(n, std::sync::atomic::Ordering::Relaxed) + n;
        if !self.on {
            return;
        }
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.elapsed().as_millis() < 200 {
            return;
        }
        *last = Instant::now();
        let secs = self.start.elapsed().as_secs_f64().max(1e-3);
        let pct = if self.total > 0 { 100.0 * d as f64 / self.total as f64 } else { 100.0 };
        eprint!("\r\x1b[2K{} {:5.1}%  {} / {}  {:.0} MB/s", self.label, pct.min(100.0), human(d), human(self.total), d as f64 / secs / 1e6);
    }
    fn finish(&self) {
        if self.on {
            eprint!("\r\x1b[2K");
        }
    }
}

fn human(b: u64) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.2} GB", b as f64 / 1e9),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / 1e6),
        b => format!("{:.0} KB", b as f64 / 1e3),
    }
}

enum Kind {
    Dir,
    File(u64),
    Symlink,
    Other,
}

/// Everything under `root`, depth first, names sorted (so the same folder always packs
/// to the same archive). Symlinks are listed, not followed.
fn scan_dir(root: &Path) -> Result<Vec<(PathBuf, PathBuf, Kind)>> {
    fn walk(root: &Path, rel: &Path, out: &mut Vec<(PathBuf, PathBuf, Kind)>) -> Result<()> {
        let dir = root.join(rel);
        let mut names: Vec<_> = std::fs::read_dir(&dir).with_context(|| format!("reading folder {}", dir.display()))?
            .map(|e| e.map(|e| e.file_name())).collect::<std::io::Result<_>>()?;
        names.sort();
        for n in names {
            let r = rel.join(&n);
            let full = root.join(&r);
            let m = std::fs::symlink_metadata(&full).with_context(|| format!("reading {}", full.display()))?;
            let ft = m.file_type();
            if ft.is_symlink() {
                out.push((full, r, Kind::Symlink));
            } else if ft.is_dir() {
                out.push((full, r.clone(), Kind::Dir));
                walk(root, &r, out)?;
            } else if ft.is_file() {
                out.push((full, r, Kind::File(m.len())));
            } else {
                out.push((full, r, Kind::Other));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, Path::new(""), &mut out)?;
    Ok(out)
}

/// Pack scanned folder items into a tar stream written to `w` (permissions, modification
/// times and symlinks kept; sockets, pipes and devices skipped). Returns `w` and the
/// number of skipped items.
fn pack_items<W: Write>(items: &[(PathBuf, PathBuf, Kind)], w: W, meter: &Meter) -> Result<(W, usize)> {
    let mut b = tar::Builder::new(w);
    b.follow_symlinks(false);
    let mut skipped = 0;
    for (full, rel, kind) in items {
        match kind {
            Kind::Other => skipped += 1,
            _ => b.append_path_with_name(full, rel).with_context(|| format!("reading {}", full.display()))?,
        }
        if let Kind::File(n) = kind {
            meter.add(*n);
        }
    }
    Ok((b.into_inner()?, skipped))
}

/// Extract a tar stream into `dest`. `unpack_in` refuses `..` components and writes
/// through symlinks that leave `dest`, and strips leading `/`. Directories are finished
/// last (deepest first) so their permissions and times aren't disturbed by their contents.
fn unpack_tar<R: std::io::Read>(r: R, dest: &Path, meter: &Meter) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let mut a = tar::Archive::new(r);
    a.set_preserve_permissions(true);
    a.set_preserve_mtime(true);
    a.set_overwrite(true);
    let mut dirs = Vec::new();
    for e in a.entries()? {
        let mut e = e?;
        if e.header().entry_type().is_dir() {
            dirs.push(e);
            continue;
        }
        let n = e.size();
        e.unpack_in(dest).with_context(|| format!("extracting into {}", dest.display()))?;
        meter.add(n);
    }
    dirs.sort_by(|a, b| b.path_bytes().cmp(&a.path_bytes()));
    for mut d in dirs {
        d.unpack_in(dest).with_context(|| format!("extracting into {}", dest.display()))?;
    }
    Ok(())
}

/// Extract a packed folder to `out`. A new folder is extracted beside it and renamed at
/// the end, so a failure never leaves a half-extracted folder under the final name; with
/// -f an existing folder is extracted into directly.
fn extract_dir<R: std::io::Read>(r: R, out: &Path, total: u64, quiet: bool) -> Result<()> {
    let meter = Meter::new("extracting ", total, quiet);
    if out.exists() {
        unpack_tar(r, out, &meter)?;
    } else {
        let partial = out.with_extension("nzip-partial");
        let guard = TempPath(partial.clone());
        unpack_tar(r, &partial, &meter)?;
        std::fs::rename(&partial, out)?;
        std::mem::forget(guard);
    }
    meter.finish();
    Ok(())
}

/// Bytes to compress: a memory map of a file, or a folder packed in memory.
enum Source {
    Map(memmap2::Mmap),
    Mem(Vec<u8>),
}
impl std::ops::Deref for Source {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Source::Map(m) => m,
            Source::Mem(v) => v,
        }
    }
}

fn read_input(path: &Path) -> Result<memmap2::Mmap> {
    let meta = std::fs::metadata(path).with_context(|| format!("opening {}", path.display()))?;
    if !meta.is_file() {
        bail!("{} is not a regular file or folder (devices, sockets and pipes can't be compressed)", path.display());
    }
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: the file is only read; concurrent modification by another process is
    // caught by the checksums on decompress.
    unsafe { memmap2::Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))
}

/// Create `path`'s temporary sibling for writing; it is renamed over `path` only after
/// the whole operation succeeded, so a failure never leaves a partial output behind.
fn create_tmp(path: &Path, force: bool) -> Result<(PathBuf, File)> {
    if path.exists() && !force {
        bail!("{} exists (use -f to overwrite)", path.display());
    }
    let tmp = path.with_extension("nzip-tmp");
    let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tmp)
        .with_context(|| format!("creating {}", tmp.display()))?;
    Ok((tmp, f))
}

/// Discards output; only valid for files without dedup (nothing is ever read back).
struct NullSink;
impl nzip::Sink for NullSink {
    fn write_at(&mut self, _: u64, _: &[u8]) -> Result<(), nzip::Error> { Ok(()) }
    fn read_at(&mut self, _: u64, _: &mut [u8]) -> Result<(), nzip::Error> {
        Err(nzip::Error::Format("internal: read from discarded output"))
    }
}

/// A temporary file that never outlives us (unlinked at once on Linux, deleted on close on Windows).
fn tempfile_in(dir: &Path) -> Result<File> {
    let p = dir.join(format!("nzip-test-{}.tmp", std::process::id()));
    let mut o = std::fs::OpenOptions::new();
    o.read(true).write(true).create_new(true);
    // Windows can't unlink an open file: ask the OS to delete it when it is closed.
    #[cfg(windows)]
    std::os::windows::fs::OpenOptionsExt::custom_flags(&mut o, 0x0400_0000); // FILE_FLAG_DELETE_ON_CLOSE
    let f = o.open(&p).with_context(|| format!("creating {}", p.display()))?;
    #[cfg(not(windows))]
    let _ = std::fs::remove_file(&p);
    Ok(f)
}

fn to_device(d: DeviceArg) -> nzip::Device {
    match d {
        DeviceArg::Auto => nzip::Device::Auto,
        DeviceArg::Gpu => nzip::Device::Gpu,
        DeviceArg::Cpu => nzip::Device::Cpu,
    }
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / 1e6 / secs.max(1e-9)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Compress { input, output, mode, level, block, threads, device, memory, no_dedup, no_transforms, force, quiet } => {
            let opt = Options {
                mode: match mode { ModeArg::Store => Mode::Store, ModeArg::Fast => Mode::Fast, ModeArg::Max => Mode::Max, ModeArg::CoreTest => Mode::CoreTest },
                level, block_size: block.unwrap_or(0), threads, dedup: !no_dedup, transforms: !no_transforms,
                device: to_device(device), mem_limit: memory.unwrap_or(0), dedup_segment: 0, dir_archive: false, unpack: true,
            };
            let is_dir = input.is_dir();
            let out_path = match output {
                Some(p) => p,
                None => with_suffix(&input, ".nz")?,
            };
            if out_path.exists() && !force {
                bail!("{} exists (use -f to overwrite)", out_path.display());
            }
            // A folder is packed first: in memory when it fits the memory budget, otherwise
            // into a temporary file beside the output.
            let mut _packed: Option<TempPath> = None;
            let src: Source = if is_dir {
                let parent = out_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
                let (din, dout) = (std::fs::canonicalize(&input)?, std::fs::canonicalize(parent)?);
                if dout.starts_with(&din) {
                    bail!("the output can't be inside the folder being compressed");
                }
                let items = scan_dir(&input)?;
                let (mut files, mut bytes) = (0u64, 0u64);
                for (_, _, k) in &items {
                    if let Kind::File(n) = k {
                        files += 1;
                        bytes += n;
                    }
                }
                if !quiet {
                    eprintln!("{}: {} files, {}", input.display(), files, human(bytes));
                }
                let est = bytes as usize + items.len() * 1536 + (1 << 20);
                let meter = Meter::new("packing    ", bytes, quiet);
                let (src, skipped) = if est <= nzip::memory_budget(opt.mem_limit) / 2 {
                    let (v, sk) = pack_items(&items, Vec::with_capacity(est), &meter)?;
                    (Source::Mem(v), sk)
                } else {
                    let tar_path = out_path.with_extension("nzip-tar");
                    _packed = Some(TempPath(tar_path.clone()));
                    let f = File::create(&tar_path).with_context(|| format!("creating {}", tar_path.display()))?;
                    let (w, sk) = pack_items(&items, BufWriter::with_capacity(8 << 20, f), &meter)?;
                    w.into_inner().map_err(|e| e.into_error())?;
                    (Source::Map(read_input(&tar_path)?), sk)
                };
                meter.finish();
                if skipped > 0 && !quiet {
                    eprintln!("note: skipped {skipped} sockets, pipes or device files");
                }
                src
            } else {
                Source::Map(read_input(&input)?)
            };
            let opt = Options { dir_archive: is_dir, ..opt };
            let data: &[u8] = &src;
            let (tmp, f) = create_tmp(&out_path, force)?;
            let t = Instant::now();
            let mut w = BufWriter::with_capacity(8 << 20, f);
            let meter = Meter::new("compressing", data.len() as u64, quiet);
            let res = nzip::compress_to_progress(data, &opt, &mut w, &|n| meter.add(n));
            meter.finish();
            let comp_len = match res.and_then(|n| w.into_inner().map_err(|e| e.into_error().into()).map(|_| n)) {
                Ok(n) => n,
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(e.into());
                }
            };
            let secs = t.elapsed().as_secs_f64();
            std::fs::rename(&tmp, &out_path)?;
            if !quiet {
                let gpu = match opt.mode {
                    Mode::Max => nzip::gpu_blocks_used() > 0,
                    _ => opt.gpu_used(data.len()),
                };
                let dev = if gpu { "GPU+CPU" } else { "CPU" };
                eprintln!("{}{} -> {}: {} -> {} bytes ({:.3} bits/byte), {:.1} MB/s [{dev}]",
                    input.display(), if is_dir { " (folder)" } else { "" }, out_path.display(), data.len(), comp_len,
                    if data.is_empty() { 0.0 } else { 8.0 * comp_len as f64 / data.len() as f64 }, mbps(data.len(), secs));
                if matches!(opt.mode, Mode::Max) && !matches!(device, DeviceArg::Cpu) && data.len() >= 64 << 20 {
                    if let Some(note) = gpu_note() {
                        eprintln!("nzip: {note}");
                    }
                }
            }
        }
        Cmd::Decompress { input, output, threads, device, memory, force, quiet } => {
            let out_path = match output {
                Some(p) => p,
                None => match input.extension() {
                    Some(e) if e == "nz" => input.with_extension(""),
                    _ => bail!("can't derive output name (input doesn't end in .nz); give one explicitly"),
                },
            };
            let data: &[u8] = &read_input(&input)?;
            let (fh, _) = nzip::info(data)?;
            let is_dir = fh.flags & nzip::format::FLAG_DIR != 0;
            if is_dir && out_path.exists() && !(force && out_path.is_dir()) {
                bail!("{} exists (use -f to extract into an existing folder)", out_path.display());
            }
            let t = Instant::now();
            let dev = to_device(device);
            let meter = Meter::new("decompressing", fh.orig_len, quiet);
            let n = if is_dir && fh.orig_len as usize <= nzip::memory_budget(memory.unwrap_or(0)) / 2 {
                // Small enough: decode the packed folder in memory, then extract.
                let mut buf = Vec::with_capacity(fh.orig_len as usize);
                let n = nzip::decompress_to_progress(data, &mut buf, threads, dev, memory.unwrap_or(0), &|n| meter.add(n))?;
                meter.finish();
                extract_dir(std::io::Cursor::new(&buf), &out_path, fh.orig_len, quiet)?;
                n
            } else {
                let target = if is_dir { out_path.with_extension("nzip-tar") } else { out_path.clone() };
                let (tmp, mut f) = create_tmp(&target, true)?;
                let guard = TempPath(tmp.clone());
                let n = nzip::decompress_to_progress(data, &mut f, threads, dev, memory.unwrap_or(0), &|n| meter.add(n))?;
                meter.finish();
                drop(f);
                if is_dir {
                    extract_dir(std::io::BufReader::with_capacity(8 << 20, File::open(&tmp)?), &out_path, fh.orig_len, quiet)?;
                } else {
                    std::fs::rename(&tmp, &out_path)?;
                    std::mem::forget(guard);
                }
                n
            };
            let secs = t.elapsed().as_secs_f64();
            if !quiet {
                eprintln!("{} -> {}{}: {} bytes, {:.1} MB/s", input.display(), out_path.display(), if is_dir { " (folder)" } else { "" },
                    n, mbps(n as usize, secs));
            }
        }
        Cmd::Test { input, threads, memory } => {
            let data: &[u8] = &read_input(&input)?;
            let (fh, _) = nzip::info(data)?;
            let t = Instant::now();
            let n = if fh.flags & (nzip::format::FLAG_DEDUP | nzip::format::FLAG_UNPACK) != 0 {
                // Dedup copies and unpacked-stream checks read earlier output back: use an
                // anonymous temporary file.
                let dir = std::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
                let mut f = tempfile_in(&dir)?;
                nzip::decompress_to(data, &mut f, threads, nzip::Device::Auto, memory.unwrap_or(0))?
            } else {
                nzip::decompress_to(data, &mut NullSink, threads, nzip::Device::Auto, memory.unwrap_or(0))?
            };
            println!("OK: {} bytes verified, {:.1} MB/s", n, mbps(n as usize, t.elapsed().as_secs_f64()));
        }
        Cmd::Gpu => print_gpu(),
        Cmd::Info { input } => {
            let data: &[u8] = &read_input(&input)?;
            let (fh, frames) = nzip::info(data)?;
            // A closed pipe (e.g. `nzip info f | head`) is a normal way to stop listing.
            let mut o = std::io::stdout().lock();
            let r = (|| -> std::io::Result<()> {
                writeln!(o, "original {} bytes{}, compressed {} bytes, {} frames, dedup: {}",
                    fh.orig_len, if fh.flags & nzip::format::FLAG_DIR != 0 { " (packed folder)" } else { "" },
                    data.len(), fh.n_frames, fh.flags & nzip::format::FLAG_DEDUP != 0)?;
                for (i, f) in frames.iter().enumerate() {
                    let codec = ["store", "zstd", "bsc", "ncore"][f.codec as usize];
                    let kind = if f.kind == nzip::format::KIND_RECIPE { "recipe" } else { "data" };
                    let mut tf = Vec::new();
                    if f.transform & nzip::format::TF_E8E9 != 0 { tf.push("e8e9".to_string()); }
                    if f.transform & nzip::format::TF_REORDER != 0 { tf.push(format!("reorder/{}", f.param)); }
                    writeln!(o, "  #{i:<4} {kind:6} {codec:5} {:>11} -> {:>11}  {}", f.raw_len, f.comp_len, tf.join(","))?;
                }
                Ok(())
            })();
            match r {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
                r => r?,
            }
        }
    }
    Ok(())
}

/// CUDA version number (e.g. 13010) as text ("13.1").
fn cuda_ver(v: i32) -> String {
    format!("{}.{}", v / 1000, v % 1000 / 10)
}

fn gb(b: usize) -> f64 {
    b as f64 / 1e9
}

/// Why a max-mode run left (some of) its sorts to the CPU, if it did.
fn gpu_note() -> Option<String> {
    let s = nzip::gpu_status();
    if !s.built {
        return s.driver_found.then(|| "this nzip was built without GPU support, so it ran on the CPU only (see `nzip gpu`)".into());
    }
    if s.error != 0 || s.devices == 0 {
        return s.driver_found.then(|| format!("GPU not used: {} (CUDA error {}; see `nzip gpu`)", nzip::cuda_error_text(s.error), s.error));
    }
    let (need, avail) = nzip::gpu_shortfall()?;
    Some(format!("some blocks were sorted on the CPU: they needed {:.1} GB of free GPU memory and {:.1} GB was free. \
        Close programs using the GPU, or use -b 512M (smaller blocks fit; slightly larger output).", gb(need), gb(avail)))
}

fn print_gpu() {
    let s = nzip::gpu_status();
    if !s.built {
        println!("GPU code in this build: no (CPU-only build)");
        println!("To use an NVIDIA GPU, build nzip from source with the CUDA Toolkit installed (see the README).");
        return;
    }
    println!("GPU code in this build: yes (CUDA runtime {})", cuda_ver(s.runtime_cuda));
    match (s.driver_found, s.driver_cuda) {
        (false, _) => println!("NVIDIA driver: not found"),
        (true, 0) => println!("NVIDIA driver: found"),
        (true, v) => println!("NVIDIA driver: found, supports CUDA up to {}", cuda_ver(v)),
    }
    if s.error != 0 || s.devices == 0 {
        println!("GPU: not usable: {} (CUDA error {})", nzip::cuda_error_text(s.error), s.error);
        return;
    }
    println!("GPUs: {}", s.devices);
    println!("GPU memory: {:.2} GB free of {:.2} GB", gb(s.free), gb(s.total));
    let m = nzip::gpu_max_sort();
    println!("Largest max-mode sort that fits now: {} MiB (each sorted byte needs ~21 bytes of GPU memory;", m >> 20);
    println!("  a 1 GiB block usually shrinks to 550-750 MiB before the sort; bigger sorts run on the CPU)");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_entry(b: &mut tar::Builder<Vec<u8>>, name: &[u8], kind: tar::EntryType, link: &[u8], data: &[u8]) {
        let mut h = tar::Header::new_gnu();
        h.as_gnu_mut().unwrap().name[..name.len()].copy_from_slice(name);
        if !link.is_empty() {
            h.as_gnu_mut().unwrap().linkname[..link.len()].copy_from_slice(link);
        }
        h.set_entry_type(kind);
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append(&h, data).unwrap();
    }

    /// Hostile archives must never write outside the destination folder.
    #[test]
    fn unpack_refuses_escapes() {
        let base = std::env::temp_dir().join(format!("nzip-escape-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("dest")).unwrap();
        let cases: [(&str, Vec<(&[u8], tar::EntryType, &[u8], &[u8])>); 3] = [
            ("dotdot", vec![(b"../evil_dotdot", tar::EntryType::Regular, b"", b"x")]),
            ("absolute", vec![(b"/tmp/nzip_evil_absolute", tar::EntryType::Regular, b"", b"x")]),
            ("symlink", vec![(b"lnk", tar::EntryType::Symlink, b"..", b""), (b"lnk/evil_symlink", tar::EntryType::Regular, b"", b"x")]),
        ];
        for (name, entries) in cases {
            let mut b = tar::Builder::new(Vec::new());
            for (n, k, l, d) in &entries {
                raw_entry(&mut b, n, *k, l, d);
            }
            let bytes = b.into_inner().unwrap();
            let tar_path = base.join(format!("{name}.tar"));
            std::fs::write(&tar_path, bytes).unwrap();
            let dest = base.join("dest").join(name);
            let meter = Meter::new("", 0, true);
            let _ = unpack_tar(File::open(&tar_path).unwrap(), &dest, &meter); // may error; must not escape
            assert!(!base.join("dest").join("evil_dotdot").exists() && !base.join("evil_dotdot").exists(), "{name}: wrote via ..");
            assert!(!Path::new("/tmp/nzip_evil_absolute").exists(), "{name}: wrote absolute path");
            assert!(!base.join("dest").join("evil_symlink").exists(), "{name}: wrote through symlink");
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
