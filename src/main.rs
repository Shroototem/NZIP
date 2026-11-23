//! nzip: universal lossless compressor.

mod archive;
mod bwtc;
mod cm;
mod codec;
mod coder;
mod container;
mod gpu;
mod lz;
mod lzx;
mod mixer;
mod par;
mod sais;
mod sys;
mod tables;
mod textx;
mod x86;

use container::{FLAG_DIR, Mode, Options};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
nzip 1.0 - universal lossless compressor

USAGE:
  nzip c [options] <file|folder> [-o out.nz]   compress (default output: <input>.nz)
  nzip d [options] <archive.nz>  [-o out]       decompress
  nzip t [options] <archive.nz>                 test archive integrity
  nzip bench [options] <file>                   compress + decompress in memory, verify
  nzip info                                     show detected hardware

MODES:
  (default)     max:  best compression that stays fast (word transform + BWT + neural CM coder)
  --fast        fast: maximum speed (LZ77 on every core)

OPTIONS:
  -t N          threads (default: all logical CPUs, reduced to fit free RAM)
  -b SIZE       block size, e.g. 64M (default: automatic from cores, free RAM and VRAM)
  -m SIZE       memory budget, e.g. 4G (default: 90% of total RAM); fewer blocks run at once
  -f            overwrite existing output
  -v            verbose
";

struct Args {
    cmd: String,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    mode: Mode,
    threads: usize,
    block: Option<usize>,
    force: bool,
    verbose: bool,
}

fn parse_size(s: &str) -> Option<usize> {
    let (num, mul) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1usize << 10),
        'm' | 'M' => (&s[..s.len() - 1], 1 << 20),
        'g' | 'G' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mul)
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or_else(|| USAGE.to_string())?;
    let mut a = Args {
        cmd,
        input: None,
        output: None,
        mode: Mode::Max,
        threads: sys::cpus(),
        block: None,
        force: false,
        verbose: false,
    };
    while let Some(x) = it.next() {
        match x.as_str() {
            "-o" => a.output = Some(it.next().ok_or("-o needs a path")?.into()),
            "-t" => a.threads = it.next().and_then(|v| v.parse().ok()).filter(|&n| n > 0).ok_or("-t needs a number")?,
            "-b" => a.block = Some(it.next().as_deref().and_then(parse_size).filter(|&n| n > 0).ok_or("-b needs a size")?),
            "-m" => {
                let m = it.next().as_deref().and_then(parse_size).filter(|&n| n > 0).ok_or("-m needs a size")?;
                sys::set_mem_limit(m as u64);
            }
            "-f" => a.force = true,
            "-v" => a.verbose = true,
            "--fast" => a.mode = Mode::Fast,
            "--max" => a.mode = Mode::Max,
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ => a.input = Some(x.into()),
        }
    }
    Ok(a)
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn create_out(p: &Path, force: bool) -> io::Result<File> {
    if !force && p.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists (use -f to overwrite)", p.display())));
    }
    File::create(p)
}

fn run(a: Args) -> io::Result<()> {
    let opts = Options { mode: a.mode, threads: a.threads, block_size: a.block, verbose: a.verbose };
    let need_input = || a.input.clone().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing input"));
    match a.cmd.as_str() {
        "c" | "compress" => {
            let input = need_input()?;
            let out_path = a.output.clone().unwrap_or_else(|| {
                let mut s = input.clone().into_os_string();
                s.push(".nz");
                s.into()
            });
            let t0 = Instant::now();
            let out = BufWriter::with_capacity(1 << 20, create_out(&out_path, a.force)?);
            let st = if input.is_dir() {
                let entries = archive::scan(&input)?;
                let (r, total, cuts) = archive::PackReader::new(&entries);
                container::compress(BufReader::with_capacity(1 << 20, r), total, FLAG_DIR, out, &opts, &cuts)?
            } else {
                let f = File::open(&input)?;
                let total = f.metadata()?.len();
                container::compress(BufReader::with_capacity(1 << 20, f), total, 0, out, &opts, &[])?
            };
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "{} -> {}: {} -> {} bytes ({:.3}%, {:.4} bpb) in {:.2}s, {:.2} MB/s [{:?}, {} blocks of {} MB, {} at once]",
                input.display(),
                out_path.display(),
                st.in_bytes,
                st.out_bytes,
                st.out_bytes as f64 * 100.0 / st.in_bytes.max(1) as f64,
                st.out_bytes as f64 * 8.0 / st.in_bytes.max(1) as f64,
                dt,
                mb(st.in_bytes) / dt,
                opts.mode,
                st.blocks,
                st.block_size >> 20,
                st.threads
            );
        }
        "d" | "decompress" | "t" | "test" => {
            let input = need_input()?;
            let test = a.cmd.starts_with('t');
            let t0 = Instant::now();
            let mut r = BufReader::with_capacity(1 << 20, File::open(&input)?);
            let h = container::read_header(&mut r)?;
            let n = if test {
                container::decompress(r, &h, io::sink(), a.threads)?
            } else {
                let out_path = a.output.clone().unwrap_or_else(|| {
                    let s = input.to_string_lossy();
                    PathBuf::from(s.strip_suffix(".nz").map(str::to_string).unwrap_or_else(|| format!("{s}.out")))
                });
                if h.flags & FLAG_DIR != 0 {
                    if !a.force && out_path.exists() {
                        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists (use -f)", out_path.display())));
                    }
                    let mut u = archive::Unpacker::new(&out_path);
                    let n = container::decompress(r, &h, &mut u, a.threads)?;
                    u.finish()?;
                    n
                } else {
                    let out = BufWriter::with_capacity(1 << 20, create_out(&out_path, a.force)?);
                    container::decompress(r, &h, out, a.threads)?
                }
            };
            if n != h.total {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("size mismatch: {n} != {}", h.total)));
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("{}: {} bytes OK in {:.2}s, {:.2} MB/s", input.display(), n, dt, mb(n) / dt);
        }
        "bench" => {
            let input = need_input()?;
            let data = std::fs::read(&input)?;
            let t0 = Instant::now();
            let mut comp = Vec::new();
            let st = container::compress(&data[..], data.len() as u64, 0, &mut comp, &opts, &[])?;
            let tc = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut r = &comp[..];
            let h = container::read_header(&mut r)?;
            let mut back = Vec::with_capacity(data.len());
            container::decompress(r, &h, &mut back, a.threads)?;
            let td = t1.elapsed().as_secs_f64();
            let ok = back == data;
            println!(
                "{}: {} -> {} ({:.4} bpb) | c {:.2}s {:.2} MB/s | d {:.2}s {:.2} MB/s | {:?} blocks {} x {}MB threads {} | {}",
                input.display(),
                data.len(),
                comp.len(),
                comp.len() as f64 * 8.0 / data.len().max(1) as f64,
                tc,
                mb(data.len() as u64) / tc,
                td,
                mb(data.len() as u64) / td,
                opts.mode,
                st.blocks,
                st.block_size >> 20,
                st.threads,
                if ok { "OK" } else { "MISMATCH" }
            );
            if !ok {
                std::process::exit(2);
            }
        }
        "bwtsize" => {
            // dev: text transform + BWT + coder on one block, with timings
            let data = std::fs::read(need_input()?)?;
            let t0 = Instant::now();
            let data = if std::env::var_os("NZ_NOTEXT").is_none() { textx::encode(&data, a.threads).unwrap_or(data) } else { data };
            let t1 = Instant::now();
            let (b, idx) = gpu::bwt(&data);
            let t2 = Instant::now();
            if let Some(p) = std::env::var_os("NZ_DUMP") {
                std::fs::write(p, &b)?;
            }
            let c = bwtc::encode(&b, a.threads);
            let t3 = Instant::now();
            let d = bwtc::decode(&c, b.len(), a.threads).unwrap();
            let t4 = Instant::now();
            let u = sais::unbwt(&d, &idx, a.threads).unwrap();
            let t5 = Instant::now();
            assert!(u == data);
            println!(
                "{} -> {} | text {:.2}s bwt {:.2}s code {:.2}s | decode {:.2}s unbwt {:.2}s",
                data.len(), c.len(),
                (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64(),
                (t4 - t3).as_secs_f64(), (t5 - t4).as_secs_f64()
            );
        }
        "lzsize" => {
            // dev: LZ engine on one file (x86 filter if detected), with timings
            let mut data = std::fs::read(need_input()?)?;
            let orig = data.clone();
            let x86 = x86::looks_like_x86(&data) && std::env::var_os("NOX86").is_none();
            if x86 {
                x86::encode(&mut data);
            }
            let t0 = Instant::now();
            let c = lzx::encode(&data, a.threads, x86);
            let t1 = Instant::now();
            let mut d = lzx::decode(&c, data.len(), a.threads, x86).unwrap();
            let t2 = Instant::now();
            if x86 {
                x86::decode(&mut d);
            }
            assert!(d == orig, "LZ round trip failed");
            println!(
                "{} -> {} (x86 {}) | c {:.2}s {:.1} MB/s | d {:.2}s {:.1} MB/s",
                orig.len(), c.len(), x86,
                (t1 - t0).as_secs_f64(), orig.len() as f64 / 1e6 / (t1 - t0).as_secs_f64(),
                (t2 - t1).as_secs_f64(), orig.len() as f64 / 1e6 / (t2 - t1).as_secs_f64()
            );
        }
        "lzsplit" => {
            // dev: x86 split streams, each through the LZ engine
            let data = std::fs::read(need_input()?)?;
            let t0 = Instant::now();
            let sp = x86::split(&data);
            let cm = lzx::encode(&sp.main, a.threads, false);
            let cf = lzx::encode(&sp.flags, a.threads, false);
            let ca = lzx::encode(&sp.addr, a.threads, false);
            let cj = lzx::encode(&sp.jump, a.threads, false);
            let t1 = Instant::now();
            if std::env::var_os("NZ_ENCONLY").is_some() {
                println!("{} -> {} (main {}) | c {:.2}s", data.len(), cm.len() + cf.len() + ca.len() + cj.len(), cm.len(), (t1 - t0).as_secs_f64());
                return Ok(());
            }
            let m = lzx::decode(&cm, sp.main.len(), a.threads, false).unwrap();
            let f = lzx::decode(&cf, sp.flags.len(), a.threads, false).unwrap();
            let ad = lzx::decode(&ca, sp.addr.len(), a.threads, false).unwrap();
            let jd = lzx::decode(&cj, sp.jump.len(), a.threads, false).unwrap();
            let back = x86::join(&m, &f, &ad, &jd, data.len()).unwrap();
            let t2 = Instant::now();
            assert!(back == data);
            println!(
                "{} -> {} (main {} -> {}, flags {} -> {}, call {} -> {}, jump {} -> {}) | c {:.2}s | d {:.2}s",
                data.len(), cm.len() + cf.len() + ca.len() + cj.len(), sp.main.len(), cm.len(), sp.flags.len(), cf.len(), sp.addr.len(), ca.len(), sp.jump.len(), cj.len(),
                (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64()
            );
        }
        "info" => {
            println!("logical CPUs : {}", sys::cpus());
            println!("RAM          : {:.1} GB total, budget {:.1} GB", sys::total_ram() as f64 / 1e9, sys::mem_budget() as f64 / 1e9);
            println!("AVX2 path    : {}", if codec::has_avx2() { "yes" } else { "no (portable path, same output)" });
            println!("GPU          : {}", gpu::describe());
            let bs = container::choose_block_size(1 << 40, Mode::Max, sys::cpus());
            println!("max block    : {} MB here (decoding it needs ~{} MB RAM)", bs >> 20, container::block_mem(Mode::Max, bs) >> 20);
        }
        _ => {
            eprint!("{USAGE}");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let verbose = a.verbose;
    if let Err(e) = run(a) {
        eprintln!("nzip: {e}");
        std::process::exit(1);
    }
    if verbose {
        if let Some(p) = sys::peak_rss() {
            eprintln!("peak RAM: {} MB", p >> 20);
        }
    }
}
