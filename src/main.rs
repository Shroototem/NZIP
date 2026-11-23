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
mod mix;
mod outfile;
mod mixer;
mod par;
mod progress;
mod rezip;
mod sais;
mod sys;
mod tables;
mod textx;
mod x86;
mod zlib;

use container::{FLAG_DIR, Mode, Options};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "\
nzip 1.0 - universal lossless compressor

USAGE:
  nzip -c [options] <file|folder> [-o out.nz]   compress (default output: <input>.nz)
  nzip -d [options] <archive.nz>  [-o out]       decompress
  nzip -t [options] <archive.nz>                 test archive integrity
  nzip --bench [options] <file>                  compress + decompress in memory, verify
  nzip --info                                    show detected hardware

MODES:
  (default)     max:  best compression that stays fast (word transform + BWT + neural CM coder)
  --fast        fast: maximum speed (LZ77 on every core)

OPTIONS:
  --cpu PCT     share of logical CPUs to use, 1-100 (default 99, so the desktop
                keeps a core)
  -T N          exact number of threads (overrides --cpu)
  -p            show a progress bar
  -b SIZE       block size, e.g. 64M (default: automatic from cores, free RAM and VRAM)
  --gpu PCT     share of free GPU memory to use, 0-100 (default 99; 0 = CPU only)
  --ram PCT     share of free RAM to use, 1-100 (default 99); fewer blocks run at once
                when it is low
  -m SIZE       exact memory budget, e.g. 4G (overrides --ram)
  -f            overwrite existing output
  -v            verbose
";

struct Args {
    cmd: String,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    mode: Mode,
    threads: usize,
    progress: bool,
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
    let mut it = std::env::args().skip(1).peekable();
    // a bare first word selects a developer command (lzsize, bwtsize, ...)
    let dev = match it.peek() {
        Some(w) if matches!(w.as_str(), "bwtsize" | "lzsize" | "lzsplit" | "mixsize" | "lzstats" | "rezip") => it.next(),
        _ => None,
    };
    let mut a = Args {
        cmd: dev.unwrap_or_default(),
        input: None,
        output: None,
        mode: Mode::Max,
        threads: 0,
        progress: false,
        block: None,
        force: false,
        verbose: false,
    };
    let mut cpu_pct = 99usize;
    let set_cmd = |a: &mut Args, c: &str| -> Result<(), String> {
        if !a.cmd.is_empty() && a.cmd != c {
            return Err(format!("choose one of -c, -d, -t, --bench, --info\n\n{USAGE}"));
        }
        a.cmd = c.to_string();
        Ok(())
    };
    while let Some(x) = it.next() {
        match x.as_str() {
            "-c" | "--compress" => set_cmd(&mut a, "c")?,
            "-d" | "--decompress" => set_cmd(&mut a, "d")?,
            "-t" | "--test" => set_cmd(&mut a, "t")?,
            "--bench" => set_cmd(&mut a, "bench")?,
            "--info" => set_cmd(&mut a, "info")?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            "-o" => a.output = Some(it.next().ok_or("-o needs a path")?.into()),
            "--cpu" => {
                let p: usize = it.next().and_then(|v| v.trim_end_matches('%').parse().ok()).filter(|p| (1..=100).contains(p)).ok_or("--cpu needs a percentage 1-100")?;
                cpu_pct = p;
            }
            "--gpu" => {
                let p: u64 = it.next().and_then(|v| v.trim_end_matches('%').parse().ok()).filter(|p| *p <= 100).ok_or("--gpu needs a percentage 0-100")?;
                gpu::set_share(p);
            }
            "--ram" => {
                let p: u64 = it.next().and_then(|v| v.trim_end_matches('%').parse().ok()).filter(|p| (1..=100).contains(p)).ok_or("--ram needs a percentage 1-100")?;
                sys::set_ram_share(p);
            }
            "-p" | "--progress" => a.progress = true,
            "-T" => a.threads = it.next().and_then(|v| v.parse().ok()).filter(|&n| n > 0).ok_or("-T needs a number")?,
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
    if a.cmd.is_empty() {
        return Err(USAGE.to_string());
    }
    if a.threads == 0 {
        a.threads = (sys::cpus() * cpu_pct / 100).max(1);
    }
    Ok(a)
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn check_len(n: u64, total: u64) -> io::Result<()> {
    if n != total {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("size mismatch: {n} != {total}")));
    }
    Ok(())
}

fn run(a: Args) -> io::Result<()> {
    let opts = Options { mode: a.mode, threads: a.threads, block_size: a.block, verbose: a.verbose };
    let need_input = || a.input.clone().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing input"));
    match a.cmd.as_str() {
        "c" | "compress" => {
            let input = need_input()?;
            let out_path = match a.output.clone() {
                Some(p) => p,
                None => outfile::archive_name(&input)?,
            };
            let t0 = Instant::now();
            let (guard, file) = outfile::OutFile::create(&out_path, a.force)?;
            let mut out = BufWriter::with_capacity(1 << 20, file);
            let mut dup_bytes = 0u64;
            let (src, total, flags, cuts): (Box<dyn io::Read>, u64, u8, Vec<u64>) = if input.is_dir() {
                // never pack the archive being written, wherever -o puts it
                let mut entries = archive::scan(&input, Some(guard.temp_path()))?;
                dup_bytes = archive::find_dups(&mut entries, opts.threads)?;
                if opts.verbose && dup_bytes > 0 {
                    eprintln!("  duplicate files: {dup_bytes} bytes stored once");
                }
                let (r, total, cuts) = archive::PackReader::new(&entries);
                (Box::new(r), total, FLAG_DIR, cuts)
            } else {
                let f = File::open(&input)?;
                let total = f.metadata()?.len();
                (Box::new(f), total, 0, Vec::new())
            };
            let bar = progress::start("compressing", total, a.progress);
            let st = container::compress(BufReader::with_capacity(1 << 20, src), total, flags, &mut out, &opts, &cuts)?;
            out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            guard.commit()?;
            drop(bar);
            let dt = t0.elapsed().as_secs_f64();
            // real input size: duplicate files count although they are stored once
            let in_bytes = st.in_bytes + dup_bytes;
            eprintln!(
                "{} -> {}: {} -> {} bytes ({:.3}%, {:.4} bpb) in {:.2}s, {:.2} MB/s [{:?}, {} blocks (largest {} MB), {} threads]",
                input.display(),
                out_path.display(),
                in_bytes,
                st.out_bytes,
                st.out_bytes as f64 * 100.0 / in_bytes.max(1) as f64,
                st.out_bytes as f64 * 8.0 / in_bytes.max(1) as f64,
                dt,
                mb(in_bytes) / dt,
                opts.mode,
                st.blocks,
                st.largest.div_ceil(1 << 20),
                opts.threads
            );
        }
        "d" | "decompress" | "t" | "test" => {
            let input = need_input()?;
            let test = a.cmd.starts_with('t');
            let t0 = Instant::now();
            let mut r = BufReader::with_capacity(1 << 20, File::open(&input)?);
            let h = container::read_header(&mut r)?;
            let bar = progress::start(if test { "testing" } else { "decompressing" }, h.total, a.progress);
            // extra: bytes of duplicate files, copied rather than decoded
            let (n, extra) = if test {
                (container::decompress(r, &h, io::sink(), a.threads)?, 0)
            } else {
                let out_path = a.output.clone().unwrap_or_else(|| outfile::unpack_name(&input));
                if h.flags & FLAG_DIR != 0 {
                    let guard = outfile::OutDir::create(&out_path, a.force)?;
                    let mut u = archive::Unpacker::new(guard.temp_path());
                    let n = container::decompress(r, &h, &mut u, a.threads)?;
                    let extra = u.finish()?;
                    check_len(n, h.total)?;
                    guard.commit()?;
                    (n, extra)
                } else {
                    let (guard, file) = outfile::OutFile::create(&out_path, a.force)?;
                    let mut out = BufWriter::with_capacity(1 << 20, file);
                    let n = container::decompress(r, &h, &mut out, a.threads)?;
                    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                    check_len(n, h.total)?;
                    guard.commit()?;
                    (n, 0)
                }
            };
            drop(bar);
            if n != h.total {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("size mismatch: {n} != {}", h.total)));
            }
            let dt = t0.elapsed().as_secs_f64();
            let all = n + extra;
            eprintln!("{}: {} bytes OK in {:.2}s, {:.2} MB/s", input.display(), all, dt, mb(all) / dt);
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
        "rezip" => {
            // dev: deflate streams our zlib port rebuilds, and the round trip
            let data = std::fs::read(need_input()?)?;
            let t0 = Instant::now();
            match rezip::expand(&data, a.threads, usize::MAX) {
                None => println!("nothing to expand"),
                Some(x) => {
                    let t1 = Instant::now();
                    let comp: usize = x.recs.iter().map(|r| r.comp).sum();
                    let text: usize = x.pieces.iter().filter(|p| p.kind == mix::TEXT).map(|p| p.len).sum();
                    let mut r = rezip::restore(&x.data, &x.recs, data.len(), a.threads).unwrap();
                    let mut t2 = Instant::now();
                    if std::env::var_os("NZ_RZT").is_some() {
                        for _ in 0..3 {
                            let t = Instant::now();
                            r = rezip::restore(&x.data, &x.recs, data.len(), a.threads).unwrap();
                            eprintln!("  restore again {:.2}s", t.elapsed().as_secs_f64());
                        }
                        t2 = t1 + (Instant::now() - t1) / 4;
                    }
                    println!(
                        "{} -> {} bytes: {} streams, {} deflated -> {} raw ({} text) | expand {:.2}s restore {:.2}s {}",
                        data.len(), x.data.len(), x.recs.len(), comp, x.data.len() - (data.len() - comp), text,
                        (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), if r == data { "OK" } else { "MISMATCH" }
                    );
                    if let Some(p) = std::env::var_os("NZ_DUMP") {
                        std::fs::write(p, &x.data)?;
                    }
                    let mut big: Vec<usize> = x.recs.iter().map(|r| r.raw).collect();
                    big.sort_unstable_by(|a, b| b.cmp(a));
                    println!("largest raw streams: {:?}", &big[..big.len().min(8)]);
                }
            }
        }
        "lzstats" => {
            // dev: cost breakdown of the LZ engine on one file (main stream if x86)
            let data = std::fs::read(need_input()?)?;
            let s = if x86::looks_like_x86(&data) && std::env::var_os("NOX86").is_none() { x86::split_chunks(&data, a.threads).0.main } else { data };
            let c = lzx::encode(&s, a.threads, false, true, 0);
            println!("{} -> {}", s.len(), c.len());
            lzx::stats(&c, s.len(), false);
        }
        "mixsize" => {
            // dev: zip-aware text / binary split; writes both streams next to NZ_DUMP
            let data = std::fs::read(need_input()?)?;
            let Some(p) = mix::zip_pieces(&data) else {
                println!("no split");
                return Ok(());
            };
            let parts = mix::gather(&data, &p);
            assert!(mix::scatter(&parts, &p).unwrap() == data);
            let [t, o, _] = parts;
            let tab = mix::write_table(&p);
            assert!(mix::read_table(&tab, data.len()).unwrap() == p);
            if let Some(d) = std::env::var_os("NZ_DUMP") {
                std::fs::write(PathBuf::from(&d).with_extension("text"), &t)?;
                std::fs::write(PathBuf::from(&d).with_extension("bin"), &o)?;
                std::fs::write(PathBuf::from(&d).with_extension("tab"), &tab)?;
            }
            println!("{} -> text {} bin {} pieces {} table {}", data.len(), t.len(), o.len(), p.len(), tab.len());
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
            let c = lzx::encode(&data, a.threads, x86, x86, 0);
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
            let (sp, lens) = x86::split_chunks(&data, a.threads);
            if let Some(d) = std::env::var_os("NZ_DUMP") {
                let d = PathBuf::from(d);
                for (n, v) in [("main", &sp.main), ("flags", &sp.flags), ("call", &sp.addr), ("jump", &sp.jump)] {
                    std::fs::write(d.join(n), v)?;
                }
            }
            let cm = lzx::encode(&sp.main, a.threads, false, true, 0);
            let cf = lzx::encode(&sp.flags, a.threads, false, true, 0);
            let ca = lzx::encode(&sp.addr, a.threads, false, true, 0);
            let cj = lzx::encode(&sp.jump, a.threads, false, true, 0);
            let t1 = Instant::now();
            if std::env::var_os("NZ_ENCONLY").is_some() {
                println!("{} -> {} (main {}) | c {:.2}s", data.len(), cm.len() + cf.len() + ca.len() + cj.len(), cm.len(), (t1 - t0).as_secs_f64());
                return Ok(());
            }
            let m = lzx::decode(&cm, sp.main.len(), a.threads, false).unwrap();
            let f = lzx::decode(&cf, sp.flags.len(), a.threads, false).unwrap();
            let ad = lzx::decode(&ca, sp.addr.len(), a.threads, false).unwrap();
            let jd = lzx::decode(&cj, sp.jump.len(), a.threads, false).unwrap();
            let back = x86::join_chunks([&m, &f, &ad, &jd], &lens, data.len(), a.threads).unwrap();
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
            println!("RAM          : {:.1} GB total, {:.1} GB free, budget {:.1} GB", sys::total_ram() as f64 / 1e9, sys::free_ram() as f64 / 1e9, sys::mem_budget() as f64 / 1e9);
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
    #[cfg(target_os = "linux")]
    if std::env::var_os("NZ_PTRACE").is_some() {
        // dev: let a sampling profiler (gdb) attach to this process
        unsafe extern "C" {
            fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        }
        unsafe { prctl(0x5961_6d61, u64::MAX, 0, 0, 0) };
    }
    outfile::install_handler();
    sys::no_throttle();
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let verbose = a.verbose;
    par::set_threads(a.threads);
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
