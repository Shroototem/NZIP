//! nzip: block-parallel lossless compressor.
//!
//! Pipeline (compress): the input is processed in **segments** sized from available RAM,
//! so inputs of any size (10+ GB) run in bounded memory. Per segment: optional
//! long-range dedup (copy references may point anywhere earlier in the input) -> split
//! the unique bytes into blocks -> per block (in bounded parallel groups):
//! incompressibility probe, transforms, codec, fall back to store if it didn't help ->
//! frames are written out as soon as the segment is done. The file header (sizes,
//! whole-file checksum, frame count) is written last.
//!
//! Decompress streams frames in memory-bounded batches to a [`Sink`] (a file for the
//! CLI), verifying checksums at every level. Dedup copies read earlier output back from
//! the sink, so no whole-file buffer is ever needed.

pub mod codec;
pub mod dedup;
mod dict;
mod expand;
pub mod format;
mod sched;
pub mod transform;

use codec::Codec;
use format::*;
use rayon::prelude::*;
use std::io::{Seek, SeekFrom, Write};
use xxhash_rust::xxh3::{xxh3_64, Xxh3Default};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("format error: {0}")]
    Format(&'static str),
    #[error("corrupt data: {0}")]
    Corrupt(&'static str),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("input too large: {0}")]
    TooLarge(&'static str),
    #[error("GPU requested but none is usable (built with GPU support: {built})")]
    NoGpu { built: bool },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Where compute-heavy stages run. `Auto` picks the fastest available hardware per
/// stage (measured): GPU when present and faster, CPU otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

/// Inputs smaller than this compress on the CPU under `Device::Auto` (GPU start-up
/// cost ~150 ms; measured crossover ~16 MB).
pub const GPU_MIN_INPUT: usize = 16 << 20;
/// bsc blocks smaller than this decompress fully on the CPU under `Device::Auto`. The
/// GPU inverse BWT only saves ~1 ms/MB, so start-up is repaid above ~60 MB (measured:
/// 51 MB block 390 ms GPU vs 376 ms CPU; 100 MB block +15% on GPU).
pub const GPU_MIN_DECODE_BLOCK: usize = 64 << 20;
/// ncore inputs/frames smaller than this are coded on the CPU under `Device::Auto`.
/// GPU cost is ~0.1 s CUDA start-up plus one lane's serial latency (~0.18 s for 64 KiB
/// lanes, however few lanes there are), against ~33 MB/s on 24 CPU threads. Measured
/// end to end: 4 MB 143 ms CPU vs 215 ms GPU; 8 MB 265 ms CPU vs 226 ms GPU.
pub const NCORE_GPU_MIN: usize = 6 << 20;

/// Max mode's block size when memory allows.
pub const MAX_MODE_BLOCK: usize = 1 << 30;

pub use bsc_sys::{cuda_error_text, GpuStatus};

/// GPU support of this build and machine, for diagnostics.
pub fn gpu_status() -> GpuStatus {
    bsc_sys::gpu_status()
}

/// Largest max-mode sort (bytes after the LZP stage) that fits the GPU's free memory
/// right now; larger ones run on the CPU, with identical output.
pub fn gpu_max_sort() -> usize {
    sched::max_gpu_sort()
}

/// If this run sorted some max-mode block on the CPU because the GPU lacked free memory:
/// (largest memory need, memory available then), in bytes.
pub fn gpu_shortfall() -> Option<(usize, usize)> {
    sched::shortfall()
}

/// True if a usable GPU is present for nzip's GPU stages.
pub fn gpu_available() -> bool {
    bsc_sys::gpu_available()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// No compression; still checksummed.
    Store,
    /// zstd (default level 3).
    Fast,
    /// Best compression (the default). Today: libbsc (LZP + BWT + adaptive QLFC) with
    /// the largest blocks memory allows (1 GiB); BWT on the GPU when a block fits in VRAM,
    /// on the CPU otherwise. The neural + context-mixing tier will replace it once it wins.
    Max,
    /// Experimental: GPU entropy core (ncore) with its order-2 test model. Exercises the
    /// Phase 2 GPU pipeline end to end; not a competitive ratio.
    CoreTest,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub mode: Mode,
    /// zstd level for Fast mode.
    pub level: i32,
    /// Block size in bytes; 0 = mode default.
    pub block_size: usize,
    /// Worker threads; 0 = all logical CPUs.
    pub threads: usize,
    pub dedup: bool,
    pub transforms: bool,
    pub device: Device,
    /// Host memory budget in bytes for buffers in flight; 0 = auto (half of the RAM
    /// currently available). Bounds memory for inputs of any size.
    pub mem_limit: usize,
    /// Input bytes per dedup segment; 0 = default (max(2 x block, 1 GiB)). Part of the
    /// output layout, never derived from RAM, so files don't depend on the machine.
    pub dedup_segment: usize,
    /// Mark the archive as a packed folder (`FLAG_DIR`); the input is its tar stream.
    pub dir_archive: bool,
    /// Max mode: unpack deflate streams found anywhere (PNG, zip, gzip, ...) and rebuild
    /// them exactly on decode (`expand.rs`).
    pub unpack: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { mode: Mode::Max, level: 3, block_size: 0, threads: 0, dedup: true, transforms: true, device: Device::Auto, mem_limit: 0, dedup_segment: 0, dir_archive: false, unpack: true }
    }
}

impl Options {
    /// Maximum block size for this run.
    ///
    /// Max mode always uses 1 GiB blocks, the best ratio (enwik9: 1.441 bpb at 128 MiB,
    /// 1.356 at 512 MiB, 1.314 at 1 GiB). Each block's sort runs on the GPU when its LZP
    /// output fits the card's free memory (~21 bytes per byte); a block that doesn't is
    /// split into sub-blocks that do (`sched::bsc_block`), so smaller cards trade a little
    /// ratio for GPU speed. Only a memory budget too small for one block (~7 bytes per
    /// byte) shrinks it. Any machine decodes any archive; decoding needs no GPU.
    pub fn max_block_size(&self, _gpu: bool) -> usize {
        let b = match (self.block_size, self.mode) {
            (0, Mode::Fast) => 4 << 20,
            (0, Mode::Max) => {
                let want = MAX_MODE_BLOCK;
                let fit = memory_budget(self.mem_limit) / work_factor(Mode::Max) / (64 << 20) * (64 << 20);
                want.min(fit.max(64 << 20))
            }
            (0, Mode::Store) => 16 << 20,
            (0, Mode::CoreTest) => 256 << 20,
            (b, _) => b,
        };
        b.clamp(64 * 1024, MAX_RAW_BLOCK)
    }

    /// Resolve the device choice for this run.
    ///
    /// Per-stage choices (measured on RTX 5060 Ti vs Ryzen 9 3900X, see bench/PHASE1.md):
    /// * Max compress: BWT on GPU when the block fits in VRAM (2-3.6x faster, identical output).
    /// * Fast (zstd) compress: CPU. nvCOMP's GPU zstd is no faster here and compresses worse.
    /// * All decompression: CPU (bsc GPU inverse is slower than the 24-thread CPU path).
    ///
    /// Auto also uses the CPU for inputs under [`GPU_MIN_INPUT`]: GPU start-up costs
    /// ~150 ms, which only pays off above ~16 MB (measured crossover). The GPU is not
    /// even probed in that case, since probing initialises the CUDA runtime.
    ///
    /// Core-test mode uses the GPU for inputs of at least [`NCORE_GPU_MIN`].
    pub fn use_gpu(&self, input_len: usize) -> Result<bool, Error> {
        if self.mode == Mode::CoreTest {
            return match self.device {
                Device::Cpu => Ok(false),
                Device::Auto => Ok(input_len >= NCORE_GPU_MIN && ncore::gpu::likely_present()),
                Device::Gpu if ncore::gpu::available() => Ok(true),
                Device::Gpu => Err(Error::NoGpu { built: bsc_sys::built_with_gpu() }),
            };
        }
        if self.mode != Mode::Max {
            return match self.device {
                Device::Gpu => Err(Error::Codec("this mode has no GPU path that beats the CPU; use max or --device auto".into())),
                _ => Ok(false),
            };
        }
        match self.device {
            Device::Cpu => Ok(false),
            Device::Auto => Ok(input_len >= GPU_MIN_INPUT && gpu_available()),
            Device::Gpu if gpu_available() => Ok(true),
            Device::Gpu => Err(Error::NoGpu { built: bsc_sys::built_with_gpu() }),
        }
    }

    /// Whether a run on `input_len` bytes actually had a usable GPU (for reporting; call
    /// after [`compress`], when the probe result is cached).
    pub fn gpu_used(&self, input_len: usize) -> bool {
        if !self.use_gpu(input_len).unwrap_or(false) {
            return false;
        }
        match self.mode {
            Mode::CoreTest => ncore::gpu::warmup(),
            // Max mode sorts a block on the CPU when the card can't take it right now.
            Mode::Max => gpu_blocks_used() > 0,
            _ => gpu_available(),
        }
    }

    fn codec(&self, gpu: bool) -> Codec {
        match self.mode {
            Mode::Store => Codec::Store,
            Mode::Fast => Codec::Zstd { level: self.level.clamp(-7, 22) },
            Mode::Max => Codec::Bsc { coder: bsc_sys::CODER_QLFC_ADAPTIVE, gpu },
            Mode::CoreTest => Codec::Ncore { gpu },
        }
    }
}

fn thread_count(t: usize) -> usize {
    if t == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { t }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    // Each call builds a fresh pool sized for this run.
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool")
}

/// Threads each block may use internally, mirroring the bsc CLI policy: if there are
/// enough blocks to keep every core busy, blocks run single-threaded in parallel;
/// otherwise each block also uses OpenMP threads inside libbsc.
fn inner_threads(total: usize, blocks: usize) -> usize {
    if blocks == 0 || total < 2 * blocks { 1 } else { total / blocks }
}

/// Cap a block's inner threads by its size (~1 thread per 256 KiB): spinning up 24 OpenMP
/// threads for a 150 KB block costs more than it saves.
fn inner_for(len: usize, inner: usize) -> usize {
    inner.min(len.div_ceil(256 << 10)).max(1)
}

/// Total worker threads worth starting for `bytes` of input (~1 per 256 KiB).
fn useful_threads(threads: usize, bytes: usize) -> usize {
    threads.min(bytes.div_ceil(256 << 10)).max(1)
}

/// Memory budget for buffers in flight: `limit` if set, else half of the memory this
/// process can actually use: the smaller of the system's available RAM (Linux
/// `MemAvailable`) and the room left under a cgroup v2 limit (containers, systemd
/// scopes). 4 GiB if neither is known; at least 256 MiB.
pub fn memory_budget(limit: usize) -> usize {
    if limit > 0 {
        return limit.max(16 << 20);
    }
    let avail = available_ram();
    let usable = match (avail, cgroup_room()) {
        (Some(a), Some(c)) => Some(a.min(c)),
        (a, c) => a.or(c),
    };
    usable.map(|b| b / 2).unwrap_or(4 << 30).max(256 << 20)
}

/// RAM the system can give us now: Linux `MemAvailable`, Windows `ullAvailPhys`.
#[cfg(not(windows))]
fn available_ram() -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok()
    }).map(|kb| kb.saturating_mul(1024))
}

#[cfg(windows)]
fn available_ram() -> Option<usize> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, memory_load: 0, total_phys: 0, avail_phys: 0,
                                 total_page_file: 0, avail_page_file: 0, total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0 };
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then(|| m.avail_phys.min(usize::MAX as u64) as usize)
}

/// Bytes left under the tightest cgroup v2 `memory.max` on this process's cgroup path.
fn cgroup_room() -> Option<usize> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = cg.lines().find_map(|l| l.strip_prefix("0::"))?.trim().trim_start_matches('/').to_string();
    let mut dir = std::path::PathBuf::from("/sys/fs/cgroup").join(rel);
    let mut best: Option<usize> = None;
    loop {
        let max = std::fs::read_to_string(dir.join("memory.max")).ok();
        let cur = std::fs::read_to_string(dir.join("memory.current")).ok();
        if let (Some(m), Some(c)) = (max, cur) {
            if let (Ok(m), Ok(c)) = (m.trim().parse::<usize>(), c.trim().parse::<usize>()) {
                let room = m.saturating_sub(c);
                best = Some(best.map_or(room, |b| b.min(room)));
            }
        }
        if !dir.pop() || dir.as_os_str() == "/sys/fs" {
            break;
        }
    }
    best
}

/// Host bytes a block's codec needs while it runs, per input byte (rough upper bounds:
/// libbsc on the CPU needs ~5n for LZP+BWT+QLFC; we also hold the input copy for
/// transforms and the output).
fn work_factor(mode: Mode) -> usize {
    match mode {
        Mode::Max => 7,
        Mode::Fast | Mode::CoreTest => 3,
        Mode::Store => 2,
    }
}

/// Input bytes per dedup segment. Fixed (not derived from RAM), so the compressed file is
/// byte-identical whatever the memory budget; memory only decides how much runs at once.
fn dedup_segment(bs: usize, opt: &Options) -> usize {
    if opt.dedup_segment > 0 { opt.dedup_segment.max(bs) } else { (2 * bs).max(1 << 30) }
}

struct EncodedFrame {
    header: FrameHeader,
    payload: Vec<u8>,
}

/// Number of blocks sorted on the GPU so far in this process.
pub fn gpu_blocks_used() -> u64 {
    sched::GPU_BLOCKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `cpu`: the run's full-size pool, for data-parallel transforms inside one block (the
/// caller may be on a small per-block pool).
fn encode_block(block: &[u8], opt: &Options, inner: usize, total: usize, gpu: bool, cpu: &rayon::ThreadPool) -> Result<EncodedFrame, Error> {
    let raw_xxh3 = xxh3_64(block);
    let store = |block: &[u8]| EncodedFrame {
        header: FrameHeader { kind: KIND_DATA, codec: CODEC_STORE, transform: 0, param: 0, raw_len: block.len() as u32,
                              comp_len: block.len() as u32, raw_xxh3, comp_xxh3: xxh3_64(block) },
        payload: block.to_vec(),
    };
    let codec = opt.codec(gpu);
    if codec == Codec::Store || codec::looks_incompressible(block) {
        return Ok(store(block));
    }
    let mut transform = 0u8;
    let mut param = 0u32;
    let mut work: Option<Vec<u8>> = None;
    if opt.transforms {
        if transform::should_e8e9(block) {
            let mut t = block.to_vec();
            transform::e8e9_encode(&mut t);
            work = Some(t);
            transform |= TF_E8E9;
        }
        // Text: replace frequent words with short codes (smaller and faster to sort).
        if matches!(codec, Codec::Bsc { .. }) && work.is_none() {
            if let Some(t) = cpu.install(|| dict::encode(block)) {
                param = t.len() as u32;
                work = Some(t);
                transform |= TF_DICT;
            }
        }
        if matches!(codec, Codec::Bsc { .. }) && transform & TF_DICT == 0 {
            let cur: &[u8] = work.as_deref().unwrap_or(block);
            // Record detection rarely pays off on small blocks and costs ~1.5 ms each.
            let rs = if cur.len() >= 1 << 20 { bsc_sys::detect_record_size(cur) } else { 1 };
            if rs > 1 {
                let mut t = cur.to_vec();
                bsc_sys::reorder_forward(&mut t, rs).map_err(|e| Error::Codec(e.to_string()))?;
                work = Some(t);
                transform |= TF_REORDER;
                param = rs as u32;
            }
        }
    }
    let input: &[u8] = work.as_deref().unwrap_or(block);
    let payload = match codec {
        // Max mode: staged LZP / BWT / QLFC with per-block GPU or CPU sorting.
        Codec::Bsc { coder, gpu } => sched::bsc_block(input, coder, total, gpu).map_err(|e| Error::Codec(e.to_string()))?.0,
        _ => codec.compress(input, inner)?,
    };
    if payload.len() >= block.len() {
        return Ok(store(block));
    }
    Ok(EncodedFrame {
        header: FrameHeader { kind: KIND_DATA, codec: codec.id(), transform, param, raw_len: block.len() as u32,
                              comp_len: payload.len() as u32, raw_xxh3, comp_xxh3: xxh3_64(&payload) },
        payload,
    })
}

/// Compress `input` into a complete nzip container in memory.
pub fn compress(input: &[u8], opt: &Options) -> Result<Vec<u8>, Error> {
    let mut cur = std::io::Cursor::new(Vec::new());
    compress_to(input, opt, &mut cur)?;
    Ok(cur.into_inner())
}

/// Compress `input` into `w`, streaming: frames are written as soon as each wave of
/// blocks is encoded and the file header is filled in at the end (hence `Seek`). Host
/// memory stays within [`memory_budget`] whatever the input size (`input` itself is
/// typically a memory map). The output does not depend on the memory budget or thread
/// count. Returns the number of bytes written.
pub fn compress_to<W: Write + Seek>(input: &[u8], opt: &Options, w: &mut W) -> Result<u64, Error> {
    compress_to_progress(input, opt, w, &|_| {})
}

/// Progress callback: called with the number of original bytes just finished (from
/// worker threads; the calls add up to the input or output size).
pub type Progress<'a> = &'a (dyn Fn(u64) + Sync);

/// [`compress_to`] with a progress callback.
pub fn compress_to_progress<W: Write + Seek>(input: &[u8], opt: &Options, w: &mut W, progress: Progress<'_>) -> Result<u64, Error> {
    if opt.unpack && opt.mode == Mode::Max && opt.transforms && input.len() >= 4096 {
        if let Some(n) = expand::compress(input, opt, w, progress)? {
            return Ok(n);
        }
    }
    let threads = useful_threads(thread_count(opt.threads), input.len());
    let gpu = opt.use_gpu(input.len())?;
    let bs = opt.max_block_size(gpu);
    let budget = memory_budget(opt.mem_limit);
    // Half the budget for codec working memory of blocks in flight, a quarter for a wave's
    // input-side buffers and a quarter for its encoded frames.
    let active = ((budget / 2) / (bs * work_factor(opt.mode))).clamp(1, threads);
    let wave_bytes = (budget / 4).max(bs);
    let dedup = opt.dedup && opt.mode != Mode::Store && input.len() >= 2 * bs;
    let pool = pool(threads);
    let start = w.stream_position()?;
    w.write_all(&[0u8; FILE_HEADER_LEN])?; // placeholder, patched at the end
    let mut written = FILE_HEADER_LEN as u64;
    let mut n_frames = 0u32;

    // Encode a list of blocks in groups of `active` (bounded codec memory).
    let encode_all = |blocks: &[&[u8]]| -> Result<Vec<EncodedFrame>, Error> {
        if opt.mode == Mode::Max {
            // Stream blocks through `active` slots (no barrier between groups) so the GPU
            // sort and the CPU stages of other blocks overlap; threads are shared inside.
            let jobs = rayon::ThreadPoolBuilder::new().num_threads(active.min(blocks.len()).max(1)).build().expect("thread pool");
            return jobs.install(|| blocks.par_iter().with_max_len(1).map(|b| {
                let r = encode_block(b, opt, 1, threads, gpu, &pool);
                progress(b.len() as u64);
                r
            }).collect());
        }
        let mut out = Vec::with_capacity(blocks.len());
        for group in blocks.chunks(active) {
            let inner = inner_threads(threads, group.len());
            let enc: Result<Vec<EncodedFrame>, Error> =
                pool.install(|| group.par_iter().map(|b| {
                    let r = encode_block(b, opt, inner_for(b.len(), inner), threads, gpu, &pool);
                    progress(b.len() as u64);
                    r
                }).collect());
            out.extend(enc?);
        }
        Ok(out)
    };
    let mut emit = |frames: &[EncodedFrame], w: &mut W| -> Result<(), Error> {
        let mut buf = Vec::with_capacity(FRAME_HEADER_LEN);
        for f in frames {
            buf.clear();
            f.header.write(&mut buf);
            w.write_all(&buf)?;
            w.write_all(&f.payload)?;
            written += (FRAME_HEADER_LEN + f.payload.len()) as u64;
        }
        n_frames = n_frames.checked_add(frames.len() as u32).ok_or(Error::TooLarge("frame count"))?;
        Ok(())
    };
    // Split evenly (e.g. 1000 MB at a 512 MiB max -> 2 x 500 MB, not 512 + 488) so blocks
    // finish together and the GPU/CPU pipeline stays balanced.
    let split = |d: &[u8]| -> usize { d.len().div_ceil(d.len().div_ceil(bs).max(1)).max(1) };

    // CUDA start-up (~0.1 s) and the whole-input hash run on their own threads, hidden
    // behind the CPU stages; the first GPU call waits for the warm-up.
    let orig_xxh3 = std::thread::scope(|s| -> Result<u64, Error> {
        if gpu {
            s.spawn(|| gpu_warmup(opt.mode));
        }
        let hash = s.spawn(|| xxh3_64(input));
        if !dedup {
            let blocks: Vec<&[u8]> = input.chunks(split(input)).collect();
            let mut i = 0;
            while i < blocks.len() {
                let mut end = i;
                let mut bytes = 0;
                while end < blocks.len() && (end == i || bytes + blocks[end].len() <= wave_bytes) {
                    bytes += blocks[end].len();
                    end += 1;
                }
                let frames = encode_all(&blocks[i..end])?;
                emit(&frames, w)?;
                i = end;
            }
        } else {
            // ~48 bytes per remembered chunk; cap the table at 1/8 of the budget.
            let mut dd = dedup::Deduper::new(bs, budget / 8 / 48);
            let seg_len = dedup_segment(bs, opt);
            let mut s0 = 0usize;
            while s0 < input.len() {
                // A wave of whole segments (at least one).
                let mut segs = Vec::new();
                let mut bytes = 0;
                while s0 < input.len() && (segs.is_empty() || bytes + seg_len.min(input.len() - s0) <= wave_bytes) {
                    let s1 = (s0 + seg_len).min(input.len());
                    let g = pool.install(|| dd.segment(input, s0, s1));
                    progress((s1 - s0 - g.unique.len()) as u64); // bytes replaced by copies
                    segs.push(g);
                    bytes += s1 - s0;
                    s0 = s1;
                }
                let per_seg: Vec<Vec<&[u8]>> = segs.iter().map(|g| g.unique.chunks(split(&g.unique)).collect()).collect();
                let all: Vec<&[u8]> = per_seg.iter().flatten().copied().collect();
                let mut frames = encode_all(&all)?.into_iter();
                for (g, blocks) in segs.iter().zip(&per_seg) {
                    let recipe = dedup::encode_ops(&g.ops);
                    if recipe.len() > MAX_RAW_BLOCK {
                        return Err(Error::TooLarge("dedup recipe"));
                    }
                    let comp = zstd::bulk::compress(&recipe, 19).map_err(|e| Error::Codec(format!("zstd: {e}")))?;
                    let h = FrameHeader { kind: KIND_RECIPE, codec: CODEC_ZSTD, transform: 0, param: blocks.len() as u32,
                                          raw_len: recipe.len() as u32, comp_len: comp.len() as u32,
                                          raw_xxh3: xxh3_64(&recipe), comp_xxh3: xxh3_64(&comp) };
                    let mut seg_frames = vec![EncodedFrame { header: h, payload: comp }];
                    seg_frames.extend(frames.by_ref().take(blocks.len()));
                    emit(&seg_frames, w)?;
                }
            }
        }
        Ok(hash.join().expect("hash thread"))
    })?;
    let mut hdr = Vec::with_capacity(FILE_HEADER_LEN);
    let flags = if dedup { FLAG_DEDUP } else { 0 } | if opt.dir_archive { FLAG_DIR } else { 0 };
    FileHeader { flags, orig_len: input.len() as u64, orig_xxh3, n_frames }.write(&mut hdr);
    w.seek(SeekFrom::Start(start))?;
    w.write_all(&hdr)?;
    w.seek(SeekFrom::Start(start + written))?;
    w.flush()?;
    Ok(written)
}

/// Pay CUDA start-up for the GPU code a mode uses (meant for a background thread).
fn gpu_warmup(mode: Mode) {
    if mode == Mode::CoreTest {
        ncore::gpu::warmup();
    } else {
        bsc_sys::gpu_warmup();
    }
}

fn decode_frame(f: &Frame<'_>, out: &mut [u8], inner: usize, gpu: bool) -> Result<(), Error> {
    let h = &f.header;
    if xxh3_64(f.payload) != h.comp_xxh3 {
        return Err(Error::Corrupt("frame payload checksum mismatch"));
    }
    if h.transform & TF_DICT != 0 {
        let n = h.param as usize;
        if n == 0 || n > MAX_RAW_BLOCK {
            return Err(Error::Corrupt("bad dictionary length"));
        }
        let mut t = vec![0u8; n];
        codec::decompress(h.codec, f.payload, &mut t, inner, gpu)?;
        dict::decode(&t, out).map_err(Error::Corrupt)?;
    } else {
        codec::decompress(h.codec, f.payload, out, inner, gpu)?;
    }
    if h.transform & TF_REORDER != 0 {
        let rs = h.param as usize;
        if rs < 2 {
            return Err(Error::Corrupt("bad record size"));
        }
        bsc_sys::reorder_reverse(out, rs).map_err(|_| Error::Corrupt("reorder reverse failed"))?;
    }
    if h.transform & TF_E8E9 != 0 {
        transform::e8e9_decode(out);
    }
    if xxh3_64(out) != h.raw_xxh3 {
        return Err(Error::Corrupt("block checksum mismatch"));
    }
    Ok(())
}

/// Where decompressed bytes go. Bytes arrive in order (`off` = bytes already written);
/// `read_at` reads back bytes already written (needed only for dedup copies).
pub trait Sink: Send {
    fn write_at(&mut self, off: u64, data: &[u8]) -> Result<(), Error>;
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> Result<(), Error>;
}

impl Sink for Vec<u8> {
    fn write_at(&mut self, off: u64, data: &[u8]) -> Result<(), Error> {
        let o = off as usize;
        if o == self.len() {
            self.extend_from_slice(data);
        } else {
            if o + data.len() > self.len() {
                self.resize(o + data.len(), 0);
            }
            self[o..o + data.len()].copy_from_slice(data);
        }
        Ok(())
    }
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let o = off as usize;
        buf.copy_from_slice(&self[o..o + buf.len()]);
        Ok(())
    }
}

/// A file sink (positional writes and reads).
#[cfg(unix)]
impl Sink for std::fs::File {
    fn write_at(&mut self, off: u64, data: &[u8]) -> Result<(), Error> {
        Ok(std::os::unix::fs::FileExt::write_all_at(self, data, off)?)
    }
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        Ok(std::os::unix::fs::FileExt::read_exact_at(self, buf, off)?)
    }
}

/// A file sink (positional writes and reads).
#[cfg(windows)]
impl Sink for std::fs::File {
    fn write_at(&mut self, mut off: u64, mut data: &[u8]) -> Result<(), Error> {
        use std::os::windows::fs::FileExt;
        while !data.is_empty() {
            let n = self.seek_write(data, off)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into());
            }
            data = &data[n..];
            off += n as u64;
        }
        Ok(())
    }
    fn read_at(&mut self, mut off: u64, mut buf: &mut [u8]) -> Result<(), Error> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = self.seek_read(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            buf = &mut buf[n..];
            off += n as u64;
        }
        Ok(())
    }
}

/// Decompress a complete nzip container into memory. `threads` = 0 uses all CPUs. Uses
/// the GPU where it is faster (`Device::Auto`).
pub fn decompress(data: &[u8], threads: usize) -> Result<Vec<u8>, Error> {
    decompress_with(data, threads, Device::Auto)
}

/// Decompress into memory with an explicit device choice.
pub fn decompress_with(data: &[u8], threads: usize, device: Device) -> Result<Vec<u8>, Error> {
    let fh = FileHeader::read(data)?;
    let cap = usize::try_from(fh.orig_len).map_err(|_| Error::TooLarge("original size"))?;
    // The header is checksummed but not trusted for huge preallocations.
    let mut out = Vec::with_capacity(cap.min(data.len().saturating_mul(64)).min(1 << 32));
    decompress_to(data, &mut out, threads, device, 0)?;
    Ok(out)
}

/// Output buffer for dedup reconstruction: holds recent output and flushes it to the
/// sink (updating the whole-file hash) once it grows past `FLUSH`.
struct OutBuf<'s> {
    sink: &'s mut dyn Sink,
    hasher: &'s mut Xxh3Default,
    progress: Progress<'s>,
    flushed: u64,
    buf: Vec<u8>,
}

impl OutBuf<'_> {
    const FLUSH: usize = 64 << 20;
    fn pos(&self) -> u64 {
        self.flushed + self.buf.len() as u64
    }
    fn push(&mut self, d: &[u8]) -> Result<(), Error> {
        self.buf.extend_from_slice(d);
        if self.buf.len() >= Self::FLUSH {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), Error> {
        if !self.buf.is_empty() {
            self.hasher.update(&self.buf);
            self.sink.write_at(self.flushed, &self.buf)?;
            self.flushed += self.buf.len() as u64;
            (self.progress)(self.buf.len() as u64);
            self.buf.clear();
        }
        Ok(())
    }
    /// Append a copy of `len` bytes from absolute offset `src` (LZ77 overlap semantics).
    fn copy(&mut self, src: u64, len: u64) -> Result<(), Error> {
        let (mut from, mut left) = (src, len);
        let dist = self.pos() - src; // caller checked src < pos
        let mut tmp = Vec::new();
        while left > 0 {
            let n = left.min(dist).min(1 << 20) as usize;
            tmp.resize(n, 0);
            // Source bytes may sit in the sink, in the buffer, or straddle both.
            let in_sink = self.flushed.saturating_sub(from).min(n as u64) as usize;
            if in_sink > 0 {
                self.sink.read_at(from, &mut tmp[..in_sink])?;
            }
            if in_sink < n {
                let b = (from + in_sink as u64 - self.flushed) as usize;
                tmp[in_sink..].copy_from_slice(&self.buf[b..b + n - in_sink]);
            }
            self.push(&tmp)?;
            from += n as u64;
            left -= n as u64;
        }
        Ok(())
    }
}

/// Decompress into `sink`, streaming: frames are decoded in parallel batches whose size
/// is bounded by [`memory_budget`] (`mem_limit`, 0 = auto), and output is written in
/// order. Returns the original size. Nothing is trusted until its checksum matches; the
/// whole-file checksum is checked at the end, so callers writing to a file should only
/// keep it on `Ok`.
///
/// GPU use (measured): the inverse BWT of bsc blocks >= [`GPU_MIN_DECODE_BLOCK`] and ncore
/// frames >= [`NCORE_GPU_MIN`] run on the GPU. CUDA start-up runs on a background thread.
pub fn decompress_to(data: &[u8], sink: &mut dyn Sink, threads: usize, device: Device, mem_limit: usize) -> Result<u64, Error> {
    decompress_to_progress(data, sink, threads, device, mem_limit, &|_| {})
}

/// [`decompress_to`] with a progress callback (original bytes written).
pub fn decompress_to_progress(data: &[u8], sink: &mut dyn Sink, threads: usize, device: Device, mem_limit: usize,
                              progress: Progress<'_>) -> Result<u64, Error> {
    let fh = FileHeader::read(data)?;
    if fh.flags & FLAG_UNPACK != 0 {
        return expand::decompress(data, &fh, sink, threads, device, mem_limit, progress);
    }
    let (fh, frames) = format::parse(data)?;
    let threads = useful_threads(thread_count(threads), fh.orig_len.min(usize::MAX as u64) as usize);
    let dedup = fh.flags & FLAG_DEDUP != 0;
    let budget = memory_budget(mem_limit);

    let big_bsc = |f: &Frame<'_>| f.header.codec == CODEC_BSC && f.header.raw_len as usize >= GPU_MIN_DECODE_BLOCK;
    let ncore_frame = |f: &Frame<'_>| f.header.kind == KIND_DATA && f.header.codec == CODEC_NCORE;
    let big_ncore = |f: &Frame<'_>| ncore_frame(f) && f.header.raw_len as usize >= NCORE_GPU_MIN;
    let gpu = match device {
        Device::Cpu => false,
        _ if !frames.iter().any(big_bsc) => false,
        Device::Auto => bsc_sys::gpu_likely_present(),
        Device::Gpu if gpu_available() => true,
        Device::Gpu => return Err(Error::NoGpu { built: bsc_sys::built_with_gpu() }),
    };
    let ncore_gpu = match device {
        Device::Cpu => false,
        Device::Auto => frames.iter().any(big_ncore) && ncore::gpu::likely_present(),
        Device::Gpu if !frames.iter().any(ncore_frame) => false,
        Device::Gpu if ncore::gpu::available() => true,
        Device::Gpu => return Err(Error::NoGpu { built: bsc_sys::built_with_gpu() }),
    };
    let use_gpu_for = |f: &Frame<'_>| match f.header.codec {
        CODEC_BSC => gpu && big_bsc(f),
        CODEC_NCORE => ncore_gpu && (device == Device::Gpu || big_ncore(f)),
        _ => false,
    };

    let pool = pool(threads);
    // Decode consecutive data frames in parallel batches bounded by the memory budget.
    let decode_batch = |batch: &[Frame<'_>]| -> Result<Vec<u8>, Error> {
        if batch.iter().any(|f| f.header.kind != KIND_DATA) {
            return Err(Error::Corrupt("unexpected frame kind"));
        }
        let total: usize = batch.iter().map(|f| f.header.raw_len as usize).sum();
        let mut buf = vec![0u8; total];
        let mut slices = Vec::with_capacity(batch.len());
        let mut rest: &mut [u8] = &mut buf;
        for f in batch {
            let (a, b) = rest.split_at_mut(f.header.raw_len as usize);
            slices.push(a);
            rest = b;
        }
        let inner = inner_threads(threads, batch.len());
        pool.install(|| batch.par_iter().zip(slices.into_par_iter())
            .try_for_each(|(f, out)| decode_frame(f, out, inner_for(out.len(), inner), use_gpu_for(f))))?;
        Ok(buf)
    };
    let batch_end = |frames: &[Frame<'_>], from: usize, to: usize| -> usize {
        // At least one frame; stop at `threads` frames or when the batch's buffers plus
        // codec working memory would exceed the budget.
        let mut end = from;
        let mut bytes = 0usize;
        while end < to {
            let r = frames[end].header.raw_len as usize;
            if end > from && (end - from >= threads.max(1) || (bytes + r) * 6 > budget) {
                break;
            }
            bytes += r;
            end += 1;
        }
        end
    };

    let mut hasher = Xxh3Default::new();
    let total = std::thread::scope(|s| -> Result<u64, Error> {
        if gpu {
            s.spawn(bsc_sys::gpu_warmup);
        }
        if ncore_gpu {
            s.spawn(ncore::gpu::warmup);
        }
        (|| -> Result<u64, Error> {
            if !dedup {
                let mut pos = 0u64;
                let mut i = 0;
                while i < frames.len() {
                    let end = batch_end(&frames, i, frames.len());
                    let buf = decode_batch(&frames[i..end])?;
                    if pos + buf.len() as u64 > fh.orig_len {
                        return Err(Error::Corrupt("frames exceed the original size"));
                    }
                    hasher.update(&buf);
                    sink.write_at(pos, &buf)?;
                    pos += buf.len() as u64;
                    progress(buf.len() as u64);
                    i = end;
                }
                return Ok(pos);
            }
            // Dedup: [recipe, data x param] per segment. Ops take literals from the
            // segment's decoded data in order, and copies from any earlier output.
            let mut ob = OutBuf { sink: &mut *sink, hasher: &mut hasher, progress, flushed: 0, buf: Vec::new() };
            let mut i = 0;
            while i < frames.len() {
                let rf = &frames[i];
                if rf.header.kind != KIND_RECIPE {
                    return Err(Error::Corrupt("missing dedup recipe"));
                }
                let n = rf.header.param as usize;
                let seg_end = i.checked_add(1 + n).filter(|&e| e <= frames.len()).ok_or(Error::Corrupt("recipe frame count"))?;
                let mut raw = vec![0u8; rf.header.raw_len as usize];
                decode_frame(rf, &mut raw, 1, false)?;
                let ops = dedup::decode_ops(&raw)?;
                drop(raw);
                // Lazily decoded literal source.
                let (mut next, mut cur, mut cur_pos) = (i + 1, Vec::new(), 0usize);
                for op in &ops {
                    let mut lit = op.lit;
                    while lit > 0 {
                        if cur_pos == cur.len() {
                            if next == seg_end {
                                return Err(Error::Corrupt("recipe literal out of range"));
                            }
                            let end = batch_end(&frames, next, seg_end);
                            cur = decode_batch(&frames[next..end])?;
                            cur_pos = 0;
                            next = end;
                            continue;
                        }
                        let take = lit.min((cur.len() - cur_pos) as u64) as usize;
                        if ob.pos() + take as u64 > fh.orig_len {
                            return Err(Error::Corrupt("recipe exceeds original size"));
                        }
                        ob.push(&cur[cur_pos..cur_pos + take])?;
                        cur_pos += take;
                        lit -= take as u64;
                    }
                    if op.copy_len > 0 {
                        if op.copy_src >= ob.pos() || op.copy_len > fh.orig_len - ob.pos() {
                            return Err(Error::Corrupt("recipe copy out of range"));
                        }
                        ob.copy(op.copy_src, op.copy_len)?;
                    }
                }
                if cur_pos != cur.len() || next != seg_end {
                    return Err(Error::Corrupt("recipe does not cover its data exactly"));
                }
                i = seg_end;
            }
            ob.flush()?;
            Ok(ob.flushed)
        })()
    })?;
    if total != fh.orig_len {
        return Err(Error::Corrupt("frame sizes don't match original size"));
    }
    if hasher.digest() != fh.orig_xxh3 {
        return Err(Error::Corrupt("whole-file checksum mismatch"));
    }
    Ok(total)
}

/// Per-frame summary for `nzip info`.
pub struct FrameInfo {
    pub kind: u8,
    pub codec: u8,
    pub transform: u8,
    pub param: u32,
    pub raw_len: u32,
    pub comp_len: u32,
}

pub fn info(data: &[u8]) -> Result<(FileHeader, Vec<FrameInfo>), Error> {
    let fh = FileHeader::read(data)?;
    if fh.flags & FLAG_UNPACK != 0 {
        let mut all = Vec::new();
        let parts = expand::parts(data, &fh)?;
        all.extend(info(parts.main)?.1);
        for (_, a, b) in parts.waves {
            all.extend(info(a)?.1);
            all.extend(info(b)?.1);
        }
        return Ok((fh, all));
    }
    let (fh, frames) = format::parse(data)?;
    let v = frames.iter().map(|f| {
        let h = f.header;
        FrameInfo { kind: h.kind, codec: h.codec, transform: h.transform, param: h.param, raw_len: h.raw_len, comp_len: h.comp_len }
    }).collect();
    Ok((fh, v))
}
