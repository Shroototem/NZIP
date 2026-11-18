//! GPU/CPU scheduling for max mode's bsc blocks.
//!
//! Each block runs LZP (CPU) -> BWT (GPU or CPU) -> QLFC (CPU) as separate stages, so a
//! block waiting for or using the GPU sort holds no CPU threads while other blocks run
//! their CPU stages. The GPU takes a block only if its sort fits the card's free memory
//! right now (decided per block, so any card size works and the output never depends on
//! it: both devices produce identical bytes). A block also goes to the CPU when the GPU
//! queue is so long that the CPU would finish it sooner (measured rates). If a GPU sort
//! ever reports an error, the GPU is switched off for the rest of the run and the block
//! is redone on the CPU from its input.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Mutex;
use std::time::Instant;

use bsc_sys::{Params, Staged};

/// Serialises GPU sorts (one at a time uses the whole card) and guards libbsc's cached
/// device storage.
static GPU: Mutex<()> = Mutex::new(());
/// Largest sort length libbsc has device storage for (it frees and regrows the cache
/// when a larger block arrives).
static GPU_CACHED: AtomicUsize = AtomicUsize::new(0);
/// Bytes queued for or being sorted on the GPU.
static GPU_QUEUED: AtomicUsize = AtomicUsize::new(0);
/// Set after a GPU error: everything after runs on the CPU.
static GPU_OFF: AtomicBool = AtomicBool::new(false);
/// Measured sort speeds in bytes/s (EWMA), seeded with RTX 5060 Ti / Ryzen 3900X values.
static GPU_RATE: AtomicU64 = AtomicU64::new(400_000_000);
static CPU_RATE: AtomicU64 = AtomicU64::new(45_000_000);
/// Blocks currently in a CPU stage (to share the CPU threads among them).
static CPU_BUSY: AtomicUsize = AtomicUsize::new(0);
/// Blocks sorted on the GPU in this process.
pub static GPU_BLOCKS: AtomicU64 = AtomicU64::new(0);
/// Largest sort that went to the CPU because it didn't fit the GPU's free memory, and
/// the memory that was available then (bytes; 0 = none).
static SHORT_SORT: AtomicUsize = AtomicUsize::new(0);
static SHORT_AVAIL: AtomicUsize = AtomicUsize::new(0);

/// `NZIP_TRACE=1`: print a stage timeline to stderr (seconds since the first block).
fn trace(what: &str, n: usize) {
    static ON: std::sync::OnceLock<Option<Instant>> = std::sync::OnceLock::new();
    if let Some(t0) = ON.get_or_init(|| std::env::var_os("NZIP_TRACE").map(|_| Instant::now())) {
        eprintln!("{:8.2} {what} {}M", t0.elapsed().as_secs_f64(), n >> 20);
    }
}

/// Device bytes kept free besides libcubwt's storage (driver, other apps' growth).
const VRAM_MARGIN: usize = 384 << 20;

/// Whether a sort of `n` bytes fits the GPU now (call while holding [`GPU`] for a
/// definitive answer).
fn fits(n: usize) -> bool {
    let cached = GPU_CACHED.load(Acquire);
    n + n / 32 <= cached // libbsc's device storage already covers it
        || available() >= need(n)
}

/// Device memory a sort may use: free memory plus libbsc's cached storage (libbsc frees
/// its cache before allocating a larger one).
fn available() -> usize {
    let per = bsc_sys::GPU_BYTES_PER_BLOCK_BYTE;
    bsc_sys::gpu_free_memory() + GPU_CACHED.load(Acquire) * per * 32 / 33
}

/// Device memory a sort of `n` bytes needs (n + n/32 slots of ~20.4 bytes; 21 per
/// input byte covers the slack), plus the margin.
pub fn need(n: usize) -> usize {
    n * bsc_sys::GPU_BYTES_PER_BLOCK_BYTE + VRAM_MARGIN
}

/// Largest sort (LZP output bytes) that fits the GPU right now (0 without a GPU).
pub fn max_gpu_sort() -> usize {
    available().saturating_sub(VRAM_MARGIN) / bsc_sys::GPU_BYTES_PER_BLOCK_BYTE
}

/// If some sort went to the CPU for lack of GPU memory: (largest such sort's memory
/// need, memory available then), in bytes.
pub fn shortfall() -> Option<(usize, usize)> {
    let n = SHORT_SORT.load(Relaxed);
    (n > 0).then(|| (need(n), SHORT_AVAIL.load(Relaxed)))
}

fn ewma(r: &AtomicU64, bytes: usize, secs: f64) {
    if secs > 0.01 {
        let v = (bytes as f64 / secs) as u64;
        let old = r.load(Relaxed);
        r.store((old * 3 + v) / 4, Relaxed);
    }
}

/// Runs `f` as a CPU stage with a fair share of `total` threads.
fn cpu_stage<T>(total: usize, len: usize, f: impl FnOnce(usize) -> T) -> T {
    let busy = CPU_BUSY.fetch_add(1, AcqRel) + 1;
    let t = (total / busy).max(1).min(len.div_ceil(256 << 10)).max(1);
    let r = f(t);
    CPU_BUSY.fetch_sub(1, AcqRel);
    r
}

/// Smallest sub-block worth splitting a block into for the GPU: below this the ratio
/// loss outweighs the speed (a card that small sorts whole blocks on the CPU).
const MIN_PIECE: usize = 64 << 20;

fn params(coder: i32, threads: usize) -> Params {
    Params { coder, threads, gpu: false, ..Default::default() }
}

fn lzp(data: &[u8], coder: i32, total: usize) -> Result<Staged, bsc_sys::BscError> {
    trace("lzp start", data.len());
    let s = cpu_stage(total, data.len(), |t| Staged::lzp(data, &params(coder, t)))?;
    trace("lzp done", s.sort_len());
    Ok(s)
}

/// Compress one block with bsc in scheduled stages. Returns the bsc data and whether a
/// sort ran on the GPU.
///
/// Blocks are as large as memory allows (best ratio). When a block's sort doesn't fit
/// the GPU's memory, it is split into equal sub-blocks that do, each a complete bsc
/// block, back to back in the frame: any card size keeps the GPU busy, and any machine
/// (GPU or CPU) decodes the result.
pub fn bsc_block(data: &[u8], coder: i32, total: usize, gpu: bool) -> Result<(Vec<u8>, bool), bsc_sys::BscError> {
    let s = lzp(data, coder, total)?;
    let n = s.sort_len();
    if gpu && n >= 1 << 20 && !GPU_OFF.load(Acquire) && !fits(n) {
        // Pieces sized from the LZP output, with slack for uneven LZP gains.
        let cap = max_gpu_sort() * 9 / 10;
        let k = n.div_ceil(cap.max(1));
        if cap >= MIN_PIECE && k >= 2 {
            drop(s);
            let piece = data.len().div_ceil(k);
            // Pieces run concurrently: one sorts on the GPU while others do LZP/coding.
            let parts: Vec<_> = std::thread::scope(|sc| {
                let hs: Vec<_> = data.chunks(piece)
                    .map(|c| sc.spawn(move || lzp(c, coder, total).and_then(|s| sort_and_code(c, s, coder, total, gpu))))
                    .collect();
                hs.into_iter().map(|h| h.join().expect("bsc piece")).collect()
            });
            let (mut out, mut any) = (Vec::new(), false);
            for r in parts {
                let (b, g) = r?;
                out.extend_from_slice(&b);
                any |= g;
            }
            return Ok((out, any));
        }
    }
    sort_and_code(data, s, coder, total, gpu)
}

/// Sort (GPU if it fits and is quicker, else CPU) and entropy-code one bsc block.
fn sort_and_code(data: &[u8], mut s: Staged, coder: i32, total: usize, gpu: bool) -> Result<(Vec<u8>, bool), bsc_sys::BscError> {
    let p = |threads| params(coder, threads);
    let n = s.sort_len();
    let mut on_gpu = false;
    let eligible = gpu && n >= 1 << 20 && !GPU_OFF.load(Acquire);
    if eligible && !fits(n) && SHORT_SORT.fetch_max(n, Relaxed) < n {
        SHORT_AVAIL.store(available(), Relaxed);
    }
    if eligible && fits(n) {
        // Queue for the GPU unless the CPU would finish this sort before the queue drains.
        let queued = GPU_QUEUED.fetch_add(n, AcqRel) + n;
        let gpu_eta = queued as f64 / GPU_RATE.load(Relaxed) as f64;
        let cpu_eta = n as f64 / CPU_RATE.load(Relaxed) as f64;
        if gpu_eta <= cpu_eta {
            let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
            if !GPU_OFF.load(Acquire) && fits(n) {
                trace("gpu start", n);
                // Record the storage libbsc is about to (re)allocate, so blocks checking
                // `fits` while this sort runs count it as theirs to reuse.
                GPU_CACHED.fetch_max(n + n / 32, AcqRel);
                let t0 = Instant::now();
                let r = s.bwt(2, true);
                trace("gpu done", n);
                if r.is_err() || bsc_sys::gpu_error() {
                    GPU_OFF.store(true, Release);
                    eprintln!("nzip: GPU sort failed; continuing on the CPU");
                    s = cpu_stage(total, data.len(), |t| Staged::lzp(data, &p(t)))?;
                } else {
                    ewma(&GPU_RATE, n, t0.elapsed().as_secs_f64());
                    on_gpu = true;
                }
            }
        }
        GPU_QUEUED.fetch_sub(n, AcqRel);
    }
    if !on_gpu {
        trace("cpu sort start", n);
        cpu_stage(total, data.len(), |t| {
            let t0 = Instant::now();
            let r = s.bwt(t, false);
            // libsais uses at most 8 threads; normalise the rate to that.
            ewma(&CPU_RATE, n * 8 / t.min(8), t0.elapsed().as_secs_f64());
            r
        })?;
    } else {
        GPU_BLOCKS.fetch_add(1, Relaxed);
    }
    trace("code start", n);
    // The coder gets every thread (the OS shares them while several blocks code): a
    // fair share fixed at its start would leave the last blocks crawling on a few
    // threads once the others finish.
    let out = cpu_stage(total, data.len(), |_| s.finish(data, &p(total.min(data.len().div_ceil(256 << 10)).max(1))))?;
    trace("code done", out.len());
    Ok((out, on_gpu))
}
