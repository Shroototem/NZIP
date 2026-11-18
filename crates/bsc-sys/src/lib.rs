//! Minimal safe bindings to libbsc (block-sorting compressor by Ilya Grebnov, Apache-2.0).
use std::os::raw::{c_int, c_uchar};
use std::sync::Once;

pub const HEADER_SIZE: usize = 28;
pub const FEATURE_FASTMODE: c_int = 1;
pub const FEATURE_MULTITHREADING: c_int = 2;
pub const FEATURE_CUDA: c_int = 8;
pub const BLOCKSORTER_BWT: c_int = 1;
pub const CODER_QLFC_STATIC: c_int = 1;
pub const CODER_QLFC_ADAPTIVE: c_int = 2;
pub const CODER_QLFC_FAST: c_int = 3;
pub const DEFAULT_LZP_HASH: c_int = 15;
pub const DEFAULT_LZP_MINLEN: c_int = 72;
/// libbsc uses `int` sizes; keep blocks safely below 2 GiB.
pub const MAX_BLOCK: usize = (1 << 30) + (1 << 29);

unsafe extern "C" {
    fn bsc_init(features: c_int) -> c_int;
    fn bsc_compress(input: *const c_uchar, output: *mut c_uchar, n: c_int, lzp_hash: c_int,
                    lzp_min: c_int, sorter: c_int, coder: c_int, features: c_int) -> c_int;
    fn bsc_block_info(hdr: *const c_uchar, hdr_size: c_int, block_size: *mut c_int,
                      data_size: *mut c_int, features: c_int) -> c_int;
    fn bsc_decompress(input: *const c_uchar, in_size: c_int, output: *mut c_uchar,
                      out_size: c_int, features: c_int) -> c_int;
    fn bsc_detect_recordsize(input: *const c_uchar, n: c_int, features: c_int) -> c_int;
    fn bsc_reorder_forward(t: *mut c_uchar, n: c_int, record: c_int, features: c_int) -> c_int;
    fn bsc_reorder_reverse(t: *mut c_uchar, n: c_int, record: c_int, features: c_int) -> c_int;
    fn omp_set_num_threads(n: c_int);
    // Internal libbsc stages (C linkage), used to run our own entropy stage on BWT output.
    fn bsc_lzp_compress(input: *const c_uchar, output: *mut c_uchar, n: c_int, hash: c_int, min_len: c_int, features: c_int) -> c_int;
    fn bsc_lzp_decompress(input: *const c_uchar, output: *mut c_uchar, n: c_int, hash: c_int, min_len: c_int, adler: *mut u32, features: c_int) -> c_int;
    fn bsc_bwt_encode(t: *mut c_uchar, n: c_int, num_indexes: *mut c_uchar, indexes: *mut c_int, features: c_int) -> c_int;
    fn bsc_bwt_decode(t: *mut c_uchar, n: c_int, index: c_int, num_indexes: c_uchar, indexes: *mut c_int, features: c_int) -> c_int;
    fn bsc_coder_compress(input: *const c_uchar, output: *mut c_uchar, n: c_int, coder: c_int, features: c_int) -> c_int;
    fn bsc_adler32(t: *const c_uchar, n: c_int, features: c_int) -> u32;
}

/// Primary index plus the auxiliary sampled indexes libbsc uses for parallel inversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BwtIndex {
    pub primary: u32,
    pub aux: Vec<u32>,
}

/// LZP stage. Returns the LZP output length, or None if LZP didn't shrink the input
/// (then the caller should use the input as is). `out` must be >= input.len().
pub fn lzp_compress(input: &[u8], out: &mut [u8], hash: i32, min_len: i32, threads: usize) -> Option<usize> {
    init();
    if input.len() < 16 || input.len() > MAX_BLOCK || out.len() < input.len() {
        return None;
    }
    let r = unsafe { bsc_lzp_compress(input.as_ptr(), out.as_mut_ptr(), input.len() as c_int, hash, min_len, features(threads)) };
    if r > 0 && (r as usize) < input.len() { Some(r as usize) } else { None }
}

/// Inverse LZP. `out.len()` must be the original size.
pub fn lzp_decompress(input: &[u8], out: &mut [u8], hash: i32, min_len: i32, threads: usize) -> Result<(), BscError> {
    init();
    if input.len() > MAX_BLOCK || out.len() > MAX_BLOCK {
        return Err(BscError(-1));
    }
    let mut adler = 0u32;
    let r = unsafe { bsc_lzp_decompress(input.as_ptr(), out.as_mut_ptr(), input.len() as c_int, hash, min_len, &mut adler, features(threads)) };
    if r as usize == out.len() { Ok(()) } else if r < 0 { Err(BscError(r)) } else { Err(BscError(-6)) }
}

/// In-place forward BWT (GPU if `gpu` and available, byte-identical either way).
pub fn bwt_encode(t: &mut [u8], threads: usize, gpu: bool) -> Result<BwtIndex, BscError> {
    init();
    if t.len() < 2 || t.len() > MAX_BLOCK {
        return Err(BscError(-1));
    }
    let mut num = 0u8;
    let mut idx = [0 as c_int; 256];
    let f = features(threads) | if gpu && gpu_available() { FEATURE_CUDA } else { 0 };
    let r = unsafe { bsc_bwt_encode(t.as_mut_ptr(), t.len() as c_int, &mut num, idx.as_mut_ptr(), f) };
    if r <= 0 {
        return Err(BscError(if r == 0 { -1 } else { r }));
    }
    Ok(BwtIndex { primary: r as u32, aux: idx[..num as usize].iter().map(|&v| v as u32).collect() })
}

/// In-place inverse BWT.
pub fn bwt_decode(t: &mut [u8], index: &BwtIndex, threads: usize, gpu: bool) -> Result<(), BscError> {
    init();
    if t.len() < 2 || t.len() > MAX_BLOCK || index.primary == 0 || index.primary as usize > t.len() || index.aux.len() > 255 {
        return Err(BscError(-6));
    }
    let mut idx = [0 as c_int; 256];
    for (d, &s) in idx.iter_mut().zip(&index.aux) {
        *d = s as c_int;
    }
    let f = features(threads) | if gpu && cfg!(bsc_cuda) { FEATURE_CUDA } else { 0 };
    let r = unsafe { bsc_bwt_decode(t.as_mut_ptr(), t.len() as c_int, index.primary as c_int, index.aux.len() as u8, idx.as_mut_ptr(), f) };
    if r == 0 { Ok(()) } else { Err(BscError(r)) }
}

#[cfg(bsc_cuda)]
unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaFree(p: *mut std::ffi::c_void) -> c_int;
    fn cudaGetLastError() -> c_int;
}

/// True if a CUDA error is pending (and clears it if it isn't sticky). After a failed
/// GPU sort the device may be unusable and the in-place buffer suspect.
pub fn gpu_error() -> bool {
    #[cfg(bsc_cuda)]
    if gpu_available() {
        return unsafe { cudaGetLastError() } != 0;
    }
    false
}

/// GPU memory libcubwt needs per input byte (measured: ~20.3 B/B at 512-640 MiB blocks),
/// plus headroom.
pub const GPU_BYTES_PER_BLOCK_BYTE: usize = 21;

/// Free GPU memory in bytes (0 if no GPU), capped by `NZIP_VRAM_LIMIT` if set.
pub fn gpu_free_memory() -> usize {
    #[cfg(bsc_cuda)]
    if gpu_available() {
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 {
            // NZIP_VRAM_LIMIT (bytes) pretends the card has at most that much free memory,
            // to test small-GPU behaviour on a big one.
            let lim = std::env::var("NZIP_VRAM_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
            return free.min(lim);
        }
    }
    0
}

/// Total GPU memory in bytes (0 if no GPU), capped by `NZIP_VRAM_LIMIT` if set. A
/// property of the card (unlike free memory), so choices made from it are repeatable.
pub fn gpu_total_memory() -> usize {
    #[cfg(bsc_cuda)]
    if gpu_available() {
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cudaMemGetInfo(&mut free, &mut total) } == 0 {
            let lim = std::env::var("NZIP_VRAM_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
            return total.min(lim);
        }
    }
    0
}

/// Create the CUDA context now (takes ~0.1 s) so the first GPU BWT doesn't pay for it.
/// Call it from a background thread while other work runs.
pub fn gpu_warmup() {
    #[cfg(bsc_cuda)]
    if gpu_available() {
        unsafe { cudaFree(std::ptr::null_mut()) };
    }
}

/// True if libbsc was built with CUDA and a usable NVIDIA GPU + driver is present.
pub fn gpu_available() -> bool {
    #[cfg(bsc_cuda)]
    {
        static GPU: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GPU.get_or_init(|| {
            let mut n: c_int = 0;
            unsafe { cudaGetDeviceCount(&mut n) == 0 && n > 0 }
        })
    }
    #[cfg(not(bsc_cuda))]
    {
        false
    }
}

/// Cheap guess (no CUDA initialisation) that an NVIDIA GPU + driver is present.
/// Used to start GPU work optimistically; libbsc falls back to the CPU if wrong.
pub fn gpu_likely_present() -> bool {
    built_with_gpu() && nvidia_driver_present()
}

/// An NVIDIA driver is installed: Linux has its device node, Windows its user-mode driver.
fn nvidia_driver_present() -> bool {
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        std::path::Path::new(&root).join("System32").join("nvcuda.dll").exists()
    } else {
        std::path::Path::new("/dev/nvidiactl").exists()
    }
}

/// True if this build contains the GPU code at all.
pub const fn built_with_gpu() -> bool {
    cfg!(bsc_cuda)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BscError(pub i32);

impl std::fmt::Display for BscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.0 {
            -1 => "bad parameter", -2 => "not enough memory", -3 => "not compressible",
            -4 => "not supported", -5 => "unexpected end of block", -6 => "data corrupt",
            _ => "unknown error",
        };
        write!(f, "libbsc error {} ({what})", self.0)
    }
}
impl std::error::Error for BscError {}

static INIT: Once = Once::new();
fn init() {
    INIT.call_once(|| unsafe {
        let cuda = if cfg!(bsc_cuda) { FEATURE_CUDA } else { 0 };
        let r = bsc_init(FEATURE_FASTMODE | FEATURE_MULTITHREADING | cuda);
        assert_eq!(r, 0, "bsc_init failed: {r}");
    });
}

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub lzp_hash: i32,
    pub lzp_min: i32,
    pub coder: i32,
    /// OpenMP threads used inside this one call (1 = single-threaded).
    pub threads: usize,
    /// Run the BWT on the GPU (output is byte-identical to the CPU path; libbsc falls
    /// back to the CPU if the GPU can't take the block).
    pub gpu: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self { lzp_hash: DEFAULT_LZP_HASH, lzp_min: DEFAULT_LZP_MINLEN, coder: CODER_QLFC_ADAPTIVE, threads: 1, gpu: false }
    }
}

fn features(threads: usize) -> c_int {
    // Always set the OpenMP thread count for the calling thread: libbsc code paths
    // that use OpenMP regardless of the MULTITHREADING flag must not oversubscribe.
    unsafe { omp_set_num_threads(threads.max(1) as c_int) };
    if threads > 1 { FEATURE_FASTMODE | FEATURE_MULTITHREADING } else { FEATURE_FASTMODE }
}

/// Compress `input` (len <= MAX_BLOCK). Returns the bsc block (header included).
pub fn compress(input: &[u8], p: &Params) -> Result<Vec<u8>, BscError> {
    init();
    if input.len() > MAX_BLOCK {
        return Err(BscError(-1));
    }
    let mut out = vec![0u8; input.len() + HEADER_SIZE];
    let n = unsafe {
        bsc_compress(input.as_ptr(), out.as_mut_ptr(), input.len() as c_int, p.lzp_hash,
                     p.lzp_min, BLOCKSORTER_BWT, p.coder,
                     features(p.threads) | if p.gpu && gpu_available() { FEATURE_CUDA } else { 0 })
    };
    if n < 0 {
        return Err(BscError(n));
    }
    out.truncate(n as usize);
    Ok(out)
}

/// Decompress one bsc block into `out`, whose length must equal the original size.
/// `gpu`: run the inverse BWT on the GPU (libbsc falls back to the CPU if it can't).
pub fn decompress(block: &[u8], out: &mut [u8], threads: usize, gpu: bool) -> Result<(), BscError> {
    init();
    if block.len() < HEADER_SIZE || block.len() > i32::MAX as usize || out.len() > MAX_BLOCK {
        return Err(BscError(-6));
    }
    let (mut bsize, mut dsize) = (0, 0);
    let feats = features(threads);
    let r = unsafe { bsc_block_info(block.as_ptr(), HEADER_SIZE as c_int, &mut bsize, &mut dsize, feats) };
    if r != 0 {
        return Err(BscError(r));
    }
    if bsize as usize != block.len() || dsize as usize != out.len() {
        return Err(BscError(-6));
    }
    let feats = feats | if gpu && cfg!(bsc_cuda) { FEATURE_CUDA } else { 0 };
    let r = unsafe { bsc_decompress(block.as_ptr(), bsize, out.as_mut_ptr(), dsize, feats) };
    if r != 0 { Err(BscError(r)) } else { Ok(()) }
}

/// Detect a fixed record size (tables, 16-bit samples, ...). 1 = none found.
pub fn detect_record_size(data: &[u8]) -> usize {
    init();
    if data.is_empty() || data.len() > MAX_BLOCK {
        return 1;
    }
    let r = unsafe { bsc_detect_recordsize(data.as_ptr(), data.len() as c_int, FEATURE_FASTMODE) };
    if r < 1 { 1 } else { r as usize }
}

pub fn reorder_forward(data: &mut [u8], record: usize) -> Result<(), BscError> {
    init();
    let r = unsafe { bsc_reorder_forward(data.as_mut_ptr(), data.len() as c_int, record as c_int, FEATURE_FASTMODE) };
    if r != 0 { Err(BscError(r)) } else { Ok(()) }
}

pub fn reorder_reverse(data: &mut [u8], record: usize) -> Result<(), BscError> {
    init();
    let r = unsafe { bsc_reorder_reverse(data.as_mut_ptr(), data.len() as c_int, record as c_int, FEATURE_FASTMODE) };
    if r != 0 { Err(BscError(r)) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let data: Vec<u8> = (0..200_000u32).map(|i| ((i * 7) % 251) as u8 ^ (i >> 9) as u8).collect();
        let c = compress(&data, &Params::default()).unwrap();
        assert!(c.len() < data.len());
        let mut out = vec![0; data.len()];
        decompress(&c, &mut out, 1, false).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn stages_roundtrip() {
        let data: Vec<u8> = b"banana bandana cabana ".iter().cycle().take(300_000).enumerate().map(|(i, &b)| b ^ ((i / 5000) as u8 & 1)).collect();
        let mut l = vec![0u8; data.len()];
        let n = lzp_compress(&data, &mut l, DEFAULT_LZP_HASH, DEFAULT_LZP_MINLEN, 1).unwrap_or(0);
        let mut t = if n > 0 { l[..n].to_vec() } else { data.clone() };
        let orig_t = t.clone();
        let idx = bwt_encode(&mut t, 4, false).unwrap();
        bwt_decode(&mut t, &idx, 4, false).unwrap();
        assert_eq!(t, orig_t);
        if n > 0 {
            let mut back = vec![0u8; data.len()];
            lzp_decompress(&t, &mut back, DEFAULT_LZP_HASH, DEFAULT_LZP_MINLEN, 1).unwrap();
            assert_eq!(back, data);
        }
    }

    #[test]
    fn gpu_output_identical_to_cpu() {
        if !gpu_available() {
            eprintln!("no GPU: skipped");
            return;
        }
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i.wrapping_mul(2654435761) >> 27) as u8 ^ (i / 1000) as u8).collect();
        let cpu = compress(&data, &Params::default()).unwrap();
        let gpu = compress(&data, &Params { gpu: true, threads: 4, ..Default::default() }).unwrap();
        assert!(cpu == gpu, "GPU and CPU bsc output differ");
    }
}

/// One bsc block being compressed in three separately scheduled stages: [`Staged::lzp`]
/// (CPU), [`Staged::bwt`] (GPU or CPU) and [`Staged::finish`] (CPU). The result is
/// byte-identical to [`compress`] (it mirrors libbsc's `bsc_compress`), so blocks decode
/// with [`decompress`] whichever device sorted them.
pub struct Staged {
    /// Header space + LZP output (sorted in place by the BWT stage).
    buf: Vec<u8>,
    n: usize,
    lz: usize,
    mode: c_int,
    index: c_int,
    num_indexes: u8,
    indexes: [c_int; 256],
    adler_in: u32,
    stored: bool,
}

impl Staged {
    /// LZP stage. `input.len()` must be <= 1 GiB.
    pub fn lzp(input: &[u8], p: &Params) -> Result<Self, BscError> {
        init();
        let n = input.len();
        if n > 1 << 30 {
            return Err(BscError(-1));
        }
        let f = features(p.threads);
        let mut s = Staged { buf: Vec::new(), n, lz: 0, mode: 0, index: 0, num_indexes: 0, indexes: [0; 256], adler_in: 0, stored: false };
        if n <= HEADER_SIZE {
            s.stored = true;
            return Ok(s);
        }
        let mut mode = BLOCKSORTER_BWT + (p.coder << 5);
        if p.lzp_min != 0 || p.lzp_hash != 0 {
            if !(4..=255).contains(&p.lzp_min) || !(10..=28).contains(&p.lzp_hash) {
                return Err(BscError(-1));
            }
            mode += (p.lzp_min << 8) + (p.lzp_hash << 16);
        }
        s.buf = vec![0u8; n + HEADER_SIZE];
        let mut lz = 0;
        if mode != (mode & 0xff) {
            lz = unsafe { bsc_lzp_compress(input.as_ptr(), s.buf.as_mut_ptr(), n as c_int, p.lzp_hash, p.lzp_min, f) };
            if lz < 0 {
                mode &= 0xff;
            }
        }
        if mode == (mode & 0xff) {
            lz = n as c_int;
            s.buf[..n].copy_from_slice(input);
        }
        if lz as usize <= HEADER_SIZE {
            mode = (mode & !0x1f) | BLOCKSORTER_BWT;
        }
        s.lz = lz as usize;
        s.mode = mode;
        s.adler_in = unsafe { bsc_adler32(input.as_ptr(), n as c_int, f) };
        Ok(s)
    }

    /// The LZP output (after [`Staged::bwt`]: its BWT).
    pub fn sorted(&self) -> &[u8] {
        if self.stored { &[] } else { &self.buf[..self.lz] }
    }

    /// Bytes the BWT stage sorts (the LZP output size).
    pub fn sort_len(&self) -> usize {
        if self.stored { 0 } else { self.lz }
    }

    /// BWT stage, in place. With `gpu`, libcubwt sorts it (falling back to the CPU if the
    /// GPU can't take it); otherwise libsais on up to 8 of `threads` CPU threads.
    pub fn bwt(&mut self, threads: usize, gpu: bool) -> Result<(), BscError> {
        if self.stored {
            return Ok(());
        }
        let f = features(threads) | if gpu && gpu_available() { FEATURE_CUDA } else { 0 };
        let mut num = 0u8;
        let r = unsafe { bsc_bwt_encode(self.buf.as_mut_ptr(), self.lz as c_int, &mut num, self.indexes.as_mut_ptr(), f) };
        if r < 0 {
            return Err(BscError(r));
        }
        self.index = r;
        self.num_indexes = if self.n < 64 * 1024 { 0 } else { num };
        Ok(())
    }

    /// Entropy-coding stage. Returns the finished bsc block (header included).
    pub fn finish(mut self, input: &[u8], p: &Params) -> Result<Vec<u8>, BscError> {
        let n = self.n;
        assert_eq!(input.len(), n);
        let f = features(p.threads);
        let store = |input: &[u8]| -> Vec<u8> {
            let mut out = vec![0u8; n + HEADER_SIZE];
            out[HEADER_SIZE..].copy_from_slice(input);
            let a = unsafe { bsc_adler32(input.as_ptr(), n as c_int, f) };
            for (i, v) in [(n + HEADER_SIZE) as u32, n as u32, 0, 0, a, a].into_iter().enumerate() {
                out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
            let h = unsafe { bsc_adler32(out.as_ptr(), 24, f) };
            out[24..28].copy_from_slice(&h.to_le_bytes());
            out
        };
        if self.stored {
            return Ok(store(input));
        }
        let lz = self.lz;
        let mut tmp = vec![0u8; lz + 4096];
        let r = unsafe { bsc_coder_compress(self.buf.as_ptr(), tmp.as_mut_ptr(), lz as c_int, p.coder, f) };
        let ni = self.num_indexes as usize;
        if r < 0 || r as usize + 1 + 4 * ni >= n {
            return Ok(store(input));
        }
        let mut result = r as usize;
        let out = &mut self.buf;
        out[HEADER_SIZE..HEADER_SIZE + result].copy_from_slice(&tmp[..result]);
        drop(tmp);
        for t in 0..ni {
            let o = HEADER_SIZE + result + 4 * t;
            out[o..o + 4].copy_from_slice(&self.indexes[t].to_le_bytes());
        }
        out[HEADER_SIZE + result + 4 * ni] = self.num_indexes;
        result += 1 + 4 * ni;
        let a_out = unsafe { bsc_adler32(out.as_ptr().add(HEADER_SIZE), result as c_int, f) };
        for (i, v) in [(result + HEADER_SIZE) as u32, n as u32, self.mode as u32, self.index as u32, self.adler_in, a_out].into_iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        let h = unsafe { bsc_adler32(out.as_ptr(), 24, f) };
        out[24..28].copy_from_slice(&h.to_le_bytes());
        out.truncate(result + HEADER_SIZE);
        out.shrink_to_fit();
        Ok(std::mem::take(out))
    }
}

/// libbsc's QLFC entropy coder alone (used to study splitting the coder into streams).
pub fn coder_compress(input: &[u8], coder: i32, threads: usize) -> Result<Vec<u8>, BscError> {
    init();
    let mut out = vec![0u8; input.len() + 4096];
    let r = unsafe { bsc_coder_compress(input.as_ptr(), out.as_mut_ptr(), input.len() as c_int, coder, features(threads)) };
    if r < 0 { return Err(BscError(r)); }
    out.truncate(r as usize);
    Ok(out)
}
