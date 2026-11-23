//! BWT front end: GPU suffix sorting when an NVIDIA GPU is usable, otherwise the CPU.
//!
//! * The CUDA driver is loaded at run time (libcuda.so.1 / nvcuda.dll); there is no build-
//!   or install-time CUDA dependency. Kernels ship as PTX and are JIT-compiled by the driver.
//! * A suffix array has exactly one correct answer, so archives are byte-identical whether
//!   the GPU or the CPU sorted them, and decompression never needs a GPU.
//! * Every GPU result is verified on the device with an exact O(n) check. Any failure, a
//!   block too big for free VRAM, or no usable GPU -> that block is sorted on the CPU.

#![allow(non_snake_case)]

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

const PTX: &str = include_str!("../gpu/sa.ptx");
const TILE: u32 = 4096; // scan tile
const RS_TILE: u32 = 2048; // radix-sort tile

type CuResult = c_int;
type DevPtr = u64;

macro_rules! api {
    ($($name:ident : fn($($arg:ty),*);)*) => {
        struct Api { $($name: unsafe extern "system" fn($($arg),*) -> CuResult,)* }
        impl Api {
            unsafe fn load(lib: *mut c_void) -> Option<Api> {
                Some(Api { $($name: {
                    let p = unsafe { sym(lib, stringify!($name)) };
                    if p.is_null() { return None; }
                    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($arg),*) -> CuResult>(p) }
                },)* })
            }
        }
    };
}

api! {
    cuInit: fn(c_uint);
    cuDeviceGetCount: fn(*mut c_int);
    cuDeviceGet: fn(*mut c_int, c_int);
    cuDeviceGetName: fn(*mut c_char, c_int, c_int);
    cuDevicePrimaryCtxRetain: fn(*mut *mut c_void, c_int);
    cuCtxSetCurrent: fn(*mut c_void);
    cuModuleLoadData: fn(*mut *mut c_void, *const c_void);
    cuModuleGetFunction: fn(*mut *mut c_void, *mut c_void, *const c_char);
    cuMemAlloc_v2: fn(*mut DevPtr, usize);
    cuMemFree_v2: fn(DevPtr);
    cuMemcpyHtoD_v2: fn(DevPtr, *const c_void, usize);
    cuMemcpyDtoH_v2: fn(*mut c_void, DevPtr, usize);
    cuMemcpyDtoD_v2: fn(DevPtr, DevPtr, usize);
    cuMemsetD8_v2: fn(DevPtr, u8, usize);
    cuMemGetInfo_v2: fn(*mut usize, *mut usize);
    cuLaunchKernel: fn(*mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void);
    cuCtxSynchronize: fn();
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(file: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const c_char) -> *mut c_void;
}

unsafe fn open_lib() -> *mut c_void {
    #[cfg(unix)]
    unsafe {
        for n in [&b"libcuda.so.1\0"[..], b"libcuda.so\0"] {
            let h = dlopen(n.as_ptr() as *const c_char, 2);
            if !h.is_null() {
                return h;
            }
        }
        std::ptr::null_mut()
    }
    #[cfg(windows)]
    unsafe {
        LoadLibraryA(b"nvcuda.dll\0".as_ptr() as *const c_char)
    }
    #[cfg(not(any(unix, windows)))]
    std::ptr::null_mut()
}

unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
    let c = CString::new(name).unwrap();
    #[cfg(unix)]
    unsafe {
        dlsym(lib, c.as_ptr())
    }
    #[cfg(windows)]
    unsafe {
        GetProcAddress(lib, c.as_ptr())
    }
    #[cfg(not(any(unix, windows)))]
    std::ptr::null_mut()
}

struct Kernels {
    rs_hist: *mut c_void,
    rs_scatter: *mut c_void,
    scan_reduce: *mut c_void,
    scan_top: *mut c_void,
    scan_down: *mut c_void,
    sa_init: *mut c_void,
    sa_flags: *mut c_void,
    sa_assign: *mut c_void,
    sa_compact: *mut c_void,
    sa_keys: *mut c_void,
    sa_isa: *mut c_void,
    sa_check: *mut c_void,
    sa_bwt: *mut c_void,
    ub_walk: *mut c_void,
    ub_count: *mut c_void,
    ub_scatter: *mut c_void,
    ub_c1: *mut c_void,
    mf_lcp: *mut c_void,
    mf_find: *mut c_void,
    mf_keys: *mut c_void,
    mf_prev: *mut c_void,
    mf_count: *mut c_void,
    mf_pack: *mut c_void,
    lz_parse: *mut c_void,
}

struct Gpu {
    api: Api,
    ctx: *mut c_void,
    k: Kernels,
    name: String,
    disabled: bool,
    /// device buffers kept between jobs (capacity, pointer): cuMemAlloc / cuMemFree cost
    /// milliseconds each and every free synchronizes the device
    cache: std::cell::RefCell<Vec<(usize, DevPtr)>>,
}

// The raw handles are only used while holding the global mutex.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// Share of free GPU memory nzip may use (the --gpu option; 0 = never use the GPU).
static GPU_PCT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(99);

pub fn set_share(pct: u64) {
    GPU_PCT.store(pct.min(100), std::sync::atomic::Ordering::Relaxed);
}

fn lock(m: &'static Mutex<Gpu>) -> std::sync::MutexGuard<'static, Gpu> {
    m.lock().unwrap()
}

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| {
        if std::env::var_os("NZIP_NO_GPU").is_some() || GPU_PCT.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return None;
        }
        match unsafe { init() } {
            Ok(g) => Some(Mutex::new(g)),
            Err(e) => {
                if std::env::var_os("NZIP_VERBOSE").is_some() {
                    eprintln!("nzip: no GPU: {e}");
                }
                None
            }
        }
    })
    .as_ref()
}

fn ck(r: CuResult, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("CUDA error {r} in {what}")) }
}

unsafe fn init() -> Result<Gpu, String> {
    unsafe {
        let lib = open_lib();
        if lib.is_null() {
            return Err("no CUDA driver".into());
        }
        let api = Api::load(lib).ok_or("CUDA driver too old")?;
        ck((api.cuInit)(0), "cuInit")?;
        let mut count = 0;
        ck((api.cuDeviceGetCount)(&mut count), "cuDeviceGetCount")?;
        if count < 1 {
            return Err("no CUDA device".into());
        }
        let mut dev = 0;
        ck((api.cuDeviceGet)(&mut dev, 0), "cuDeviceGet")?;
        let mut name = [0 as c_char; 128];
        ck((api.cuDeviceGetName)(name.as_mut_ptr(), 128, dev), "cuDeviceGetName")?;
        let name = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        let mut ctx = std::ptr::null_mut();
        ck((api.cuDevicePrimaryCtxRetain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        ck((api.cuCtxSetCurrent)(ctx), "cuCtxSetCurrent")?;
        let ptx = CString::new(PTX).unwrap();
        let mut module = std::ptr::null_mut();
        ck(
            (api.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void),
            "cuModuleLoadData (GPU or driver too old for the kernels)",
        )?;
        let f = |n: &str| -> Result<*mut c_void, String> {
            let mut p = std::ptr::null_mut();
            let c = CString::new(n).unwrap();
            ck((api.cuModuleGetFunction)(&mut p, module, c.as_ptr()), n)?;
            Ok(p)
        };
        let k = Kernels {
            rs_hist: f("rs_hist")?,
            rs_scatter: f("rs_scatter")?,
            scan_reduce: f("scan_reduce")?,
            scan_top: f("scan_top")?,
            scan_down: f("scan_down")?,
            sa_init: f("sa_init")?,
            sa_flags: f("sa_flags")?,
            sa_assign: f("sa_assign")?,
            sa_compact: f("sa_compact")?,
            sa_keys: f("sa_keys")?,
            sa_isa: f("sa_isa")?,
            sa_check: f("sa_check")?,
            sa_bwt: f("sa_bwt")?,
            ub_walk: f("ub_walk")?,
            ub_count: f("ub_count")?,
            ub_scatter: f("ub_scatter")?,
            ub_c1: f("ub_c1")?,
            mf_lcp: f("mf_lcp")?,
            mf_find: f("mf_find")?,
            mf_keys: f("mf_keys")?,
            mf_prev: f("mf_prev")?,
            mf_count: f("mf_count")?,
            mf_pack: f("mf_pack")?,
            lz_parse: f("lz_parse")?,
        };
        Ok(Gpu { api, ctx, k, name, disabled: false, cache: std::cell::RefCell::new(Vec::new()) })
    }
}

/// Device bytes needed to sort a block of `n` bytes.
pub fn vram_need(n: usize) -> u64 {
    // text n, sa 4n, rank 4n, keys 2x8n, vals 2x4n, histogram n/4, + slack
    n as u64 * 34 + (64 << 20)
}

/// Device buffers left by `sa_build`.
struct SaDev<'a> {
    text: Buf<'a>,
    sa: Buf<'a>,
    rank: Buf<'a>,
    ka: Buf<'a>,
    kb: Buf<'a>,
    va: Buf<'a>,
    vb: Buf<'a>,
    hist: Buf<'a>,
    sums: Buf<'a>,
    err: Buf<'a>,
}

struct Buf<'a> {
    g: &'a Gpu,
    p: DevPtr,
    cap: usize,
}

impl Drop for Buf<'_> {
    fn drop(&mut self) {
        // back to the cache for the next job
        self.g.cache.borrow_mut().push((self.cap, self.p));
    }
}

/// Largest total the buffer cache keeps (more is freed).
const CACHE_MAX: usize = 6 << 30;

#[inline(always)]
fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

impl Gpu {
    fn alloc(&self, bytes: usize) -> Result<Buf<'_>, String> {
        let bytes = bytes.max(4);
        {
            // smallest cached buffer that fits, unless it is far too big
            let mut c = self.cache.borrow_mut();
            let best = c.iter().enumerate().filter(|(_, b)| b.0 >= bytes && b.0 <= bytes * 2 + (16 << 20)).min_by_key(|(_, b)| b.0).map(|(i, _)| i);
            if let Some(i) = best {
                let (cap, p) = c.swap_remove(i);
                return Ok(Buf { g: self, p, cap });
            }
        }
        // round sizes up so later jobs of similar size reuse the buffer
        let cap = if bytes > 1 << 20 { bytes.next_multiple_of(bytes.next_power_of_two() / 8) } else { bytes };
        let mut p = 0;
        if unsafe { (self.api.cuMemAlloc_v2)(&mut p, cap) } != 0 {
            // out of memory: drop the cache and try once more
            self.trim(0);
            ck(unsafe { (self.api.cuMemAlloc_v2)(&mut p, cap) }, "cuMemAlloc")?;
        }
        Ok(Buf { g: self, p, cap })
    }

    /// Frees cached buffers until at most `keep` bytes stay cached.
    fn trim(&self, keep: usize) {
        let mut c = self.cache.borrow_mut();
        c.sort_by_key(|&(cap, _)| std::cmp::Reverse(cap));
        let mut total: usize = c.iter().map(|&(cap, _)| cap).sum();
        while total > keep {
            let Some((cap, p)) = c.pop() else { break };
            unsafe { (self.api.cuMemFree_v2)(p) };
            total -= cap;
        }
    }

    fn cached(&self) -> u64 {
        self.cache.borrow().iter().map(|&(cap, _)| cap as u64).sum()
    }

    /// Launch `f` over `n` threads in blocks of `block`.
    fn launch(&self, f: *mut c_void, n: u32, block: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        self.launch_grid(f, n.div_ceil(block), block, args)
    }

    fn launch_grid(&self, f: *mut c_void, grid: u32, block: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        ck(
            unsafe {
                (self.api.cuLaunchKernel)(f, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())
            },
            "cuLaunchKernel",
        )
    }

    /// Free device memory nzip may use: the --gpu share of what is free now.
    fn free_mem(&self) -> u64 {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { (self.api.cuMemGetInfo_v2)(&mut free, &mut total) };
        // cached buffers are ours to reuse
        (free as u64 + self.cached()) / 100 * GPU_PCT.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn read_u32(&self, p: DevPtr) -> Result<u32, String> {
        let mut v = 0u32;
        ck(unsafe { (self.api.cuMemcpyDtoH_v2)(&mut v as *mut u32 as *mut c_void, p, 4) }, "read")?;
        Ok(v)
    }

    /// In-place scan of `n` u32 at `buf` (sum or max, exclusive or inclusive).
    fn scan(&self, buf: DevPtr, n: u32, sums: DevPtr, mx: bool, inclusive: bool) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let tiles = n.div_ceil(TILE);
        let (mut b, mut o, mut n_, mut s) = (buf, buf, n, sums);
        let (mut m, mut inc, mut t) = (mx as i32, inclusive as i32, tiles);
        self.launch_grid(self.k.scan_reduce, tiles, 256, &mut [arg(&mut b), arg(&mut n_), arg(&mut s), arg(&mut m)])?;
        self.launch_grid(self.k.scan_top, 1, 256, &mut [arg(&mut s), arg(&mut t), arg(&mut m)])?;
        self.launch_grid(
            self.k.scan_down,
            tiles,
            256,
            &mut [arg(&mut b), arg(&mut o), arg(&mut n_), arg(&mut s), arg(&mut m), arg(&mut inc)],
        )
    }

    /// Stable LSD radix sort of (key, val) pairs by the given 8-bit digit shifts.
    /// Returns true if the result ended up in the alternate buffers.
    #[allow(clippy::too_many_arguments)]
    fn sort(&self, k0: DevPtr, v0: DevPtr, k1: DevPtr, v1: DevPtr, n: u32, shifts: &[u32], hist: DevPtr, sums: DevPtr) -> Result<bool, String> {
        let tiles = n.div_ceil(RS_TILE);
        let (mut ki, mut vi, mut ko, mut vo) = (k0, v0, k1, v1);
        let mut flipped = false;
        for &sh in shifts {
            let (mut n_, mut sh_, mut h, mut t) = (n, sh, hist, tiles);
            self.launch_grid(self.k.rs_hist, tiles, 256, &mut [arg(&mut ki), arg(&mut n_), arg(&mut sh_), arg(&mut h), arg(&mut t)])?;
            self.scan(hist, tiles * 256, sums, false, false)?;
            self.launch_grid(
                self.k.rs_scatter,
                tiles,
                256,
                &mut [arg(&mut ki), arg(&mut vi), arg(&mut ko), arg(&mut vo), arg(&mut n_), arg(&mut sh_), arg(&mut h), arg(&mut t)],
            )?;
            std::mem::swap(&mut ki, &mut ko);
            std::mem::swap(&mut vi, &mut vo);
            flipped = !flipped;
        }
        Ok(flipped)
    }

    /// Suffix array of `s` on the device (verified by the caller via `err`): returns the
    /// text, SA, inverse SA and scratch buffers (ka/kb are free after this, n*8 bytes each).
    fn sa_build(&self, s: &[u8]) -> Result<SaDev<'_>, String> {
        let n = s.len();
        let n32 = n as u32;
        let tiles = n32.div_ceil(RS_TILE) as usize;
        let text = self.alloc(n)?;
        let sa = self.alloc(n * 4)?;
        let rank = self.alloc(n * 4)?;
        let ka = self.alloc(n * 8)?;
        let kb = self.alloc(n * 8)?;
        let va = self.alloc(n * 4)?;
        let vb = self.alloc(n * 4)?;
        let hist = self.alloc(tiles * 256 * 4)?;
        // block sums for scans over up to max(n, histogram) elements
        let sums = self.alloc((tiles.max((tiles * 256).div_ceil(TILE as usize)) + 16) * 4)?;
        let err = self.alloc(4)?;
        unsafe {
            ck((self.api.cuMemcpyHtoD_v2)(text.p, s.as_ptr() as *const c_void, n), "upload")?;
            ck((self.api.cuMemsetD8_v2)(err.p, 0, 4), "memset")?;
        }
        let (mut tp, mut np) = (text.p, n32);
        let (mut kc, mut vc, mut kx, mut vx) = (ka.p, va.p, kb.p, vb.p);
        self.launch(self.k.sa_init, n32, 256, &mut [arg(&mut tp), arg(&mut np), arg(&mut kc), arg(&mut vc)])?;
        let bits = 32 - n32.leading_zeros(); // rank+1 <= n fits in `bits` bits
        let mut m = n32;
        let mut l = 7u32; // suffixes are sorted by their first `l` bytes after each round
        let mut round0 = 1i32;
        let (mut sap, mut rp) = (sa.p, rank.p);
        let trace = std::env::var_os("NZIP_TRACE").is_some();
        let t0 = std::time::Instant::now();
        loop {
            if trace {
                unsafe { (self.api.cuCtxSynchronize)() };
                eprintln!("  gpu round l={l:>8} active={m:>10} t={:.3}s", t0.elapsed().as_secs_f64());
            }
            let shifts: Vec<u32> = if round0 == 1 {
                (0..8).map(|i| i * 8).collect()
            } else {
                let d = bits.div_ceil(8);
                (0..d).map(|i| i * 8).chain((0..d).map(|i| 32 + i * 8)).collect()
            };
            let tt = std::time::Instant::now();
            let lap = |name: &str, tt: &mut std::time::Instant| {
                if trace {
                    unsafe { (self.api.cuCtxSynchronize)() };
                    eprintln!("      {name}: {:.1} ms", tt.elapsed().as_secs_f64() * 1e3);
                    *tt = std::time::Instant::now();
                }
            };
            let mut tt = tt;
            if self.sort(kc, vc, kx, vx, m, &shifts, hist.p, sums.p)? {
                std::mem::swap(&mut kc, &mut kx);
                std::mem::swap(&mut vc, &mut vx);
            }
            lap("sort", &mut tt);
            // scratch: kx as two u32 arrays A, B; vx as keep flags -> compaction index
            let (mut a, mut b) = (kx, kx + m as u64 * 4);
            let mut m_ = m;
            self.launch(self.k.sa_flags, m, 256, &mut [arg(&mut kc), arg(&mut m_), arg(&mut round0), arg(&mut a), arg(&mut b)])?;
            self.scan(a, m, sums.p, true, true)?;
            self.scan(b, m, sums.p, true, true)?;
            lap("flags+scans", &mut tt);
            let mut keep = vx;
            self.launch(
                self.k.sa_assign,
                m,
                256,
                &mut [arg(&mut kc), arg(&mut vc), arg(&mut m_), arg(&mut round0), arg(&mut a), arg(&mut b), arg(&mut sap), arg(&mut rp), arg(&mut keep)],
            )?;
            lap("assign", &mut tt);
            if round0 == 0 {
                l = l.saturating_mul(2);
            }
            let last_keep = self.read_u32(keep + (m as u64 - 1) * 4)?;
            self.scan(keep, m, sums.p, false, false)?;
            let total = self.read_u32(keep + (m as u64 - 1) * 4)? + last_keep;
            if total == 0 {
                break;
            }
            // compact survivors (+ their new ranks) into kc, then build next keys into kx/vx
            let (mut list, mut list_rank) = (kc, kc + m as u64 * 4);
            let mut tot = total;
            self.launch(
                self.k.sa_compact,
                m,
                256,
                &mut [arg(&mut vc), arg(&mut keep), arg(&mut m_), arg(&mut tot), arg(&mut a), arg(&mut list), arg(&mut list_rank)],
            )?;
            lap("compact", &mut tt);
            let (mut hh, mut nn) = (l, n32);
            self.launch(
                self.k.sa_keys,
                total,
                256,
                &mut [arg(&mut list), arg(&mut list_rank), arg(&mut tot), arg(&mut hh), arg(&mut nn), arg(&mut rp), arg(&mut kx), arg(&mut vx)],
            )?;
            std::mem::swap(&mut kc, &mut kx);
            std::mem::swap(&mut vc, &mut vx);
            lap("keys", &mut tt);
            m = total;
            round0 = 0;
            if l as u64 > 2 * n as u64 + 16 {
                return Err("suffix sort did not converge".into());
            }
        }
        // verify: isa (into the rank buffer), then the exact adjacency check
        let mut ep = err.p;
        self.launch(self.k.sa_isa, n32, 256, &mut [arg(&mut sap), arg(&mut np), arg(&mut rp), arg(&mut ep)])?;
        self.launch(self.k.sa_check, n32, 256, &mut [arg(&mut tp), arg(&mut sap), arg(&mut rp), arg(&mut np), arg(&mut ep)])?;
        Ok(SaDev { text, sa, rank, ka, kb, va, vb, hist, sums, err })
    }

    /// LZ match lists for every position of `s` (see mf_find in sa.cu); positions in
    /// segments where `skip(seg)` is set get empty lists.
    fn match_lists(&self, s: &[u8], seg_bits: u32, skip: &[bool]) -> Result<(Vec<u32>, Vec<(u32, u32)>), String> {
        const C: usize = 6;
        const W: usize = 1 + 2 * C;
        let n = s.len();
        let n32 = n as u32;
        let trace = std::env::var_os("NZIP_TRACE").is_some();
        let mut tt = std::time::Instant::now();
        let mut lap = |what: &str| {
            if trace {
                unsafe { (self.api.cuCtxSynchronize)() };
                eprintln!("    mf {what}: {:.0} ms", tt.elapsed().as_secs_f64() * 1e3);
                tt = std::time::Instant::now();
            }
        };
        let d = self.sa_build(s)?;
        lap("suffix array");
        let (mut tp, mut np, mut sap, mut isap) = (d.text.p, n32, d.sa.p, d.rank.p);
        // nearest earlier occurrence per gram length: stable sort (key = gram, value = position)
        const LEVELS: usize = 6;
        const GRAMS: [u32; LEVELS] = [3, 4, 6, 8, 12, 16];
        let prev = self.alloc(n * 4 * LEVELS)?;
        for (v, &l) in GRAMS.iter().enumerate() {
            // key bytes: the gram itself up to 8 bytes, else a 32-bit hash
            let shifts: Vec<u32> = (0..if l <= 8 { l } else { 4 }).map(|i| i * 8).collect();
            let (mut kp, mut vp, mut l_) = (d.ka.p, d.va.p, l);
            self.launch(self.k.mf_keys, n32, 256, &mut [arg(&mut tp), arg(&mut np), arg(&mut l_), arg(&mut kp), arg(&mut vp)])?;
            let flipped = self.sort(d.ka.p, d.va.p, d.kb.p, d.vb.p, n32, &shifts, d.hist.p, d.sums.p)?;
            let (mut ks, mut vs) = if flipped { (d.kb.p, d.vb.p) } else { (d.ka.p, d.va.p) };
            let mut pp = prev.p + (v * n * 4) as u64;
            self.launch(self.k.mf_prev, n32, 256, &mut [arg(&mut ks), arg(&mut vs), arg(&mut np), arg(&mut pp)])?;
        }
        lap("gram levels");
        let mut prevp = prev.p;
        let mut lcpp = d.kb.p;
        self.launch(self.k.mf_lcp, n32, 256, &mut [arg(&mut tp), arg(&mut sap), arg(&mut np), arg(&mut lcpp)])?;
        lap("lcp");
        unsafe { ck((self.api.cuCtxSynchronize)(), "sync")? };
        let e = self.read_u32(d.err.p)?;
        if e != 0 {
            return Err(format!("GPU suffix array failed verification ({e})"));
        }
        // one pass over all positions when the output fits (each pass costs syncs)
        let chunk = n.clamp(1, 16 << 20);
        let outb = self.alloc(chunk * W * 4)?;
        let skipd = self.alloc(skip.len())?;
        let flags: Vec<u8> = skip.iter().map(|&b| b as u8).collect();
        let counts = self.alloc(chunk * 4)?;
        let packed = self.alloc(chunk * C * 8)?;
        unsafe { ck((self.api.cuMemcpyHtoD_v2)(skipd.p, flags.as_ptr() as *const c_void, flags.len()), "upload")? };
        let mut idx = Vec::with_capacity(n + 1);
        let mut m: Vec<(u32, u32)> = Vec::with_capacity(n * 2);
        let mut offs = vec![0u32; chunk];
        let mut p0 = 0usize;
        while p0 < n {
            let cnt = chunk.min(n - p0);
            let (mut p0_, mut cnt_, mut outp) = (p0 as u32, cnt as u32, outb.p);
            self.launch(
                self.k.mf_find,
                cnt as u32,
                128,
                &mut [arg(&mut tp), arg(&mut sap), arg(&mut isap), arg(&mut lcpp), arg(&mut prevp), arg(&mut np), arg(&mut p0_), arg(&mut cnt_), arg(&mut outp)],
            )?;
            let (mut sk, mut sb, mut cp) = (skipd.p, seg_bits, counts.p);
            self.launch(self.k.mf_count, cnt as u32, 256, &mut [arg(&mut outp), arg(&mut sk), arg(&mut sb), arg(&mut p0_), arg(&mut cnt_), arg(&mut cp)])?;
            let last = self.read_u32(counts.p + (cnt as u64 - 1) * 4)?;
            // exclusive scan in place: counts -> offsets (then counts are re-read for the pack)
            let offsd = self.alloc(cnt * 4)?;
            unsafe { ck((self.api.cuMemcpyDtoD_v2)(offsd.p, counts.p, cnt * 4), "copy")? };
            self.scan(offsd.p, cnt as u32, d.sums.p, false, false)?;
            let total = self.read_u32(offsd.p + (cnt as u64 - 1) * 4)? as usize + last as usize;
            let (mut op, mut pk) = (offsd.p, packed.p);
            self.launch(self.k.mf_pack, cnt as u32, 256, &mut [arg(&mut outp), arg(&mut op), arg(&mut cp), arg(&mut cnt_), arg(&mut pk)])?;
            lap("find");
            let base = m.len() as u32;
            m.resize(m.len() + total, (0, 0));
            unsafe {
                ck((self.api.cuMemcpyDtoH_v2)(offs.as_mut_ptr() as *mut c_void, offsd.p, cnt * 4), "download")?;
                ck((self.api.cuMemcpyDtoH_v2)(m[base as usize..].as_mut_ptr() as *mut c_void, packed.p, total * 8), "download")?;
            }
            idx.extend(offs[..cnt].iter().map(|&o| base + o));
            p0 += cnt;
            lap("download");
        }
        idx.push(m.len() as u32);
        Ok((idx, m))
    }

    /// Optimal parse of the 64 KiB segments [seg0, seg0 + nseg) of `s` (see lz_parse in
    /// sa.cu). `idx`/`pairs` are the match lists of positions [ib, ib + idx.len() - 1), with
    /// idx relative to `pairs`. Returns, per segment, None when it is dense (coded as
    /// literals) or its tokens in order as (word, d).
    #[allow(clippy::too_many_arguments)]
    fn lz_parse(&self, s: &[u8], ib: usize, idx: &[u32], pairs: &[(u32, u32)], dense: &[bool], seg0: usize, nseg: usize, prices: &[LzPrices], lit: &[u16]) -> Result<Vec<Option<Vec<(u32, u32)>>>, String> {
        let n = s.len();
        let span = idx.len() - 1;
        let up = |b: &Buf, src: *const c_void, len: usize| unsafe { ck((self.api.cuMemcpyHtoD_v2)(b.p, src, len), "upload") };
        let text = self.alloc(n + 64)?;
        up(&text, s.as_ptr() as *const c_void, n)?;
        let di = self.alloc(idx.len() * 4)?;
        up(&di, idx.as_ptr() as *const c_void, idx.len() * 4)?;
        let dp = self.alloc(pairs.len() * 8)?;
        if !pairs.is_empty() {
            up(&dp, pairs.as_ptr() as *const c_void, pairs.len() * 8)?;
        }
        let fl: Vec<u8> = dense.iter().map(|&b| b as u8).collect();
        let dd = self.alloc(fl.len())?;
        up(&dd, fl.as_ptr() as *const c_void, fl.len())?;
        let psz = std::mem::size_of::<LzPrices>() * prices.len();
        let dpr = self.alloc(psz)?;
        up(&dpr, prices.as_ptr() as *const c_void, psz)?;
        let mut pstep = (prices.len() > 1) as u32;
        let dl = self.alloc(lit.len() * 2)?;
        up(&dl, lit.as_ptr() as *const c_void, lit.len() * 2)?;
        let ba = self.alloc(span * 4)?;
        let bd = self.alloc(span * 4)?;
        let on = self.alloc(nseg * 4)?;
        let (mut tp, mut n32, mut s0, mut ns, mut ib32) = (text.p, n as u32, seg0 as u32, nseg as u32, ib as u32);
        let (mut ip, mut pp, mut dp_, mut prp, mut lp, mut bap, mut bdp, mut onp) = (di.p, dp.p, dd.p, dpr.p, dl.p, ba.p, bd.p, on.p);
        const WARPS: u32 = 8;
        self.launch_grid(
            self.k.lz_parse,
            (nseg as u32).div_ceil(WARPS),
            32 * WARPS,
            &mut [arg(&mut tp), arg(&mut n32), arg(&mut s0), arg(&mut ns), arg(&mut ib32), arg(&mut ip), arg(&mut pp), arg(&mut dp_), arg(&mut prp), arg(&mut pstep), arg(&mut lp), arg(&mut bap), arg(&mut bdp), arg(&mut onp)],
        )?;
        let mut a = vec![0u32; span];
        let mut d = vec![0u32; span];
        let mut flags = vec![0u32; nseg];
        if std::env::var_os("NZIP_TRACE").is_some() {
            let t0 = std::time::Instant::now();
            unsafe { ck((self.api.cuCtxSynchronize)(), "lz_parse")? };
            eprintln!("    lz_parse kernel: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        unsafe {
            ck((self.api.cuCtxSynchronize)(), "lz_parse")?;
            ck((self.api.cuMemcpyDtoH_v2)(a.as_mut_ptr() as *mut c_void, ba.p, span * 4), "download")?;
            ck((self.api.cuMemcpyDtoH_v2)(d.as_mut_ptr() as *mut c_void, bd.p, span * 4), "download")?;
            ck((self.api.cuMemcpyDtoH_v2)(flags.as_mut_ptr() as *mut c_void, on.p, nseg * 4), "download")?;
        }
        let mut out = Vec::with_capacity(nseg);
        for k in 0..nseg {
            if flags[k] != 0 {
                out.push(None);
                continue;
            }
            let st = ((seg0 + k) << 16) - ib;
            let en = (((seg0 + k + 1) << 16).min(n)) - ib;
            let mut toks = Vec::new();
            let mut j = en;
            while j > st {
                let w = a[j - 1];
                let l = (w & 511) as usize;
                if l == 0 || l > j - st {
                    return Err("GPU parse returned a broken chain".into());
                }
                toks.push((w, d[j - 1]));
                j -= l;
            }
            toks.reverse();
            out.push(Some(toks));
        }
        Ok(out)
    }

    fn bwt(&self, s: &[u8]) -> Result<(Vec<u8>, Vec<u32>), String> {
        let n = s.len();
        let n32 = n as u32;
        let d = self.sa_build(s)?;
        let (mut tp, mut np, mut sap) = (d.text.p, n32, d.sa.p);
        let kx = d.ka.p;
        let err = &d.err;
        // BWT + chain indices on the device: only n + 4k bytes come back
        let k = crate::sais::chains_for(n);
        let seg = n.div_ceil(k) as u32;
        let idxb = self.alloc(k * 4)?;
        unsafe { ck((self.api.cuMemsetD8_v2)(idxb.p, 0, k * 4), "memset")? };
        let (mut outp, mut ip, mut sg) = (kx, idxb.p, seg);
        self.launch(self.k.sa_bwt, n32, 256, &mut [arg(&mut tp), arg(&mut sap), arg(&mut np), arg(&mut sg), arg(&mut outp), arg(&mut ip)])?;
        let mut out = vec![0u8; n];
        let mut idx = vec![0u32; k];
        unsafe {
            ck((self.api.cuCtxSynchronize)(), "sync")?;
            let e = self.read_u32(err.p)?;
            if e != 0 {
                return Err(format!("GPU suffix array failed verification ({e})"));
            }
            ck((self.api.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, outp, n), "download")?;
            ck((self.api.cuMemcpyDtoH_v2)(idx.as_mut_ptr() as *mut c_void, idxb.p, k * 4), "download")?;
        }
        Ok((out, idx))
    }
}

/// Smallest block worth sending to the GPU (below this the CPU is as fast).
const GPU_MIN: usize = 4 << 20;

/// BWT of `s`: on the GPU if possible, else on the CPU. Same result either way.
pub fn bwt(s: &[u8]) -> (Vec<u8>, Vec<u32>) {
    if s.len() >= GPU_MIN && s.len() < (u32::MAX as usize >> 1) {
        // the whole GPU turn (queueing, the job) runs without a CPU token, taken back
        // only after the GPU is released, so neither cores nor the GPU wait on the other
        if let Some(r) = crate::par::idle(|| gpu_bwt(s)) {
            return r;
        }
    }
    crate::sais::bwt(s)
}

fn gpu_bwt(s: &[u8]) -> Option<(Vec<u8>, Vec<u32>)> {
    {
        if let Some(m) = gpu() {
            let mut g = lock(m);
            if !g.disabled {
                unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
                if g.free_mem() >= vram_need(s.len()) {
                    let r = g.bwt(s);
                    g.trim(CACHE_MAX);
                    match r {
                        Ok(r) => return Some(r),
                        Err(e) => {
                            // never trust the device again in this run; the CPU redoes the block
                            eprintln!("nzip: GPU disabled for this run ({e}); using the CPU");
                            g.disabled = true;
                        }
                    }
                }
            }
        }
    }
    None
}

/// LZ match lists for every position of `s` from the GPU (a suffix array and its
/// nearest neighbours), or None when no GPU can take it. Index `p` covers
/// `m[idx[p]..idx[p + 1]]`, pairs (len, dist - 1) with increasing length.
pub fn match_lists(s: &[u8], seg_bits: u32, skip: &[bool]) -> Option<(Vec<u32>, Vec<(u32, u32)>)> {
    if s.len() < GPU_MIN || s.len() >= (u32::MAX as usize >> 1) {
        return None;
    }
    // the GPU job runs without a CPU token (taken back once the GPU is released)
    crate::par::idle(|| gpu_match_lists(s, seg_bits, skip))
}

fn gpu_match_lists(s: &[u8], seg_bits: u32, skip: &[bool]) -> Option<(Vec<u32>, Vec<(u32, u32)>)> {
    let m = gpu()?;
    // a busy GPU is not waited for: the CPU match finder takes this piece, so the GPU and
    // every core stay busy (which pieces the GPU gets then depends on timing; any choice
    // decodes the same)
    let mut g = match m.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::WouldBlock) if std::env::var_os("NZ_GPUWAIT").is_none() => return None,
        Err(_) => lock(m),
    };
    if g.disabled {
        return None;
    }
    unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
    if g.free_mem() < vram_need(s.len()) + s.len() as u64 * 24 {
        return None;
    }
    let t0 = std::time::Instant::now();
    let r = g.match_lists(s, seg_bits, skip);
    g.trim(CACHE_MAX);
    match r {
        Ok(r) => {
            if std::env::var_os("NZIP_TRACE").is_some() {
                eprintln!("  gpu match lists: {} MB in {:.3}s", s.len() >> 20, t0.elapsed().as_secs_f64());
            }
            Some(r)
        }
        Err(e) => {
            eprintln!("nzip: GPU disabled for this run ({e}); using the CPU");
            g.disabled = true;
            None
        }
    }
}

/// Static LZ prices for the GPU parser, in 1/16 bit (layout = LzP in sa.cu).
#[repr(C)]
#[derive(Clone)]
pub struct LzPrices {
    pub lit_flag: [[u32; 4]; 12],
    pub match_flag: [[u32; 4]; 12],
    pub rep_flag: [[u32; 2]; 12],
    pub rep0: [[u32; 2]; 12],
    pub rep0long: [[[u32; 2]; 4]; 12],
    pub rep1: [[u32; 2]; 12],
    pub rep2: [[u32; 2]; 12],
    pub mlen: [[u32; 272]; 4],
    pub rlen: [[u32; 272]; 4],
    pub slot: [[u32; 64]; 4],
    pub small: [[u32; 128]; 4],
    pub align: [u32; 16],
}

/// GPU optimal parse of whole 64 KiB segments (see `Gpu::lz_parse`), or None when no GPU
/// can take it.
#[allow(clippy::too_many_arguments)]
pub fn lz_parse(s: &[u8], ib: usize, idx: &[u32], pairs: &[(u32, u32)], dense: &[bool], seg0: usize, nseg: usize, prices: &[LzPrices], lit: &[u16]) -> Option<Vec<Option<Vec<(u32, u32)>>>> {
    let m = gpu()?;
    let mut g = lock(m);
    if g.disabled {
        return None;
    }
    unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
    let t0 = std::time::Instant::now();
    match g.lz_parse(s, ib, idx, pairs, dense, seg0, nseg, prices, lit) {
        Ok(r) => {
            if std::env::var_os("NZIP_TRACE").is_some() {
                eprintln!("  gpu parse: {nseg} segments in {:.3}s", t0.elapsed().as_secs_f64());
            }
            Some(r)
        }
        Err(e) => {
            eprintln!("nzip: GPU disabled for this run ({e}); using the CPU");
            g.disabled = true;
            None
        }
    }
}

/// Smallest inverse BWT worth the GPU (driver start-up, transfers).
const UNBWT_MIN: usize = 32 << 20;

/// Inverse BWT on the GPU (`b` = BWT output, `rows` = start row of each chain of `seg`
/// text bytes; rows[0] is the primary index), or None to let the CPU do it. The result is
/// checked: every chain must end exactly where the next one starts.
pub fn unbwt(b: &[u8], rows: &[u32], seg: usize) -> Option<Vec<u8>> {
    let n = b.len();
    if n < UNBWT_MIN || n >= u32::MAX as usize >> 1 {
        return None;
    }
    let m = gpu()?;
    let mut g = lock(m);
    if g.disabled {
        return None;
    }
    unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
    // text n, psi 4(n+1), output n, per-tile counts 4*256*(n/UB_T) = n/2, + slack
    let need = n as u64 * 7 + (64 << 20);
    if g.free_mem() < need {
        return None;
    }
    match g.unbwt(b, rows, seg) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("nzip: GPU disabled for this run ({e}); using the CPU");
            g.disabled = true;
            None
        }
    }
}

const UB_T: usize = 2048;

impl Gpu {
    fn unbwt(&self, b: &[u8], rows: &[u32], seg: usize) -> Result<Vec<u8>, String> {
        let n = b.len();
        let nrows = n + 1;
        let k = n.div_ceil(seg);
        let tiles = nrows.div_ceil(UB_T);
        let db = self.alloc(n)?;
        let dpsi = self.alloc(nrows * 4)?;
        let dcnt = self.alloc(tiles * 256 * 4)?;
        let dsums = self.alloc(((tiles * 256).div_ceil(TILE as usize) + 16) * 4)?;
        let dc1 = self.alloc(257 * 4)?;
        let drows = self.alloc(k * 4)?;
        let dout = self.alloc(n)?;
        let dfin = self.alloc(k * 4)?;
        unsafe {
            ck((self.api.cuMemcpyHtoD_v2)(db.p, b.as_ptr() as *const c_void, n), "upload")?;
            ck((self.api.cuMemcpyHtoD_v2)(drows.p, rows.as_ptr() as *const c_void, k * 4), "upload rows")?;
        }
        let (mut pb, mut nr, mut p1, mut nt) = (db.p, nrows as u32, rows[0], tiles as u32);
        let (mut cnt, mut psi, mut c1) = (dcnt.p, dpsi.p, dc1.p);
        self.launch(self.k.ub_count, tiles as u32, 64, &mut [arg(&mut pb), arg(&mut nr), arg(&mut p1), arg(&mut nt), arg(&mut cnt)])?;
        self.scan(dcnt.p, (tiles * 256) as u32, dsums.p, false, false)?;
        self.launch_grid(self.k.ub_c1, 1, 256, &mut [arg(&mut cnt), arg(&mut nt), arg(&mut nr), arg(&mut c1)])?;
        self.launch(self.k.ub_scatter, tiles as u32, 64, &mut [arg(&mut pb), arg(&mut nr), arg(&mut p1), arg(&mut nt), arg(&mut cnt), arg(&mut psi)])?;
        let (mut r, mut kk, mut sg, mut nn, mut o, mut f) = (drows.p, k as u32, seg as u32, n as u32, dout.p, dfin.p);
        self.launch(self.k.ub_walk, k as u32, 64, &mut [arg(&mut psi), arg(&mut r), arg(&mut kk), arg(&mut sg), arg(&mut nn), arg(&mut c1), arg(&mut o), arg(&mut f)])?;
        ck(unsafe { (self.api.cuCtxSynchronize)() }, "inverse BWT")?;
        let mut out = vec![0u8; n];
        let mut fin = vec![0u32; k];
        unsafe {
            ck((self.api.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, dout.p, n), "download")?;
            ck((self.api.cuMemcpyDtoH_v2)(fin.as_mut_ptr() as *mut c_void, dfin.p, k * 4), "download")?;
        }
        if (0..k.saturating_sub(1)).any(|t| fin[t] != rows[t + 1]) || fin[k - 1] != 0 {
            return Err("inverse BWT check failed".into());
        }
        Ok(out)
    }
}

/// Free device memory if a GPU is usable (for block sizing).
pub fn free_vram() -> Option<u64> {
    let m = gpu()?;
    let g = m.lock().unwrap();
    if g.disabled {
        return None;
    }
    unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
    Some(g.free_mem())
}

pub fn describe() -> String {
    match gpu() {
        Some(m) => {
            let g = m.lock().unwrap();
            unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
            format!("{} ({:.1} GB free) - suffix sorting and LZ match finding", g.name, g.free_mem() as f64 / 1e9)
        }
        None => "none usable (CPU suffix sorting; output is identical)".into(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu_matches_cpu() {
        if super::gpu().is_none() {
            eprintln!("no GPU: skipped");
            return;
        }
        let mut rng = 99u64;
        for (len, alpha) in [(1usize << 20, 256u64), (3 << 20, 4), (2_000_001, 2), (1 << 20, 1)] {
            let s: Vec<u8> = (0..len)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((rng >> 33) % alpha) as u8
                })
                .collect();
            let g = super::bwt(&s);
            let c = crate::sais::bwt(&s);
            assert!(g == c, "len {len} alpha {alpha}");
            let m = super::gpu().unwrap().lock().unwrap();
            assert!(!m.disabled, "GPU fell back on len {len} alpha {alpha}");
        }
    }
}

#[cfg(test)]
mod dbg {
    use super::*;
    #[test]
    fn radix_and_scan() {
        let Some(m) = gpu() else { return };
        let g = m.lock().unwrap();
        unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
        for n in [5u32, 4096, 4097, 100_000, 1_000_003] {
            let mut rng = n as u64;
            let keys: Vec<u64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); rng >> 3 }).collect();
            let vals: Vec<u32> = (0..n).collect();
            let nn = n as usize;
            let tiles = n.div_ceil(RS_TILE) as usize;
            let (k0, k1, v0, v1) = (g.alloc(nn * 8).unwrap(), g.alloc(nn * 8).unwrap(), g.alloc(nn * 4).unwrap(), g.alloc(nn * 4).unwrap());
            let hist = g.alloc(tiles * 256 * 4).unwrap();
            let sums = g.alloc((tiles * 256).div_ceil(4096) * 4 + 64).unwrap();
            unsafe {
                (g.api.cuMemcpyHtoD_v2)(k0.p, keys.as_ptr() as *const c_void, nn * 8);
                (g.api.cuMemcpyHtoD_v2)(v0.p, vals.as_ptr() as *const c_void, nn * 4);
            }
            // scan check on vals (sum, exclusive) using v1 copy
            unsafe { (g.api.cuMemcpyHtoD_v2)(v1.p, vals.as_ptr() as *const c_void, nn * 4) };
            g.scan(v1.p, n, sums.p, false, false).unwrap();
            let mut sc = vec![0u32; nn];
            unsafe { (g.api.cuCtxSynchronize)(); (g.api.cuMemcpyDtoH_v2)(sc.as_mut_ptr() as *mut c_void, v1.p, nn * 4) };
            let mut acc = 0u32; let mut bad = None;
            for i in 0..nn { if sc[i] != acc && bad.is_none() { bad = Some((i, sc[i], acc)); } acc = acc.wrapping_add(i as u32); }
            assert!(bad.is_none(), "scan n={n} first bad {:?}", bad);
            let shifts: Vec<u32> = (0..8).map(|i| i * 8).collect();
            let fl = g.sort(k0.p, v0.p, k1.p, v1.p, n, &shifts, hist.p, sums.p).unwrap();
            assert!(!fl);
            let mut ko = vec![0u64; nn];
            let mut vo = vec![0u32; nn];
            unsafe {
                (g.api.cuCtxSynchronize)();
                (g.api.cuMemcpyDtoH_v2)(ko.as_mut_ptr() as *mut c_void, k0.p, nn * 8);
                (g.api.cuMemcpyDtoH_v2)(vo.as_mut_ptr() as *mut c_void, v0.p, nn * 4);
            }
            let mut want: Vec<(u64, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
            want.sort_by_key(|x| x.0);
            let got: Vec<(u64, u32)> = ko.into_iter().zip(vo).collect();
            let first = (0..nn).find(|&i| got[i] != want[i]);
            assert!(first.is_none(), "sort n={n} first bad {:?}", first.map(|i| (i, got[i], want[i])));
        }
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    #[test]
    #[ignore]
    fn sort_speed() {
        let Some(m) = gpu() else { return };
        let g = m.lock().unwrap();
        unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
        let n = 150_000_000u32;
        let nn = n as usize;
        let tiles = n.div_ceil(RS_TILE) as usize;
        let (k0, k1, v0, v1) = (g.alloc(nn * 8).unwrap(), g.alloc(nn * 8).unwrap(), g.alloc(nn * 4).unwrap(), g.alloc(nn * 4).unwrap());
        let hist = g.alloc(tiles * 256 * 4).unwrap();
        let sums = g.alloc((tiles * 256 / 4096 + 64) * 4 + nn / 1000).unwrap();
        let mut rng = 1u64;
        let keys: Vec<u64> = (0..n).map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); rng }).collect();
        unsafe { (g.api.cuMemcpyHtoD_v2)(k0.p, keys.as_ptr() as *const c_void, nn * 8); (g.api.cuCtxSynchronize)(); }
        let sync = || unsafe { (g.api.cuCtxSynchronize)(); };
        let (mut ki, mut vi, mut ko, mut vo) = (k0.p, v0.p, k1.p, v1.p);
        let mut th = 0.0; let mut ts = 0.0; let mut tc = 0.0;
        for sh in [0u32, 8, 16, 24] {
            let (mut n_, mut sh_, mut h, mut t) = (n, sh, hist.p, tiles as u32);
            let a = std::time::Instant::now();
            g.launch_grid(g.k.rs_hist, tiles as u32, 256, &mut [arg(&mut ki), arg(&mut n_), arg(&mut sh_), arg(&mut h), arg(&mut t)]).unwrap(); sync();
            let b = std::time::Instant::now();
            g.scan(hist.p, tiles as u32 * 256, sums.p, false, false).unwrap(); sync();
            let c = std::time::Instant::now();
            g.launch_grid(g.k.rs_scatter, tiles as u32, 256, &mut [arg(&mut ki), arg(&mut vi), arg(&mut ko), arg(&mut vo), arg(&mut n_), arg(&mut sh_), arg(&mut h), arg(&mut t)]).unwrap(); sync();
            let d = std::time::Instant::now();
            th += (b - a).as_secs_f64(); ts += (c - b).as_secs_f64(); tc += (d - c).as_secs_f64();
            std::mem::swap(&mut ki, &mut ko); std::mem::swap(&mut vi, &mut vo);
        }
        eprintln!("per pass: hist {:.1} ms, scan {:.1} ms, scatter {:.1} ms", th * 250.0, ts * 250.0, tc * 250.0);
        let a = std::time::Instant::now();
        g.scan(v0.p, n, sums.p, true, true).unwrap(); sync();
        eprintln!("scan of n: {:.1} ms", a.elapsed().as_secs_f64() * 1e3);
    }
}
