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
}

struct Gpu {
    api: Api,
    ctx: *mut c_void,
    k: Kernels,
    name: String,
    disabled: bool,
}

// The raw handles are only used while holding the global mutex.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| {
        if std::env::var_os("NZIP_NO_GPU").is_some() {
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
        };
        Ok(Gpu { api, ctx, k, name, disabled: false })
    }
}

/// Device bytes needed to sort a block of `n` bytes.
pub fn vram_need(n: usize) -> u64 {
    // text n, sa 4n, rank 4n, keys 2x8n, vals 2x4n, histogram n/4, + slack
    n as u64 * 34 + (64 << 20)
}

struct Buf<'a> {
    g: &'a Gpu,
    p: DevPtr,
}

impl Drop for Buf<'_> {
    fn drop(&mut self) {
        unsafe { (self.g.api.cuMemFree_v2)(self.p) };
    }
}

#[inline(always)]
fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

impl Gpu {
    fn alloc(&self, bytes: usize) -> Result<Buf<'_>, String> {
        let mut p = 0;
        ck(unsafe { (self.api.cuMemAlloc_v2)(&mut p, bytes.max(4)) }, "cuMemAlloc")?;
        Ok(Buf { g: self, p })
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

    fn free_mem(&self) -> u64 {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { (self.api.cuMemGetInfo_v2)(&mut free, &mut total) };
        free as u64
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

    fn bwt(&self, s: &[u8]) -> Result<(Vec<u8>, Vec<u32>), String> {
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
const GPU_MIN: usize = 1 << 20;

/// BWT of `s`: on the GPU if possible, else on the CPU. Same result either way.
pub fn bwt(s: &[u8]) -> (Vec<u8>, Vec<u32>) {
    if s.len() >= GPU_MIN && s.len() < (u32::MAX as usize >> 1) {
        if let Some(m) = gpu() {
            let mut g = m.lock().unwrap();
            if !g.disabled {
                unsafe { (g.api.cuCtxSetCurrent)(g.ctx) };
                if g.free_mem() >= vram_need(s.len()) {
                    match g.bwt(s) {
                        Ok(r) => return r,
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
    crate::sais::bwt(s)
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
            format!("{} ({:.1} GB free) - suffix sorting in max mode", g.name, g.free_mem() as f64 / 1e9)
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
