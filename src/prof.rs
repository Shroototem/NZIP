//! Dev-only sampling profiler (tune builds, NZ_PROF=<file>): SIGPROF every 1 ms of process
//! CPU time records the interrupted instruction address; `dump` writes them as offsets into
//! the binary, one per line, for `addr2line -f -i -C -e <binary>`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const CAP: usize = 1 << 22;
static SAMPLES: [AtomicU64; CAP] = [const { AtomicU64::new(0) }; CAP];
static N: AtomicUsize = AtomicUsize::new(0);

#[repr(C)]
struct SigAction {
    handler: usize,
    mask: [u64; 16],
    flags: i32,
    restorer: usize,
}

#[repr(C)]
struct ItimerVal {
    interval: [i64; 2],
    value: [i64; 2],
}

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
    fn setitimer(which: i32, new: *const ItimerVal, old: *mut ItimerVal) -> i32;
}

extern "C" fn on_prof(_sig: i32, _info: *mut u8, ctx: *mut u8) {
    // x86_64 Linux ucontext_t: uc_mcontext.gregs starts at byte 40, RIP is gregs[16]
    let rip = unsafe { *(ctx.add(40 + 16 * 8) as *const u64) };
    let i = N.fetch_add(1, Ordering::Relaxed);
    if i < CAP {
        SAMPLES[i].store(rip, Ordering::Relaxed);
    }
}

pub fn start() {
    if crate::sys::dev_var("NZ_PROF").is_none() {
        return;
    }
    let a = SigAction { handler: on_prof as *const () as usize, mask: [0; 16], flags: 4 | 0x1000_0000, restorer: 0 };
    let t = ItimerVal { interval: [0, 1000], value: [0, 1000] };
    unsafe {
        sigaction(27, &a, std::ptr::null_mut());
        setitimer(2, &t, std::ptr::null_mut());
    }
}

pub fn dump() {
    let Some(path) = crate::sys::dev_var("NZ_PROF") else { return };
    let t = ItimerVal { interval: [0, 0], value: [0, 0] };
    unsafe { setitimer(2, &t, std::ptr::null_mut()) };
    // load address of the binary: first mapping of our own executable
    let exe = std::fs::read_link("/proc/self/exe").unwrap_or_default();
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let base = maps
        .lines()
        .find(|l| l.ends_with(exe.to_str().unwrap_or("?")))
        .and_then(|l| u64::from_str_radix(l.split('-').next()?, 16).ok())
        .unwrap_or(0);
    let n = N.load(Ordering::Relaxed).min(CAP);
    let mut s = String::with_capacity(n * 10);
    s.push_str(&format!("#dump {:x} {:x}\n", dump as *const () as usize, base));
    for v in SAMPLES.iter().take(n) {
        let r = v.load(Ordering::Relaxed);
        s.push_str(&format!("{:x}\n", r.wrapping_sub(base)));
    }
    let _ = std::fs::write(&path, s);
    let _ = std::fs::write(format!("{path}.maps"), &maps);
}
