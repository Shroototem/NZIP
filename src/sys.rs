//! Hardware detection: logical CPUs and RAM, on Linux and Windows.

use std::sync::atomic::{AtomicU64, Ordering};

/// Developer switch from the environment (tuning, tracing, forcing a code path). Only
/// builds with `--features tune` read them: a release binary behaves the same whatever
/// the environment holds.
pub fn dev_var(name: &str) -> Option<String> {
    #[cfg(feature = "tune")]
    {
        std::env::var(name).ok()
    }
    #[cfg(not(feature = "tune"))]
    {
        let _ = name;
        None
    }
}

pub fn dev_flag(name: &str) -> bool {
    dev_var(name).is_some()
}

pub fn cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Asks Windows not to power-throttle this process: Windows 11 otherwise may run a
/// console program whose window is not in front on efficiency cores at low clocks (Intel
/// hybrid CPUs). No effect elsewhere (priority is `set_priority`'s job).
pub fn no_throttle() {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Throttle {
            version: u32,
            control_mask: u32,
            state_mask: u32,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn SetProcessInformation(p: *mut core::ffi::c_void, class: i32, info: *const core::ffi::c_void, size: u32) -> i32;
        }
        // ProcessPowerThrottling = 4; EXECUTION_SPEED in the control mask, clear in the
        // state mask = never throttle
        let t = Throttle { version: 1, control_mask: 1, state_mask: 0 };
        unsafe { SetProcessInformation(GetCurrentProcess(), 4, &t as *const Throttle as *const core::ffi::c_void, std::mem::size_of::<Throttle>() as u32) };
    }
}

static MEM_LIMIT: AtomicU64 = AtomicU64::new(0);

/// Sets the memory budget (the -m option).
pub fn set_mem_limit(bytes: u64) {
    MEM_LIMIT.store(bytes, Ordering::Relaxed);
}

/// Total physical RAM (or the container's limit, if lower).
pub fn total_ram() -> u64 {
    imp::total_ram().unwrap_or(4 << 30)
}

static RAM_PCT: AtomicU64 = AtomicU64::new(99);

/// Sets the share of free RAM nzip may plan to use (the --ram option).
pub fn set_ram_share(pct: u64) {
    RAM_PCT.store(pct.clamp(1, 100), Ordering::Relaxed);
}

/// RAM free for use right now (other programs' memory stays theirs).
pub fn free_ram() -> u64 {
    imp::free_ram().unwrap_or_else(total_ram).min(total_ram())
}

/// Bytes nzip may plan to use: the -m value, else the --ram share (default 99%) of the RAM
/// free at start. Block sizes and the number of blocks in flight stay inside it, in both
/// directions.
pub fn mem_budget() -> u64 {
    static FREE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    match MEM_LIMIT.load(Ordering::Relaxed) {
        0 => (*FREE.get_or_init(free_ram) / 100 * RAM_PCT.load(Ordering::Relaxed)).max(256 << 20),
        m => m,
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn total_ram() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = s.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line[9..].trim().trim_end_matches("kB").trim().parse().ok()?;
        let mut total = kb * 1024;
        // cgroup v2 limit (containers), if any
        if let Ok(l) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
            if let Ok(lim) = l.trim().parse::<u64>() {
                total = total.min(lim);
            }
        }
        Some(total)
    }

    pub fn free_ram() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kb: u64 = line[13..].trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    }
}

#[cfg(windows)]
mod imp {
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
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }
    pub fn total_ram() -> Option<u64> {
        let mut m = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        let ok = unsafe { GlobalMemoryStatusEx(&mut m) };
        if ok != 0 { Some(m.total_phys.min(m.total_virtual)) } else { None }
    }

    pub fn free_ram() -> Option<u64> {
        let mut m = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        let ok = unsafe { GlobalMemoryStatusEx(&mut m) };
        if ok != 0 { Some(m.avail_phys.min(m.avail_virtual)) } else { None }
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn total_ram() -> Option<u64> {
        None
    }
    pub fn free_ram() -> Option<u64> {
        None
    }
}

/// Peak resident memory of this process in bytes, if the OS reports it.
pub fn peak_rss() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        let l = s.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = l[6..].trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Process priority (the --priority option): how much nzip yields the CPU (and disk) to
/// other programs. It never changes how fast nzip runs on an idle machine.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Priority {
    /// below normal (Unix nice 10, Windows BELOW_NORMAL): the desktop stays responsive
    Low,
    /// only when nothing else wants the CPU (nice 19, Windows IDLE)
    Idle,
    /// unchanged
    Normal,
}

/// Lowers this process's CPU (and, on Linux, I/O) priority. Call before any thread starts:
/// on Linux the setting is per thread and new threads inherit it.
pub fn set_priority(p: Priority) {
    if p == Priority::Normal {
        return;
    }
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        }
        // PRIO_PROCESS = 0, who 0 = this process (this thread on Linux)
        unsafe { setpriority(0, 0, if p == Priority::Idle { 19 } else { 10 }) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn syscall(n: i64, ...) -> i64;
        }
        #[cfg(target_arch = "x86_64")]
        const IOPRIO_SET: i64 = 251;
        #[cfg(target_arch = "aarch64")]
        const IOPRIO_SET: i64 = 30;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // IOPRIO_WHO_PROCESS = 1; best effort class (2) at its lowest level (7), or the idle class (3)
            let prio: i64 = if p == Priority::Idle { 3 << 13 } else { (2 << 13) | 7 };
            unsafe { syscall(IOPRIO_SET, 1i64, 0i64, prio) };
        }
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn SetPriorityClass(p: *mut core::ffi::c_void, class: u32) -> i32;
        }
        // BELOW_NORMAL_PRIORITY_CLASS / IDLE_PRIORITY_CLASS
        unsafe { SetPriorityClass(GetCurrentProcess(), if p == Priority::Idle { 0x40 } else { 0x4000 }) };
    }
}

/// Raises the soft limit on open files to the hard limit (Unix): folders of many files are
/// read on every thread at once.
pub fn raise_open_files() {
    #[cfg(target_os = "linux")]
    {
        #[repr(C)]
        struct Rlimit {
            cur: u64,
            max: u64,
        }
        unsafe extern "C" {
            fn getrlimit(res: i32, r: *mut Rlimit) -> i32;
            fn setrlimit(res: i32, r: *const Rlimit) -> i32;
        }
        const RLIMIT_NOFILE: i32 = 7;
        let mut r = Rlimit { cur: 0, max: 0 };
        unsafe {
            if getrlimit(RLIMIT_NOFILE, &mut r) == 0 && r.cur < r.max {
                r.cur = r.max;
                setrlimit(RLIMIT_NOFILE, &r);
            }
        }
    }
}

/// Reads `buf.len()` bytes at `off` without moving a shared file position.
pub fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            match f.seek_read(&mut buf[done..], off + done as u64)? {
                0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                k => done += k,
            }
        }
        Ok(())
    }
}

