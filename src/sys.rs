//! Hardware detection: logical CPUs and RAM, on Linux and Windows.

use std::sync::atomic::{AtomicU64, Ordering};

pub fn cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
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

/// Bytes nzip may plan to use: the -m value, else 90% of total RAM. Block sizes and the
/// number of blocks in flight are chosen to stay inside it, in both directions.
pub fn mem_budget() -> u64 {
    match MEM_LIMIT.load(Ordering::Relaxed) {
        0 => total_ram() / 10 * 9,
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
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn total_ram() -> Option<u64> {
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
