//! Minimal scoped thread pool helpers (std only).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Run `f` over `items` on up to `threads` threads; results in input order.
pub fn map<T: Sync, U: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    if threads <= 1 || n == 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<U>> = (0..n).map(|_| None).collect();
    let slots: Vec<Mutex<&mut Option<U>>> = out.iter_mut().map(Mutex::new).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.min(n) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let r = f(&items[i]);
                **slots[i].lock().unwrap() = Some(r);
            });
        }
    });
    drop(slots);
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Run `f(index, chunk)` over mutable chunks in parallel.
pub fn for_chunks<T: Send>(chunks: Vec<&mut [T]>, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let work: Vec<Mutex<Option<&mut [T]>>> = chunks.into_iter().map(|c| Mutex::new(Some(c))).collect();
    let next = AtomicUsize::new(0);
    let n = work.len();
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let c = work[i].lock().unwrap().take().unwrap();
                f(i, c);
            });
        }
    });
}

/// Global memory pool for heavy per-thread work (e.g. LZ sub-blocks): work that needs
/// `bytes` waits until that much of the budget is free, so running every block on every
/// thread never exceeds the memory budget. Units are MiB.
struct Pool {
    free: Mutex<u64>,
    cv: std::sync::Condvar,
    cap: u64,
}

fn pool() -> &'static Pool {
    static P: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        // half the budget: the other half holds block buffers
        let cap = (crate::sys::mem_budget() / 2 >> 20).max(1);
        Pool { free: Mutex::new(cap), cv: std::sync::Condvar::new(), cap }
    })
}

/// Runs `f` once `bytes` of the memory pool are available (always runs eventually: a
/// request larger than the whole pool just waits for exclusive use).
pub fn with_memory<R>(bytes: u64, f: impl FnOnce() -> R) -> R {
    let p = pool();
    let need = (bytes >> 20).max(1).min(p.cap);
    {
        let mut free = p.free.lock().unwrap();
        while *free < need {
            free = p.cv.wait(free).unwrap();
        }
        *free -= need;
    }
    let r = f();
    *p.free.lock().unwrap() += need;
    p.cv.notify_all();
    r
}

