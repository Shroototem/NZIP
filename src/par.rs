//! Minimal scoped thread pool helpers (std only).

use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// CPU tokens: at most `limit` threads do work at once, across all nested `map` calls.
/// The calling thread of a `map` always works (it already holds a token: the main thread's
/// own, or the one it took as a helper); helpers take a token per item and give it back
/// after, so nested maps share the cores and none can deadlock.
static FREE: AtomicIsize = AtomicIsize::new(isize::MAX);

/// Caps the number of threads working at once (the main thread counts as one).
pub fn set_threads(n: usize) {
    FREE.store(n.max(1) as isize - 1, Ordering::SeqCst);
}

fn take() -> bool {
    let mut f = FREE.load(Ordering::Relaxed);
    while f > 0 {
        match FREE.compare_exchange_weak(f, f - 1, Ordering::Acquire, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(v) => f = v,
        }
    }
    false
}

fn give() {
    FREE.fetch_add(1, Ordering::Release);
}

/// Runs item indices from `next` until none are left; helpers hold a token per item.
fn drain(next: &AtomicUsize, n: usize, helper: bool, mut work: impl FnMut(usize)) {
    loop {
        if helper {
            // wait for a free core, unless the work is gone
            let mut spins = 0u32;
            while !take() {
                if next.load(Ordering::Relaxed) >= n {
                    return;
                }
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i < n {
            work(i);
        }
        if helper {
            give();
        }
        if i >= n {
            return;
        }
    }
}

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
    let run = |helper: bool| drain(&next, n, helper, |i| **slots[i].lock().unwrap() = Some(f(&items[i])));
    std::thread::scope(|s| {
        for _ in 1..threads.min(n) {
            s.spawn(|| run(true));
        }
        run(false);
    });
    drop(slots);
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Run `f(index, chunk)` over mutable chunks in parallel.
pub fn for_chunks<T: Send>(chunks: Vec<&mut [T]>, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let work: Vec<Mutex<Option<&mut [T]>>> = chunks.into_iter().map(|c| Mutex::new(Some(c))).collect();
    let next = AtomicUsize::new(0);
    let n = work.len();
    let run = |helper: bool| drain(&next, n, helper, |i| f(i, work[i].lock().unwrap().take().unwrap()));
    std::thread::scope(|s| {
        for _ in 1..threads.clamp(1, n.max(1)) {
            s.spawn(|| run(true));
        }
        run(false);
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

