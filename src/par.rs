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

thread_local! {
    /// Whether this thread holds a CPU token (threads start holding one, except decode
    /// pipeline workers, which only take one for actual work).
    static HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

fn held() -> bool {
    HELD.with(|h| h.get())
}

fn set_held(v: bool) {
    HELD.with(|h| h.set(v));
}

/// Waits for a free CPU token and takes it.
fn wait_token() {
    let mut spins = 0u32;
    while !take() {
        spins += 1;
        if spins < 64 {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

/// Runs `f` holding a CPU token, taking one first if this thread has none.
pub fn with_token<R>(f: impl FnOnce() -> R) -> R {
    if held() {
        return f();
    }
    wait_token();
    set_held(true);
    let r = f();
    set_held(false);
    give();
    r
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
    with_token(|| map_held(items, threads, f))
}

fn map_held<T: Sync, U: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    if threads <= 1 || n == 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let fin = AtomicUsize::new(0);
    let mut out: Vec<Option<U>> = (0..n).map(|_| None).collect();
    let slots: Vec<Mutex<&mut Option<U>>> = out.iter_mut().map(Mutex::new).collect();
    let run = |helper: bool| {
        drain(&next, n, helper, |i| {
            **slots[i].lock().unwrap() = Some(f(&items[i]));
            fin.fetch_add(1, Ordering::Release);
        })
    };
    let mut lent = false;
    std::thread::scope(|s| {
        for _ in 1..threads.min(n) {
            s.spawn(|| run(true));
        }
        run(false);
        // helpers still busy: lend this core while waiting for them
        if fin.load(Ordering::Acquire) < n {
            give();
            set_held(false);
            lent = true;
        }
    });
    if lent {
        wait_token();
        set_held(true);
    }
    drop(slots);
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// One `map_ranked` call's jobs as every thread sees them.
struct Ranked {
    /// job indices, costliest first
    order: Vec<usize>,
    cost: Vec<u64>,
    /// jobs claimed (changed only under RANK)
    next: AtomicUsize,
    /// jobs not finished yet
    left: AtomicUsize,
    run: &'static (dyn Fn(usize) + Sync),
}

/// Jobs of every running `map_ranked` call: a free thread always takes the costliest job
/// left in any of them, so the long ones start first wherever they come from.
static RANK: Mutex<Vec<&'static Ranked>> = Mutex::new(Vec::new());

fn claim() -> Option<(&'static Ranked, usize)> {
    let list = RANK.lock().unwrap();
    let mut best: Option<(&'static Ranked, u64)> = None;
    for &r in list.iter() {
        let k = r.next.load(Ordering::Relaxed);
        if k < r.order.len() && best.is_none_or(|b| r.cost[r.order[k]] > b.1) {
            best = Some((r, r.cost[r.order[k]]));
        }
    }
    let (r, _) = best?;
    let k = r.next.fetch_add(1, Ordering::Relaxed);
    Some((r, r.order[k]))
}

static RUNNING: AtomicUsize = AtomicUsize::new(0);
fn run_claimed((r, i): (&'static Ranked, usize)) {
    if crate::sys::dev_flag("NZ_TTRACE") {
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let t = T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64();
        eprintln!("  claim {t:.3}s cost {} running {} free {}", r.cost[i], RUNNING.load(Ordering::Relaxed), FREE.load(Ordering::Relaxed));
    }
    RUNNING.fetch_add(1, Ordering::Relaxed);
    (r.run)(i);
    RUNNING.fetch_sub(1, Ordering::Relaxed);
    r.left.fetch_sub(1, Ordering::Release);
}

/// `map` for jobs of known relative cost: the jobs of all calls running at once (blocks
/// decoded side by side) share one queue, costliest first, so no long job starts last.
pub fn map_ranked<T: Sync, U: Send>(items: &[T], threads: usize, cost: impl Fn(&T) -> u64, f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let n = items.len();
    if n == 0 || threads <= 1 {
        return with_token(|| items.iter().map(f).collect());
    }
    // a caller without a token registers its jobs at once and then works like a helper
    let own = held();
    let mut out: Vec<Option<U>> = (0..n).map(|_| None).collect();
    {
        let slots: Vec<Mutex<&mut Option<U>>> = out.iter_mut().map(Mutex::new).collect();
        let job = |i: usize| **slots[i].lock().unwrap() = Some(f(&items[i]));
        let job: &(dyn Fn(usize) + Sync) = &job;
        // SAFETY: the job is reachable only through `r`, which leaves RANK once every job
        // is claimed, and this function returns only after every claimed job has finished.
        let job: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(job) };
        let cost: Vec<u64> = items.iter().map(cost).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(cost[i]));
        let r = Ranked { order, cost, next: AtomicUsize::new(0), left: AtomicUsize::new(n), run: job };
        let rs: &'static Ranked = unsafe { &*(&r as *const Ranked) };
        RANK.lock().unwrap().push(rs);
        let open = || r.next.load(Ordering::Relaxed) < n;
        let helper = || {
            loop {
                let mut spins = 0u32;
                while !take() {
                    if !open() {
                        return;
                    }
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
                let c = if open() { claim() } else { None };
                let more = c.is_some();
                if let Some(c) = c {
                    let was = held();
                    set_held(true);
                    run_claimed(c);
                    set_held(was);
                }
                give();
                if !more || !open() {
                    return;
                }
            }
        };
        std::thread::scope(|s| {
            for _ in 1..threads.min(n) {
                s.spawn(&helper);
            }
            if own {
                while open() {
                    match claim() {
                        Some(c) => run_claimed(c),
                        None => break,
                    }
                }
            } else {
                helper();
            }
        });
        RANK.lock().unwrap().retain(|&x| !std::ptr::eq(x, rs));
        // jobs of ours still running on other calls' threads: lend the core meanwhile
        if !own {
            while r.left.load(Ordering::Acquire) > 0 {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        } else if r.left.load(Ordering::Acquire) > 0 {
            give();
            while r.left.load(Ordering::Acquire) > 0 {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            while !take() {
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    }
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Run `f(index, chunk)` over mutable chunks in parallel.
pub fn for_chunks<T: Send>(chunks: Vec<&mut [T]>, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    with_token(|| for_chunks_held(chunks, threads, f))
}

fn for_chunks_held<T: Send>(chunks: Vec<&mut [T]>, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
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


/// Ordered pipeline: `next` yields jobs (None = no more); up to `workers` jobs run at
/// once, each on a worker thread holding a CPU token; `done` gets the results in job
/// order. A finished job waits for earlier ones only while fewer than `ahead` results are
/// parked, so one slow job never idles the other workers the way a batch barrier does.
/// The calling thread only reads and writes, so it lends its token to the workers.
pub fn pipeline<J: Send, R: Send>(
    workers: usize,
    ahead: usize,
    tokens: bool,
    mut next: impl FnMut() -> std::io::Result<Option<J>>,
    work: impl Fn(J) -> R + Sync,
    mut done: impl FnMut(R) -> std::io::Result<()>,
) -> std::io::Result<()> {
    use std::collections::BTreeMap;
    use std::sync::mpsc;
    let workers = workers.max(1);
    let (job_tx, job_rx) = mpsc::channel::<(usize, J)>();
    let (res_tx, res_rx) = mpsc::channel::<(usize, R)>();
    let job_rx = Mutex::new(job_rx);
    give();
    let r = std::thread::scope(|s| {
        for _ in 0..workers {
            let res_tx = res_tx.clone();
            let (job_rx, work) = (&job_rx, &work);
            s.spawn(move || {
                loop {
                    let job = job_rx.lock().unwrap().recv();
                    let Ok((i, j)) = job else { break };
                    // without `tokens`, a job starts at once and takes a token only for
                    // the work inside it (decoding: every block's pieces get queued early)
                    let r = if tokens {
                        wait_token();
                        let r = work(j);
                        give();
                        r
                    } else {
                        set_held(false);
                        work(j)
                    };
                    if res_tx.send((i, r)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(res_tx);
        let mut sent = 0usize;
        let mut out = 0usize;
        let mut parked: BTreeMap<usize, R> = BTreeMap::new();
        let mut eof = false;
        let res = (|| -> std::io::Result<()> {
            loop {
                while !eof && sent - out - parked.len() < workers && parked.len() < ahead {
                    match next()? {
                        Some(j) => {
                            job_tx.send((sent, j)).map_err(|_| std::io::Error::other("worker gone"))?;
                            sent += 1;
                        }
                        None => eof = true,
                    }
                }
                if out == sent {
                    return Ok(());
                }
                let (i, r) = res_rx.recv().map_err(|_| std::io::Error::other("worker gone"))?;
                parked.insert(i, r);
                while let Some(r) = parked.remove(&out) {
                    out += 1;
                    done(r)?;
                }
            }
        })();
        drop(job_tx);
        res
    });
    // take the caller's token back
    while !take() {
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    r
}

/// Runs `f` (a wait: a lock, a GPU job) without this thread's CPU token, so another
/// thread can compute meanwhile; takes a token again before returning.
pub fn idle<R>(f: impl FnOnce() -> R) -> R {
    give();
    let r = f();
    let mut spins = 0u32;
    while !take() {
        spins += 1;
        if spins < 64 {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
    r
}
