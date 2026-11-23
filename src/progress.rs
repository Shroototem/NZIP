//! Optional progress bar on stderr (the -p option).
//!
//! Finished blocks are counted exactly; inside a block the engines report the bytes they
//! have coded, so long blocks still move the bar. The bar shows whichever is further.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

static ON: AtomicBool = AtomicBool::new(false);
static TOTAL: AtomicU64 = AtomicU64::new(0);
static BLOCKS: AtomicU64 = AtomicU64::new(0);
static WORK: AtomicU64 = AtomicU64::new(0);

/// An engine coded `bytes` more of its input.
#[inline]
pub fn work(bytes: usize) {
    if ON.load(Ordering::Relaxed) {
        WORK.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// A block of `bytes` input bytes is finished.
pub fn block(bytes: u64) {
    if ON.load(Ordering::Relaxed) {
        BLOCKS.fetch_add(bytes, Ordering::Relaxed);
    }
}

pub struct Bar {
    stop: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Starts drawing a bar for `total` bytes, labelled `what`; it is erased when dropped.
pub fn start(what: &str, total: u64, enabled: bool) -> Bar {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    if !enabled {
        return Bar { stop, thread: None };
    }
    TOTAL.store(total.max(1), Ordering::Relaxed);
    BLOCKS.store(0, Ordering::Relaxed);
    WORK.store(0, Ordering::Relaxed);
    ON.store(true, Ordering::Relaxed);
    let what = what.to_string();
    let s2 = stop.clone();
    let thread = std::thread::spawn(move || {
        let t0 = Instant::now();
        while !s2.load(Ordering::Relaxed) {
            draw(&what, t0);
            for _ in 0..5 {
                if s2.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        eprint!("\r{:79}\r", "");
        let _ = std::io::stderr().flush();
    });
    Bar { stop, thread: Some(thread) }
}

fn draw(what: &str, t0: Instant) {
    let total = TOTAL.load(Ordering::Relaxed);
    let done = BLOCKS.load(Ordering::Relaxed).max(WORK.load(Ordering::Relaxed).min(total * 99 / 100)).min(total);
    let frac = done as f64 / total as f64;
    let secs = t0.elapsed().as_secs_f64();
    let speed = done as f64 / 1e6 / secs.max(1e-3);
    let eta = if done > 0 { ((total - done) as f64 / (done as f64 / secs)).round() as u64 } else { 0 };
    const W: usize = 24;
    let fill = ((frac * W as f64) as usize).min(W);
    let bar: String = "#".repeat(fill) + &".".repeat(W - fill);
    let eta = if done > 0 { format!("ETA {}:{:02}", eta / 60, eta % 60) } else { "ETA --:--".into() };
    eprint!("\r{what} [{bar}] {:5.1}%  {:.0}/{:.0} MB  {speed:.1} MB/s  {eta}  ", frac * 100.0, done as f64 / 1e6, total as f64 / 1e6);
    let _ = std::io::stderr().flush();
}

impl Drop for Bar {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        ON.store(false, Ordering::Relaxed);
    }
}
