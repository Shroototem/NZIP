//! Cost of splitting QLFC into K independent streams (what a GPU coder would need).
//!   split <file> <offset MiB> <block MiB>
use rayon::prelude::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let off = a[2].parse::<usize>().unwrap() << 20;
    let bs = a[3].parse::<usize>().unwrap() << 20;
    let b = &data[off..(off + bs).min(data.len())];
    let p = bsc_sys::Params { threads: 24, ..Default::default() };
    let mut s = bsc_sys::Staged::lzp(b, &p).unwrap();
    s.bwt(24, true).unwrap();
    let t = s.sorted();
    let t0 = Instant::now();
    let whole = bsc_sys::coder_compress(t, bsc_sys::CODER_QLFC_ADAPTIVE, 24).unwrap().len();
    println!("n={} lz={} default(64-128 streams): {} B  {:.2}s", b.len() >> 20, t.len() >> 20, whole, t0.elapsed().as_secs_f64());
    for k in [256usize, 1024, 4096, 16384] {
        let c = t.len().div_ceil(k);
        let t0 = Instant::now();
        let sum: usize = t.par_chunks(c).map(|x| bsc_sys::coder_compress(x, bsc_sys::CODER_QLFC_ADAPTIVE, 1).unwrap().len()).sum();
        println!("  {k:>5} streams ({:>4} KiB each): {sum} B  {:+.2}%  {:.2}s", c >> 10, 100.0 * (sum as f64 / whole as f64 - 1.0), t0.elapsed().as_secs_f64());
    }
}
