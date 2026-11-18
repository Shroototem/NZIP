//! Times each bsc stage on blocks of a file and checks the staged output equals
//! bsc_compress.   cargo run --release -p bsc-sys --example stages -- <file> <block MiB> [threads] [blocks]
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let bs = a[2].parse::<usize>().unwrap() << 20;
    let t = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(24);
    let nb = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let skip = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(0);
    let use_gpu = a.get(6).map(|v| v == "1").unwrap_or(true);
    let p = bsc_sys::Params { threads: t, ..Default::default() };
    bsc_sys::gpu_warmup();
    println!("free VRAM {} MiB", bsc_sys::gpu_free_memory() >> 20);
    for (i, b) in data.chunks(bs).enumerate().skip(skip).take(nb) {
        let t0 = Instant::now();
        let s = bsc_sys::Staged::lzp(b, &p).unwrap();
        let t_lzp = t0.elapsed().as_secs_f64();
        let lz = s.sort_len();
        let mut g = bsc_sys::Staged::lzp(b, &p).unwrap();
        let t1 = Instant::now();
        let fits = bsc_sys::gpu_free_memory().saturating_sub(256 << 20) / 21 >= lz;
        g.bwt(t, use_gpu).unwrap();
        let t_gpu = t1.elapsed().as_secs_f64();
        let mut c = s;
        let t2 = Instant::now();
        c.bwt(t, false).unwrap();
        let t_cpu = t2.elapsed().as_secs_f64();
        let t3 = Instant::now();
        let out = c.finish(b, &p).unwrap();
        let t_code = t3.elapsed().as_secs_f64();
        let og = g.finish(b, &p).unwrap();
        let t4 = Instant::now();
        let reference = bsc_sys::compress(b, &bsc_sys::Params { gpu: false, ..p }).unwrap();
        let t_whole = t4.elapsed().as_secs_f64();
        println!("blk {i}: n={} lz={} ({:.0}%) fits={fits} lzp {t_lzp:.2}s bwt gpu {t_gpu:.2}s cpu {t_cpu:.2}s code {t_code:.2}s | whole cpu {t_whole:.2}s | out {} same={} gpu_same={}",
            b.len() >> 20, lz >> 20, 100.0 * lz as f64 / b.len() as f64, out.len(), out == reference, og == reference);
    }
}
