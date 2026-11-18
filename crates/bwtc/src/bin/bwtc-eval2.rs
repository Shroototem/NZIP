//! Ratio experiment for model v2 (rank flags). Usage: bwtc-eval2 FILE [seg KiB ...]
use bwtc::{model::Tables, model2::{Config2, Lane2}};
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read");
    let segs: Vec<usize> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![0, 256, 64] };
    let mut l = vec![0u8; data.len()];
    let n = bsc_sys::lzp_compress(&data, &mut l, bsc_sys::DEFAULT_LZP_HASH, bsc_sys::DEFAULT_LZP_MINLEN, 24).unwrap_or(0);
    let mut bwt = if n > 0 { l[..n].to_vec() } else { data.clone() };
    bsc_sys::bwt_encode(&mut bwt, 24, true).expect("bwt");
    let p = bsc_sys::Params { coder: bsc_sys::CODER_QLFC_ADAPTIVE, threads: 24, ..Default::default() };
    let q = bsc_sys::compress(&data, &p).unwrap().len();
    println!("{}: bsc QLFC adaptive {q} bytes ({:.4} bpb)", args[1], 8.0 * q as f64 / data.len() as f64);
    let t = Tables::new();
    for &kib in &segs {
        let seg = if kib == 0 { bwt.len() } else { kib * 1024 };
        let lanes = bwt.len().div_ceil(seg);
        let t0 = std::time::Instant::now();
        let (bits, nd) = bwt.par_chunks(seg).map(|s| {
            let mut lane = Lane2::new(Config2::default(), &t);
            let (mut bits, mut nd) = (0f64, 0u64);
            let mut f = |p: u32, b: Option<u32>| {
                let b = b.unwrap();
                let pp = if b == 1 { p } else { 4096 - p };
                bits -= (pp as f64 / 4096.0).log2();
                nd += 1;
                b
            };
            for &b in s { lane.code(&t, Some(b), &mut f); }
            (bits, nd)
        }).reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let bytes = bits / 8.0 + 16.0 * lanes as f64;
        println!("  v2 seg {:>6} KiB ({:>5} lanes): {:>11.0} bytes ({:.4} bpb, {:+.2}% vs QLFC), {:.2} decisions/byte, {:.1}s",
            seg / 1024, lanes, bytes, 8.0 * bytes / data.len() as f64, 100.0 * (bytes / q as f64 - 1.0), nd as f64 / bwt.len() as f64, t0.elapsed().as_secs_f64());
    }
}
