//! Ratio experiment: code length of candidate models on LZP+BWT output, by segment size,
//! compared against bsc's own QLFC coder on the identical block.
//! Usage: bwtc-eval FILE [segment sizes in KiB...]
use bwtc::model::{prior_from_counts, quant, Config, Lane, Tables};
use rayon::prelude::*;

fn cost_bits(seg: &[u8], cfg: Config, prior: Option<&[u16]>, t: &Tables) -> f64 {
    let mut lane = Lane::new(cfg, prior);
    let mut bits = 0f64;
    for &b in seg {
        lane.code_byte(t, b as u32, |p1, bit| {
            let p = if bit == 1 { p1 } else { 4096 - p1 };
            bits -= (p as f64 / 4096.0).log2();
        });
    }
    bits
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read");
    let segs_kib: Vec<usize> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![0, 4096, 1024, 256, 64] };

    // LZP + BWT exactly as bsc does it
    let mut l = vec![0u8; data.len()];
    let n = bsc_sys::lzp_compress(&data, &mut l, bsc_sys::DEFAULT_LZP_HASH, bsc_sys::DEFAULT_LZP_MINLEN, 24).unwrap_or(0);
    let mut bwt = if n > 0 { l[..n].to_vec() } else { data.clone() };
    let t0 = std::time::Instant::now();
    bsc_sys::bwt_encode(&mut bwt, 24, true).expect("bwt");
    eprintln!("LZP {} -> {} bytes, BWT {:.2}s", data.len(), bwt.len(), t0.elapsed().as_secs_f64());

    // Reference: bsc QLFC adaptive / static on the whole block
    for (name, coder) in [("bsc QLFC adaptive", bsc_sys::CODER_QLFC_ADAPTIVE), ("bsc QLFC static", bsc_sys::CODER_QLFC_STATIC)] {
        let p = bsc_sys::Params { coder, threads: 24, ..Default::default() };
        let c = bsc_sys::compress(&data, &p).unwrap();
        println!("{name:28} {:>12} bytes  {:.4} bpb (whole bsc pipeline)", c.len(), 8.0 * c.len() as f64 / data.len() as f64);
    }

    // Order-1 byte-pair counts over the whole BWT block -> quantised prior
    let mut counts = vec![0u64; 256 * 256];
    let mut prev = 0usize;
    for &b in &bwt {
        counts[prev * 256 + b as usize] += 1;
        prev = b as usize;
    }
    let q: Vec<u8> = counts.iter().map(|&c| quant(c)).collect();
    let prior = prior_from_counts(&q);
    let prior_cost = zstd::bulk::compress(&q, 19).unwrap().len();
    let t = Tables::new();

    let configs: Vec<(&str, Config)> = vec![
        ("o0+o1 mix+APM", Config::default()),
        ("o0+o1 mix, no APM", Config { use_apm: false, ..Default::default() }),
        ("o1 only+APM", Config { use_order0: false, ..Default::default() }),
        ("rates 3/6", Config { fast_rate: 3, slow_rate: 6, ..Default::default() }),
        ("rates 5/8", Config { fast_rate: 5, slow_rate: 8, ..Default::default() }),
    ];
    for &kib in &segs_kib {
        let seg = if kib == 0 { bwt.len() } else { kib * 1024 };
        let lanes = bwt.len().div_ceil(seg);
        for (name, cfg) in &configs {
            if kib != 0 && *name != "o0+o1 mix+APM" && kib != 256 {
                continue; // full config sweep only at whole-block and 256 KiB
            }
            for use_prior in [false, true] {
                let p = if use_prior { Some(&prior[..]) } else { None };
                let bits: f64 = bwt.par_chunks(seg).map(|s| cost_bits(s, *cfg, p, &t)).sum();
                let bytes = bits / 8.0 + if use_prior { prior_cost as f64 } else { 0.0 } + 16.0 * lanes as f64;
                println!("seg {:>6} KiB ({:>5} lanes) {:22} prior={:5} {:>12.0} bytes  {:.4} bpb",
                    if kib == 0 { bwt.len() / 1024 } else { kib }, lanes, name, use_prior, bytes, 8.0 * bytes / data.len() as f64);
            }
        }
    }
    eprintln!("prior table cost: {prior_cost} bytes");
}
