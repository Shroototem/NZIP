//! Throughput of the entropy core: CPU reference (all cores) vs GPU, by lane size.
//! Usage: ncore-bench FILE [lane KiB ...]
use ncore::{cpu, gpu, model::O2Model, quantize, rans};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read");
    let lanes: Vec<usize> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![1024, 256, 64, 16] };
    let mb = data.len() as f64 / 1e6;
    println!("{}: {:.1} MB, GPU available: {}", args[1], mb, gpu::available());

    // Cost split on one CPU core: model (weights+quantize+update) vs rANS coding.
    let n = data.len().min(4 << 20);
    let mut m = O2Model::new();
    let (mut r, mut f) = ([0u32; 256], [0u32; 256]);
    let t0 = Instant::now();
    let mut cf = Vec::with_capacity(n);
    for &b in &data[..n] {
        m.weights(&mut r);
        quantize(&r, &mut f);
        let cum: u32 = f[..b as usize].iter().sum();
        cf.push((cum, f[b as usize]));
        m.update(b as u32);
    }
    let t_model = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut x = rans::L;
    let mut w = Vec::with_capacity(n);
    for &(c, q) in cf.iter().rev() {
        rans::put(&mut x, c, q, &mut w);
    }
    let t_rans = t1.elapsed().as_secs_f64();
    println!("cost split (1 core, {} bytes): model {:.1} ns/B, rANS {:.1} ns/B ({:.1}% of total)",
        n, 1e9 * t_model / n as f64, 1e9 * t_rans / n as f64, 100.0 * t_rans / (t_model + t_rans));

    for &kib in &lanes {
        let lane = kib * 1024;
        let t = Instant::now();
        let c = cpu::encode(&data, lane);
        let ce = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let d = cpu::decode(&c).unwrap();
        let cd = t.elapsed().as_secs_f64();
        assert!(d == data);
        print!("lane {:>5} KiB ({:>6} lanes): {:.3} bpb | CPU enc {:6.1} dec {:6.1} MB/s",
            kib, data.len().div_ceil(lane), 8.0 * c.len() as f64 / data.len() as f64, mb / ce, mb / cd);
        if gpu::available() {
            let t = Instant::now();
            let (g, ke) = gpu::encode(&data, lane).unwrap();
            let ge = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (d2, kd) = gpu::decode(&g).unwrap();
            let gd = t.elapsed().as_secs_f64();
            assert!(g == c && d2 == data, "GPU mismatch");
            print!(" | GPU enc {:6.1} (kernel {:6.1}) dec {:6.1} (kernel {:6.1}) MB/s",
                mb / ge, mb / (ke as f64 / 1e3), mb / gd, mb / (kd as f64 / 1e3));
        }
        println!();
    }
}
