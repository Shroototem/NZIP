//! GPU must be bit-exact with the CPU reference, in both directions.
use ncore::{cpu, gpu};

fn inputs() -> Vec<Vec<u8>> {
    let mut x = 0x1234_5678u32;
    let mut noise = |n: usize| -> Vec<u8> { (0..n).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x as u8 }).collect() };
    let text: Vec<u8> = b"It was the best of times, it was the worst of times; ".iter().cycle().take(400_000).copied().collect();
    let mut mixed = text.clone();
    mixed.extend(noise(150_000));
    mixed.extend(std::iter::repeat_n(7u8, 90_000)); // long run: exercises counter halving
    vec![vec![], vec![42], noise(1000), text, mixed, noise(70_000)]
}

#[test]
fn gpu_matches_cpu_both_directions() {
    if !gpu::available() {
        eprintln!("no GPU: skipped");
        return;
    }
    for data in inputs() {
        for lane in [1000usize, 4096, 65_536, 70_001, 1 << 20] {
            let c = cpu::encode(&data, lane);
            let (g, _) = gpu::encode(&data, lane).unwrap();
            assert!(c == g, "GPU stream differs from CPU (len {}, lane {lane})", data.len());
            assert_eq!(gpu::decode(&c).unwrap().0, data, "GPU decode of CPU stream");
            assert_eq!(cpu::decode(&g).unwrap(), data, "CPU decode of GPU stream");
        }
    }
}

#[test]
fn gpu_rejects_corruption() {
    if !gpu::available() {
        return;
    }
    let data: Vec<u8> = b"abcabcabdabcabce".iter().cycle().take(200_000).copied().collect();
    let c = cpu::encode(&data, 8192);
    for p in [20usize, 100, c.len() / 2, c.len() - 3] {
        let mut bad = c.clone();
        bad[p] ^= 0x5A;
        match gpu::decode(&bad) {
            Err(_) => {}
            Ok((d, _)) => assert!(d != data || p >= c.len(), "corruption at {p} went unnoticed with identical output"),
        }
    }
}

/// A tiny simulated card (NZIP_VRAM_LIMIT) forces many small lane batches; streams must
/// stay byte-identical to the CPU reference.
#[test]
fn small_vram_multi_batch_matches_cpu() {
    if !ncore::gpu::available() {
        return;
    }
    let mut d = Vec::new();
    let mut x = 7u32;
    while d.len() < 3_000_000 {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        d.extend_from_slice(if x % 3 == 0 { b"alpha beta " } else { b"gamma\n" });
        d.push(x as u8);
    }
    let lane = 20_000; // 150 lanes
    let cpu = ncore::cpu::encode(&d, lane);
    // ~25 MB of usable memory after the 512 MiB reserve: a handful of lanes per batch.
    unsafe { std::env::set_var("NZIP_VRAM_LIMIT", ((512u64 << 20) + (1 << 20) + 25_000_000).to_string()) };
    let (gpu, _) = ncore::gpu::encode(&d, lane).expect("gpu encode");
    let dec = ncore::gpu::decode(&cpu).expect("gpu decode").0;
    unsafe { std::env::remove_var("NZIP_VRAM_LIMIT") };
    assert!(gpu == cpu, "GPU stream differs under a small VRAM limit");
    assert_eq!(dec, d);
}
