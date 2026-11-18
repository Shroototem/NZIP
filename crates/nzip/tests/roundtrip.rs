//! Round-trip and corruption tests for the whole pipeline.

use nzip::{compress, decompress, Mode, Options};
use proptest::prelude::*;

fn opts(mode: Mode, block: usize, threads: usize) -> Options {
    Options { mode, block_size: block, threads, ..Options::default() }
}

const MODES: [Mode; 3] = [Mode::Store, Mode::Fast, Mode::Max];

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut r = rng(seed);
    (0..n).map(|_| r() as u8).collect()
}

fn text(n: usize, seed: u64) -> Vec<u8> {
    let words = ["the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ", "compression ", "\n", "of ", "data "];
    let mut r = rng(seed);
    let mut v = Vec::with_capacity(n + 16);
    while v.len() < n {
        v.extend_from_slice(words[(r() % words.len() as u64) as usize].as_bytes());
    }
    v.truncate(n);
    v
}

/// Synthetic x86-like code: many CALLs to a few targets using relative displacements.
fn x86ish(n: usize, seed: u64) -> Vec<u8> {
    let mut r = rng(seed);
    let targets: Vec<u32> = (0..64).map(|_| (r() % n as u64) as u32).collect();
    let mut v = Vec::with_capacity(n + 16);
    while v.len() < n {
        match r() % 4 {
            0 => {
                let t = targets[(r() % 64) as usize];
                let rel = t.wrapping_sub(v.len() as u32 + 5);
                v.push(0xE8);
                v.extend_from_slice(&rel.to_le_bytes());
            }
            1 => v.extend_from_slice(&[0x48, 0x89, 0xC7]),
            2 => v.extend_from_slice(&[0x8B, 0x45, (r() % 32) as u8 * 4]),
            _ => v.push(0x90),
        }
    }
    v.truncate(n);
    v
}

/// Fixed 12-byte records with slowly varying fields (table-like).
fn table(n: usize, seed: u64) -> Vec<u8> {
    let mut r = rng(seed);
    let mut v = Vec::with_capacity(n + 12);
    let mut i = 0u32;
    while v.len() < n {
        v.extend_from_slice(&i.to_le_bytes());
        v.extend_from_slice(&((i / 7) as u16).to_le_bytes());
        v.extend_from_slice(&[(r() % 4) as u8, 0, 0x41, 0x42, (i % 3) as u8, 0xFF]);
        i += 1;
    }
    v.truncate(n);
    v
}

fn check(data: &[u8], o: &Options) {
    let c = compress(data, o).expect("compress");
    let d = decompress(&c, o.threads).expect("decompress");
    assert!(d == data, "round trip mismatch: mode {:?}, len {}", o.mode, data.len());
}

#[test]
fn edge_sizes() {
    for mode in MODES {
        for n in [0usize, 1, 2, 5, 100, 65535, 65536, 65537, 3 * 65536 + 7] {
            check(&text(n, n as u64), &opts(mode, 64 * 1024, 4));
            check(&noise(n, n as u64), &opts(mode, 64 * 1024, 4));
        }
    }
}

#[test]
fn structured_inputs_all_modes() {
    let inputs = [text(700_000, 1), x86ish(700_000, 2), table(700_000, 3), noise(300_000, 4), vec![0u8; 500_000]];
    for data in &inputs {
        for mode in MODES {
            for (block, threads) in [(64 * 1024, 1), (256 * 1024, 8), (0, 0)] {
                check(data, &opts(mode, block, threads));
            }
        }
    }
}

#[test]
fn transforms_trigger() {
    let x = x86ish(1 << 20, 9);
    let c = compress(&x, &opts(Mode::Max, 0, 4)).unwrap();
    let (_, frames) = nzip::info(&c).unwrap();
    assert!(frames.iter().any(|f| f.transform & nzip::format::TF_E8E9 != 0), "e8e9 not applied to x86-like data");
    assert_eq!(decompress(&c, 4).unwrap(), x);

    let t = table(1 << 20, 10);
    let c = compress(&t, &opts(Mode::Max, 0, 4)).unwrap();
    let (_, frames) = nzip::info(&c).unwrap();
    assert!(frames.iter().any(|f| f.transform & nzip::format::TF_REORDER != 0), "record reorder not applied to table");
    assert_eq!(decompress(&c, 4).unwrap(), t);
}

#[test]
fn incompressible_is_stored() {
    let n = noise(4 << 20, 11);
    for mode in [Mode::Fast, Mode::Max] {
        let c = compress(&n, &opts(mode, 1 << 20, 4)).unwrap();
        let (_, frames) = nzip::info(&c).unwrap();
        // Data frames are stored; a streaming encoder may also emit an (empty) dedup recipe.
        assert!(frames.iter().filter(|f| f.kind == nzip::format::KIND_DATA).all(|f| f.codec == nzip::format::CODEC_STORE));
        assert!(c.len() < n.len() + 4096, "store overhead too big: {}", c.len() - n.len());
        assert_eq!(decompress(&c, 4).unwrap(), n);
    }
}

#[test]
fn dedup_far_repeats() {
    // 2 MiB of text repeated after 3 MiB of other data; 256 KiB blocks so the repeat is
    // far outside any one block.
    let a = text(2 << 20, 21);
    let mut data = a.clone();
    data.extend_from_slice(&noise(3 << 20, 22));
    data.extend_from_slice(&a);
    data.extend_from_slice(&text(100_000, 23));
    for mode in [Mode::Fast, Mode::Max] {
        let o = opts(mode, 256 * 1024, 8);
        let c = compress(&data, &o).unwrap();
        let (fh, _) = nzip::info(&c).unwrap();
        assert!(fh.flags & nzip::format::FLAG_DEDUP != 0, "dedup not used");
        assert_eq!(decompress(&c, 8).unwrap(), data);
        let no = Options { dedup: false, ..o };
        let c2 = compress(&data, &no).unwrap();
        assert!(c.len() < c2.len(), "dedup didn't help: {} vs {}", c.len(), c2.len());
    }
}

/// Inputs bigger than the memory budget: many waves, several dedup segments with copies
/// that cross segments. The file must be byte-identical whatever the budget, and decode
/// identically with a tiny decode budget too.
#[test]
fn output_independent_of_memory_budget() {
    let a = text(3 << 20, 71);
    let mut data = a.clone();
    data.extend_from_slice(&x86ish(2 << 20, 72));
    data.extend_from_slice(&noise(1 << 20, 73));
    data.extend_from_slice(&a); // repeat far away, in a later segment
    data.extend_from_slice(&text(1 << 20, 74));
    for (mode, dedup) in [(Mode::Fast, true), (Mode::Max, true), (Mode::Max, false), (Mode::Store, false)] {
        let base = Options { dedup, dedup_segment: 2 << 20, ..opts(mode, 256 * 1024, 8) };
        let big = compress(&data, &Options { mem_limit: 1 << 30, ..base.clone() }).unwrap();
        let small = compress(&data, &Options { mem_limit: 16 << 20, ..base.clone() }).unwrap();
        assert!(big == small, "{mode:?} dedup={dedup}: output depends on the memory budget");
        let (fh, frames) = nzip::info(&big).unwrap();
        assert_eq!(fh.flags & nzip::format::FLAG_DEDUP != 0, dedup && mode != Mode::Store);
        if dedup && mode != Mode::Store {
            let recipes = frames.iter().filter(|f| f.kind == nzip::format::KIND_RECIPE).count();
            assert!(recipes >= 4, "expected several dedup segments, got {recipes}");
        }
        let mut out = Vec::new();
        nzip::decompress_to(&big, &mut out, 8, nzip::Device::Auto, 16 << 20).unwrap();
        assert_eq!(out, data, "{mode:?} dedup={dedup}");
    }
}

#[test]
fn deterministic_across_thread_counts() {
    let data = text(3 << 20, 31);
    for mode in MODES {
        let a = compress(&data, &opts(mode, 1 << 20, 1)).unwrap();
        let b = compress(&data, &opts(mode, 1 << 20, 24)).unwrap();
        assert!(a == b, "output differs by thread count in {mode:?}");
    }
}

#[test]
fn gpu_and_cpu_produce_identical_files() {
    if !nzip::gpu_available() {
        eprintln!("no GPU: skipped");
        return;
    }
    let mut data = text(3 << 20, 51);
    data.extend_from_slice(&x86ish(1 << 20, 52));
    data.extend_from_slice(&table(1 << 20, 53));
    for block in [1 << 20, 0] {
        let mut o = opts(Mode::Max, block, 8);
        o.device = nzip::Device::Gpu;
        let g = compress(&data, &o).unwrap();
        o.device = nzip::Device::Cpu;
        let c = compress(&data, &o).unwrap();
        assert!(g == c, "GPU and CPU containers differ (block {block})");
        assert_eq!(decompress(&g, 8).unwrap(), data);
    }
}

#[test]
fn core_test_mode_gpu_and_cpu_interoperate() {
    let mut data = text(900_000, 61);
    data.extend_from_slice(&x86ish(200_000, 62));
    let mut o = opts(Mode::CoreTest, 0, 8);
    o.device = nzip::Device::Cpu;
    let c_cpu = compress(&data, &o).unwrap();
    assert_eq!(nzip::decompress_with(&c_cpu, 8, nzip::Device::Cpu).unwrap(), data);
    if nzip::gpu_available() {
        o.device = nzip::Device::Gpu;
        let c_gpu = compress(&data, &o).unwrap();
        assert!(c_gpu == c_cpu, "GPU and CPU ncore containers differ");
        // Forced GPU decode (Auto would pick the CPU for an input this small).
        assert_eq!(nzip::decompress_with(&c_cpu, 8, nzip::Device::Gpu).unwrap(), data);
        assert_eq!(nzip::decompress_with(&c_gpu, 8, nzip::Device::Cpu).unwrap(), data);
        assert_eq!(nzip::decompress_with(&c_gpu, 8, nzip::Device::Auto).unwrap(), data);
    }
}

#[test]
fn truncation_is_rejected() {
    let data = text(400_000, 41);
    for mode in MODES {
        let c = compress(&data, &opts(mode, 64 * 1024, 2)).unwrap();
        for cut in [0, 1, 39, 40, 41, 79, 80, c.len() / 2, c.len() - 1] {
            assert!(decompress(&c[..cut], 2).is_err(), "truncated at {cut} accepted");
        }
        let mut longer = c.clone();
        longer.push(0);
        assert!(decompress(&longer, 2).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn roundtrip_arbitrary(data in proptest::collection::vec(any::<u8>(), 0..200_000), m in 0usize..3, b in 0usize..3) {
        let block = [64 * 1024, 100_000, 0][b];
        check(&data, &opts(MODES[m], block, 3));
    }

    #[test]
    fn roundtrip_runs(runs in proptest::collection::vec((any::<u8>(), 1usize..5000), 0..200), m in 0usize..3) {
        let data: Vec<u8> = runs.iter().flat_map(|&(b, n)| std::iter::repeat_n(b, n)).collect();
        check(&data, &opts(MODES[m], 64 * 1024, 2));
    }

    /// Any single-byte corruption must be detected (error), never panic or return wrong data.
    #[test]
    fn corruption_detected(seed in any::<u64>(), pos_frac in 0.0f64..1.0, xor in 1u8..=255, m in 0usize..3) {
        let data = text(150_000 + (seed % 50_000) as usize, seed);
        let c = compress(&data, &opts(MODES[m], 64 * 1024, 2)).unwrap();
        let mut bad = c.clone();
        let pos = ((c.len() - 1) as f64 * pos_frac) as usize;
        bad[pos] ^= xor;
        prop_assert!(decompress(&bad, 2).is_err(), "corruption at {} of {} not detected", pos, c.len());
    }
}
