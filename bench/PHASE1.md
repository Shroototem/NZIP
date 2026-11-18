# Phase 1 report: container, classic tiers, GPU-first

Date: 2026-11-17. Machine: RTX 5060 Ti 16 GB, Ryzen 9 3900X (24 threads), 125 GB RAM.
Reproduce (after `source .sandbox/env.sh`):
- `cargo build --release && cargo test --release --workspace`
- `python bench/baselines.py --codecs nzip-fast,nzip-balanced --sets silesia,enwik8,enwik9`
- `python bench/baselines.py --report`

## What was built

- **Rust workspace.**
  - `crates/bsc-sys`: vendored libbsc 3.4 including libcubwt GPU BWT, with safe bindings.
  - `crates/nzip`: library.
  - `crates/nzip-cli`: the `nzip` binary.
- **Container v1** ([format.rs](../crates/nzip/src/format.rs)): every frame header and payload is checksummed (xxh3) before any decoder touches it, then each block and the whole file are checked after decoding.
- **Tiers:**
  - `store`;
  - `fast` (zstd-3, 4 MiB blocks, all cores);
  - `balanced` (bsc LZP+BWT+adaptive QLFC, up to 512 MiB blocks).
- **Automatic store** for incompressible blocks: entropy probe plus a zstd-1 trial.
- **Long-range dedup** ([dedup.rs](../crates/nzip/src/dedup.rs)): content-defined chunks, used only for repeats further away than a block.
- **Transforms:**
  - E8E9 for x86 code, chosen by an empirical trial;
  - bsc record reordering for tables.
- **GPU-first device selection** (`--device auto|gpu|cpu`, default `auto`), detected at run time:

| Stage | Runs on | Why (measured) |
|---|---|---|
| balanced compress, input ≥ 16 MB | **GPU** BWT + CPU LZP/QLFC, pipelined across blocks | GPU BWT 4.4–6.6× faster than 24-thread CPU BWT. Output is byte-identical. |
| balanced compress, input < 16 MB | CPU | GPU start-up (~150 ms) exceeds the gain below ~16 MB |
| fast (zstd) compress | CPU | nvCOMP GPU zstd: 1.56 GB/s at 3.22 bpb, vs our CPU path at 1.7 GB/s and 2.85 bpb |
| all decompression | CPU | bsc GPU inverse is slower (153 vs 191 MB/s), and output is CPU-decodable by design |
| no GPU / no driver | CPU | statically linked CUDA runtime loads the driver at run time, so the binary falls back |

- **Tests:** 18 in total, all passing:
  - round trips across modes, block sizes and thread counts (property-based);
  - every single-byte corruption detected;
  - truncation rejected;
  - output identical for 1 vs 24 threads;
  - **GPU and CPU produce byte-identical files**;
  - the no-GPU fallback, checked with `CUDA_VISIBLE_DEVICES=""`.

## Results

Every round trip was verified byte for byte. MB/s is wall clock including process start and file I/O.

### enwik9 (1 GB)

| | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| **nzip balanced (GPU)** | **169,415,210** | **1.355** | **224.6** | 249.7 |
| bsc -b100e2 (best bsc setting tried) | 182,014,570 | 1.456 | 75.8 | 263.3 |
| bsc default | 195,724,082 | 1.566 | 106.8 | 419.0 |
| zstd -19 | 235,328,939 | 1.883 | 12.7 | 557.5 |
| **nzip fast** | 313,361,673 | 2.507 | **1725.6** | **3088.3** |
| zstd -3 -T0 | 312,109,709 | 2.497 | 1125.2 | 664.1 |

nzip balanced is **7% smaller than bsc's best and 3× faster to compress**. The GPU makes 512 MiB BWT blocks affordable, and those are what give the better ratio.

### enwik8 (100 MB, one block)

| | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| **nzip balanced (GPU)** | 20,803,078 | 1.664 | **138.5** | 183.7 |
| bsc -b100e2 | 20,803,016 | 1.664 | 55.0 | 173.3 |
| nzip fast | 35,575,811 | 2.846 | 1108.7 | 2506.3 |
| zstd -3 -T0 | 35,445,335 | 2.836 | 1215.1 | 743.5 |

### Silesia (12 files, 211.9 MB; each file compressed separately, all under 52 MB)

| | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| **nzip balanced** | 46,518,222 | 1.756 | 75.0 | 135.4 |
| bsc -b100e2 | 46,536,916 | 1.757 | 58.5 | 128.1 |
| xz -9e | 48,456,100 | 1.829 | 2.4 | 136.0 |
| nzip fast | 66,049,368 | 2.493 | 570.7 | 1468.7 |
| zstd -3 -T0 | 66,216,569 | 2.499 | 414.8 | 795.3 |

On tiny files (Canterbury, Calgary), nzip is ~25% slower than the bsc/zstd CLIs. Process start-up and thread-pool setup dominate there; the ratio is identical.

## Where the time goes (one 100 MB block, 24 threads)

| Stage | CPU | GPU |
|---|---:|---:|
| LZP | 0.03 s | (CPU) |
| BWT | 1.26 s | **0.29 s** (+0.1 s one-time CUDA start-up, now overlapped with hashing) |
| QLFC adaptive (entropy coding) | 0.33 s | (CPU) |

With the BWT on the GPU, **QLFC entropy coding on the CPU is now the bottleneck** for single-block inputs. For multi-block inputs the GPU BWT of one block overlaps CPU QLFC of another.

## Open items and next speedups

- QLFC on the GPU, or a GPU-friendly entropy coder for BWT output (rANS on GPU). This is the remaining single-block bottleneck. It overlaps with the Phase 2 GPU rANS work.
- Pipelining within a single block (BWT/QLFC overlap via sub-block QLFC).
- Faster GPU inverse BWT, to make GPU decompression worth using.
- 1 GB blocks would compress enwik9 further (164.2 MB measured) but need about 21 GB of GPU memory. A GPU BWT that streams through host memory could lift that limit.
- Small-file overhead: strip debug info from the release binary and create the thread pool lazily.
