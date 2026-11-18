# Phase 2 report: GPU entropy core (`crates/ncore`)

Date: 2026-11-17. Machine: RTX 5060 Ti 16 GB, Ryzen 9 3900X (24 threads).
Reproduce (after `source .sandbox/env.sh`):
- `cargo test --release -p ncore`
- `target/release/ncore-bench data/bench/enwik8/enwik8 256 64 16`

## What was built

The core the neural tier will plug into: **model → 256-way distribution → exact integer frequencies → rANS**, on many independent lanes at once.

| Piece | Design |
|---|---|
| Lanes | Input split into contiguous lanes (default 64 KiB), coded independently. **One GPU warp per lane**: each of its 32 threads owns 8 of the 256 symbols. |
| Quantiser ([lib.rs](../crates/ncore/src/lib.rs)) | Weights → frequencies ≥ 1 summing exactly to 2^16, using one 64-bit division per byte; the remainder goes to the most likely symbol. The neural tier will feed network outputs through the same function. |
| rANS ([rans.rs](../crates/ncore/src/rans.rs)) | 32-bit state, 16-bit renormalisation, 2^16 scale. Frames of 64 Ki symbols, each encoded backwards from a (cum, freq) buffer and flushed separately. |
| GPU kernels ([ncore.cu](../crates/ncore/cuda/ncore.cu)) | Warp-wide sum/argmax/scan for the distribution, ballot-based symbol search, model updates in global memory. **The whole decode loop runs inside one kernel** (no host round-trips). Lanes are processed in batches of 2,048 to bound GPU memory. |
| CPU reference ([cpu.rs](../crates/ncore/src/cpu.rs)) | The specification. Same integer maths, lanes run in parallel with rayon. It is also the **GPU-less fallback decoder**. |
| Test model ([model.rs](../crates/ncore/src/model.rs)) | Adaptive order-1 + hashed order-2 frequency counts. It exists to exercise the core with realistic memory traffic and is **not** a competitive model. |
| Integration | New container codec `ncore` (id 3) and experimental `nzip c -m core-test`: GPU encode, GPU decode, CPU fallback. |

## Exit criteria

| Criterion | Result |
|---|---|
| Bit-exact GPU vs CPU | **Yes.** Streams are byte-identical for every input and lane size tested (empty, 1 byte, noise, text, long runs that force counter halving; lanes from 1,000 B to 1 MiB). Each side decodes the other's output. The same holds for whole containers. |
| Coding ≪ model cost | **Yes.** rANS is **1.5%** of per-byte cost (5 ns of 326 ns on one CPU core); the model is the rest. |
| No host round-trips in decode | **Yes.** One kernel per 2,048-lane batch decodes every byte of its lanes; the host copies only the input words and the output. |

## Throughput (enwik8, 100 MB)

| Lane size | Lanes | bits/byte (test model) | CPU enc / dec (24 threads) | GPU enc / dec, wall | GPU enc / dec, kernel only |
|---|---:|---:|---:|---:|---:|
| 256 KiB | 382 | 3.679 | 34.3 / 32.4 MB/s | 212.7 / 321.2 MB/s | 341 / 356 MB/s |
| **64 KiB** | 1,526 | 3.759 | 34.6 / 32.9 MB/s | **255.8 / 470.3 MB/s** | 344 / **561** MB/s |
| 16 KiB | 6,104 | 4.007 | 34.7 / 33.1 MB/s | 261.6 / 445.6 MB/s | 422 / 544 MB/s |

- The GPU is **7–14× faster** than all 24 CPU threads. Before the quantiser fix (256 divisions per byte → 1) it was 12.5 MB/s CPU and 234 / 370 MB/s GPU.
- End to end through the CLI (`-m core-test`), after the start-up fix below:
  - compress: **318 MB/s** reported, 377 ms per 100 MB including process start (was 177 MB/s);
  - decompress: **327–335 MB/s** reported, 357 ms per 100 MB decode-only (was 280 MB/s); 32.5 MB/s CPU-only (`CUDA_VISIBLE_DEVICES=""`), byte-identical.
- **Lane size trades parallelism for cold start.** This count-based model loses 2–9% going from 256 KiB to 16 KiB lanes; the BWT-output CM experiment lost under 1%. The neural tier will need a **pretrained starting point** (the plan's prior) so that 64 KiB lanes start warm.

## What this tells Phase 3

1. **The plumbing is not the bottleneck.** At 1.5% of cost, the neural tier's speed is set entirely by the model kernel. That is exactly what the persistent on-chip megakernel (research item N1) targets.
2. **The CPU fallback is ~10–15× slower than the GPU**, as expected. The neural tier's CPU decoder will be correspondingly slow; it is a correctness fallback, not a fast path.
3. **Encode is slower than decode** on the GPU: 344 vs 561 MB/s kernel time. The single-thread backward rANS pass per frame and per-lane copies cost more than they should. It can be fixed by spreading the backward pass across the warp; it is not urgent.
4. **Determinism held on the first try** with plain integer arithmetic plus deterministic tie-breaking (lowest index wins argmax). The same discipline carries over to INT8 neural kernels.

## Start-up fix (after the first report)

The CLI was slower than `ncore-bench` because the benchmark starts the GPU before timing and the CLI cannot. Measured per-run costs ([startup.rs](../crates/ncore/examples/startup.rs): `cargo run --release -p ncore --example startup <file>`):

| Cost | Time | What was done |
|---|---:|---|
| Driver probe + CUDA context creation | ~15 + ~92 ms | `ncore::gpu::warmup()` starts on a background thread at the very start of `compress`/`decompress`; the first GPU call waits for it. During compression it is hidden behind hashing, dedup and probes. Decompression has little CPU work to hide it behind. |
| GPU used for tiny inputs | 1 KB file: ~170 ms | Inputs/frames under 6 MiB use the CPU under `--device auto` (`NCORE_GPU_MIN`), and the GPU is not touched at all. 1 KB now takes **3 ms**. |
| Encode: 1,526 per-lane copies + Rust re-packing | ~100 ms | A pack kernel gathers all lanes into one buffer and a single copy writes it straight into the output after the header: **~33 ms**. |
| Decode: word-array conversion + extra 100 MB copy | ~35 ms + a copy | The stream's word bytes go to the GPU as they are, and `decode_into` writes into nzip's buffer directly. |
| First kernel after start-up is slower | +50–70 ms | **Not fixable in software.** A compute or memory-heavy warm-up only moves the cost, since the GPU reaches full clocks on its own timer. A driver clock setting (e.g. locked clocks) is a system change left to the user. |

Measured crossover through the CLI (best of 3, ms):

| Input | 1 MB | 2 MB | 4 MB | 8 MB | 16 MB | 32 MB |
|---|---:|---:|---:|---:|---:|---:|
| compress CPU / GPU | 100 / 228 | 104 / 217 | 143 / 215 | 265 / 226 | 509 / 232 | 996 / 236 |
| decompress CPU / GPU | 108 / 219 | 113 / 219 | 152 / 228 | 286 / 226 | 544 / 245 | 1070 / 270 |

**GPU time is nearly flat from 1 to 32 MB.** All lanes run at once, so a run costs at least one lane's serial latency (~180 ms for 64 KiB at this model's speed) however small the input is. For the neural tier this means per-lane latency, not just throughput, sets small-file speed. Shorter lanes, or faster per-byte steps inside a warp, are the levers. This goes into the Phase 3 kernel design.

The remaining gap to the in-process benchmark on 100 MB decode (~357 ms vs ~210 ms) is CUDA start-up plus the slow first kernel; it shrinks in relative terms as files grow. Writing a 100 MB output can add up to ~80 ms more when the disk is still flushing an earlier file.
