# Phase 0 report: baselines and GPU measurements

Date: 2026-11-17. Machine: RTX 5060 Ti 16 GB, Ryzen 9 3900X, 125 GB RAM.
Reproduce (after `source .sandbox/env.sh`):
- `python bench/baselines.py --report`
- `python bench/gpu_micro.py`

Raw data: `bench/results/baselines.jsonl`, `bench/results/gpu_micro.json`.

Every baseline round trip was verified byte for byte (516/516 for the fast codecs).

## 1. GPU: what the card can actually do

| Measurement | Result | What it means |
|---|---|---|
| INT8 tensor-core GEMM (int32 accumulate) | **~145–155 TOPS peak**; 115–145 at decode shapes with K ≥ 1024; only 45–70 at K = 256 | Usable budget ≈ **125 TOPS**. Layers should be ≥ 512–1024 wide to use it. |
| FP16 / BF16 GEMM | **~45 TFLOPS** | **INT8 is about 3× faster than FP16 on this card.** The integer-only decision costs nothing in speed; it is the *fast* path. |
| FP8 GEMM | ~150 TFLOPS | Same as INT8, so FP8 has no speed advantage. That makes it easy to rule out, given we already excluded it for portability. |
| DRAM copy bandwidth | 324 GB/s (spec 448) | |
| L2-resident read | 1.35–1.53 TB/s at 16–24 MB; falls to ~350 GB/s at 48 MB | Keep hot weights plus activations **under about 24 MB**. |
| Kernel launch | 5.4 µs plain, **0.7 µs** in a CUDA Graph | Graphs are required. A persistent kernel would be better still for small models. |

### Decode-step throughput for byte-level models

One model call per byte, B parallel streams, captured in a CUDA Graph:

| Model (dense INT8) | Params | GEMM-only ceiling (≈ fused kernel) | Stock PyTorch, unfused |
|---|---:|---:|---:|
| d256 × 2 layers | 0.7 M | **41.6 MB/s** | 7.6 MB/s |
| d384 × 4 | 3.1 M | **13.1 MB/s** | 2.7 MB/s |
| d512 × 4 | 5.4 M | **8.5 MB/s** | 2.2 MB/s |
| d512 × 8 | 10.6 M | 4.3 MB/s | 1.1 MB/s |
| d768 × 8 | 23.8 M | 2.3 MB/s | 0.65 MB/s |
| d1024 × 12 | 63 M | 1.0 MB/s | 0.32 MB/s |
| d1536 × 12 | 142 M | 0.5 MB/s | 0.2 MB/s |

(Best batch size shown; B = 2048 was usually best, and B = 8192 got *slower* because activations spill out of L2.)

**Rule of thumb, measured:** decode MB/s ≈ 125 TOPS ÷ (2 × params used per byte).

Findings:

1. **Fused kernels are mandatory.** Stock PyTorch reaches only 10–30% of the GEMM ceiling. The elementwise work between GEMMs (requantization, recurrence, CDF) moves int32 tensors through memory and dominates. In a custom kernel that work lives in registers, in the GEMM epilogue. This confirms Phase 2–3's plan for hand-written CUDA.
2. **The neural tier must be small per byte.** 10–30 MB/s means about **1.5–4M params touched per byte**, not the 2–8M the plan assumed. Context models and the mixer are nearly free next to that.
3. **Max mode (patching + MoE) is confirmed as the right idea, at the lower end of the estimate.** About 18M ops per byte works out to a ceiling of about 7 MB/s. Assuming fused kernels reach 70–80% of the ceiling, that gives **about 4–6 MB/s** (plan said 3–10). A dense 142M byte model would manage 0.5 MB/s, which shows why the redesign was needed.
4. **Ternary weights** give no compute gain on this GPU (the INT8 path does the work). Their value here is fitting large weights into the ~24 MB L2 sweet spot, plus CPU speed. That matches PLAN.md section 9.

## 2. Classic compressor baselines

Sorted by size; MB/s is wall-clock including process start. That makes small-file speeds pessimistic for fast codecs.

### enwik8 (100 MB)

| codec | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| bsc -b100e2 | 20,803,016 | **1.664** | 55.0 | 173.3 |
| bsc -m6e2 -G (GPU ST) | 21,971,914 | 1.758 | 157.9 | 14.8 |
| bsc (default) | 22,463,250 | 1.797 | 71.6 | 318.3 |
| xz -9e | 24,831,648 | 1.987 | 1.7 | 129.6 |
| zstd --ultra -22 --long=31 | 25,272,471 | 2.022 | 1.7 | 660.5 |
| brotli -q11 | 25,742,001 | 2.059 | 0.55 | 389.6 |
| zstd -19 | 26,944,227 | 2.156 | 5.8 | 783.1 |
| bzip2 -9 | 29,008,758 | 2.321 | 15.7 | 31.4 |
| zstd -3 | 35,445,335 | 2.836 | 1215 | 743.5 |
| lz4 -1 | 57,004,012 | 4.560 | 2597 | 1585 |

Published, not measured here, for reference on enwik8:
- ts_zip 13,825,741 (1.106 bpb);
- NNCP 14,915,298 (1.19);
- cmix v19 14,837,987 (1.19).

### Silesia (12 files, 211.9 MB, mixed text and binaries)

| codec | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| bsc -b100e2 | 46,536,916 | **1.757** | 58.5 | 128.1 |
| bsc (default) | 47,428,196 | 1.790 | 59.1 | 200.2 |
| xz -9e | 48,456,004 | 1.829 | 2.2 | 121.5 |
| brotli -q11 | 49,564,563 | 1.871 | 0.58 | 370.8 |
| zstd -22 --long | 52,364,240 | 1.977 | 2.4 | 755.0 |
| zstd -19 | 52,895,350 | 1.997 | 4.0 | 795.9 |
| zstd -3 | 66,216,569 | 2.499 | 414.8 | 795.3 |

Canterbury and Calgary: same ordering, with bsc on top. Full tables come from `python bench/baselines.py --report`.

### enwik9 (partial: the run was stopped after zstd -19)

| codec | bytes | bits/byte | comp MB/s | decomp MB/s |
|---|---:|---:|---:|---:|
| zstd -19 | 235,328,939 | 1.883 | 12.7 | 557.5 |
| zstd -3 | 312,109,709 | 2.497 | 1125 | 664.1 |

Not yet run on enwik9: zstd -22 --long, xz -9e, bsc variants, lz4.

### Dropped: paq8px and cmix

Stopped and removed at your request. They run at kB/s, too slow to be useful baselines for a speed-and-ratio tool. For ratio-only reference on enwik8, published cmix v19 is 14,837,987 bytes (1.19 bpb).

### Not run: ts_zip and NNCP

Both are distributed only as prebuilt binaries (closed-source LibNC), so building from source isn't possible. Running them needs your OK.

## 3. What this changes in the plan

| Plan item | Before | After Phase 0 |
|---|---|---|
| INT8 vs float | assumed similar | INT8 = **3× FP16** on this GPU. Integer-only is also the fastest option. |
| `neural` tier size | 2–8M params, 10–30 MB/s | **1.5–4M params per byte** for 10–30 MB/s (fused). |
| `neural` tier bar to clear | "close to paq8 / ts_zip" | Must beat **bsc: 1.66 bpb on enwik8 at 55/173 MB/s**. Target **≤1.4 bpb at ≥10 MB/s**, or that tier isn't worth having over bsc. |
| `balanced` tier | BWT on GPU | bsc on CPU is already 55–70 MB/s at the best mid-tier ratio. bsc's GPU sort-transform mode compresses at 158 MB/s but decompresses at only 15 MB/s and gives a worse ratio. **Use CPU bsc**, with GPU as an optional compress-only accelerator. |
| `max` tier | 3–10 MB/s | **4–6 MB/s** realistic (ceiling ~7) for the patching+MoE design. |
| Weight budget in L2 | ≤ 20M INT8 params | Effective L2 sweet spot is **about 24 MB** including activations, and **B ≈ 2048 streams**. |
| Kernels | fused "should" help | Mandatory: stock ops reach only 10–30% of the ceiling. |
