# Optimization research, round 2 (after Phase 1)

Date: 2026-11-17. The goal is more **speed and ratio together**. Each item below lists its measured or published evidence, the expected gain, the effort, and where it fits in the plan.

## 1. Where the time goes now (measured on this machine)

One 100 MB block (enwik8), 24 CPU threads, RTX 5060 Ti:

| Stage | Compress | Decompress |
|---|---:|---:|
| LZP preprocessing (CPU) | 0.03 s | ~0.03 s |
| BWT / inverse BWT | **GPU 0.29 s** (CPU 1.26 s) | CPU 0.18 s, **GPU 0.095 s** |
| QLFC entropy coding (CPU) | **0.33 s** | **0.32 s** |

The 400 MB block shows the same shape: GPU BWT 0.95 s, QLFC 1.0–1.1 s, inverse BWT 0.68 s on CPU vs 0.35 s on GPU.

**Conclusion:** the CPU entropy coder (QLFC) is now the largest single cost in **both** directions. Inverse BWT on the GPU is 2× faster than on the CPU, but nzip doesn't use it yet.

Two other facts from this machine:
- The project sits on a **SATA SSD** (Samsung 870 EVO, ~530 MB/s). Fast mode at 1.7 GB/s and balanced decompression on uncached files will hit the disk limit before anything else.
- **Unified memory to fit 1 GB BWT blocks was tried and rejected:** 47.6 MB/s, slower than the CPU's 56.8 MB/s. It also made 512 MiB blocks slower (161 vs 225 MB/s). The patch was reverted.

## 2. Ranked recommendations

| # | Optimization | Evidence | Expected gain | Effort | Fits in |
|---|---|---|---|---|---|
| ~~**1**~~ | ~~**GPU entropy stage for BWT output**~~ **Tested 2026-11-17 and rejected; see section 4.** (replaces CPU QLFC). Static per-segment models (two-pass: histogram, then rANS), thousands of interleaved lanes. | bsc's *static* QLFC is only **+0.7%** larger than adaptive on enwik8 (20.94 vs 20.80 MB), so a static model loses little. A GPU rANS with static order-1 tables runs at **2–4 GB/s resident** on an RTX 4080 Laptop ([arXiv 2610.02236](https://arxiv.org/html/2610.02236)). | Balanced single-block **~2× both ways** (est. compress ~250–300 MB/s on enwik8, decompress ~400–500 MB/s). The GPU can afford richer static contexts (order-2 over MTF/RLE symbols), which could win back the 0.7%. | Medium. It reuses the Phase 2 GPU rANS core. | **Phase 2, first consumer** |
| **2** | **GPU inverse BWT on decompression** for blocks ≥ 16 MB, with the CUDA warm-up overlapped with entropy decoding | Measured: 0.095 s vs 0.18 s per 100 MB; output identical | +15–20% decompression speed now, more once #1 lands | Low | Now (small patch) |
| **3** | **I/O pipelining**: read block *k+1* and write block *k−1* while computing block *k*. Today the whole input is read and hashed before compression starts. | 1 GB from this SSD takes ~1.9 s just to read, comparable to the 4.4 s compression | Up to ~30% wall clock on **uncached** files; nothing on cached ones | Medium | Now / Phase 6 |
| **4** | **Text transforms before BWT**: split out digit runs into side streams (STC), then capitalization and word-dictionary transforms | STC: **−2.63 MB on enwik9** (−1.6%) with the coder held fixed ([arXiv 2606.03570](https://arxiv.org/html/2606.03570v2)). Dictionary transforms are standard in top LTCB entries (fx2-cmix etc.). | −2 to −5% on text, ~0 on binaries; CPU-parallel and cheap at run time | Medium | Phase 1b |
| **5** | **Context-aware post-BWT coding (M03 family)** instead of QLFC. M03 models the BWT's implicit suffix-tree intervals rather than MTF ranks. | Same 1 GB block on enwik9: bsc-m03 **160.3 MB** vs bsc -e2 163.9 MB (**−2.2%**) (LTCB). But m03 is single-threaded at ~160 ns/byte (~6 MB/s) and GPL-3.0. | −2% ratio. Interval subtrees are independent, so it parallelizes (CPU workers or GPU). The idea must be reimplemented, not copied (license). | High | Phase 4–5 (balanced+) |
| **6** | **Whole-file BWT beyond GPU memory**: sort partitions on the GPU, then merge the BWTs (gap-array / BWT-merge style), or use a leaner GPU suffix sorter | Measured 1 GB block: **164.2 MB vs 169.4 MB** at 512 MiB (−3.1%). libcubwt needs ~20.3 bytes of GPU memory per input byte, so 16 GB caps blocks at ~640 MiB. | −3% on files > 640 MB | High (research-grade) | Later |
| **7** | **Small-file overhead**: strip debug info (the release binary is 30 MB), create the thread pool lazily, skip GPU probing for small inputs (done) | Canterbury/Calgary run ~25% slower than the CLIs | Small files on par with zstd/bsc | Low | Now |

### For the neural tiers (Phases 2–5)

| # | Technique | Evidence | Why it matters here |
|---|---|---|---|
| N1 | **Persistent megakernel with weights held on-chip** (registers + shared memory), looping over decode steps with no relaunches | Persistent RNNs keep recurrent weights on-chip across timesteps ([Diamos et al., ICML 2016](https://mlanthology.org/icml/2016/diamos2016icml-persistent); [Sparse Persistent RNNs](https://arxiv.org/pdf/1804.10223)). A persistent megakernel has been built and run correctly on sm_120 (RTX 5090). | 36 SMs × (256 KB registers + ~100 KB shared memory) ≈ **12–13 MB on chip**, so a 2–4M-param INT8 neural-tier model fits **entirely on chip**. That removes L2/DRAM traffic and launches per step, the overheads that held stock PyTorch to 10–30% of peak in Phase 0. **Highest-value Phase 3 kernel technique.** Watch for register spills. |
| N2 | **GPU rANS engineering details**: 32-bit state with 16-bit renormalization, ~8 KiB segments per lane, table-lookup (LUT) decode, cached frequency tables, warp-level histograms (`__match_any_sync`) | [arXiv 2610.02236](https://arxiv.org/html/2610.02236): 2–4 GB/s resident encode and decode, CPU fallback decoder at 215 MB/s | A ready blueprint for the Phase 2 core. It confirms entropy coding won't bottleneck the neural tier, which runs at 10–30 MB/s. |
| N3 | **Masked/diffusion-LM parallel decoding**: K model calls per window instead of one per token; positions committed by lowest entropy | [arXiv 2608.11249](https://arxiv.org/pdf/2608.11249): 1.01 → 166 kbit/s as tokens per step rise, but the compressed fraction worsens from 0.127 to 0.246 (8B model on an H100) | Ratio drops quickly. Useful only where there are too few streams to batch (small files in max mode). **Optional, low priority.** |
| N4 | **Gated linear networks (GLN)** as the GPU context-mixing stage: purely local online updates, vectorizable | [arXiv 1712.01897](https://arxiv.org/pdf/1712.01897) | A GPU-friendly, cheap way to "keep learning from the file" every symbol. Updates must be done in fixed point to stay deterministic. Candidate for Phase 4. |
| N5 | **Small models win on throughput; quantization matters** | 30-LLM code-compression study ([arXiv 2607.24192](https://www.alphaxiv.org/abs/2607.24192.md)), as cited by 2608.11249 | Supports the plan's small, INT8, patch-based design. |

### Checked and rejected

- **Unified memory for > VRAM BWT blocks**: slower than the CPU (measured, above).
- **nvCOMP GPU zstd for fast mode**: worse ratio (3.22 vs 2.85 bpb) and not faster than our CPU path (Phase 1).
- **Speculative / multi-token heads for autoregressive models**: bpb doubles (Phase 0 research).

## 3. Proposed order

1. **Now (small):** #2 GPU inverse BWT on decode, and #7 small-file overhead.
2. **Phase 2, the GPU rANS core:** build it (N2 blueprint) and plug it in first as the **GPU entropy stage for BWT** (#1). That gives balanced mode an immediate ~2× and validates the core before the neural models use it.
3. **Phase 1b (can run in parallel):** #4 text transforms and #3 I/O pipelining.
4. **Phase 3:** neural-tier kernels built as on-chip persistent megakernels (N1).
5. **Phase 4–5:** GLN mixing (N4); M03-style coding (#5) for a stronger balanced tier.
6. **Later:** whole-file BWT beyond VRAM (#6).

## 4. Result: GPU entropy stage for BWT output (rejected)

Tested on enwik8 (LZP+BWT output, 98.6 MB) with a CPU simulation of GPU-friendly models (`crates/bwtc`, `bwtc-eval`/`bwtc-eval2`). Integer arithmetic throughout; code length measured exactly.

| Model | Lanes (segment) | Size | vs bsc QLFC (20,802,998) |
|---|---|---:|---:|
| bit-level o0+o1 mixing + APM (bcm-like) | 1 (whole block) | 22,698,878 | +9.1% |
| same, tuned rates 5/8 | 1 | 22,487,685 | +8.1% |
| same | 377 (256 KiB) | 22,806,861 | +9.6% |
| same | 1,506 (64 KiB) | 22,906,227 | +10.1% |
| same + per-block static order-1 prior | 1,506 | 22,842,203 | +9.8% |
| rank-flag decomposition (MTF ranks 0–2 + escape), 3.3 decisions/byte | 1 | 23,015,943 | +10.6% |

What this shows:
1. **Splitting into many independent lanes is cheap:** 1 → 1,506 lanes costs only +0.9%. GPU-scale parallelism is not the problem.
2. **Simple GPU-friendly models lose 8–11% to QLFC.** QLFC is a heavily tuned model with dozens of fitted parameters and ~2.5 MB of adaptive state per lane. My earlier "static QLFC is only +0.7%" argument was wrong: bsc's "static" mode still uses QLFC's adaptive context modelling.
3. **QLFC itself on the GPU is not promising:**
   - ~2.5 MB of randomly accessed state per lane means only ~12 lanes' state fits in the 32 MB L2, so lanes would wait on DRAM.
   - The rank/run logic is branchy, so threads in a warp diverge.
   - It would have to beat 24 CPU threads at ~300 MB/s.

**Decision:**
- The balanced tier keeps CPU QLFC; no ratio is traded for speed.
- Its speed comes from the GPU BWT plus the GPU/CPU pipeline across blocks: 231 MB/s compress, 309 MB/s decompress on enwik9.
- The GPU entropy core (Phase 2) is still needed for the **neural tier**, where the network supplies the probabilities and entropy coding is the cheap part.

## Sources

- [Cartesian Byte-Bank Models and Warp Lag Screening for GPU Lossless Compression (arXiv 2610.02236)](https://arxiv.org/html/2610.02236)
- [Diffuse to Compress (arXiv 2608.11249)](https://arxiv.org/pdf/2608.11249)
- [STC: Reversible Digit-Context Decomposition for BWT-Family Text Compression (arXiv 2606.03570)](https://arxiv.org/html/2606.03570v2)
- [bsc-m03](https://github.com/IlyaGrebnov/bsc-m03), [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html)
- [libcubwt](https://github.com/IlyaGrebnov/libcubwt), [Jampack](https://github.com/loxxous/Jampack)
- [Parallel Lossless Data Compression on the GPU (Patel et al.)](https://escholarship.org/content/qt7bs1x3qn/qt7bs1x3qn.pdf)
- [GPU-Accelerated BWT Construction for Large Collections (arXiv 1401.7457)](https://arxiv.org/pdf/1401.7457)
- [Persistent RNNs (ICML 2016)](https://mlanthology.org/icml/2016/diamos2016icml-persistent), [Sparse Persistent RNNs (arXiv 1804.10223)](https://arxiv.org/pdf/1804.10223)
- [Online Learning with Gated Linear Networks (arXiv 1712.01897)](https://arxiv.org/pdf/1712.01897)
- [Recoil: Parallel rANS Decoding (ResearchGate)](https://www.researchgate.net/publication/373920773_Recoil_Parallel_rANS_Decoding_with_Decoder-Adaptive_Scalability)
- [LLM-based Source Code Compression via Thresholded Symbol Ranking (arXiv 2607.24192)](https://www.alphaxiv.org/abs/2607.24192.md)
