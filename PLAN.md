# Plan: GPU+CPU Neural Lossless Compressor (working name `nzip`)

Status: **Phase 3a done** (see [bench/PHASE3.md](bench/PHASE3.md)); Phase 2 done (see [bench/PHASE2.md](bench/PHASE2.md)); earlier: [PHASE1](bench/PHASE1.md), [PHASE0](bench/PHASE0.md). GPU throughput and the classic baselines are now measured. Figures still marked *(est.)* depend on kernels that aren't written yet.

---

## 1. What info.md tells us, and what it leaves out

info.md covers the core idea: **compression = prediction + entropy coding**. A model gives `P(next symbol | past)`, and an arithmetic or ANS coder spends `-log2 P` bits on each symbol. Better prediction means a smaller file. info.md also covers the extras: model size counts toward the "honest" size, online (prequential) adaptation, and ANS as the fast entropy coder.

It does not cover the three problems that decide whether a real tool works:

1. **Speed.** Neural compressors are slow. ts_zip does up to about 1 MB/s on an RTX 4090, and NNCP and cmix run at kB/s. zstd does hundreds of MB/s.
2. **Determinism.** The decoder has to produce *bit-identical* probabilities to the encoder's. One bit of float drift corrupts the rest of the stream. Float GPU math is not bit-identical across GPUs, drivers or batch sizes.
3. **Generality.** An LLM trained on text does little for binaries, logs, tables or data that is already compressed.

The plan below is built around these three problems.

## 2. Your hardware (measured on this machine)

| | |
|---|---|
| GPU | RTX 5060 Ti 16 GB (Blackwell, sm_120), 36 SMs, 448 GB/s GDDR7, 32 MB L2 (spec sheet) |
| CPU | Ryzen 9 3900X, 12C/24T, AVX2 (no AVX-512), 64 MB L3 |
| RAM | 125 GB |
| Toolchain | CUDA 13.4, Rust/cargo, gcc/clang, cmake, Python 3.14 + PyTorch 2.14.1 (cu130, in `.sandbox/venv`) |
| **Measured** | INT8 GEMM ~145–155 TOPS peak (~125 usable at decode shapes), FP16 only ~45 TFLOPS, DRAM 324 GB/s, L2 sweet spot ≤ ~24 MB at ~1.5 TB/s, graph launch 0.7 µs |

What this means for the design: decode speed ≈ 125 TOPS ÷ (2 × params used per byte), measured. A dense 142M byte model would do 0.5 MB/s, so small per-byte models, patching/MoE for the big tier, many parallel streams and fused kernels are all required. INT8 is the fastest number format on this card, about 3× FP16.

## 3. Research findings that shape the design

| Finding | Source | Design consequence |
|---|---|---|
| In a conventional LLM compressor, CPU arithmetic coding plus copying probability vectors device-to-host takes **60–68% of decode time**. Moving interval selection to the GPU and sending only targets gave **5.8× encode / 3.0× decode** speedups. | arXiv 2611.05871 | **The whole decode loop stays on the GPU.** No per-symbol host round-trips. |
| Capturing only the target interval (O(1) transfer instead of O(vocab)) gave **18.96×**. At large batch sizes the coder, not the model, becomes the bottleneck. | arXiv 2610.17586 | The integer CDF and the rANS coder run on the GPU, fused with the model step. |
| Multi-token heads and speculative decoding speed up decode only by about 2×, while bpb **doubles or triples** (0.88 → 1.78). Rank coding costs 0.4–0.8 bpb. | arXiv 2611.05871 | **Rejected.** One model call per symbol is the honest price, so we make each call cheap. |
| Float determinism only works with batch-invariant custom kernels, and even then only on the *same* hardware. The 7B speculative setup needed FP32. | arXiv 2611.05871 | **Use integer-only inference** (INT8 weights, INT32 accumulation, fixed-point activations). Integer math gives the same result in any order, so it is bit-exact on any GPU or CPU. |
| Integer-only post-training quantization makes neural codecs cross-platform deterministic with marginal ratio loss. | arXiv 2202.07513, arXiv 2603.25260 | Use quantization-aware training (QAT) aimed at exactly the integer kernel we ship. |
| ts_zip: RWKV-169M, 8-bit weights, 1.084 bpb on enwik9, reproducible across GPU/CPU/thread count, up to 1 MB/s on a 4090. | bellard.org/ts_zip | Recurrent (RNN-style) models are a proven fit. Hardware-independent results are achievable. |
| NNCP (2023): 0.853 bpb on enwik9 (107.3 MB total). cmix v19: 0.892 bpb. | bellard.org/nncp | Upper bound on ratio, at very low speed. |
| tufazip (Oct 2026): 163M transformer with online training and replay; **105.39 MB** on enwik9, #2 on LTCB. Increasing the replay window cut the payload from 107.22 to 105.92 MB. | tufalabs.ai | Online adaptation and replay is worth about 1% at the top end. Use it only in max mode, because it is expensive. |
| Nacrith: 135M LLM **plus classical context mixing** reaches 0.939 bpb on enwik8. | arXiv 2602.19626 | Mixing a neural net with cheap context models (n-gram and match models) helps a lot. |
| Hutter Prize record (Jul 2026): fx2-cmix-T, 100.4 MB on enwik9. It uses preprocessing transforms plus cmix plus a transformer. | hutter1.net / fx2-cmix | Preprocessing transforms (text and dictionary transforms, E8E9 for executables) still pay off. |
| GPU rANS/ANS decodes at 90+ GB/s (Recoil, Turing) to about 590 GB/s (DietGPU, H100). nvCOMP ANS reaches about 180 GB/s on B200. | arXiv 2306.12141, NVIDIA | Entropy coding costs almost nothing once it is on the GPU. The model is the only real cost. |
| libcubwt: GPU BWT construction and inversion (same author as libbsc). | github.com/IlyaGrebnov/libcubwt | A fast, strong mid tier is available without writing it ourselves. |

## 4. Architecture

```
            ┌──────────────────────── CPU (Rust, 24 threads) ────────────────────────┐
 input ───► │ analyze & split ─► dedup (CDC+hash) ─► transforms (text/E8E9/delta) ─► │
            │ per-block router: store │ fast │ balanced │ neural (GPU or CPU worker)  │
            └───────┬───────────────────────────────────────────────┬────────────────┘
                    │ blocks → B parallel streams                   │ blocks
            ┌───────▼────────────── GPU (CUDA, one fused step) ─────┐  ┌──▼──────────────┐
            │ for each step, for all B streams in lockstep:         │  │ CPU neural/CM    │
            │   INT8 recurrent model (tensor cores, weights in L2)  │  │ workers (AVX2,   │
            │ + GPU context models (order-1..4 hashed, match model) │  │ same integer     │
            │ + integer logistic mixer + SSE/APM                    │  │ math) pull blocks│
            │ → integer CDF (sum 2^16, min freq 1) → rANS per stream│  │ from same queue  │
            └───────┬───────────────────────────────────────────────┘  └──┬──────────────┘
                    └──────────────► container (block table, model hash, xxh3 per block)
```

### 4.1 Tiers (one format, chosen per block)

| Mode | Engine | Target speed *(est.)* | Target ratio |
|---|---|---|---|
| `store` | Detect incompressible data (JPEG/MP4/zip…) with a fast entropy probe, then store it raw | memcpy | 1.0 |
| `fast` | zstd-3, 4 MiB blocks on all CPU cores (nvCOMP GPU zstd measured slower and worse ratio) | **measured 1.7 / 3.1 GB/s** on enwik9 | zstd-3 class |
| `balanced` | libbsc: **BWT on GPU** (libcubwt, byte-identical output) + LZP/QLFC on CPU, up to 512 MiB blocks; CPU decode | **measured 225 / 250 MB/s** on enwik9 (139 / 184 on enwik8) | measured 1.355 bpb enwik9 (bsc best: 1.456), 1.664 enwik8 |
| `neural` (default for compressible data) | Small INT8 byte-level model (**~1.5–4M params per byte**) + GPU context mixing | **10–30 MB/s** (measured ceiling: 3.1M → 13 MB/s, 0.7M → 42 MB/s) | must beat bsc: target **≤1.4 bpb** on enwik8 |
| `max` | Patch-based sparse (MoE) model, ~150M total / ~30M active params, + mixing + file adaptation (see 4.4) | **4–6 MB/s** (measured ceiling ~7) | aim for ≤1.0 bpb on enwik9-class text |

We reuse proven code (zstd, libbsc/libcubwt, nvCOMP) for the classic tiers. The engineering effort goes into the neural tiers, where the research gains are.

### 4.2 Neural engine details

- **Byte-level model.** Required for "general" data. Text-only token models are an optional later add-on for the `max` text path, where tokens give about 4 bytes per model call.
- **Recurrent architecture.** RWKV-7-style linear recurrence, with a minGRU/LSTM-style model as the alternative. The architecture is picked in a bake-off (Phase 3).
  - Why recurrent: O(1) cost per step, no KV cache, unbounded context.
  - Batched across B streams, each step becomes a real GEMM (`[B×d]·[d×d]`), which keeps the tensor cores busy.
  - Watch-out: RWKV-style matrix state per stream costs memory bandwidth. State size is one of the bake-off criteria.
- **Encoder faster than decoder.** The encoder knows every byte, so it can run layer by layer over whole blocks as large GEMMs, with only the cheap elementwise recurrence done in sequence. Integer math makes this bit-identical to the decoder's step-by-step path. The decoder is inherently one step at a time.
- **Fused, persistent decode step.** CUDA Graph or a persistent kernel, so per-step launch overhead goes away (a bottleneck called out in 2610.17586). Model, context models, mixer, CDF and rANS decode all run in one pass, and the decoded byte feeds straight into the next step on the GPU.
- **Weights in L2.** A model of ≤~20M INT8 params fits in the 32 MB L2, so each step is bound by compute, not by memory bandwidth.
- **Context mixing on the GPU, per stream.** Order-1 to order-4 hashed counters and a match model, which catches long repeats the neural net misses. Mixed in the logistic domain with integer weights (paq/cmix style, which is deterministic), then an SSE/APM stage.
- **Streams vs ratio.** More streams means more parallelism, but each stream starts colder. Mitigations:
  - a pretrained prior;
  - a minimum of ~256 KB–1 MB per stream;
  - an optional shared "warm-up" prefix;
  - in archive mode, many files become many streams for free.
  - Small inputs (<~1 MB) go to CPU workers.
- **CPU use.** Preprocessing, dedup, container I/O, the fast and balanced tiers, and **CPU neural workers** (AVX2 integer kernels running exactly the same math) that pull blocks from the same queue as the GPU. Throughput from both devices adds up. The same CPU path is also the **GPU-less decoder**: any file decodes on any machine, just more slowly.

### 4.3 Determinism contract (the most important rule)

- All prediction math is integer or fixed-point. Nonlinearities (sigmoid, exp, softmax) use fixed-point LUTs or polynomials. Frequencies are quantized to sum to 2^16 with a minimum of 1, so every symbol stays codable.
- A bit-exact **CPU reference implementation** is the specification. CUDA kernels are tested against it on random and adversarial inputs (CI fuzzing).
- The container stores a model hash and a codec version, and the decoder refuses to run on a mismatch. Each block carries an xxh3 checksum.
- What we give up: Blackwell's FP4/FP8 tensor throughput (FP accumulation order is not portable). Decided: no same-hardware-only mode, so files always decode on other machines.

### 4.4 Fast `max` mode: big model, adaptive, ~10× faster than the first draft

**Why the first draft was slow.** Decode speed is roughly `usable GPU ops ÷ (ops per byte)`. A dense 100M-param byte-level model costs about 2 × 100M = 0.2 GOP per byte. At ~100 TOPS usable INT8 *(est.)* that is about 0.5 MB/s, and backprop on every symbol for online learning roughly triples the cost. Batching more streams cannot beat that limit. The only way through it is **fewer ops per byte, with the model's capacity kept**. Four changes do that:

**1. Run the big model once per patch, not once per byte** (Byte Latent Transformer, arXiv 2412.09871)
- A large *global* model runs once per patch of bytes. A small *local* byte model (~3M params) runs every byte, conditioned on the global model's output.
- Patch boundaries come from the local model's own uncertainty: long patches on predictable data, short ones on surprising data. The decoder computes that uncertainty from bytes it has already decoded, so it finds the same boundaries with no side information.
- Works on any bytes, with no fixed vocabulary. BLT reports matching Llama 3 at the same training compute with up to 50% fewer inference FLOPs, and its curves improve as patches and the model both grow.
- Simpler fallback if patching underperforms: a byte-fallback BPE tokenizer trained on the mixed corpus (~4 bytes per call on text, ~1.5–2 on binaries).
- Batching detail: every step, the local model runs for all B streams. The global model runs on a compacted subset, gathered from the streams whose patch just ended (about B/5 per step), which is still a large GEMM batch.

**2. Sparse mixture-of-experts global model**
- About 150M total params (knowledge), but only ~30M active per patch (cost). Experts stored as INT4 and unpacked to INT8 in registers. That halves weight bandwidth and stays integer-exact.

**3. Adaptation that costs nothing at decode time.** Three layers, from fastest to slowest:
- *Within the stream:* RWKV-7-style recurrent state updates use a delta rule. The state is effectively fast weights that learn the current file as it goes, at no extra cost.
- *Every symbol:* the integer mixer, SSE and match model adapt online (cheap; already in the neural tier).
- *Per file:* the **encoder** fine-tunes a small LoRA adapter on the file (a sample of it for speed), quantizes it to INT8 and **stores it in the container** (~0.2–2 MB). The decoder just loads it, so decode speed is unaffected.
  - Training is ordinary float PyTorch-style training. Only the quantized result is transmitted, so training does not need to be deterministic.
  - The encoder includes the adapter only when it saves more bytes than it costs (it measures both).
  - arXiv 2604.02343 reports domain LoRA adapters roughly doubling the arithmetic-coding ratio on LLM-generated text.
  - Encode cost is set with `--adapt 0..3`.

**4. (Optional, Phase 5b) True online learning without stalling the decoder.** Gradients from chunk *k* are applied from chunk *k+2* onwards. Training runs on a separate CUDA stream while chunk *k+1* decodes, so it is hidden behind decoding. Encoder and decoder do the same delayed integer-SGD updates (this part does need deterministic integer training). This is the tufazip-style gain (their replay change: 107.22 → 105.92 MB) without the per-symbol stall.

**Speed estimate *(est.)*:**
- global: 2 × 30M active ÷ ~5 bytes per patch ≈ 12 MOP/byte;
- local: 2 × 3M ≈ 6 MOP/byte;
- total ≈ 18 MOP/byte, about **11× less** than the dense design.

At ~100–150 TOPS usable, that is about **5–8 MB/s**, and closer to 10 MB/s on predictable text with long patches. Weight bandwidth is not the limit: ~150 MB of experts per global step at 448 GB/s ≈ 0.33 ms, amortized over ~B × 5 bytes per step.

**Ternary weights** (section 9) are a candidate for the global model and experts. They shrink the weights enough to mostly fit in L2, and they speed up the CPU path.

### 4.5 Our own model, and training it

- **Training stack:** PyTorch in a venv, quantization-aware training that simulates the exact integer kernel, then export with a bit-exactness check against the CPU reference.
- **Corpus (general):** text (Wikipedia minus enwik, books, code), logs, JSON/CSV, ELF/PE binaries, uncompressed image and audio samples, source tarballs.
- **Held-out eval set:** Silesia, Canterbury, Calgary, enwik8/9 (excluded from training so results are honest), plus a sample of your own files.
- **Compute estimate *(est.)*:**
  - neural-tier model (~5M params, ~10B bytes): about 3e17 FLOPs, a few hours on the 5060 Ti;
  - max-tier model (~150M total / ~30M active MoE + 3M local): training cost scales with *active* params, so about 2–6e18 FLOPs, roughly 1–2 days *(est.)*. Alternative: distill from an open Apache-2.0 RWKV-7 checkpoint to speed up training.
- **Weights ship with the binary:** ~5 MB for the neural tier, ~75 MB for max (INT4 experts).

## 5. Implementation phases

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Benchmarks & microbenchmarks** | Sandbox (`.sandbox/env.sh`, venv, local caches). Benchmark harness and corpus. Baselines measured here: zstd, xz, brotli, bzip2, lz4, bsc. paq8px/cmix were dropped as too slow; ts_zip/NNCP are pending your OK (binary-only). INT8 GEMM microbenchmarks on the 5060 Ti. | Real numbers replace every *(est.)* in this plan. |
| **1. Container + classic tiers** ✅ | Rust CLI (`nzip c/d/t/i`), block container, checksums, store detection, dedup, transforms, zstd and bsc tiers, GPU BWT with CPU fallback. | Done: 18 tests pass; balanced is 7% smaller and 3× faster than bsc's best on enwik9. |
| **1b. Quick wins** ✅ (GPU inverse BWT, small-file overhead) | GPU inverse BWT on decompression (measured 2× faster than CPU), small-file overhead (strip debug info, lazy thread pool). Then text transforms (digit split, capitalization/dictionary) and I/O pipelining. | Decompression +15–20%; small files on par with CLIs; −2 to −5% on text. See [research/OPTIMIZATIONS.md](research/OPTIMIZATIONS.md). |
| **2. GPU entropy core** ✅ | Batched GPU rANS (32-bit state, 16-bit renormalization, ~8 KiB lanes, LUT decode), integer CDFs, CPU reference implementation. First consumer: a GPU-only order-2 context model, then the neural tier, with a persistent decode loop. (A GPU entropy stage for BWT output was tested and rejected: 8–11% worse ratio than CPU QLFC; see research/OPTIMIZATIONS.md section 4.) | Done: byte-identical GPU/CPU streams; rANS 1.5% of cost; GPU decode 470 MB/s vs 33 MB/s on 24 CPU threads (test model). |
| **3a. Model bake-off** ✅ | Done ([bench/PHASE3.md](bench/PHASE3.md)). 3.47 GB decontaminated corpus; five architectures at ~2M decode MACs/byte. **Gated DeltaNet wins** (1.714 bpb val mean; RWKV-7 1.760, SWA 1.934, minGRU 2.024, LSTM 2.107), but its 96 KB/lane state makes naive decoding bandwidth-bound (~1.6 MB/s est.). Model alone loses to bsc on every Silesia file; **model + adaptive context mixing beats bsc on text by 6–12%** (4 MB lanes). | Pick architecture ✅, measure mixing ✅. |
| **3b. Neural tier core** | Delta-rule model with a small INT8 state (e.g. 8×32×32 heads), trained on multi-GB at 2–3 sizes; integer context-mixing stage (counters, match, record contexts, mixer, SSE) pulled forward from Phase 4; ≥4 MB lanes; QAT to integer kernels; persistent GPU kernels with state on-chip; CPU reference. | Beat bsc on enwik8/Silesia at ≥10 MB/s decode. |
| **4. Mixing** | GPU context models, match model, integer mixer (or fixed-point gated linear network), SSE. Optional: M03-style context-aware BWT coding for a stronger balanced tier (−2% measured by bsc-m03). CPU neural workers and a joint scheduler. | Measurable bpb gain over model-only, and CPU+GPU throughput above GPU alone. |
| **5. Max mode** | Patch-based local+global model, MoE experts (INT4), entropy patching (BPE fallback), per-file LoRA adapter stored in the container. **5b:** delayed online updates on a second CUDA stream. | ≥3 MB/s decode on this GPU, with ratio within reach of ts_zip/NNCP on enwik8/9. |
| **6. Hardening** | Fuzzing, a corrupt-input-safe decoder, streaming I/O, docs, packaging. | Ready for real use. |

Each phase ends with a short benchmark report so you can redirect early.

## 6. Risks

- **Speed estimates are unverified.** The 5060 Ti's INT8 tensor throughput and real decode-step latency are the big unknowns, and Phase 0 measures them first.
- **Integer QAT may cost some ratio.** The literature says the loss is marginal. We will measure it.
- **Max-mode complexity.** Entropy patching, MoE routing and uneven patch lengths across batched streams are the hardest kernels in the project. BLT's gains were measured on LM benchmarks, not on compression. The BPE-tokenizer fallback limits the downside.
- **Multi-stream cold start** hurts small files. Mitigated by routing small files to CPU and by a pretrained prior.
- **Model generality.** Data unlike the training mix gains little. Context mixing and the `balanced`/`fast` fallback (chosen per block by trial on a sample) keep the worst case near zstd/bsc.
- **Training time and data downloads** (tens of GB) are needed for Phases 3 and 5.

## 7. Decisions (answered 2026-11-17)

1. **Language:** Rust (CLI, container, CPU paths) + CUDA C++ kernels. Python/PyTorch is used **only for offline pretraining**, as a build-time tool that outputs a weights file. Python never ships and never runs during compress or decompress.
   - Training that happens *inside the tool* is Rust + CUDA: the per-file LoRA adapter (4.4) and the delayed online updates (Phase 5b). These are small custom backward passes over a frozen base, not general training.
   - The source of truth stays in Rust: the Rust integer reference implementation is exposed to Python via PyO3. Training's QAT simulation and the export step are checked bit-for-bit against it.
   - Revisit Burn (a Rust training framework) for pretraining if its CUDA backend matures enough for ternary QAT and MoE.
2. **Portability:** files must decode on other machines. That means integer-only inference everywhere and no FP8/FP4 mode. A language model is only deterministic across hardware if its math is integer; float LMs are not.
3. **Data:** a mix of everything (text, code, logs, binaries, tables, media). The training corpus is broad and the byte-level/patch design stays. The BPE tokenizer is only a fallback.
4. **Speed floor:** not answered. Assumed: `neural` ≥10 MB/s as the default, `max` opt-in.
5. **Sandbox:** every install, cache, download and build output stays inside this project folder (see section 8).
6. **GPU first, CPU fallback** (2026-11-17): each stage runs on the fastest hardware available, measured per stage (`--device auto`, the default). The CPU is used only when there is no GPU or when it is measured faster (small inputs, decompression, zstd). GPU-produced files must stay byte-identical to, or at least decodable by, the CPU path.
7. **Speed + compression together**: no speedup is accepted if it costs ratio unless it is an explicit lower tier.
8. **Any size, any RAM, any VRAM** (2026-11-17): inputs of any size (10+ GB) stream with bounded memory; the budget comes from available RAM and cgroup limits, and GPU work (block sizes, lane and model batches) is sized from the card's free VRAM at run time, never from constants tuned to this card. The compressed file must not depend on the RAM budget. Done for the classic tiers and the GPU coder ([bench/SCALING.md](bench/SCALING.md)); every neural kernel must follow it too.
9. **Modes** (2026-11-17): `max` is the default and means the best compression available (today: bsc pipeline with 1 GiB blocks, GPU BWT when a block fits in VRAM; the neural + context-mixing tier takes over when it wins), plus `fast` and `store`. CLI: `nzip -c <input> <output>`, `nzip -d <input> <output>` (device flag is `-D`).

## 8. Sandbox layout

Nothing is written outside the project folder (it can be renamed or moved; paths are found at run time). The system CUDA 13.4, Rust and gcc already installed are *used*, not modified.

```
test/
  .sandbox/
    env.sh          # source this: sets every path below
    venv/           # Python venv (PyTorch etc.)           VIRTUAL_ENV
    pip-cache/      #                                      PIP_CACHE_DIR
    hf/             # model weights / HF datasets           HF_HOME
    torch/          #                                      TORCH_HOME
    cargo/          # crate registry + installed tools     CARGO_HOME
    rustup/         # only if a different toolchain is needed  RUSTUP_HOME
    tmp/            #                                      TMPDIR
  data/
    bench/          # Silesia, Canterbury, Calgary, enwik8/9 (held out)
    train/          # training corpus
    downloads/      # raw downloads, each in its own dir, treated as untrusted
  target/           # Rust build output                   CARGO_TARGET_DIR
  models/           # our trained/exported weights
```

## 9. Ternary weights (BitNet b1.58): assessment

**Verdict: yes, as a candidate for the large models, measured in the bake-off. Not as a blanket choice.**

Where ternary is a clear win:
- **Determinism:** with weights in {-1, 0, +1}, a matmul is just integer adds and subtracts of INT8 activations. It is exactly reproducible on any hardware, so it fits our integer-only rule perfectly.
- **Size and cache:** about 1.6–2 bits per weight. The max-mode MoE (~150M params) becomes ~30–38 MB instead of ~75 MB at INT4, so most of it can sit in the 5060 Ti's 32 MB L2. The shipped binary is smaller too.
- **CPU workers and the GPU-less decoder:** this is where ternary shines. bitnet.cpp reports 2.37×–6.17× speedups on x86 over llama.cpp, using lookup-table kernels with no multiplies. That directly speeds up CPU workers and decoding on machines without an NVIDIA GPU.

Where it doesn't help, or costs:
- **GPU compute:** tensor cores have no ternary instruction. Weights get unpacked to INT8 and run through the same INT8 tensor-core path, so ops per second are unchanged. Our batched design is compute-bound, so the GPU gain is mainly cache and bandwidth, not 3–5×. The GPU speed comes from patching and MoE (4.4). Ternary adds to it modestly.
- **Quality at small sizes:** BitNet b1.58's parity with FP16 was shown at ~3B params. At hundreds of millions of params there is a measurable gap, and our models are 3M–150M. In compression, any prediction loss becomes bigger files 1:1. The real question is whether **4–5× more ternary params for the same bytes and bandwidth beats fewer INT8 params**. That is a bake-off measurement, not something to assume.
- **Training:** it has to be trained ternary from the start (quantization-aware training with straight-through estimators); converting a finished model doesn't work well. We train our own anyway, so this costs nothing extra. It does rule out simply converting an open RWKV-7 checkpoint.

How we'd use it:
- **Ternary candidates:** max-mode global model and MoE experts (large and size/bandwidth-sensitive), and possibly the CPU-worker model.
- **Stay INT8:** embeddings, output head and norms (BitNet keeps these at higher precision too), the small ~3M local byte model, and the neural-tier model (already fits in L2, where quality matters more than size).
- **Experiment:** ternary weights already have about ⅓ zeros. Training with a structured 2:4 sparsity constraint would let Blackwell's sparse INT8 tensor cores run up to 2× faster. Speculative, but cheap to try once the ternary training pipeline exists.

## Sources

- [The Invocation Bottleneck in LLM-Based Lossless Compression (arXiv 2611.05871)](https://arxiv.org/abs/2611.05871)
- [High-Throughput LLM-Based Compression and What It Takes (arXiv 2610.17586)](https://arxiv.org/html/2610.17586)
- [Post-Training Quantization for Cross-Platform Learned Image Compression (arXiv 2202.07513)](https://arxiv.org/pdf/2202.07513)
- [Towards Practical Lossless Neural Compression for LiDAR Point Clouds (arXiv 2603.25260)](https://arxiv.org/pdf/2603.25260)
- [ts_zip — Fabrice Bellard](https://bellard.org/ts_zip/)
- [NNCP — Fabrice Bellard](https://bellard.org/nncp/) and [NNCP v2 paper](https://bellard.org/nncp/nncp_v2.1.pdf)
- [tufazip: Learning to Compress a Billion Bytes of Wikipedia](https://tufalabs.ai/research/tufazip/)
- [Nacrith: Neural Lossless Compression via Ensemble (arXiv 2602.19626)](https://arxiv.org/pdf/2602.19626)
- [Hutter Prize (Wikipedia)](https://en.wikipedia.org/wiki/Hutter_Prize), [fx2-cmix](https://github.com/kaitz/fx2-cmix)
- [Recoil: Parallel rANS Decoding (arXiv 2306.12141)](https://arxiv.org/pdf/2306.12141)
- [nvCOMP algorithm selection](https://docs.nvidia.com/cuda/nvcomp/algorithm_selection.html)
- [libcubwt](https://github.com/IlyaGrebnov/libcubwt)
- [The Era of 1-bit LLMs: BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764)
- [bitnet.cpp: Fast and Lossless BitNet b1.58 Inference on CPUs (arXiv 2410.16144)](https://arxiv.org/html/2410.16144v2)
- [Low-bit GEMM kernels on CPU/GPU (arXiv 2508.06753)](https://arxiv.org/pdf/2508.06753v2)
- [Byte Latent Transformer: Patches Scale Better Than Tokens (arXiv 2412.09871)](https://arxiv.org/abs/2412.09871)
- [Haiku to Opus in Just 10 bits: domain LoRA for LLM compression (arXiv 2604.02343)](https://arxiv.org/html/2604.02343)
- [RWKV-7 "Goose" (arXiv 2503.14456)](https://www.arxiv.org/abs/2503.14456)
- [Language Modeling Is Compression (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/3cbf627fa24fb6cb576e04e689b9428b-Paper-Conference.pdf)
