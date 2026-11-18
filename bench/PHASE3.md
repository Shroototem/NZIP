# Phase 3a report: corpus, model bake-off, and the case for mixing

Date: 2026-11-17. Machine: RTX 5060 Ti 16 GB, Ryzen 9 3900X, 125 GB RAM.
Reproduce (after `source .sandbox/env.sh`):
- corpus: `python -I train/corpus.py`
- bake-off: `bash train/bakeoff.sh 256M` (results in `models/bakeoff/results.jsonl`)
- decode-speed estimate: `python -I bench/decode_model.py`
- mixing: `python -I train/dump_probs.py models/bakeoff/gdn.pt <file> <file>.p16`, then
  `target/release/mixlab <lane> <file>...`

## 1. Training corpus (`data/train`, 3.47 GB + 32 MB held out)

| Category | Source | Train | Dropped as benchmark overlap |
|---|---|---:|---:|
| web | FineWeb-edu sample | 896 MB | 1.5 MB |
| books | PG-19 (Gutenberg), benchmark titles excluded | 695 MB | 31.3 MB |
| wiki | German Wikipedia XML (markup without English Wikipedia content) | 596 MB | 5.7 MB |
| code | /usr/include, Python stdlib, Rust crate sources | 374 MB | 88.6 MB |
| docs | /usr/share/doc, man pages | 246 MB | 10.9 MB |
| bin | ELF executables and libraries | 443 MB | 175.4 MB |
| data | XML/JSON/SVG/PO/CSV under /usr/share | 218 MB | 23.8 MB |

Every document was checked against Canterbury, Calgary, Silesia and the enwik8 prefix with anchored 32-byte substring hashes (`train/contam.py`), and dropped if it shared more than a few anchors. The benchmarks are therefore honest held-out tests.

## 2. Bake-off (float, equal decode cost ~2M MACs/byte, 256 MB of training each)

Bits per byte; lower is better. Benchmarks are coded in independent 256 KiB lanes with state carried inside each lane, as the real coder would.

| Model | Params | Train time | web | books | wiki | code | docs | bin | data | **val mean** | enwik8 10M | calgary | cantrbry | silesia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gated DeltaNet** | 1.92M | 6 min | 1.806 | 1.865 | 1.846 | 1.092 | 1.161 | 2.917 | 1.314 | **1.714** | 2.054 | 2.375 | 1.558 | 3.112 |
| RWKV-7 | 1.95M | 20 min | 1.824 | 1.875 | 1.877 | 1.138 | 1.219 | 3.040 | 1.343 | 1.760 | 2.034 | 2.381 | 1.556 | 3.200 |
| Sliding-window transformer | 1.43M | 7 min | 1.950 | 2.041 | 2.059 | 1.275 | 1.425 | 3.229 | 1.562 | 1.934 | 2.258 | 2.591 | 2.433 | 3.599 |
| minGRU | 2.38M | 12 min | 1.899 | 1.927 | 1.998 | 1.613 | 1.574 | 3.515 | 1.642 | 2.024 | 2.300 | 2.842 | 2.918 | 4.383 |
| LSTM | 2.17M | 6 min | 1.943 | 1.989 | 2.085 | 1.729 | 1.705 | 3.529 | 1.770 | 2.107 | 2.402 | 2.922 | 3.002 | 4.536 |

- **The delta-rule family wins clearly**, most of all on code, docs and data, where its fast-weight state learns repeats in context. Gated DeltaNet trains 3× faster than RWKV-7 here.
- LSTM and minGRU degrade on long lanes: their state drifts beyond the 2,048-byte training windows.

## 3. Estimated decode speed (`bench/decode_model.py`)

Per decode step: INT8 tensor-core GEMMs for all lanes plus one read and one write of each lane's state, captured in CUDA graphs. Real fused kernels should do better on the compute part.

| Model | State per lane | Best MB/s (lanes) | MB/s at 256 lanes |
|---|---:|---:|---:|
| LSTM | 2.8 KB | 11.3 (4,096) | 5.2 |
| minGRU | 2.2 KB | 7.5 (4,096) | 1.3 |
| Gated DeltaNet | 96 KB | 1.6 (16,384) | 1.0 |
| RWKV-7 | 96 KB | 1.6 (16,384) | 1.0 |
| Sliding-window transformer | 576 KB | 0.3 | 0.3 |

**The models with the best ratio are memory-bandwidth bound.** 96 KB of state per lane, read and written every byte, from DRAM. Remedies, in order: a smaller state (head_dim 32 is ¼ the size), INT8 state, and keeping the state on-chip in a persistent kernel (~12 MB of registers + shared memory holds ~130–500 lanes, depending on state size).

## 4. Where a model alone loses (Gated DeltaNet vs bsc, per Silesia file)

| File | GDN | bsc | | File | GDN | bsc |
|---|---:|---:|---|---|---:|---:|
| dickens | 2.153 | 1.773 | | reymont | 3.339 | 1.186 |
| mozilla | 3.806 | 2.485 | | samba | 2.183 | 1.448 |
| mr | 2.202 | 1.770 | | sao | 7.395 | 5.162 |
| nci | 1.164 | 0.289 | | webster | 2.902 | 1.244 |
| ooffice | 4.368 | 3.328 | | x-ray | 4.212 | 3.546 |
| osdb | 6.233 | 1.765 | | xml | 2.439 | 0.588 |

A small pretrained model cannot learn a file's own statistics. It loses everywhere, worst on repetitive or unusual data, and even on English (dickens) at this training budget.

## 5. Mixing a neural model with adaptive context models (`crates/mixlab`)

A paq-style bitwise mixer (float, as an estimate of what is achievable):
- inputs: the neural model's distribution, adaptive context models of order 0–4, 6 and 8, and a match model;
- two online-learned logistic mixers (selected by bit position × match length, and by the previous byte);
- two SSE stages.

Everything restarts at each lane. Samples are the first 4 MB of each file; bsc is nzip balanced on the same 4 MB.

| Sample (4 MB) | NN only | CM only | NN + CM, 1 MB lanes | NN + CM, 4 MB lanes | bsc | NN+CM vs bsc |
|---|---:|---:|---:|---:|---:|---:|
| enwik8 | 2.023 | 1.960 | 1.735 | **1.705** | 1.946 | **−12.4%** |
| dickens | 2.082 | 1.872 | 1.777 | **1.746** | 1.851 | **−5.7%** |
| sao | 7.356 | 5.126 | 5.151 | **5.111** | 5.156 | −0.9% |
| mozilla | 6.814 | 6.241 | 6.236 | **6.229** | 6.246 | −0.3% |
| xml | 2.517 | 0.484 | 0.460 | 0.464 | **0.446** | +4.0% |
| osdb | 6.231 | 2.028 | 2.132 | 2.027 | **1.849** | +9.6% |

(CM-only numbers are for 4 MB lanes.)

- **Mixing is what makes the neural tier viable.** The neural model and the adaptive models are complementary: the network knows language and code in general, and the context models learn the file. Together they beat bsc on text by 6–12% even with a 2M-parameter model trained for 6 minutes.
- **Longer lanes help** the adaptive part (osdb 2.13 → 2.03). The neural tier should use lanes of ≥4 MB, which still gives 250 lanes per GB.
- osdb-style fixed-size records need record/column contexts (standard in paq). That's the main gap left against bsc.

## 6. Decisions and next steps

1. **Architecture:** the delta-rule family (Gated DeltaNet first, RWKV-7 as a backup). Next, train variants with a smaller, INT8-friendly state (e.g. 8 heads × 32 × 32 instead of 4 × 64 × 64) to remove the bandwidth limit, and measure ratio against speed.
2. **Mixing moves from Phase 4 into the neural tier's core.** An integer context-mixing stage (counters, match model, record contexts, mixer, SSE) runs alongside the network, on GPU threads per lane or on CPU workers.
3. **Scale training.** 256 MB was enough to rank the architectures, but the loss was still falling. The chosen model gets multi-GB runs and 2–3 sizes (2M, 4M, 8M).
4. **Honest target check:**
   - the goal is ≤1.4 bpb on all of enwik8 at ≥10 MB/s;
   - today: 1.705 on a 4 MB lane, against bsc's 1.664 on the whole file with 100 MB blocks;
   - closing the gap needs a stronger network (more training and size) plus long-range help (a dedup or match model across lanes).
   - Speed depends on solving the state-bandwidth problem above.
5. INT8/ternary QAT happens once the float architecture and sizes are settled. Ternary matters less here because the bottleneck is state, not weights.
