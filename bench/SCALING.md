# Any input size, any RAM, any VRAM

Date: 2026-11-17. Machine: RTX 5060 Ti 16 GB, Ryzen 9 3900X, 125 GB RAM, SATA SSD.

## What changed

| Area | Before | Now |
|---|---|---|
| Compress output | Whole container built in RAM, then written | **Streamed** to the file wave by wave; the 40-byte file header is patched at the end |
| Dedup | Whole-input analysis, full copy of the unique bytes in RAM | **Per segment** (fixed 1 GiB, or 2× block), with one global fingerprint table, so copies still reach anywhere earlier in the file. Table capped at 1/8 of the memory budget |
| Block concurrency | Every block of the input at once (10 GB balanced ≈ 60 GB RAM) | At most `active` blocks hold codec memory at once, from the budget: half for codec working memory, a quarter for a wave's input buffers and a quarter for its encoded frames |
| Decompress | Whole output (plus a second buffer) in RAM | **Streamed** to a sink (the output file) in memory-bounded batches. Dedup copies read earlier output back from the file. `nzip test` keeps no output at all (a temporary file only when dedup copies need it) |
| Memory budget | n/a | Half of what this process can use: `min(MemAvailable, room under the cgroup v2 memory.max)`, so containers and systemd scopes are respected. Override with `-M/--memory` |
| Output vs. budget | n/a | **The compressed file does not depend on the memory budget or thread count**: block and segment boundaries never depend on RAM, only how much runs at once does |
| GPU coder (ncore) batches | Fixed 2,048 lanes (~3 GB VRAM); whole input copied to the GPU | Lanes per batch from **free VRAM** (`cudaMemGetInfo`, 512 MiB left for others); only each batch's data crosses PCIe |
| bsc GPU blocks | Block size from free VRAM (Phase 1) | Unchanged. libbsc serialises GPU BWT calls, and falls back to the CPU inverse BWT if a card can't fit a block when decoding. Any file decodes on any card, or on none |
| Testing small cards | n/a | `NZIP_VRAM_LIMIT=<bytes>` makes nzip size its GPU work as if the card had that much free memory |
| Format | v1 | **v2**: with dedup, frames come as [recipe (param = data frames in segment), data…] per segment |

## Test: 10.9 GB mixed input, full RAM vs a real 5 GB cap

Input (`bench/make_bigmix.py`): enwik9, Silesia, 2.9 GiB of ELF binaries, 1 GiB of /usr/share data files, 0.8 GiB of headers and Python source, 3.4 GiB of already-compressed downloads, then enwik9 again (an exact repeat ~9 GiB later). Balanced mode, default settings.

The 5 GB run is inside a user systemd scope with `MemoryMax=5G` and no swap: going over would get the process killed. nzip found the limit on its own (cgroup), with no flags. Peak memory is private memory (RssAnon + RssShmem, `bench/peakmem.py`); file pages from the mmap'd input are page cache and excluded.

| | Full RAM (budget ≈ 60 GB) | 5 GB cap (budget ≈ 2.5 GB) |
|---|---:|---:|
| Compress | 140.7 MB/s, 77.8 s | 126.1 MB/s, 86.8 s (−10%) |
| Compress peak memory | 20.3 GB | **3.4 GB** |
| Decompress | 146.8 MB/s, 74.5 s | 118.3 MB/s, 92.5 s (−19%) |
| Decompress peak memory | 2.7 GB | **2.2 GB** |
| Compressed size | 4,540,967,247 B (3.325 bpb) | **byte-identical file** |
| Round trip | identical to the input | identical to the input |

- Dedup found the far repeat: the second enwik9 costs almost nothing.
- 3.4 GiB of the input was already compressed and was stored, which dominates the 3.325 bpb.
- Under the cap, decompression is slower because the page cache for a 10.9 GB output must be recycled inside 5 GB; compression is slower because fewer blocks run at once.

## Regression check (normal-size files)

| File | Size before → after | Speed now (compress / decompress, incl. writing the output) |
|---|---|---|
| enwik8, balanced | 20,803,078 → 20,803,078 | 125 / 203 MB/s |
| enwik9, balanced | 169,415,210 → 169,415,210 | 217 / 297 MB/s |
| enwik8, core-test | 46,985,366 → 46,985,366 | 300 / 288 MB/s |
| 1 KB file, balanced | — | 7 ms round trip |
| empty file | — | 40-byte container, round trip OK |

Tests: 13 container tests, including `output_independent_of_memory_budget` (16 MiB vs 1 GiB budgets, 4 modes, several dedup segments with cross-segment copies, decoded with a 16 MiB budget), plus `small_vram_multi_batch_matches_cpu` (ncore under a ~25 MB simulated card: many batches, byte-identical to the CPU).
