# nzip records: best runs and best configurations

Kept up to date after every measured improvement. Machine unless noted: Ryzen 9 3900X
(12C/24T), 128 GB RAM, RTX 5060 Ti 16 GB (PCIe Gen3 x8 in this slot), CachyOS.

**Target (set 2026-11-18):** enwik9 → ~100 MB at ~100 MB/s compress *and* decompress,
GPU + CPU together, any card size, and decoding gives the same bytes on any GPU, CPU or
mix of the two.

## 0.4.0 comparison tables (2026-11-18)

AtO game-project folder (6,100,838,400-byte tar, 128,647 files, 62% PNG), this machine,
all tools multi-threaded, page cache warm, every round trip checked byte for byte:

| Tool | Size | Ratio | Compress | Decompress |
|---|---:|---:|---:|---:|
| **nzip 0.4.0 max** | **2,283,175,900** | **37.4%** | **154 MB/s** (39.5 s) | **166 MB/s** (36.8 s) |
| nzip 0.3.2 max | 2,643,139,605 | 43.3% | ~77 MB/s (79 s) | ~150 MB/s |
| 7-Zip 9 ultra (LZMA2, all threads) | 2,911,236,680 | 47.7% | 32 MB/s (190.9 s) | 744 MB/s (8.2 s) |
| xz -9 -T0 | 2,939,979,520 | 48.2% | 35 MB/s (174.6 s) | 803 MB/s (7.6 s) |
| zstd -19 --long=31 -T0 | 2,992,980,615 | 49.1% | 25 MB/s (242.4 s) | 709 MB/s (8.6 s) |
| nzip 0.4.0 fast | 3,490,144,906 | 57.2% | 1,220 MB/s (5.0 s) | 1,906 MB/s (3.2 s) |
| zstd -3 -T0 | 3,524,938,231 | 57.8% | 1,906 MB/s (3.2 s) | 1,196 MB/s (5.1 s) |

enwik9 (1,000,000,000 bytes):

| Tool | Size | Compress | Decompress |
|---|---:|---:|---:|
| fx2-cmix-T (Hutter Prize record, Jul 2026) | 100,424,672 | tens of hours | tens of hours |
| **nzip 0.4.0 max** | **161,635,432** | **171 MB/s** (5.86 s) | **318 MB/s** (3.14 s) |
| bsc (best setting) | 182.0 MB | 76 MB/s | 263 MB/s |
| xz / LZMA2 ultra (published) | ~211.8 MB | ~1.7 MB/s | — |
| zstd -19 | 235.3 MB | 13 MB/s | 558 MB/s |
| zstd -3 | 312.1 MB | 1,125 MB/s | 664 MB/s |
| nzip 0.4.0 fast | 313.4 MB | 1,725 MB/s | 2,989 MB/s |
| NVIDIA nvCOMP zstd (GPU) | 339.0 MB | 969 MB/s | 2,492 MB/s |

## enwik9 (1,000,000,000 bytes)

| Date | Build / config | Size (bytes) | bpb | Compress | Decompress | Notes |
|---|---|---:|---:|---:|---:|---|
| 2026-11-18 | 0.4.0 `-m max` (package build) | **161,635,432** | 1.293 | **171 MB/s** (5.86 s) | **318 MB/s** (3.14 s) | Same bytes as 0.3.2 (no deflate streams in enwik9) |
| 2026-11-18 | 0.3.2 `-m max` + word dictionary (default) | 161,635,432 | 1.293 | **172 MB/s** (5.8 s) | 308 MB/s (3.25 s) | **Best overall.** Text shrinks to 596 MB → LZP 578 MB → GPU sort 1.3 s |
| 2026-11-18 | 0.3.2 `-m max` without dictionary (1 GiB block) | 164,205,860 | 1.314 | 55 MB/s | 295 MB/s | Its sort on the CPU (doesn't fit 16 GB) |
| 2026-11-18 | 0.3.2 `-b 640M` (blocks that fit a 16 GB card) | 169,415,210 | 1.355 | **251 MB/s** | 315 MB/s | Fastest max-family run |
| 2026-11-18 | 0.3.2 `-m fast` | 313.4 MB | 2.507 | 1,725 MB/s | 2,989 MB/s | zstd-3, 4 MiB blocks |

Reference points (others):

| Tool | Size | Speed | Source |
|---|---:|---|---|
| fx2-cmix-T (Hutter Prize, 2026-07-24) | 100,424,672 | tens of hours each way, 1 core | Wikipedia Hutter Prize table |
| cmix-obias (2026-07-19) | 108,521,870 | tens of hours | same |
| bsc (best setting) | 182.0 MB | 76 / 263 MB/s | measured here |
| xz / LZMA2 ultra | ~211.8 MB | ~1.7 MB/s | published |
| zstd -19 | 235.3 MB | 13 / 558 MB/s | measured here |
| NVIDIA nvCOMP zstd (GPU) | 339.0 MB | 969 / 2,492 MB/s | measured here |

## Real folder: AtO project (6,100,838,400-byte tar, 128,647 files)

| Date | Config | Size (bytes) | Compress | Notes |
|---|---|---:|---:|---|
| 2026-11-18 | 0.3.2 default max (1 GiB, staged GPU/CPU pipeline) | **2,643,139,605** | 77.2 MB/s | Best ratio. 3 blocks sorted on GPU, 3 on CPU |
| 2026-11-18 | 0.3.2 `-b 640M` | 2,658,518,143 | **179 MB/s** (34.0 s) | All but one block on GPU; fastest |
| 2026-11-18 | 0.3.2 `-D cpu` (1 GiB) | 2,643,139,605 | 64.1 MB/s | Same bytes as GPU runs |
| 2026-11-18 | 0.4.0: universal deflate tokens, 512 MiB waves, lean candidate check | **2,283,175,900** (−13.6%) | **159.0 MB/s (38.8 s)** | Decompress 170 MB/s (36 s). Best overall: smaller AND ~2x faster than 0.3.2 |
| 2026-11-18 | dev: tokens, 512 MiB waves (full-output candidate check, 7.3 s) | 2,283,175,900 | 142.8 MB/s (43.1 s) | |
| 2026-11-18 | dev: tokens, 1 GiB waves | 2,273,839,308 | 72.6 MB/s (84.4 s) | Literal blocks ~700 MB after LZP: too big for 16 GB VRAM, 6 CPU sorts |
| 2026-11-18 | dev: expanded plain + main/waves layout + streamed waves | 2,438,839,006 | 64.1 MB/s (95.6 s) | Unpacking 53 GB of pixels dominated |
| 2026-11-18 | dev: universal deflate unpacker v1 (1 GiB waves) | **2,412,300,116** (−8.7%) | 51.8 MB/s (118 s) | 4,589 of 4,597 PNG streams rebuilt exactly; decompress 112 MB/s (54.5 s). Loses folder-wide dedup across waves |
| 2026-11-17 | 0.3.1 max 1 GiB, CPU | 2,648,699,653 | 56.3 MB/s | Before pipeline (folder input, older packer) |
| 2026-11-17 | 0.3.1 `-b 512M`, pure GPU queue | 2,676,198,168 | 91.4 MB/s | Old best scheduler |

Decompress: 131–150 MB/s CPU-only (also with `-M 3G`), byte-identical.

### AtO folder breakdown (2026-11-18, current max, per-type tars)

| Group | Original | Compressed | Time |
|---|---:|---:|---:|
| PNG (4,597 files) | 3,681 MB | 2,197 MB | 81 s (CPU sort: LZP output doesn't fit VRAM) |
| Unity YAML text + .cs/.meta/.txt | 1,865 MB | 29.8 MB | 2 s (dictionary applied) |
| Other (fonts, dll, wav, …) | 243 MB | 123 MB | 2 s |
| OGG | 305 MB | 297 MB | 4 s |

PNG facts (199-file sample, 145 MB): all RGBA/RGB 8-bit, zlib FLEVEL 0 (fastest deflate),
every row uses the Up filter, pixel data is 14.4× the PNG size (2.1 GB), 66.6% of RGBA pixels
have alpha 0 and 57% of those keep non-zero RGB (must be preserved).

| Sample coding (145 MB of PNG) | Size |
|---|---:|
| PNG bytes → nzip max (today) | 92.4 MB |
| Inflated, Up-filtered bytes → max | 84.3 MB |
| Unfiltered pixels → max | 95.5 MB |
| MED (LOCO-I) residuals, interleaved RGBA → max | **74.0 MB (−20%)** |
| MED residuals planar / planar + subtract-green | 91.6 / 97.2 MB (worse) |
| cjxl lossless e1 / e3 (184 of 199 files, pixels only) | 122.8 / 131.6 MB (per file, no cross-file redundancy) |

### Universal deflate rebuild (crates/unpack, 2026-11-18)

| 145 MB fpng PNG sample | preflate-rs 0.7.6 | **nzip redeflate (ours)** |
|---|---:|---:|
| Exact rebuilds | 199/199 | 199/199 |
| Recipe size | 19,219,857 B (13.2%) | **83,245 B (0.057%)** |
| Unpack + verify / rebuild | 4.77 s / 1.67 s | **1.64 s / 0.68 s** |
| Sample through nzip (unpack mode) | — | 78.4 MB vs 92.4 MB today (−15%), 5.7 s |

No encoder-specific code: block headers stored verbatim, encoder choices coded by an adaptive
model (rep-distance match length, maxlen cap, stride-aligned candidate, literal phase).

### Deflate tokens vs expanded plain (145 MB PNG sample)

| Representation | Bytes to compress | Final size |
|---|---:|---:|
| PNG bytes (before) | 145 MB | 92.4 MB |
| Expanded plain (pixels) | 2,097 MB | 75.7 MB |
| **Tokens: literals + copy records** | **310 MB** | **72.5 MB** (pipeline: 73.5 MB, 2.7 s) |
| Copy records split into run/len/dist streams | — | copies 16.1 MB vs 15.3 MB interleaved (worse) |

### Findings (2026-11-18)

1. **PNG-heavy data is a deflate problem, not an image problem.** 62% of the folder is PNG
   written by fpng (fastest deflate, Up filter). Compressing PNG bytes directly leaves ~20%
   on the table.
2. **Rebuild recipes must not depend on guessing the encoder.** preflate-rs guesses a zlib
   encoder: 13.2% recipes on fpng output. Our model-based rebuild learned fpng from the
   data (0.057%), but the best answer was simpler still: **store the deflate tokens**
   (literals + copy records + verbatim block headers). That rebuilds any encoder's output
   exactly with no model at all, and never expands the data (pixels would be 14x).
3. **Tokens beat full decompression on both axes:** 310 MB vs 2,097 MB to compress for the
   145 MB sample, 72.5 MB vs 75.7 MB final.
4. **Speed comes from fitting the GPU and never waiting:** 512 MiB waves keep every literal
   block's sort under 16 GB of VRAM (1 GiB waves forced six 35-40 s CPU sorts); waves stream
   with no group barrier; the main bytes compress alongside.
5. **Fake stream starts are common** (528,713 candidates for 4,589 real streams). zlib-style
   Huffman-set validation, then a checksum only for suspicious candidates, keeps the check
   at 2.9 s; correctness never depends on it because every stream is rebuilt and compared.
6. **Generic structure detection (row/element size) is not solved yet**: the measures that
   find row lengths are off by ±1 pixel and cheap judges mispredict BWT gains. Parked.

## Best configuration (current code)

- Max mode: 1 GiB blocks always; staged LZP → BWT → QLFC per block (`crates/nzip/src/sched.rs`).
- A block's sort goes to the GPU only if `21 × LZP length` fits free VRAM (libbsc's cached
  storage counted as reusable); else, or when the GPU queue would take longer than a CPU sort,
  the CPU sorts it. Identical bytes either way.
- The coder stage gets all threads (the OS shares them); LZP and CPU sorts get a fair share.
- Text blocks (≥1 MiB, ≥50% letters/spaces, ≤10% high bytes): word dictionary transform
  (`crates/nzip/src/dict.rs`, `TF_DICT`): 60,000 words, codes 1-3 high bytes (32 / 64×128 /
  32×128²), most frequent → shortest, sorted by reversed spelling within each length; CAP/UP
  flags; kept only if ≥15% smaller. Output depends only on the data (4 MiB chunks).
- Universal deflate unpacking (`crates/unpack`, `crates/nzip/src/expand.rs`, `FLAG_UNPACK`):
  candidates after zlib headers / gzip headers / zip local headers; pass 1 measures each
  (Huffman decode only; Adler-32 only for short or barely-expanding bare-zlib candidates);
  pass 2 splits accepted streams into tokens in waves of <= 512 MiB, literals and copies
  compressed as separate archives; everything else compressed once as the main archive.
- `NZIP_TRACE=1` prints a stage timeline (and dictionary / unpack timings).

## Measured dead ends (don't retry without a new idea)

| Idea | Result |
|---|---|
| GPU sort with managed memory (swap over PCIe) for blocks larger than VRAM | 1 GiB block: 28.1 s GPU vs 21.0 s CPU. Worse on Gen3 x8 |
| GPU queue capped at 3 + CPU overflow (0.3.1 GpuTicket) | 67 MB/s vs 91 MB/s for the pure queue |
| Splitting QLFC into many streams for a GPU coder | +0.7% at 1,024 streams, +1.3% at 4,096, +3–3.6% at 16,384 |
| One GPU OOM inside libcubwt | Can corrupt the *next* GPU sort. Never attempt a sort that might not fit; on any CUDA error, disable the GPU and redo the block on the CPU |
| Word dictionary variants on enwik8 (bsc after) | freq order −1.0%; alpha −1.2%; suffix order −1.4%; 60k words with 3-byte codes −1.45%; 150k words −1.3%. Gains plateau ~1.5% for BWT |
| fpng-specific replay (port of fpng's encoder) | Dropped before testing: the user wants one universal tool, and the universal rebuild costs 0.057% anyway |
| Unpack candidates tried in sequential batches | 34 s on the sample (fake zlib headers decoding garbage, serial real streams). Fixed: zlib-style Huffman validation, adler check for bare zlib headers, two-pass plan: 5.7 s |
| Automatic MED structure transform on unpacked pixels | Row length found within ±1 pixel, element size often wrong; judges (magnitude sum, zstd-1, order-2 model) all mispredict BWT gains. Wrong params cost +15-50%. Moot with tokens (no pixels) |
| Expanded-plain prediction model (redeflate) | Works (0.057% recipes) but 53 GB of pixels made compress 95-118 s; replaced by tokens |
| Dictionary on the AtO folder | Not applied (blocks fail the text check), so no change |

## Stage timings (one block, 24 threads)

| Data | LZP | Sort GPU | Sort CPU (≤8 thr) | QLFC |
|---|---:|---:|---:|---:|
| AtO 1 GiB → 524 MB after LZP | 0.22 s | 1.2 s | 12.0 s | 3.6 s |
| AtO 1 GiB → ~1000 MB after LZP | 0.32 s | doesn't fit | 20–24 s | 6.5–8 s |
| enwik9 640 MiB → 600 MB after LZP | — | — | — | 1.4 s |
