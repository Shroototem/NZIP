# nzip vs 7-Zip LZMA2 Ultra

Measured 2026-11-27 on the working tree on top of commit 9133381 (not committed yet).
7-Zip numbers are the fixed reference in [7z-ultra.md](7z-ultra.md) (7z was not re-run).

- Machine: AMD Ryzen 9 3900X (12 cores / 24 threads), files in page cache, output on tmpfs.
- nzip defaults: `--cpu 99`, `--gpu 99`, `--ram 99`, `--priority low`.
- Compress: `nzip -c [--fast] -f <folder> -o out.nz`, wall clock (fast: best of 3, max: 1 run).
- Decompress: `nzip -t out.nz` (test, no files written), the same as the reference `7z t`.
- Every archive was also extracted with `nzip -d` and compared with `diff -r`: all identical.
- Sizes vs 7z: negative = smaller than 7z (better).

## Max mode (default)

| Set | 7z bytes | nzip bytes | vs 7z | 7z compress | nzip compress | 7z decompress | nzip decompress |
|---|---:|---:|---:|---:|---:|---:|---:|
| obj | 72,980,940 | 60,667,441 | **-16.9%** | 41.24 s | **28.25 s** | 1.05 s | **1.03 s** |
| incr | 239,233,439 | 249,098,422 | +4.1% | 37.69 s | **28.25 s** | 2.61 s | **2.55 s** |
| exe | 156,142,573 | 154,052,740 | **-1.3%** | 46.31 s | **21.84 s** | **1.56 s** | 1.67 s |
| jar | 639,777,998 | 279,321,437 | **-56.3%** | 31.71 s | **27.85 s** | **3.76 s** | 3.78 s |

## Fast mode (`--fast`)

| Set | 7z bytes | nzip bytes | vs 7z | 7z compress | nzip compress | 7z decompress | nzip decompress |
|---|---:|---:|---:|---:|---:|---:|---:|
| obj | 72,980,940 | 94,057,392 | +28.9% | 41.24 s | **0.56 s** (74x) | 1.05 s | **0.11 s** |
| incr | 239,233,439 | 372,774,609 | +55.8% | 37.69 s | **0.85 s** (44x) | 2.61 s | **0.19 s** |
| exe | 156,142,573 | 214,664,240 | +37.5% | 46.31 s | **0.78 s** (59x) | 1.56 s | **0.16 s** |
| jar | 639,777,998 | 505,256,762 | **-21.0%** | 31.71 s | **0.56 s** (57x) | 3.76 s | **0.15 s** |

A second fast run under heavier background load gave 0.62 / 0.97 / 0.89 / 0.65 s.

## Minecraft world sample (`data/bench/world`, no 7z reference)

The largest file of each type from the world folder, 83,376,780 bytes in 13 files:
region file `r.3.44.mca` (14 MB), `DistantHorizons.sqlite` (69 MB, its LOD data is
XZ-compressed inside the database), and the .dat, .dat_old, .json, .sc, .zip, .conf,
.txt, .lock, .bak, .gitignore and .gitattributes files.

| Mode | Before (9133381) | Now | Compress | Decompress |
|---|---:|---:|---:|---:|
| max | 81,409,141 | **78,517,783** (-3.6%) | 3.76 s | 0.57 s |
| fast | 81,919,266 | 81,919,266 | 0.09 s | 0.03 s |

On the region file alone, max mode now gives 10.23 MB instead of 12.90 MB. Its 1024
chunks are zlib streams, which rezip now finds and rebuilds bit for bit.

## Fast mode: this round vs the start of the session

Same machine state, back to back (best of 3, output byte-identical):

| Set | Before | Now |
|---|---:|---:|
| obj | 0.668 s | **0.620 s** |
| incr | 0.970 s | 0.971 s |
| exe | 0.913 s | **0.887 s** |
| jar | 0.715 s | **0.653 s** |
| world | 0.112 s | 0.108 s |

## Changes in this round

- Rezip finds zlib streams (header 78 01/5E/9C/DA, Adler-32 trailer checked): Minecraft
  region chunks, PDF / git objects. Max mode only; in fast mode the rebuild costs far
  more than it gains (jar decode 0.63 s -> 10.5 s), so it stays a dev option (`NZ_FREZIP`).
- Compression reads each block on its own worker (no single reader thread), blocks are
  written on a writer thread, and more finished blocks may wait behind a slow one.
- Dedup: per-thread read buffers, four-lane gear hash, parallel index / segment building,
  faster chunk keys. Same chunks and archives as before.
- Huffman bit writer with fixed buffers and four literal writers in registers.
- `--priority low|idle|normal`, default `low` (Linux: nice 10 + I/O best effort 7;
  Windows: BELOW_NORMAL). Cost on an idle machine: fast 0-3%, max incr about 4%.

## Still behind 7z

- incr size: +4.1%.
- exe decompress: 1.67 s vs 1.56 s; jar decompress 3.78 s vs 3.76 s (within run-to-run
  noise; earlier runs gave 3.67-3.94 s).
