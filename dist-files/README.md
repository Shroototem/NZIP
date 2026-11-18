# nzip — test build 0.4.0 (Linux x86-64)

A fast lossless compressor that uses the GPU when it helps and the CPU otherwise. This is
an **early test build**: the classic tiers are done and benchmarked; the neural tiers are
still being trained and are not in this build.

> Keep your originals. Every file is checksummed (each block and the whole file), and
> `nzip test` verifies an archive without writing anything, but this is pre-release
> software. The archive format may change between test builds. This build reads 0.3.x
> archives; archives it writes need 0.4.0 or newer.

## Requirements

- Linux x86-64 with an **AVX2** CPU (x86-64-v3): Intel Core 4th gen (Haswell, 2013) or
  newer, AMD Ryzen / Excavator or newer. Check with `grep -o avx2 /proc/cpuinfo | head -1`.
- glibc 2.38 or newer: Ubuntu 24.04+, Debian 13+, Fedora 39+, RHEL/Rocky/Alma 10,
  Arch / CachyOS / Manjaro / EndeavourOS, openSUSE Tumbleweed. Check with `ldd --version`.
- Optional: an NVIDIA GPU (GTX 16xx / RTX 20xx or newer) with driver **580 or newer**
  (CUDA 13). Without one, or with an older driver, everything runs on the CPU and the
  results are identical — just slower in max mode.

Nothing to install: unpack and run. `lib/` holds the OpenMP runtime the binary uses.

## Quick start

```bash
./nzip -c <input> <output>            # compress, max (best) compression, the default
./nzip -c -m fast <input> <output>    # compress fast
./nzip -c -m store <input> <output>   # store only (checksummed)
./nzip -d <input.nz> <output>         # decompress
./nzip -t <input.nz>                  # verify all checksums, write nothing
./nzip -i <input.nz>                  # show the archive's blocks
./selftest.sh                         # round-trip check on generated data
./compare.sh <file>                   # nzip vs zstd/xz/brotli/bzip2 if installed
```

Folders work the same way: `./nzip -c <folder>` writes `<folder>.nz` (files, subfolders,
permissions, modification times and symlinks), and `./nzip -d <folder>.nz` restores the
folder. Extraction refuses entries that would land outside the destination.

The output name is optional: `-c` writes `<input>.nz`, `-d` strips `.nz`. The long forms
`compress`/`decompress`/`test`/`info` (or `c`/`d`/`t`/`i`) work too.

## Modes

| Mode | What it is | Typical speed (RTX 5060 Ti + Ryzen 9 3900X) | Ratio |
|---|---|---|---|
| `-m max` (default) | Best compression. Deflate streams anywhere in the input (PNG, zip, docx, jar, gzip, PDF, …) are unpacked into their tokens and rebuilt bit-exact on decode; text gets a word dictionary; then LZP + BWT block sorting + adaptive context coding, long-range dedup. GPU and CPU work as a pipeline: each block's sort runs on the GPU when it fits the card's memory at that moment, otherwise on the CPU | 6 GB game-project folder: 159 MB/s compress, 170 MB/s decompress; text (enwik9): ~170 MB/s compress, ~300 MB/s decompress | smaller than xz -9 and zstd -19 |
| `-m fast` | zstd-3 on all CPU cores, 4 MiB blocks | 1.4–1.7 GB/s compress, 2–3 GB/s decompress | like zstd -3 |
| `-m store` | No compression, still checksummed | disk speed | 1.0 |

Already-compressed data (JPEG, MP4, zip, …) is detected and stored, so it costs no time.

## Options worth knowing

| Option | Meaning |
|---|---|
| `-D auto\|gpu\|cpu` | Where heavy stages run. `auto` (default) picks per block; `cpu` forces CPU only. The archive is byte-identical either way |
| `-t N` | Threads (default: all) |
| `-M 4G` | Memory budget for buffers (default: half of available RAM, respecting container limits). Files of any size work. A smaller budget is slower; below ~7 GB, max mode also uses smaller blocks (slightly bigger archive) |
| `-b 640M` | Block size (default: 1 GiB in max mode). Smaller blocks that fit your GPU are faster on big non-text inputs, ~0.5–3% bigger |
| `-f` | Overwrite an existing output |
| `--no-dedup`, `--no-transforms` | Turn off long-range dedup / the deflate unpacking, dictionary, executable and table transforms |

## Measured results (enwik9 = 1 GB of Wikipedia; Silesia = 212 MB mixed corpus, 12 files compressed separately)

A real 6.1 GB game-project folder (128,647 files, 62% PNG), all tools on all threads:

| Tool | Size | Ratio | Compress | Decompress |
|---|---:|---:|---:|---:|
| **nzip 0.4.0 max** | **2,283,175,900** | **37.4%** | **154 MB/s** (39.5 s) | **166 MB/s** (36.8 s) |
| nzip 0.3.2 max | 2,643,139,605 | 43.3% | ~77 MB/s (79 s) | ~150 MB/s |
| 7-Zip 9 ultra (LZMA2, all threads) | 2,911,236,680 | 47.7% | 32 MB/s (190.9 s) | 744 MB/s (8.2 s) |
| xz -9 -T0 | 2,939,979,520 | 48.2% | 35 MB/s (174.6 s) | 803 MB/s (7.6 s) |
| zstd -19 --long=31 -T0 | 2,992,980,615 | 49.1% | 25 MB/s (242.4 s) | 709 MB/s (8.6 s) |
| nzip 0.4.0 fast | 3,490,144,906 | 57.2% | 1,220 MB/s (5.0 s) | 1,906 MB/s (3.2 s) |
| zstd -3 -T0 | 3,524,938,231 | 57.8% | 1,906 MB/s (3.2 s) | 1,196 MB/s (5.1 s) |


| Tool | enwik9 | Silesia | enwik9 compress / decompress |
|---|---:|---:|---:|
| nzip max | **161.6 MB** | **46.5 MB** | 172 / 308 MB/s |
| bsc (best setting) | 182.0 MB | 46.5 MB | 76 / 263 MB/s |
| xz -9e | — | 48.5 MB | ~2 / 125 MB/s |
| zstd -19 | 235.3 MB | 52.9 MB | 13 / 558 MB/s |
| nzip fast | 313.4 MB | 66.0 MB | 1,725 / 2,989 MB/s |
| zstd -3 | 312.1 MB | 66.2 MB | 1,125 / 664 MB/s |
| NVIDIA nvCOMP zstd (GPU, from host memory) | 339.0 MB | 70.6 MB | 969 / 2,492 MB/s |

Large inputs: a 6.1 GB game-project folder (128,647 files, 62% PNG) compresses to 2.28 GB at
159 MB/s and restores byte-identical at 170 MB/s; 0.3.2 made 2.64 GB at ~75 MB/s.
For reference, the Hutter Prize record for enwik9 (July 2026) is 100.4 MB, taking tens of
hours each way.

## What to report

Please send: the command, file type and size, the speed and size nzip printed, your CPU,
GPU + driver (`nvidia-smi`), distro, and the output of `./selftest.sh`. For a comparison,
`./compare.sh <file>` prints a table you can paste.

## Known limitations of this build

- Linux x86-64 only (no Windows/macOS build yet). glibc 2.38+ and an AVX2 CPU.
- No neural mode yet; the GPU is used for BWT sorting in max mode (a block's sort needs
  ~21 bytes of GPU memory per byte after the dictionary and LZP; larger ones run on the CPU).
- Max mode's 1 GiB blocks need ~7 GB of RAM to compress (it uses smaller blocks on
  machines with less) and up to ~6 GB to decompress without a GPU.
- Archives from this build won't open in future builds if the format changes; keep originals.

Third-party components and their licenses: see `THIRD_PARTY_NOTICES.md`.
