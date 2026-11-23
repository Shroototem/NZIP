# nzip

A universal lossless compressor in pure Rust for Linux and Windows. It uses every CPU core,
an NVIDIA GPU when one is present, and stays inside a memory budget, so it also runs on
low-RAM machines and on inputs of any size (files or folders, 100 GB+).

```
nzip c [--fast] [options] <file|folder> [-o out.nz]   compress
nzip d [options] <archive.nz> [-o out]                 decompress (file or folder)
nzip t <archive.nz>                                    test integrity
nzip info                                              show detected hardware
```

| Option | Meaning |
|---|---|
| *(default)* | **max**: best compression that stays fast |
| `--fast` | **fast**: maximum speed (LZ77 on every core) |
| `-t N` | threads (default: all logical CPUs) |
| `-m SIZE` | memory budget, e.g. `8G` (default: 90% of total RAM) |
| `-b SIZE` | block size (default: automatic from cores, RAM and VRAM) |
| `-f`, `-v` | overwrite output, verbose (per-block info, peak RAM) |

## How max mode works

```
block ─► word transform (text) / x86 filter (code) ─► BWT ─► CM coder ─► archive
                                                       │
                                  GPU suffix sort ─────┘ (CPU SA-IS if no GPU)
```

1. **Block planning.** Input is streamed in waves of blocks. Block size and the number of
   blocks in flight come from the core count, free VRAM and the RAM budget. Folders cut
   blocks at file and type boundaries, so unrelated data never shares a block.
2. **Transforms.** Text gets a reversible word dictionary built from the block itself, so
   it works for any language. x86/x64 code gets an E8/E9 branch filter. Data that is
   already compressed (JPEG, MP4, zip...) is detected and stored.
3. **BWT.** The suffix array is built on the GPU (prefix doubling over a radix sort, own
   CUDA kernels) and checked on the device with an exact O(n) test. Without a GPU, or if
   anything fails, the CPU (SA-IS) builds it. A suffix array has exactly one correct
   answer, so the archive is byte-identical either way.
4. **Coder.** Each byte is a small decision tree (same as last byte? same as the last
   other byte? else 8 literal bits). Each decision is predicted by adaptive counters,
   mixed by an online-trained neural mixer (16-bit fixed point), refined by SSE, and
   arithmetic-coded. The output is split into independent segments so one block uses
   every core in both directions.

## Correct on every machine

* All modelling is integer arithmetic, with no floating point on the coding path. The
  AVX2 and portable code paths produce identical bytes, picked at run time.
* The GPU is only used to build the suffix array, and every result is verified. Decoding
  never needs a GPU.
* Every block carries a CRC-32. Segment counts, chunk sizes and the BWT index count depend
  only on the data, never on the machine or the thread count.
* The CUDA driver is loaded at run time (`libcuda.so.1` / `nvcuda.dll`), so one binary
  runs everywhere.

## Memory

| | per block in flight |
|---|---|
| max (compress or decompress) | ~7x block size (blocks are at most 256 MB, so at most ~1.8 GB) |
| fast | ~3x block size (4 MB blocks) |

With `-m 8G`, a 10 GB file compressed at 93 MB/s with a 2.4 GB peak and decompressed with
a 4.3 GB peak. A machine with less RAM runs fewer blocks at once and gets the same bytes.

## Results (Ryzen 9 3900X, 24 threads, RTX 5060 Ti 16 GB; every run verified)

Compared with 7-Zip 26.06 LZMA2 Ultra (`-mx=9`, all threads):

| Data | 7-Zip Ultra | nzip max | Size vs 7-Zip |
|---|---:|---:|---:|
| enwik8 (100 MB text) | 24,799,463 · 56.4 s · 0.96 s | 21,002,521 · 2.2 s · 1.5 s | **−15.3%** |
| enwik9 (1 GB text) | 210,604,362 · 212.5 s · 3.1 s | 175,916,660 · 9.4 s · 10.8 s | **−16.5%** |
| Silesia folder (212 MB) | 48,688,243 · 139.5 s · 3.9 s | 46,250,824 · 3.0 s · 3.4 s | **−5.0%** |
| Firefox folder (381 MB, mostly x64 DLLs) | 93,338,606 · 205 s · 4.0 s | 114,698,850 · 5.5 s · 6.4 s | +22.9% |

Columns: size in bytes · compress time · decompress time.

nzip compresses 20–70x faster than 7-Zip Ultra. 7-Zip decompresses faster: LZMA decoding
is cheap, while a BWT + context-mixing decoder costs about as much as its encoder
(enwik9: 3.1 s vs 10.8 s). For reference, the enwik9 record (fx2-cmix, 100.4 MB) takes tens
of hours on one core.

Fast mode on enwik9: 474,584,207 bytes, 1.27 GB/s compress, 2.24 GB/s decompress.

Known weak spot: executables and other binary code. BWT scrambles the fixed-position
structure that LZ77 coders exploit. The next planned engine is an LZ77 parser with
context-mixed coding for binary blocks.

## Building

```
cargo build --release              # Linux; output target/release/nzip
scripts/build-linux.sh             # target/dist/nzip-<ver>-linux-x86_64.tar.gz
scripts/build-windows.sh           # cross-build target/dist/nzip-<ver>-windows-x86_64.zip
scripts/build-ptx.sh               # only after editing gpu/sa.cu (needs nvcc)
cargo test --release               # unit tests (GPU test is skipped without a GPU)
```

No dependencies beyond the Rust standard library.
