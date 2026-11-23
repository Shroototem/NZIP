# nzip

A universal lossless compressor in pure Rust for Linux and Windows. It uses every CPU core,
an NVIDIA GPU when one is present, and stays inside a memory budget, so it also runs on
low-RAM machines and on inputs of any size (files or folders, 100 GB+).

```
nzip -c [--fast] [options] <file|folder> [-o out.nz]   compress
nzip -d [options] <archive.nz> [-o out]                 decompress (file or folder)
nzip -t <archive.nz>                                    test integrity
nzip --bench <file>                                     compress + decompress in memory, verify
nzip --info                                             show detected hardware
```

| Option | Meaning |
|---|---|
| *(default)* | **max**: best compression that stays fast |
| `--fast` | **fast**: maximum speed (LZ77 on every core) |
| `--cpu PCT` | share of logical CPUs to use (default 99) |
| `-T N` | exact thread count (overrides `--cpu`) |
| `-p` | progress bar |
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

Compared with 7-Zip 26.06 LZMA2 Ultra (`-mx=9`, all threads; reference numbers are kept in
[bench/7z-ultra.md](bench/7z-ultra.md)). nzip uses its default 99% of CPU threads.

| Data | 7-Zip Ultra | nzip max | Size vs 7-Zip |
|---|---:|---:|---:|
| enwik8 (100 MB text) | 24,799,463 · 40.3 s · 0.80 s | 21,187,957 · 1.2 s · 0.70 s | **−14.6%** |
| enwik9 (1 GB text) | 210,604,362 · 212.5 s · 3.1 s | 177,404,158 · 5.9 s · 4.0 s | **−15.8%** |
| Silesia folder (212 MB) | 48,688,243 · 39.4 s · 1.51 s | 43,687,156 · 4.1 s · 1.14 s | **−10.3%** |
| Portal 2 folder (238 MB, DLLs + NN weights) | 143,808,833 · 73.7 s · 4.5 s | 140,176,362 · 7.5 s · 3.0 s | **−2.5%** |
| Firefox folder (381 MB, x64 DLLs + jars) | 93,338,587 · 78.3 s · 3.03 s | 96,636,809 · 19.0 s · 1.77 s | +3.5% |
| TestSet folder (4.4 MB, mixed small files) | 3,344,553 · 0.32 s · 0.09 s | 3,343,699 · 0.21 s · 0.08 s | −0.03% |

Columns: size in bytes · compress time · decompress time.

Open gaps: Firefox size (7-Zip compresses the whole folder as one stream, so small DLLs
reuse xul.dll's code), enwik9 decode speed (4.0 s vs 3.1 s), and already-compressed
content (PNG, gzip, fonts) that only recompression could shrink further.

Inverse BWT runs on the GPU when one is present (CPU fallback; the result is checked).

Fast mode on enwik9: 474,584,207 bytes, 1.27 GB/s compress, 2.24 GB/s decompress.

## Building

```
cargo build --release              # Linux; output target/release/nzip
scripts/build-linux.sh             # target/dist/nzip-<ver>-linux-x86_64.tar.gz
scripts/build-windows.sh           # cross-build target/dist/nzip-<ver>-windows-x86_64.zip
scripts/build-ptx.sh               # only after editing gpu/sa.cu (needs nvcc)
cargo test --release               # unit tests (GPU test is skipped without a GPU)
```

No dependencies beyond the Rust standard library.
