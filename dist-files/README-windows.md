# nzip — test build 0.4.0 (Windows x86-64, CPU only)

A fast lossless compressor. This Windows build runs everything on the CPU; the Linux build
also uses NVIDIA GPUs (see "GPU build" below). Archives are identical across Windows and
Linux: anything one makes, the other decompresses.

> Early test build: this Windows binary was cross-compiled from Linux and **has not been
> run on Windows yet**. Keep your originals. Every block and the whole file are
> checksummed, and `nzip -t` verifies an archive without writing anything.

## Requirements

- Windows 10 or 11, 64-bit (the Universal C Runtime is part of the OS).
- Any x86-64 CPU from roughly 2009 on (x86-64-v2: SSE4.2, POPCNT).
- Keep `libomp.dll` next to `nzip.exe` (multi-threading inside each block).

## Quick start (Command Prompt or PowerShell)

```
nzip.exe -c <input> <output>            compress, max (best) compression, the default
nzip.exe -c -m fast <input> <output>    compress fast
nzip.exe -c -m store <input> <output>   store only (checksummed)
nzip.exe -d <input.nz> <output>         decompress
nzip.exe -t <input.nz>                  verify all checksums, write nothing
nzip.exe -i <input.nz>                  show the archive's blocks
```

Folders work the same way: `nzip.exe -c C:\path\to\folder` writes `folder.nz`, and
`nzip.exe -d folder.nz` restores it. Symbolic links inside a folder can only be restored
on Windows with Developer Mode enabled or as administrator.

## Modes

| Mode | What it is |
|---|---|
| `-m max` (default) | Best compression: deflate streams anywhere in the input (PNG, zip, docx, jar, gzip, PDF, ...) are unpacked into their tokens and rebuilt bit-exact on decode; text gets a word dictionary; then BWT block sorting + context coding with long-range dedup |
| `-m fast` | zstd-3 on all cores |
| `-m store` | No compression, still checksummed |

Max mode on the CPU alone is slower than the Linux build with a GPU (the block sort is the
step the GPU accelerates).

## Options worth knowing

| Option | Meaning |
|---|---|
| `-t N` | Threads (default: all) |
| `-M 4G` | Memory budget for buffers (default: half of the available RAM) |
| `-b 640M` | Block size (default 1 GiB in max mode) |
| `-f` | Overwrite an existing output |
| `nzip.exe gpu` | Show GPU support, driver status and free GPU memory |
| `--no-dedup`, `--no-transforms` | Turn off long-range dedup / the deflate unpacking, dictionary, executable and table transforms |

## GPU build (build it on Windows)

NVIDIA's CUDA compiler needs Microsoft's compiler on Windows, so a GPU-enabled Windows
`nzip.exe` must be built on a Windows PC:

1. Install Visual Studio 2022 (Desktop development with C++), the CUDA Toolkit 13
   (sets `CUDA_PATH`) and Rust (https://rustup.rs, MSVC toolchain).
   Use an NVIDIA driver at least as new as the toolkit (CUDA 13 installers no longer
   include the driver).
2. In a new terminal, in the source folder:
   `set NZIP_REQUIRE_CUDA=1` (PowerShell: `$env:NZIP_REQUIRE_CUDA=1`), then
   `cargo build --release -p nzip-cli`.
   With `NZIP_REQUIRE_CUDA` set, the build stops with the reason if it can't find CUDA,
   instead of quietly making a CPU-only build (it prints a
   "building WITHOUT GPU support: ..." warning otherwise).
3. The result is `target\release\nzip.exe`. Check it with `nzip.exe gpu`: it shows
   whether the build has GPU code, whether the driver works (with the CUDA error in
   plain words if not) and the largest block sort that fits the GPU's free memory.

The GPU sorts a block only if it fits the free GPU memory (~21 bytes per byte), so on a
16 GB card the default 1 GiB blocks are borderline: Windows and open programs (browsers,
games, Discord) hold GPU memory too. After a max-mode run, nzip says when blocks went to
the CPU for lack of memory; close GPU-heavy programs or use `-b 512M`.

Third-party components and their licenses: see `THIRD_PARTY_NOTICES.md`.
