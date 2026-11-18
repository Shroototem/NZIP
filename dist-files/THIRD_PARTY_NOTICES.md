# Third-party components in this build

| Component | Use | License |
|---|---|---|
| libbsc 3.4.0 by Ilya Grebnov (includes libsais and libcubwt) | Max mode: LZP, BWT (CPU and GPU), QLFC coder | Apache-2.0 (`LICENSE.libbsc`) |
| zstd 1.5.7 (Meta) via the `zstd`, `zstd-safe`, `zstd-sys` crates | Fast mode, recipe frames | BSD-3-Clause (zstd-sys: BSD-3-Clause AND BSD-2-Clause) |
| NVIDIA CUDA Runtime (cudart_static, cudadevrt), CUDA 13 | GPU kernels; linked statically, loads the driver at run time | NVIDIA CUDA Toolkit EULA (redistributable runtime components) |
| CUB / Thrust (part of the CUDA toolkit, used by libcubwt) | GPU primitives | BSD-3-Clause / Apache-2.0 |
| GNU libstdc++ (statically linked) and libgomp (`lib/libgomp.so.1`), Linux build | C++ runtime and OpenMP threading for libbsc | GPL-3.0 with the GCC Runtime Library Exception |
| LLVM libc++, libunwind, compiler-rt (statically linked) and LLVM OpenMP (`libomp.dll`), Windows build only, from llvm-mingw 20261104 | C++ runtime and OpenMP threading | Apache-2.0 WITH LLVM-exception |
| mingw-w64 runtime (Windows build only) | Windows C runtime startup code | ZPL-2.1 and public-domain parts |
| xxhash-rust 0.8.21 | xxh3 checksums | BSL-1.0 |
| rayon, rayon-core, crossbeam-deque/-epoch/-utils, either | Parallelism | MIT OR Apache-2.0 |
| clap, clap_builder, clap_derive, clap_lex, anstream, anstyle, anstyle-parse, anstyle-query, colorchoice, is_terminal_polyfill, utf8parse, heck | Command line | MIT OR Apache-2.0 |
| strsim 0.11.1 | Command line suggestions | MIT |
| anyhow, thiserror, thiserror-impl | Error handling | MIT OR Apache-2.0 |
| memmap2, libc | File mapping | MIT OR Apache-2.0 |
| tar 0.4.46, filetime, xattr, bitflags, cfg-if | Folder packing and restoring | MIT OR Apache-2.0 |
| rustix, linux-raw-sys | System calls (used by the folder code) | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| proc-macro2, quote, syn, unicode-ident (build-time macros) | Code generation | MIT OR Apache-2.0 (unicode-ident also Unicode-3.0) |

Full license texts: Apache-2.0 https://www.apache.org/licenses/LICENSE-2.0 · MIT
https://opensource.org/license/mit · BSD-3-Clause https://opensource.org/license/bsd-3-clause ·
BSL-1.0 https://www.boost.org/LICENSE_1_0.txt · GCC Runtime Library Exception
https://www.gnu.org/licenses/gcc-exception-3.1.html · LLVM exception https://llvm.org/LICENSE.txt ·
mingw-w64 https://www.mingw-w64.org · CUDA EULA
https://docs.nvidia.com/cuda/eula/
