#!/usr/bin/env bash
# Cross-build nzip for Windows x86-64 (CPU only) into dist/, from Linux.
#
#   bash scripts/build-windows.sh            (after `source .sandbox/env.sh`)
#
# Uses the sandboxed llvm-mingw toolchain and Windows Rust std (.sandbox/win): see
# README notes. A GPU build for Windows must be made on Windows (Visual Studio + CUDA):
#   cargo build --release -p nzip-cli        (CUDA found through CUDA_PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
L=$(ls -d "$ROOT"/.sandbox/win/llvm-mingw-*-x86_64 | head -1)
RUST="$ROOT/.sandbox/win/rust"   # official Rust 1.100 (the distro rustc rejects official std builds)
T=x86_64-pc-windows-gnullvm
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="nzip-$VER-windows-x86_64"
export PATH="$L/bin:$PATH"
export CC_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang
export CXX_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang++
export AR_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-llvm-ar
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER=x86_64-w64-mingw32-clang
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_RUSTFLAGS="-C target-cpu=x86-64-v2"
export RUSTC="$RUST/bin/rustc"
export CARGO_TARGET_DIR="$ROOT/target/windows"
nice -n 10 "$RUST/bin/cargo" build --release -p nzip-cli --target "$T" -j "${JOBS:-8}"
OUT="dist/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"
cp "$CARGO_TARGET_DIR/$T/release/nzip.exe" "$OUT/"
llvm-strip "$OUT/nzip.exe"
# Runtime DLLs the exe imports from the toolchain (OpenMP, and libc++/libunwind if dynamic).
for dll in $(x86_64-w64-mingw32-objdump -p "$OUT/nzip.exe" | sed -n 's/^\s*DLL Name: //p'); do
  [ -f "$L/x86_64-w64-mingw32/bin/$dll" ] && cp "$L/x86_64-w64-mingw32/bin/$dll" "$OUT/"
done
cp dist-files/README-windows.md "$OUT/README.md"
cp dist-files/THIRD_PARTY_NOTICES.md "$OUT/"
cp third_party/libbsc/LICENSE "$OUT/LICENSE.libbsc"
(cd dist && rm -f "$NAME.zip" && python3 -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', '$NAME')" && sha256sum "$NAME.zip" > "$NAME.zip.sha256")
echo "built dist/$NAME.zip"
echo "imports: $(x86_64-w64-mingw32-objdump -p "$OUT/nzip.exe" | sed -n 's/^\s*DLL Name: //p' | sort -u | tr '\n' ' ')"
ls "$OUT"
