#!/usr/bin/env bash
# Cross-build nzip.exe for Windows x86-64 from Linux, into target/dist/.
# Pure Rust, no DLLs: the GPU path loads nvcuda.dll at run time if present.
# Needs the llvm-mingw toolchain and the official Rust Windows std in .sandbox/win.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
L=$(ls -d "$ROOT"/.sandbox/win/llvm-mingw-*-x86_64 | head -1)
RUST="$ROOT/.sandbox/win/rust"
T=x86_64-pc-windows-gnullvm
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="nzip-$VER-windows-x86_64"
export PATH="$L/bin:$PATH"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER=x86_64-w64-mingw32-clang
export RUSTC="$RUST/bin/rustc"
export CARGO_TARGET_DIR="$ROOT/target/windows"
"$RUST/bin/cargo" build --release --target "$T"
OUT="target/dist/$NAME"
rm -rf "${OUT:?}" && mkdir -p "$OUT"
cp "$CARGO_TARGET_DIR/$T/release/nzip.exe" README.md LICENSE "$OUT/" 2>/dev/null || cp "$CARGO_TARGET_DIR/$T/release/nzip.exe" "$OUT/"
(cd target/dist && rm -f "$NAME.zip" && python3 -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', '$NAME')" && sha256sum "$NAME.zip" > "$NAME.zip.sha256")
echo "built target/dist/$NAME.zip"
echo "imports: $(x86_64-w64-mingw32-objdump -p "$OUT/nzip.exe" | sed -n 's/^\s*DLL Name: //p' | sort -u | tr '\n' ' ')"
