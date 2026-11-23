#!/usr/bin/env bash
# Build the Linux x86-64 release into target/dist/. One binary for every x86-64 CPU (AVX2 path
# picked at run time) with optional NVIDIA GPU use (libcuda loaded at run time).
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="nzip-$VER-linux-x86_64"
cargo build --release
OUT="target/dist/$NAME"
rm -rf "${OUT:?}" && mkdir -p "$OUT"
cp target/release/nzip "$OUT/"
[ -f README.md ] && cp README.md "$OUT/"
tar -C target/dist -czf "target/dist/$NAME.tar.gz" "$NAME"
(cd target/dist && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "built target/dist/$NAME.tar.gz"
