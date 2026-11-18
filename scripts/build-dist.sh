#!/usr/bin/env bash
# Build a distributable nzip package for Linux x86-64 into dist/.
#
#   bash scripts/build-dist.sh            (after `source .sandbox/env.sh`)
#
# Portable settings: libstdc++ linked statically, extra GPU
# architectures + PTX, CUDA runtime static (driver loaded at run time, CPU fallback
# without a GPU), libgomp shipped in lib/ and found through an $ORIGIN rpath.
# Requires on the target: glibc >= 2.38 and an AVX2 CPU (what the shipped libgomp and the
# static libstdc++ from this CachyOS machine need).
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="nzip-$VER-linux-x86_64"
export NZIP_PORTABLE=1
export CARGO_TARGET_DIR="$PWD/target/dist"
export RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN/lib -C link-arg=-Wl,--enable-new-dtags'
nice -n 10 cargo build --release -p nzip-cli -j "${JOBS:-8}"
OUT="dist/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT/lib"
cp "$CARGO_TARGET_DIR/release/nzip" "$OUT/"
strip --strip-debug "$OUT/nzip"
cp "$(g++ -print-file-name=libgomp.so.1)" "$OUT/lib/" 2>/dev/null || cp /usr/lib/libgomp.so.1 "$OUT/lib/"
cp dist-files/README.md dist-files/THIRD_PARTY_NOTICES.md dist-files/selftest.sh dist-files/compare.sh "$OUT/"
cp LICENSE "$OUT/LICENSE" 2>/dev/null || true
cp third_party/libbsc/LICENSE "$OUT/LICENSE.libbsc"
chmod +x "$OUT/nzip" "$OUT/selftest.sh" "$OUT/compare.sh"
(cd dist && tar -czf "$NAME.tar.gz" "$NAME" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "built dist/$NAME.tar.gz"
