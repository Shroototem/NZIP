#!/usr/bin/env bash
# Compare nzip with common compressors on one file. Uses whichever of zstd, xz, brotli,
# bzip2, gzip and lz4 are installed. Prints a Markdown table (paste it in your report).
#   ./compare.sh <file>
set -u
[ $# -eq 1 ] && [ -f "$1" ] || { echo "usage: $0 <file>"; exit 1; }
F=$(realpath -- "$1")            # resolve before changing directory
[ -n "$F" ] && [ -f "$F" ] || { echo "cannot resolve $1"; exit 1; }
NZ=$(cd "$(dirname "$0")" && pwd)/nzip
T=$(mktemp -d "${TMPDIR:-/tmp}/nzip-compare.XXXXXX")
trap 'rm -rf "$T"' EXIT
SIZE=$(stat -c %s "$F")
cat "$F" > /dev/null  # warm the page cache so every tool reads from memory

now() { date +%s.%N; }
row() { # name compress-cmd decompress-cmd compressed-file
  local s e c d
  s=$(now); eval "$2" >/dev/null 2>&1 || { echo "| $1 | failed | | | |"; return; }; e=$(now); c=$(echo "$e - $s" | awk '{print $1-$3}')
  local out; out=$(stat -c %s "$4")
  s=$(now); eval "$3" >/dev/null 2>&1; e=$(now); d=$(echo "$e - $s" | awk '{print $1-$3}')
  cmp -s "$F" "$T/back" && ok=ok || ok=MISMATCH
  awk -v n="$1" -v sz="$SIZE" -v o="$out" -v c="$c" -v d="$d" -v ok="$ok" \
    'BEGIN{printf "| %s | %\047d | %.3f | %.1f | %.1f | %s |\n", n, o, 8*o/sz, sz/c/1e6, sz/d/1e6, ok}'
}

echo "File: $(basename "$F"), $SIZE bytes. CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
command -v nvidia-smi >/dev/null && echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo
echo "| tool | compressed bytes | bits/byte | compress MB/s | decompress MB/s | round trip |"
echo "|---|---:|---:|---:|---:|---|"
row "nzip max" "$NZ -c -q -f '$F' $T/a.nz" "$NZ -d -q -f $T/a.nz $T/back" "$T/a.nz"
row "nzip fast" "$NZ -c -q -f -m fast '$F' $T/b.nz" "$NZ -d -q -f $T/b.nz $T/back" "$T/b.nz"
command -v zstd >/dev/null && {
  row "zstd -3" "zstd -q -f -3 -T0 '$F' -o $T/z3" "zstd -q -f -d $T/z3 -o $T/back" "$T/z3"
  row "zstd -19" "zstd -q -f -19 -T0 '$F' -o $T/z19" "zstd -q -f -d $T/z19 -o $T/back" "$T/z19"; }
command -v xz >/dev/null && row "xz -9 (multi-thread)" "xz -9 -T0 -k -c '$F' > $T/x" "xz -d -c $T/x > $T/back" "$T/x"
command -v brotli >/dev/null && row "brotli -9" "brotli -q 9 -f -c '$F' > $T/br" "brotli -d -c $T/br > $T/back" "$T/br"
command -v bzip2 >/dev/null && row "bzip2 -9" "bzip2 -9 -k -c '$F' > $T/bz" "bzip2 -d -c $T/bz > $T/back" "$T/bz"
command -v gzip >/dev/null && row "gzip -6" "gzip -6 -c '$F' > $T/gz" "gzip -d -c $T/gz > $T/back" "$T/gz"
command -v lz4 >/dev/null && row "lz4 -1" "lz4 -q -f -1 '$F' $T/l4" "lz4 -q -f -d $T/l4 $T/back" "$T/l4"
true
