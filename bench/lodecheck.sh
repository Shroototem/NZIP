#!/bin/bash
# Per-type timing on samples of the 99 GB lodestone build tree (data/bench/lodestone/<type>,
# ~1 GB each, copied with the matching <type>.list). Usage: bench/lodecheck.sh [type...]
cd "$(dirname "$0")/.."
N=./target/release/nzip
S=${TMPDIR:-/tmp}/nzip-lodecheck
mkdir -p "$S"
types=${*:-$(ls -d data/bench/lodestone/*/ | xargs -n1 basename)}
printf "%-8s %14s %14s %7s %8s %8s %9s\n" type bytes packed ratio comp_s dec_s comp_MB/s
for t in $types; do
  d=data/bench/lodestone/$t
  [ -d "$d" ] || continue
  in=$(du -sb "$d" | cut -f1)
  t0=$(date +%s.%N)
  $N -c -f "$d" -o "$S/$t.nz" 2>/dev/null || { echo "$t: compress failed"; continue; }
  t1=$(date +%s.%N)
  $N -t "$S/$t.nz" >/dev/null 2>&1 || echo "$t: TEST FAILED"
  t2=$(date +%s.%N)
  out=$(stat -c %s "$S/$t.nz")
  awk -v t=$t -v i=$in -v o=$out -v a=$t0 -v b=$t1 -v c=$t2 'BEGIN {printf "%-8s %14d %14d %6.2f%% %8.2f %8.2f %9.1f\n", t, i, o, 100*o/i, b-a, c-b, i/1e6/(b-a)}'
  rm -f "$S/$t.nz"
done
