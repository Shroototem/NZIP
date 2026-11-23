#!/bin/bash
# Quick LZ engine check (dev): xul.dll via the x86 split, plus binary files from Silesia
# and omni.ja's binary part. Usage: bench/lzcheck.sh [scratch dir holding m1.bin]
cd "$(dirname "$0")/.."
S=${1:-/tmp}
N=./target/release/nzip
a=$(NZ_ENCONLY=1 NZ_SUB=8 $N lzsplit "data/bench/FireFox/Mozilla Firefox/xul.dll" | awk '{print $3, $NF}')
t=0
for f in $S/m1.bin data/bench/silesia/mozilla data/bench/silesia/sao data/bench/silesia/x-ray data/bench/silesia/ooffice; do
  [ -f "$f" ] || continue
  r=$(NZ_SUB=8 $N lzsize "$f" | awk '{print $3}'); t=$((t+r)); printf "%s=%s " "$(basename $f)" "$r"
done
echo "| xul $a | others $t"
