#!/usr/bin/env bash
# Phase 3a bake-off: train every candidate on the same number of bytes, then evaluate.
# Usage: bash train/bakeoff.sh [bytes]   (after `source .sandbox/env.sh`)
set -u
BYTES=${1:-256M}
cd "$(dirname "$0")/.."
for a in gdn lstm rwkv7 mingru swa; do
  python -I train/train.py --arch "$a" --bytes "$BYTES" --out "models/bakeoff/$a.pt" > "models/bakeoff/$a.log" 2>&1
  echo "$a done: $(tail -1 models/bakeoff/$a.log)"
done
