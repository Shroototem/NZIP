#!/usr/bin/env bash
# Phase 3b: Gated DeltaNet variants with smaller recurrent state (same 256 MB budget).
set -u
cd "$(dirname "$0")/.."
run() { python -I train/train.py --arch gdn --kw "$2" --bytes 256M --out "models/variants/$1.pt" > "models/variants/$1.log" 2>&1; echo "$1: $(tail -1 models/variants/$1.log)"; }
mkdir -p models/variants
run h8x32_l3 '{"d":256,"layers":3,"heads":8,"head_dim":32}'
run h4x32_l3 '{"d":256,"layers":3,"heads":4,"head_dim":32}'
run h8x32_l4 '{"d":256,"layers":4,"heads":8,"head_dim":32}'
run h10x32_d320 '{"d":320,"layers":3,"heads":10,"head_dim":32}'
