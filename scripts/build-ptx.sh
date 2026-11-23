#!/bin/sh
# Regenerates the PTX embedded in the binary. Only needed after editing gpu/sa.cu;
# normal builds use the committed gpu/sa.ptx and need no CUDA toolkit.
set -e
cd "$(dirname "$0")/.."
nvcc -ptx -O3 -arch=compute_75 -o gpu/sa.ptx gpu/sa.cu
echo "wrote gpu/sa.ptx"
