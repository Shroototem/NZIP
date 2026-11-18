"""Dump a trained model's next-byte distributions for a file, for mixing experiments.

Output: <out> = raw little-endian u16, 256 values per input byte, each row summing to
exactly 65536 with every entry >= 1 (same rule as ncore's quantiser: floor of the scaled
probability plus one, remainder to the most likely byte). Row i predicts byte i.
The model runs in lanes of --lane bytes with state carried inside a lane.

Usage: python -I train/dump_probs.py <checkpoint> <file> <out> [--max 4M] [--lane 1M]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402
from train import parse_size  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("file")
    ap.add_argument("out")
    ap.add_argument("--max", default="4M")
    ap.add_argument("--lane", default="1M")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cuda")
    m = models.build(ck["arch"], **ck["kw"]).cuda().eval()
    m.load_state_dict(ck["state"])
    data = Path(a.file).read_bytes()[: parse_size(a.max)]
    lane = parse_size(a.lane)
    out = open(a.out, "wb")
    for s0 in range(0, len(data), lane):
        x = torch.frombuffer(bytearray(data[s0:s0 + lane]), dtype=torch.uint8).cuda().long()[None]
        inp = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        state = None
        for c in range(0, x.shape[1], 8192):
            with torch.autocast("cuda", torch.bfloat16):
                logits, state = m(inp[:, c:c + 8192], state)
            p = torch.softmax(logits[0].float(), -1)
            f = (p * (65536 - 256)).floor().to(torch.int64) + 1
            rem = 65536 - f.sum(-1)
            f[torch.arange(f.shape[0]), p.argmax(-1)] += rem
            out.write(f.to(torch.int32).cpu().numpy().astype("<u2").tobytes())
    out.close()
    print(f"{a.file}: {len(data)} bytes -> {a.out}")
    _ = np


if __name__ == "__main__":
    main()
