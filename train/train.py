"""Train and evaluate one bake-off candidate (float, bf16 autocast).

Usage (after `source .sandbox/env.sh`):
  python -I train/train.py --arch gdn --bytes 400M --out models/bakeoff/gdn.pt
  python -I train/train.py --arch gdn --eval-only --out models/bakeoff/gdn.pt

Training: random T-byte windows drawn from the corpus categories with fixed mixture
weights, AdamW, warm-up + cosine decay. Evaluation: bits per byte on each category's
held-out split and on the benchmark sets (never trained on), coded the way the real coder
works: independent lanes of `--lane` bytes, processed in order with state carried.
Results are appended to models/bakeoff/results.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
BENCH = ROOT / "data" / "bench"
MIX = {"web": 0.18, "books": 0.14, "wiki": 0.14, "code": 0.16, "docs": 0.10, "bin": 0.16, "data": 0.12}


def parse_size(s):
    m = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(s[:-1]) * m[s[-1]]) if s[-1] in m else int(s)


class Sampler:
    def __init__(self, T, seed=0):
        self.T = T
        self.data = {c: torch.from_numpy(np.fromfile(TRAIN / f"{c}.train.bin", dtype=np.uint8)) for c in MIX}
        self.cats = list(MIX)
        self.p = np.array([MIX[c] for c in self.cats])
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        cats = self.rng.choice(len(self.cats), size=B, p=self.p)
        rows = []
        for ci in cats:
            d = self.data[self.cats[ci]]
            o = int(self.rng.integers(0, len(d) - self.T - 1))
            rows.append(d[o:o + self.T + 1])
        x = torch.stack(rows).pin_memory().cuda(non_blocking=True).long()
        return x[:, :-1], x[:, 1:]


def eval_sets():
    sets = {f"val/{c}": (TRAIN / f"{c}.val.bin").read_bytes() for c in MIX}
    sets["enwik8_10M"] = (BENCH / "derived" / "enwik8_10M").read_bytes()
    for name in ("calgary", "cantrbry", "silesia"):
        sets[name] = b"".join(p.read_bytes() for p in sorted((BENCH / name).iterdir()) if p.is_file())
    return sets


@torch.no_grad()
def bpb(model, data: bytes, lane: int, chunk: int, group: int):
    """Bits per byte coding `data` as independent lanes (state carried within a lane)."""
    model.eval()
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    n_full = len(buf) // lane
    total_bits = 0.0
    pieces = []
    if n_full:
        pieces.append(buf[: n_full * lane].view(n_full, lane))
    if len(buf) % lane:
        pieces.append(buf[n_full * lane:].view(1, -1))
    for lanes in pieces:
        L = lanes.shape[1]
        for g in range(0, lanes.shape[0], group):
            x = lanes[g:g + group].cuda().long()
            # Input is the previous byte; the first byte of a lane is predicted from byte 0.
            inp = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
            state = None
            for s in range(0, L, chunk):
                with torch.autocast("cuda", torch.bfloat16):
                    logits, state = model(inp[:, s:s + chunk], state)
                total_bits += F.cross_entropy(logits.float().reshape(-1, 256), x[:, s:s + chunk].reshape(-1),
                                              reduction="sum").item() / math.log(2)
    model.train()
    return total_bits / len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(models.ARCHS))
    ap.add_argument("--kw", default="{}", help="JSON model kwargs (size)")
    ap.add_argument("--bytes", default="400M", help="training bytes")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lane", type=int, default=256 * 1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    kw = json.loads(a.kw)
    model = models.build(a.arch, **kw).cuda()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(models.describe(model), flush=True)
    train_secs = 0.0
    steps = 0
    if not a.eval_only:
        total = parse_size(a.bytes)
        steps = total // (a.B * a.T)
        warm = max(1, steps // 50)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
        sampler = Sampler(a.T, a.seed)
        t0 = time.time()
        ema = None
        for step in range(steps):
            x, y = sampler.batch(a.B)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            l = loss.item() / math.log(2)
            ema = l if ema is None else 0.98 * ema + 0.02 * l
            if step % 200 == 0 or step == steps - 1:
                el = time.time() - t0
                print(f"step {step:6d}/{steps}  train {ema:.3f} bpb  {(step + 1) * a.B * a.T / el / 1e6:.2f} MB/s  {el / 60:.1f} min",
                      flush=True)
        train_secs = time.time() - t0
        torch.save({"arch": a.arch, "kw": kw, "state": model.state_dict()}, out)
    else:
        ck = torch.load(out, map_location="cuda")
        model.load_state_dict(ck["state"])
    res = {"arch": a.arch, "kw": kw, "params": sum(p.numel() for p in model.parameters()),
           "decode_macs": model.decode_macs(), "train_bytes": steps * a.B * a.T, "train_secs": round(train_secs, 1),
           "lane": a.lane}
    t0 = time.time()
    for name, data in eval_sets().items():
        res[name] = round(bpb(model, data, a.lane, chunk=8192, group=32), 4)
        print(f"  {name:14s} {res[name]:.4f} bpb", flush=True)
    res["eval_secs"] = round(time.time() - t0, 1)
    vals = [res[f"val/{c}"] for c in MIX]
    res["val_mean"] = round(sum(vals) / len(vals), 4)
    print(f"  val mean       {res['val_mean']:.4f} bpb   (eval {res['eval_secs']} s)", flush=True)
    with open(out.parent / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
