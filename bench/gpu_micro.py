#!/usr/bin/env python3
"""Phase 0 GPU microbenchmarks for the RTX 5060 Ti.

Measures the numbers PLAN.md had to guess:
  1. Dense GEMM throughput: INT8 (int32 accumulate), FP16, BF16, FP8 - at decode-like shapes.
  2. DRAM bandwidth and L2-resident read bandwidth.
  3. Kernel launch latency, plain vs CUDA Graph replay.
  4. A simulated batched decode step for INT8 recurrent-style byte models of several sizes,
     captured in a CUDA Graph -> bytes/s upper-ish bound with stock PyTorch ops (unfused).

Writes JSON to bench/results/gpu_micro.json and prints a summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "gpu_micro.json"
dev = torch.device("cuda")


def timeit(fn, iters: int = 50, warmup: int = 10) -> float:
    """Median-ish: total time / iters, after warmup, with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters


def gemm_bench() -> list[dict]:
    res = []
    shapes = [(m, k, n) for m in (256, 1024, 4096, 16384) for (k, n) in ((256, 1024), (512, 2048), (1024, 4096), (2048, 8192))]
    shapes += [(8192, 8192, 8192)]
    for m, k, n in shapes:
        row = dict(m=m, k=k, n=n)
        a8 = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
        b8 = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=dev)
        bt8 = b8.t().contiguous().t()  # column-major B, what cuBLASLt int8 likes
        try:
            t = timeit(lambda: torch._int_mm(a8, bt8))
            row["int8_tops"] = 2 * m * k * n / t / 1e12
        except Exception as ex:  # noqa: BLE001
            row["int8_err"] = str(ex)[:80]
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            a, b = a8.to(dt), b8.to(dt)
            t = timeit(lambda: a @ b)
            row[f"{name}_tflops"] = 2 * m * k * n / t / 1e12
        try:
            af = a8.to(torch.float16).to(torch.float8_e4m3fn)
            bf = b8.to(torch.float16).to(torch.float8_e4m3fn).t().contiguous().t()
            one = torch.tensor(1.0, device=dev)
            t = timeit(lambda: torch._scaled_mm(af, bf, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            row["fp8_tflops"] = 2 * m * k * n / t / 1e12
        except Exception as ex:  # noqa: BLE001
            row["fp8_err"] = str(ex)[:80]
        res.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return res


def bandwidth_bench() -> dict:
    out = {}
    big = torch.empty(2 * 1024**3 // 4, dtype=torch.float32, device=dev)
    dst = torch.empty_like(big)
    t = timeit(lambda: dst.copy_(big), iters=20)
    out["dram_copy_GBps"] = 2 * big.numel() * 4 / t / 1e9  # read + write
    del big, dst
    for mb in (4, 8, 16, 24, 48, 256):
        x = torch.ones(mb * 1024**2 // 2, dtype=torch.float16, device=dev)
        t = timeit(lambda: x.sum(), iters=200)
        out[f"read_{mb}MB_GBps"] = x.numel() * 2 / t / 1e9
    return out


def launch_bench() -> dict:
    x = torch.zeros(1024, device=dev)
    n = 200

    def chain():
        for _ in range(n):
            x.add_(1)

    t_plain = timeit(chain, iters=20) / n
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()
    t_graph = timeit(g.replay, iters=50) / n
    return dict(launch_us_plain=t_plain * 1e6, launch_us_graph=t_graph * 1e6)


class Int8RecurrentSim:
    """Shape-faithful stand-in for one decode step of an INT8 byte model.

    Per layer: x[B,d] -> int8 GEMM to [B,4d] (gates/mix) -> requant -> elementwise recurrence
    on a state [B,d] -> int8 GEMM [B,d]x[d,d] (output proj) -> residual. Then head [B,d]x[d,256],
    a fixed-point-ish softmax/CDF stand-in, and next-byte selection (argmax as a proxy for the
    rANS decode, which is cheap). Not the real model - just its cost profile.
    """

    def __init__(self, B: int, d: int, L: int):
        self.B, self.d, self.L = B, d, L
        r = lambda *s: torch.randint(-64, 64, s, dtype=torch.int8, device=dev)  # noqa: E731
        self.emb = r(256, d)
        self.w_in = [r(4 * d, d).t() for _ in range(L)]   # column-major [d,4d]
        self.w_out = [r(d, d).t() for _ in range(L)]
        self.head = r(256, d).t()
        self.state = [torch.zeros(B, d, dtype=torch.int32, device=dev) for _ in range(L)]
        self.tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.params = 256 * d + L * 5 * d * d + 256 * d

    def step(self):
        x = self.emb[self.tok]
        for l in range(self.L):
            h = torch._int_mm(x, self.w_in[l])                     # [B,4d] int32
            h8 = (h >> 8).clamp_(-128, 127).to(torch.int8)
            g, v = h8[:, : 2 * self.d], h8[:, 2 * self.d:]
            decay = g[:, : self.d].to(torch.int32) + 128
            self.state[l].mul_(decay).add_(v[:, : self.d].to(torch.int32) << 8).bitwise_right_shift_(8)
            y = (self.state[l] >> 4).clamp(-128, 127).to(torch.int8)
            o = torch._int_mm(y, self.w_out[l])
            x = ((o >> 8) + x.to(torch.int32)).clamp_(-128, 127).to(torch.int8)
        logits = torch._int_mm(x, self.head)                       # [B,256]
        cdf = torch.cumsum((logits >> 6).clamp_(0, 4095) + 1, dim=1)  # integer CDF stand-in
        self.tok = torch.argmax(cdf.diff(dim=1, prepend=cdf[:, :1] * 0), dim=1)


def decode_sim_bench() -> list[dict]:
    res = []
    configs = [(256, 2), (384, 4), (512, 4), (512, 8), (768, 8), (1024, 8), (1024, 12), (1536, 12)]
    for d, L in configs:
        for B in (512, 2048, 8192):
            m = Int8RecurrentSim(B, d, L)
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        m.step()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    m.step()
                t = timeit(g.replay, iters=100)
            except Exception as ex:  # noqa: BLE001
                res.append(dict(d=d, L=L, B=B, err=str(ex)[:100]))
                print(res[-1], flush=True)
                continue
            row = dict(d=d, L=L, B=B, params_M=round(m.params / 1e6, 2), step_us=round(t * 1e6, 1),
                       MBps=round(B / t / 1e6, 2),
                       eff_int8_tops=round(2 * m.params * B / t / 1e12, 1))
            res.append(row)
            print(row, flush=True)
            del m, g
            torch.cuda.empty_cache()
    return res


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    info = dict(name=props.name, sm=f"{props.major}.{props.minor}", sms=props.multi_processor_count,
                mem_GB=round(props.total_memory / 1e9, 1), l2_MB=getattr(props, "L2_cache_size", 0) / 2**20,
                torch=torch.__version__, cuda=torch.version.cuda, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(info)
    out = dict(info=info)
    out["launch"] = launch_bench(); print(out["launch"])
    out["bandwidth"] = bandwidth_bench(); print(out["bandwidth"])
    out["gemm"] = gemm_bench()
    out["decode_sim"] = decode_sim_bench()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
