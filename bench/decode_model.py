"""Estimate step-by-step decode speed of each bake-off architecture on this GPU.

A decoder advances every lane by one byte per step. Per step it must (a) run the model's
matrix multiplies for B lanes (INT8 in, INT32 accumulate: torch._int_mm on tensor cores)
and (b) read and write each lane's recurrent state (or attention window). This script
times both, captured in CUDA graphs so launch overhead is excluded (a fused persistent
kernel removes it too), and reports:

  MB/s     = B / step_time            (throughput, many lanes)
  lane KB/s = 1 / step_time / 1000    (one lane's speed: sets small-file latency)

It is an estimate of what fused integer kernels can reach, not a kernel: elementwise
work besides the state update is ignored, and nothing is fused across layers.

Usage: python -I bench/decode_model.py [--json out.json]   (GPU should be idle)
"""
import argparse
import json

import torch

# Per architecture: GEMMs per step as (K, N) for a [B x K] x [K x N] product, and recurrent
# state elements per lane (stored as int16) that are read + written each step.
# Shapes mirror train/models.py defaults (~2M decode MACs/byte each).
d_l, d_g, d_s = 352, 224, 192
ARCH = {
    "lstm": {"gemms": [(2 * d_l, 4 * d_l)] * 2 + [(d_l, 256)], "state": 2 * 2 * d_l},
    "mingru": {"gemms": [(d_g, 2 * d_g), (d_g, d_g), (d_g, 3 * d_g), (3 * d_g, d_g)] * 5 + [(d_g, 256)], "state": 5 * d_g},
    # GDN / RWKV-7: q,k,v,gate,out projections + MLP per layer; matrix state 4 heads x 64 x 64.
    "gdn": {"gemms": [(256, 256)] * 5 * 3 + [(256, 512), (512, 256)] * 3 + [(256, 256)], "state": 3 * 4 * 64 * 64},
    "rwkv7": {"gemms": [(256, 256)] * 5 * 3 + [(256, 512), (512, 256)] * 3 + [(256, 256)], "state": 3 * 4 * 64 * 64},
    # Sliding window: K and V for 512 positions x 192 dims x 3 layers (int8 here).
    "swa": {"gemms": [(d_s, 3 * d_s), (d_s, d_s), (d_s, 4 * d_s), (4 * d_s, d_s)] * 3 + [(d_s, 256)],
            "state": 3 * 2 * 512 * d_s // 2},
}


def graph_time(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 1000.0


def pad8(x):
    return (x + 7) // 8 * 8


def step_time(spec, B):
    B = max(B, 32)
    xs = {}
    ws = []
    for K, N in spec["gemms"]:
        K, N = pad8(K), pad8(N)
        if K not in xs:
            xs[K] = torch.randint(-64, 64, (B, K), device="cuda", dtype=torch.int8)
        ws.append((K, torch.randint(-64, 64, (K, N), device="cuda", dtype=torch.int8)))

    def gemms():
        for K, w in ws:
            torch._int_mm(xs[K], w)

    t_gemm = graph_time(gemms)
    st = torch.zeros(B, spec["state"], device="cuda", dtype=torch.int16)
    one = torch.ones(1, device="cuda", dtype=torch.int16)

    def state():
        st.add_(one)  # one read + one write of every state element

    t_state = graph_time(state)
    del xs, ws, st
    return t_gemm, t_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("| arch | MACs/byte | state KB/lane | best MB/s (lanes) | MB/s @256 lanes | MB/s @4096 lanes | lane KB/s @4096 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, spec in ARCH.items():
        macs = sum(K * N for K, N in spec["gemms"])
        rows = {}
        for B in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            try:
                tg, ts = step_time(spec, B)
            except torch.OutOfMemoryError:
                break
            rows[B] = {"gemm_s": tg, "state_s": ts, "mbps": B / (tg + ts) / 1e6, "lane_kbps": 1 / (tg + ts) / 1e3}
            torch.cuda.empty_cache()
        best = max(rows, key=lambda b: rows[b]["mbps"])
        res[name] = {"macs": macs, "state_elems": spec["state"], "rows": rows}
        print(f"| {name} | {macs / 1e6:.2f}M | {spec['state'] * 2 / 1024:.1f} | {rows[best]['mbps']:.1f} ({best}) | "
              f"{rows[256]['mbps']:.1f} | {rows.get(4096, rows[best])['mbps']:.1f} | {rows.get(4096, rows[best])['lane_kbps']:.1f} |",
              flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
