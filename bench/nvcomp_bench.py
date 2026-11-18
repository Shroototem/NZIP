"""Benchmark NVIDIA nvCOMP (GPU compression library) on the benchmark sets.

For every codec variant and data set it reports:
  * compressed size (nvCOMP native bitstream, headers included);
  * kernel speed: input already in GPU memory, output left there (how NVIDIA quotes it);
  * end-to-end speed: from host memory and back (H2D copy + codec + D2H copy), which is
    what a file compressor actually pays and is comparable to the CPU tools.
Speeds are best of 3 runs after a warm-up. Round trip is verified on the GPU.

Silesia is measured as one buffer (its 12 files concatenated), which helps nvCOMP
slightly versus per-file runs.

Usage (after `source .sandbox/env.sh`): python -I bench/nvcomp_bench.py
Appends rows to bench/results/baselines.jsonl with codec names "nvcomp-...".
"""
import json
import time
from pathlib import Path

import torch
from nvidia import nvcomp

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data" / "bench"
SETS = {
    "enwik8": [B / "enwik8" / "enwik8"],
    "silesia": sorted(p for p in (B / "silesia").iterdir() if p.is_file()),
    "enwik9": [B / "enwik9" / "enwik9"],
}
K = 1 << 10
VARIANTS = [
    ("nvcomp-lz4", dict(algorithm="LZ4")),
    ("nvcomp-snappy", dict(algorithm="Snappy")),
    ("nvcomp-zstd", dict(algorithm="Zstd")),
    ("nvcomp-zstd-1M", dict(algorithm="Zstd", uncomp_chunk_size=1024 * K)),
    ("nvcomp-gdeflate-1", dict(algorithm="GDeflate", algorithm_type=1)),
    ("nvcomp-gdeflate-5", dict(algorithm="GDeflate", algorithm_type=5)),
    ("nvcomp-deflate-1", dict(algorithm="Deflate", algorithm_type=1)),
    ("nvcomp-deflate-5", dict(algorithm="Deflate", algorithm_type=5)),
    ("nvcomp-ans", dict(algorithm="ANS")),
    ("nvcomp-bitcomp", dict(algorithm="Bitcomp")),
    ("nvcomp-cascaded", dict(algorithm="Cascaded")),
]


def sync():
    torch.cuda.synchronize()


def best(f, n=3):
    f()
    sync()  # warm-up
    t = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        r = f()
        sync()
        t.append(time.perf_counter() - t0)
    return min(t), r


def run(name, kw, data: bytes):
    host = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    dev = host.cuda()
    codec = nvcomp.Codec(**kw)
    arr = nvcomp.as_array(dev)
    t_enc, enc = best(lambda: codec.encode(arr))
    t_dec, dec = best(lambda: codec.decode(enc))
    out = torch.from_dlpack(dec).view(torch.uint8).flatten()
    ok = out.numel() == dev.numel() and torch.equal(out, dev)
    size = enc.buffer_size

    def e2e_enc():
        e = codec.encode(nvcomp.as_array(host.cuda()))
        return torch.from_dlpack(e.cuda()).view(torch.uint8).cpu()

    t_e2e_enc, comp_host = best(e2e_enc)

    def e2e_dec():
        d = codec.decode(enc) if comp_host is None else codec.decode(nvcomp.as_array(comp_host.cuda()))
        return torch.from_dlpack(d).view(torch.uint8).flatten().cpu()

    try:
        t_e2e_dec, back = best(e2e_dec)
        ok = ok and torch.equal(back, host)
    except Exception:  # some codecs need the original Array metadata to decode
        t_e2e_dec = float("nan")
    del dev, arr, enc, dec, out
    torch.cuda.empty_cache()
    return size, t_enc, t_dec, t_e2e_enc, t_e2e_dec, ok


def main():
    gpu = torch.cuda.get_device_name()
    out = ROOT / "bench" / "results" / "baselines.jsonl"
    for set_name, files in SETS.items():
        data = b"".join(p.read_bytes() for p in files)
        n = len(data)
        print(f"\n### {set_name} ({n:,} bytes) on {gpu}")
        print("| codec | compressed | bpb | kernel comp MB/s | kernel decomp MB/s | e2e comp MB/s | e2e decomp MB/s | ok |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
        for name, kw in VARIANTS:
            try:
                size, te, td, tee, ted, ok = run(name, kw, data)
            except Exception as e:  # unsupported option/data type: report and move on
                print(f"| {name} | error: {str(e)[:60]} |")
                continue
            mb = n / 1e6
            print(f"| {name} | {size:,} | {8 * size / n:.3f} | {mb / te:,.0f} | {mb / td:,.0f} | {mb / tee:,.0f} | "
                  f"{mb / ted:,.0f} | {'ok' if ok else 'FAIL'} |", flush=True)
            with open(out, "a") as f:
                f.write(json.dumps({"codec": name, "set": set_name, "files": len(files), "orig": n, "comp": size,
                                    "comp_s": tee, "decomp_s": ted, "kernel_comp_s": te, "kernel_decomp_s": td,
                                    "ok": bool(ok), "note": "nvCOMP end-to-end incl. PCIe copies"}) + "\n")


if __name__ == "__main__":
    main()
