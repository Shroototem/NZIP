#!/usr/bin/env python3
"""Phase 0 baseline benchmark harness.

Runs existing compressors over the benchmark corpora, verifies every round trip
byte-for-byte, and appends one JSON line per (codec, file) to a results file.

Usage (after `source .sandbox/env.sh`):
  python bench/baselines.py --codecs zstd-3,xz-9e --sets cantrbry,enwik8
  python bench/baselines.py --list
  python bench/baselines.py --report            # print summary tables
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bench"
RESULTS = ROOT / "bench" / "results" / "baselines.jsonl"
SANDBOX = ROOT / ".sandbox"
NZIP = ROOT / "target" / "release" / "nzip"

# Corpus sets: name -> list of files (files are compressed one at a time).
def corpus_sets() -> dict[str, list[Path]]:
    def files(d: str) -> list[Path]:
        return sorted(p for p in (DATA / d).iterdir() if p.is_file())
    return {
        "cantrbry": files("cantrbry"),
        "calgary": files("calgary"),
        "silesia": files("silesia"),
        "enwik8_10M": [DATA / "derived" / "enwik8_10M"],
        "enwik8": [DATA / "enwik8" / "enwik8"],
        "enwik9": [DATA / "enwik9" / "enwik9"],
    }

# Codec table. {i} = input, {o} = output. A command whose output is "-" writes to stdout.
# threads: "mt" = uses all cores, "st" = single-threaded (safe to run several in parallel).
CODECS: dict[str, dict] = {
    "lz4-1":        dict(c="lz4 -q -f -1 {i} {o}",                    d="lz4 -q -f -d {i} {o}", threads="st"),
    "zstd-3":       dict(c="zstd -q -f -3 -T0 {i} -o {o}",            d="zstd -q -f -d {i} -o {o}", threads="mt"),
    "zstd-19":      dict(c="zstd -q -f -19 -T0 {i} -o {o}",           d="zstd -q -f -d {i} -o {o}", threads="mt"),
    "zstd-22-long": dict(c="zstd -q -f --ultra -22 --long=31 -T0 {i} -o {o}",
                         d="zstd -q -f -d --long=31 {i} -o {o}", threads="mt"),
    "bzip2-9":      dict(c="bzip2 -9 -c {i}", d="bzip2 -d -c {i}", stdout=True, threads="st"),
    "xz-9e":        dict(c="xz -9e -T0 -c {i}", d="xz -d -T0 -c {i}", stdout=True, threads="mt"),
    "xz-9e-st":     dict(c="xz -9e -T1 -c {i}", d="xz -d -c {i}", stdout=True, threads="st"),
    "brotli-11":    dict(c="brotli -q 11 -f -o {o} {i}",              d="brotli -d -f -o {o} {i}", threads="st"),
    "bsc":          dict(c="bsc e {i} {o}",                           d="bsc d {i} {o}", threads="mt"),
    "bsc-b100e2":   dict(c="bsc e {i} {o} -b100e2",                   d="bsc d {i} {o}", threads="mt"),
    "bsc-b1000e2":  dict(c="bsc e {i} {o} -b1000e2",                  d="bsc d {i} {o}", threads="mt"),
    "bsc-m6e2-gpu": dict(c="bsc e {i} {o} -b100m6e2 -G",              d="bsc d {i} {o} -G", threads="mt"),
    "nzip-fast":    dict(c=f"{NZIP} c -f -q -m fast {{i}} {{o}}",     d=f"{NZIP} d -f -q {{i}} {{o}}", threads="mt"),
    "nzip-max": dict(c=f"{NZIP} c -f -q -m max {{i}} {{o}}", d=f"{NZIP} d -f -q {{i}} {{o}}", threads="mt"),
}


def run(cmd: str, i: Path, o: Path, stdout: bool) -> float:
    args = [a.format(i=i, o=o) for a in cmd.split()]
    t0 = time.perf_counter()
    if stdout:
        with open(o, "wb") as f:
            subprocess.run(args, stdout=f, stderr=subprocess.DEVNULL, check=True)
    else:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return time.perf_counter() - t0


def bench_one(codec: str, f: Path, set_name: str, workdir: Path) -> dict:
    spec = CODECS[codec]
    workdir.mkdir(parents=True, exist_ok=True)
    comp, back = workdir / "c.bin", workdir / "d.bin"
    for p in (comp, back):
        p.unlink(missing_ok=True)
    tc = run(spec["c"], f, comp, spec.get("stdout", False))
    td = run(spec["d"], comp, back, spec.get("stdout", False))
    ok = filecmp.cmp(f, back, shallow=False)
    rec = dict(codec=codec, set=set_name, file=f.name, orig=f.stat().st_size,
               comp=comp.stat().st_size, t_comp=round(tc, 4), t_decomp=round(td, 4),
               roundtrip_ok=ok, threads=spec["threads"], ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    shutil.rmtree(workdir, ignore_errors=True)
    return rec


def report(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    latest: dict[tuple, dict] = {}
    for r in rows:  # keep latest run per (codec, set, file)
        if "error" in r:
            print(f"error: {r['codec']} {r['set']}/{r['file']}: {r['error']}")
            continue
        latest[(r["codec"], r["set"], r["file"])] = r
    agg: dict[tuple, dict] = {}
    for r in latest.values():
        a = agg.setdefault((r["set"], r["codec"]), dict(orig=0, comp=0, tc=0.0, td=0.0, n=0, ok=True))
        a["orig"] += r["orig"]; a["comp"] += r["comp"]; a["tc"] += r["t_comp"]; a["td"] += r["t_decomp"]
        a["n"] += 1; a["ok"] &= r["roundtrip_ok"]
    sets = sorted({k[0] for k in agg}, key=lambda s: list(corpus_sets()).index(s) if s in corpus_sets() else 99)
    for s in sets:
        print(f"\n### {s}\n")
        print("| codec | files | compressed bytes | ratio | bits/byte | comp MB/s | decomp MB/s | round trip |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
        items = sorted(((c, a) for (ss, c), a in agg.items() if ss == s), key=lambda x: x[1]["comp"])
        for c, a in items:
            mb = a["orig"] / 1e6
            print(f"| {c} | {a['n']} | {a['comp']:,} | {a['orig']/a['comp']:.3f} | {8*a['comp']/a['orig']:.3f} "
                  f"| {mb/a['tc']:.2f} | {mb/a['td']:.2f} | {'ok' if a['ok'] else 'FAIL'} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    ap.add_argument("--sets", default="cantrbry,calgary")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--tag", default="", help="suffix for the scratch dir so parallel runs don't collide")
    args = ap.parse_args()
    results = Path(args.results)
    if args.list:
        for k, v in CODECS.items():
            print(f"{k:14s} [{v['threads']}]  {v['c']}")
        print("sets:", ", ".join(f"{k}({len(v)})" for k, v in corpus_sets().items()))
        return 0
    if args.report:
        report(results)
        return 0
    sets = corpus_sets()
    results.parent.mkdir(parents=True, exist_ok=True)
    work = SANDBOX / "tmp" / f"bench-{os.getpid()}{args.tag}"
    failures = 0
    for codec in [c for c in args.codecs.split(",") if c]:
        for s in args.sets.split(","):
            for f in sets[s]:
                try:
                    rec = bench_one(codec, f, s, work)
                except subprocess.CalledProcessError as e:
                    rec = dict(codec=codec, set=s, file=f.name, error=str(e))
                failures += not rec.get("roundtrip_ok", False)
                with open(results, "a") as out:
                    out.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
