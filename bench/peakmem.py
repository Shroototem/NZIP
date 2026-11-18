"""Run a command and report wall time and peak private memory.

Peak private memory = max over time of RssAnon + RssShmem of the process (polled every
20 ms). File pages the process maps (e.g. the input via mmap) are excluded: they are
page cache, reclaimable, and not something the program allocated.

Usage: python -I bench/peakmem.py [--cap 5G] -- <command...>
  --cap  run inside a user systemd scope with MemoryMax=<cap> and no swap, so going over
         the cap kills the process (a real low-RAM machine, not an estimate).
Prints one JSON line: {"secs":..., "peak_mb":..., "rc":...}
"""
import json
import subprocess
import sys
import time


def private_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            v = {l.split(":")[0]: l.split()[1] for l in f if l.startswith(("RssAnon", "RssShmem", "Name"))}
        return int(v.get("RssAnon", 0)) + int(v.get("RssShmem", 0)), v.get("Name", "")
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None, ""


def main():
    args = sys.argv[1:]
    cap = None
    if args[:1] == ["--cap"]:
        cap, args = args[1], args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if cap:
        args = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0"] + args
    t0 = time.perf_counter()
    p = subprocess.Popen(args)
    peak = 0
    while p.poll() is None:
        kb, _ = private_kb(p.pid)
        if kb:
            peak = max(peak, kb)
        time.sleep(0.02)
    secs = time.perf_counter() - t0
    print(json.dumps({"secs": round(secs, 3), "peak_mb": round(peak / 1024, 1), "rc": p.returncode}), flush=True)


if __name__ == "__main__":
    main()
