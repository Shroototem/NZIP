"""Build an ~11 GB mixed test input (for any-size / low-RAM tests) from data on disk.

Layout: enwik9 | Silesia | ~3 GB ELF binaries | ~1 GB /usr/share data | headers +
Python stdlib | already-compressed downloads (parquet, bz2) | enwik9 again (an exact
far repeat, ~9 GB after the first copy, to exercise long-range dedup).

Usage: python -I bench/make_bigmix.py <out-file>
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB = 1 << 30


def walk(roots, exts=None, elf=False, limit=GB):
    total = 0
    for r in roots:
        for dp, _, fns in sorted(os.walk(r)):
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                if exts and not fn.lower().endswith(exts):
                    continue
                try:
                    if elf:
                        with open(p, "rb") as f:
                            if f.read(4) != b"\x7fELF":
                                continue
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if total + sz > limit:
                    return
                total += sz
                yield p


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    b = ROOT / "data" / "bench"
    dl = ROOT / "data" / "downloads"
    parts = [("enwik9", [b / "enwik9" / "enwik9"]),
             ("silesia", sorted((b / "silesia").glob("*"))),
             ("elf", walk(["/usr/lib", "/usr/bin"], elf=True, limit=3 * GB)),
             ("share", walk(["/usr/share"], exts=(".xml", ".json", ".svg", ".txt", ".po", ".html", ".css", ".js", ".csv", ".ttf"), limit=GB)),
             ("code", walk(["/usr/include", "/usr/lib/python3.14"], limit=GB)),
             ("compressed", [*sorted((dl / "fineweb").glob("*")), *sorted((dl / "pg19").glob("*")), *sorted((dl / "dewiki").glob("*"))]),
             ("enwik9-again", [b / "enwik9" / "enwik9"])]
    total = 0
    with open(out, "wb") as o:
        for name, files in parts:
            start = total
            for p in files:
                try:
                    with open(p, "rb") as f:
                        while chunk := f.read(64 << 20):
                            o.write(chunk)
                            total += len(chunk)
                except OSError:
                    continue
            print(f"{name:13s} {(total - start) / GB:6.2f} GiB", flush=True)
    print(f"total         {total / GB:6.2f} GiB ({total} bytes)")


if __name__ == "__main__":
    main()
