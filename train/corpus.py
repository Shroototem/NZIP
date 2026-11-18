"""Build the sandboxed training corpus in data/train/.

Categories (each written as <cat>.train.bin and <cat>.val.bin, documents shuffled and
concatenated):
  web    FineWeb-edu sample (modern web prose)
  books  PG-19 (Project Gutenberg), minus books that appear in the benchmark sets
  wiki   German Wikipedia XML (wiki markup; English Wikipedia is avoided so enwik8/9 stay
         honest held-out tests)
  code   C/C++ headers, Python stdlib, Rust crate sources (local, read-only)
  docs   /usr/share/doc text/HTML and man pages (local)
  bin    ELF executables and shared libraries (local)
  data   XML/JSON/SVG/CSV/PO and similar structured files (local)

Every document is checked against the held-out benchmark files (Canterbury, Calgary,
Silesia, enwik8 prefix) with anchored 32-byte substring hashes; documents that share
more than a few anchors are dropped (see contam.py).

Run (after `source .sandbox/env.sh`):  python -I train/corpus.py [--only cat,cat]
Downloads are untrusted input: they are only parsed as data (parquet/bz2/XML text).
"""
import argparse
import bz2
import gzip
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contam  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "downloads"
OUT = ROOT / "data" / "train"
MB = 1 << 20

# Per-category byte budgets (train + val).
BUDGET = {"web": 900 * MB, "books": 700 * MB, "wiki": 600 * MB, "code": 450 * MB,
          "docs": 250 * MB, "bin": 450 * MB, "data": 300 * MB}
VAL_BYTES = 4 * MB
MAX_DOC = 4 * MB  # cap any single local file so one huge file can't dominate

# Benchmark texts that also exist as Project Gutenberg books.
BOOK_EXCLUDE = ["dickens", "alice", "paradise lost", "as you like it", "madding crowd",
                "looking glass", "shakespeare"]


def local_files(roots, exts=None, elf=False, min_size=64):
    out = []
    for r in roots:
        for dp, dns, fns in os.walk(r, followlinks=False):
            for fn in fns:
                p = os.path.join(dp, fn)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                if exts is not None and not fn.lower().endswith(exts):
                    continue
                try:
                    if os.path.getsize(p) < min_size:
                        continue
                    if elf:
                        with open(p, "rb") as f:
                            if f.read(4) != b"\x7fELF":
                                continue
                except OSError:
                    continue
                out.append(p)
    return sorted(out)


def read_local(paths, gunzip=False):
    for p in paths:
        try:
            with open(p, "rb") as f:
                b = f.read(MAX_DOC if not gunzip else 1 << 30)
            if gunzip:
                b = gzip.decompress(b)[:MAX_DOC]
            yield b
        except (OSError, EOFError, gzip.BadGzipFile):
            continue


def parquet_texts(paths, col="text", meta_filter=None):
    import pyarrow.parquet as pq
    for p in paths:
        f = pq.ParquetFile(p)
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg).to_pylist()
            for row in t:
                if meta_filter and not meta_filter(row):
                    continue
                s = row.get(col)
                if s:
                    yield s.encode("utf-8", "surrogatepass")


def book_ok(row):
    title = (row.get("short_book_title") or "").lower()
    return not any(x in title for x in BOOK_EXCLUDE)


def wiki_pages(path):
    """Split a MediaWiki XML dump into <page>...</page> documents (raw bytes, markup kept)."""
    buf = b""
    with bz2.open(path, "rb") as f:
        while True:
            chunk = f.read(8 * MB)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.find(b"<page>")
                e = buf.find(b"</page>", s + 1) if s >= 0 else -1
                if s < 0 or e < 0:
                    break
                yield buf[s:e + 7] + b"\n"
                buf = buf[e + 7:]
            if len(buf) > 64 * MB:  # pathological page; drop it
                buf = b""


def sources():
    usr_share = "/usr/share"
    return {
        "web": lambda: parquet_texts(sorted((DL / "fineweb").glob("*.parquet"))),
        "books": lambda: parquet_texts(sorted((DL / "pg19").glob("*.parquet")), meta_filter=book_ok),
        "wiki": lambda: wiki_pages(next((DL / "dewiki").glob("*.bz2"))),
        "code": lambda: read_local(local_files(["/usr/include", "/usr/lib/python3.14", str(ROOT / ".sandbox/cargo/registry/src")],
                                                exts=(".h", ".hpp", ".c", ".cc", ".cpp", ".py", ".rs", ".js", ".sh", ".pl"))),
        "docs": lambda: _chain(read_local(local_files(["/usr/share/doc"], exts=(".txt", ".html", ".htm", ".md", ".rst", ".tex"))),
                               read_local(local_files(["/usr/share/man"], exts=(".gz",)), gunzip=True)),
        "bin": lambda: read_local(local_files(["/usr/bin", "/usr/lib"], elf=True, min_size=4096)),
        "data": lambda: read_local(local_files([usr_share], exts=(".xml", ".json", ".svg", ".csv", ".po", ".desktop", ".ini",
                                                                 ".conf", ".yaml", ".yml", ".tsv", ".dat", ".txt"))),
    }


def _chain(*its):
    for it in its:
        yield from it


def build(cat, gen, bench_index, seed=1):
    budget = BUDGET[cat]
    docs, total, dropped, dropped_bytes = [], 0, 0, 0
    for d in gen():
        if not d:
            continue
        if contam.is_contaminated(d, bench_index):
            dropped += 1
            dropped_bytes += len(d)
            continue
        docs.append(d)
        total += len(d)
        if total >= budget * 1.15:  # a little extra, then shuffle and trim
            break
    random.Random(seed).shuffle(docs)
    val, train, vb = [], [], 0
    for d in docs:
        if vb < VAL_BYTES:
            val.append(d)
            vb += len(d)
        else:
            train.append(d)
    tr = b"".join(train)[: budget - vb]
    va = b"".join(val)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cat}.train.bin").write_bytes(tr)
    (OUT / f"{cat}.val.bin").write_bytes(va)
    print(f"{cat:6s} docs {len(docs):8d}  train {len(tr) / MB:7.1f} MB  val {len(va) / MB:5.1f} MB  "
          f"dropped {dropped} docs ({dropped_bytes / MB:.1f} MB) as benchmark overlap", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cats = a.only.split(",") if a.only else list(BUDGET)
    idx = contam.bench_index()
    src = sources()
    for c in cats:
        build(c, src[c], idx)


if __name__ == "__main__":
    main()
