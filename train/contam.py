"""Benchmark-contamination check with anchored substring hashes.

Every 32-byte window gets a 64-bit hash; windows whose hash is 0 mod 32 are "anchors"
(content-defined, so the same text yields the same anchors wherever it appears). The
benchmark files' anchors form a sorted index; a training document is contaminated if
more than MAX_HITS of its anchors appear in that index and they are not just
boilerplate (anchors shared with many documents are excluded as common phrases).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
W = 32
MOD = 32
MAX_HITS = 8
_M = [np.uint64(c) for c in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93)]


def anchors(b: bytes) -> np.ndarray:
    """Hashes of all anchor windows of b (unsorted, may repeat)."""
    n = len(b) - W + 1
    if n <= 0:
        return np.empty(0, np.uint64)
    pad = b + b"\0" * 8
    out = []
    with np.errstate(over="ignore"):
        for k in range(8):  # windows starting at k, k+8, k+16, ...
            words = np.frombuffer(pad, dtype="<u8", count=(len(pad) - k) // 8, offset=k)
            m = (n - k + 7) // 8  # windows with start k + 8i < n
            if m <= 0:
                continue
            h = np.zeros(m, np.uint64)
            for j in range(4):
                h ^= words[j:j + m] * _M[j]
                h = (h << np.uint64(31)) | (h >> np.uint64(33))
            h *= _M[0]
            h ^= h >> np.uint64(29)
            out.append(h[(h % np.uint64(MOD)) == 0])
    return np.concatenate(out) if out else np.empty(0, np.uint64)


def bench_files():
    fs = []
    for d in ("cantrbry", "calgary", "silesia"):
        fs += sorted(p for p in (BENCH / d).rglob("*") if p.is_file())
    fs.append(BENCH / "derived" / "enwik8_10M")
    return fs


def bench_index() -> np.ndarray:
    hs = [np.unique(anchors(p.read_bytes())) for p in bench_files()]
    # Drop anchors that occur in several different benchmark files: these are generic
    # strings (padding, common code idioms), not evidence of copied content.
    allh, counts = np.unique(np.concatenate(hs), return_counts=True)
    return allh[counts <= 2]


def _count_in(a: np.ndarray, index: np.ndarray) -> int:
    """How many of the (unique) values in `a` occur in the sorted `index` (binary search)."""
    pos = np.searchsorted(index, a)
    pos[pos == index.size] = 0
    return int((index[pos] == a).sum())


def hits(d: bytes, index: np.ndarray) -> int:
    a = np.unique(anchors(d))
    return _count_in(a, index) if a.size else 0


def is_contaminated(d: bytes, index: np.ndarray) -> bool:
    if len(d) < W:
        return False
    a = np.unique(anchors(d))
    if a.size == 0:
        return False
    h = _count_in(a, index)
    # Long runs of zeros/padding produce one repeated anchor; require several distinct hits
    # and a non-trivial fraction for long documents.
    return h > MAX_HITS and h > 0.002 * a.size
