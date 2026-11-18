"""Word-replacing transform prototype (measure ratio before writing it in Rust).
usage: wrt.py <in> <out> [K] [c1] [order] [minlen]"""
import re, sys, collections
E, CAP, UP = 0x01, 0x02, 0x03          # escape, Capitalised, ALLCAPS flags
SPECIAL = {E, CAP, UP} | set(range(128, 256))

def build(data, K, minlen):
    cnt = collections.Counter(m.group().lower() for m in re.finditer(rb'[A-Za-z]+', data))
    return cnt

def codes(n, c1, c2=None):
    """c1 one-byte codes, c2 two-byte prefixes, the remaining high values 3-byte prefixes."""
    if c2 is None:
        c2 = 128 - c1
    out = [bytes([128 + i]) for i in range(min(n, c1))]
    for a in range(c2):
        for b in range(128):
            if len(out) >= n: return out
            out.append(bytes([128 + c1 + a, 128 + b]))
    for a in range(128 - c1 - c2):
        for b in range(128):
            for c in range(128):
                if len(out) >= n: return out
                out.append(bytes([128 + c1 + c2 + a, 128 + b, 128 + c]))
    return out

def encode(data, K=40000, c1=64, order="freq", minlen=2, c2=None):
    cnt = build(data, K, minlen)
    # value of a word ~ freq * (len - ~1.5)
    cand = [w for w, f in cnt.items() if len(w) >= minlen and f >= 4]
    cand.sort(key=lambda w: -cnt[w] * (len(w) - 1.5))
    cand = cand[:K]
    cand.sort(key=lambda w: -cnt[w])          # most frequent get 1-byte codes
    cl = codes(len(cand), c1, c2)
    cand = cand[:len(cl)]
    if order == "alpha":                       # sort within each code-length class
        one = sorted(cand[:c1]); two = sorted(cand[c1:]); cand = one + two
    elif order == "suffix":
        n2 = c1 + (c2 if c2 is not None else 128 - c1) * 128
        r = lambda ws: sorted(ws, key=lambda w: w[::-1])
        cand = r(cand[:c1]) + r(cand[c1:n2]) + r(cand[n2:])
    table = {w: cl[i] for i, w in enumerate(cand)}
    out = bytearray()
    for m in re.finditer(rb'[A-Za-z]+|[^A-Za-z]+', data):
        t = m.group()
        if 65 <= t[0] <= 90 or 97 <= t[0] <= 122:
            lo = t.lower()
            c = table.get(lo)
            if c is None:
                out += t
            elif t == lo:
                out += c
            elif t[0:1].isupper() and t[1:] == lo[1:]:
                out.append(CAP); out += c
            elif t.isupper() and len(t) > 1:
                out.append(UP); out += c
            else:
                out += t
        else:
            for b in t:
                if b in SPECIAL:
                    out.append(E)
                out.append(b)
    return b"\n".join(cand), bytes(out)

if __name__ == "__main__":
    a = sys.argv
    K = int(a[3]) if len(a) > 3 else 40000
    c1 = int(a[4]) if len(a) > 4 else 64
    order = a[5] if len(a) > 5 else "freq"
    minlen = int(a[6]) if len(a) > 6 else 2
    c2 = int(a[7]) if len(a) > 7 else None
    data = open(a[1], 'rb').read()
    d, t = encode(data, K, c1, order, minlen, c2)
    open(a[2], 'wb').write(t)
    open(a[2] + '.dict', 'wb').write(d)
    print(f"K={K} c1={c1} order={order} minlen={minlen}: {len(data)} -> {len(t)} bytes ({100*len(t)/len(data):.1f}%), dict {len(d)} bytes")
