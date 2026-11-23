// nzip GPU suffix sorting: prefix doubling over a stable LSD radix sort.
// Rebuild the embedded PTX with scripts/build-ptx.sh after editing.
//
// Targets sm_75+ (the CUDA 13 minimum). Results are verified on the device
// (sa_isa + sa_check) before use; any failure sends the block to the CPU sorter.

typedef unsigned int u32;
typedef unsigned long long u64;

#define WARPS 8
#define THREADS 256
#define IPL 16
#define TILE 4096

__device__ __forceinline__ u32 lanemask_lt() {
    u32 m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

// Mask of lanes holding the same value (256 = "no item"). Hardware match on sm_70+;
// the PTX targets sm_75, so every GPU that can load these kernels has it.
__device__ __forceinline__ u32 match9(u32 d) { return __match_any_sync(0xffffffffu, d); }

// ------------------------------------------------------------------ radix sort pass

#define RS_IPL 8
#define RS_TILE (THREADS * RS_IPL)  // 2048 items per tile

extern "C" __global__ void rs_hist(const u64* __restrict__ keys, u32 n, u32 shift,
                                   u32* __restrict__ hist, u32 ntiles) {
    __shared__ u32 h[256];
    h[threadIdx.x] = 0;
    __syncthreads();
    u32 base = blockIdx.x * RS_TILE;
    for (u32 k = threadIdx.x; k < RS_TILE; k += THREADS) {
        u32 i = base + k;
        if (i < n) atomicAdd(&h[(u32)(keys[i] >> shift) & 255u], 1u);
    }
    __syncthreads();
    hist[threadIdx.x * ntiles + blockIdx.x] = h[threadIdx.x];
}

__device__ u32 block_scan(u32 v, int mx, u32* total);

// Stable scatter. Items of a tile are ranked in index order (warp-major, then iteration,
// then lane), staged in shared memory sorted by digit, then written out so that runs of
// equal digits go to consecutive global addresses (coalesced).
// hs = exclusive scan of the digit-major histogram.
extern "C" __global__ void rs_scatter(const u64* __restrict__ kin, const u32* __restrict__ vin,
                                      u64* __restrict__ kout, u32* __restrict__ vout, u32 n,
                                      u32 shift, const u32* __restrict__ hs, u32 ntiles) {
    __shared__ u32 cnt[WARPS][256];
    __shared__ u32 goff[256];
    __shared__ u32 toff[256];
    __shared__ u64 sk[RS_TILE];
    __shared__ u32 sv[RS_TILE];
    u32 t = threadIdx.x, w = t >> 5, lane = t & 31;
    for (u32 i = t; i < WARPS * 256; i += THREADS) (&cnt[0][0])[i] = 0;
    goff[t] = hs[t * ntiles + blockIdx.x];
    __syncthreads();
    u32 tile0 = blockIdx.x * RS_TILE;
    u32 base = tile0 + w * 32 * RS_IPL;
    for (int it = 0; it < RS_IPL; it++) {
        u32 i = base + it * 32 + lane;
        u32 d = i < n ? ((u32)(kin[i] >> shift) & 255u) : 256u;
        u32 m = match9(d);
        if (d < 256u && lane == (u32)(__ffs(m) - 1)) cnt[w][d] += __popc(m);
    }
    __syncthreads();
    // per-warp exclusive offsets for digit t, and the tile total for digit t
    u32 tot = 0;
    for (int k = 0; k < WARPS; k++) {
        u32 c = cnt[k][t];
        cnt[k][t] = tot;
        tot += c;
    }
    u32 all;
    toff[t] = block_scan(tot, 0, &all);  // where digit t starts inside the tile
    __syncthreads();
    u32 lt = lanemask_lt();
    for (int it = 0; it < RS_IPL; it++) {
        u32 i = base + it * 32 + lane;
        u64 key = 0;
        u32 val = 0, d = 256u;
        if (i < n) {
            key = kin[i];
            val = vin[i];
            d = (u32)(key >> shift) & 255u;
        }
        u32 m = match9(d);
        u32 pos = 0;
        if (d < 256u) pos = toff[d] + cnt[w][d] + __popc(m & lt);
        __syncwarp();
        if (d < 256u && lane == (u32)(__ffs(m) - 1)) cnt[w][d] += __popc(m);
        __syncwarp();
        if (d < 256u) {
            sk[pos] = key;
            sv[pos] = val;
        }
    }
    __syncthreads();
    u32 len = n - tile0 < RS_TILE ? n - tile0 : RS_TILE;
    for (u32 j = t; j < len; j += THREADS) {
        u64 key = sk[j];
        u32 d = (u32)(key >> shift) & 255u;
        u32 dst = goff[d] + (j - toff[d]);
        kout[dst] = key;
        vout[dst] = sv[j];
    }
}

// ------------------------------------------------------------------ scans (sum or max)

__device__ __forceinline__ u32 op2(u32 a, u32 b, int mx) { return mx ? (a > b ? a : b) : a + b; }

// Block-wide exclusive scan of one value per thread; *total = block aggregate.
__device__ u32 block_scan(u32 v, int mx, u32* total) {
    __shared__ u32 ws[WARPS];
    u32 lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    u32 x = v;
    for (int o = 1; o < 32; o <<= 1) {
        u32 y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= (u32)o) x = op2(x, y, mx);
    }
    u32 ex = __shfl_up_sync(0xffffffffu, x, 1);
    if (lane == 0) ex = 0;
    if (lane == 31) ws[w] = x;
    __syncthreads();
    if (w == 0) {
        u32 z = lane < WARPS ? ws[lane] : 0;
        for (int o = 1; o < WARPS; o <<= 1) {
            u32 y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= (u32)o) z = op2(z, y, mx);
        }
        if (lane < WARPS) ws[lane] = z;
    }
    __syncthreads();
    u32 pre = w ? ws[w - 1] : 0;
    *total = ws[WARPS - 1];
    __syncthreads();
    return op2(pre, ex, mx);
}

extern "C" __global__ void scan_reduce(const u32* __restrict__ in, u32 n, u32* __restrict__ sums, int mx) {
    u32 base = blockIdx.x * TILE + threadIdx.x * IPL;
    u32 acc = 0;
    for (int k = 0; k < IPL; k++) {
        u32 i = base + k;
        if (i < n) acc = op2(acc, in[i], mx);
    }
    u32 total;
    block_scan(acc, mx, &total);
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

// Single block: exclusive scan of m block sums, in place.
extern "C" __global__ void scan_top(u32* sums, u32 m, int mx) {
    u32 carry = 0;
    for (u32 c = 0; c < m; c += TILE) {
        u32 base = c + threadIdx.x * IPL;
        u32 v[IPL];
        u32 acc = 0;
        for (int k = 0; k < IPL; k++) {
            u32 i = base + k;
            v[k] = i < m ? sums[i] : 0;
            u32 a = acc;
            acc = op2(acc, v[k], mx);
            v[k] = a;
        }
        u32 total;
        u32 pre = block_scan(acc, mx, &total);
        for (int k = 0; k < IPL; k++) {
            u32 i = base + k;
            if (i < m) sums[i] = op2(carry, op2(pre, v[k], mx), mx);
        }
        carry = op2(carry, total, mx);
        __syncthreads();
    }
}

extern "C" __global__ void scan_down(const u32* in, u32* out, u32 n, const u32* __restrict__ sums,
                                     int mx, int inclusive) {
    u32 base = blockIdx.x * TILE + threadIdx.x * IPL;
    u32 v[IPL];
    u32 acc = 0;
    for (int k = 0; k < IPL; k++) {
        u32 i = base + k;
        u32 x = i < n ? in[i] : 0;
        u32 a = acc;
        acc = op2(acc, x, mx);
        v[k] = inclusive ? acc : a;
    }
    u32 total;
    u32 pre = op2(sums[blockIdx.x], block_scan(acc, mx, &total), mx);
    for (int k = 0; k < IPL; k++) {
        u32 i = base + k;
        if (i < n) out[i] = op2(pre, v[k], mx);
    }
}

// ------------------------------------------------------------------ suffix array steps

// Initial keys: 7 symbols (byte+1, 0 past the end) packed 9 bits each.
extern "C" __global__ void sa_init(const unsigned char* __restrict__ s, u32 n, u64* key, u32* val) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 k = 0;
    for (u32 j = 0; j < 7; j++) {
        u64 p = (u64)i + j;
        u64 c = p < n ? (u64)s[p] + 1 : 0;
        k = (k << 9) | c;
    }
    key[i] = k;
    val[i] = i;
}

// A[j] = j if a new primary group starts at j, B[j] = j if a new key run starts at j.
extern "C" __global__ void sa_flags(const u64* __restrict__ key, u32 m, int round0, u32* A, u32* B) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    u64 k = key[j];
    bool fk = j == 0 || key[j - 1] != k;
    bool fp = round0 ? (j == 0) : (j == 0 || (key[j - 1] >> 32) != (k >> 32));
    A[j] = fp ? j : 0;
    B[j] = fk ? j : 0;
}

// After inclusive max-scans A, B hold the start of each element's primary group / key run.
// Writes the SA slot, the new rank (only where it changed), the keep flag, and leaves the
// new rank in A[j] so compaction can carry it (saves a random read when building keys).
extern "C" __global__ void sa_assign(const u64* __restrict__ key, const u32* __restrict__ val, u32 m,
                                     int round0, u32* A, const u32* __restrict__ B,
                                     u32* sa, u32* rank, u32* keep) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    u32 prim = round0 ? 0 : (u32)(key[j] >> 32);
    u32 p = val[j];
    u32 a = A[j], b = B[j];
    sa[prim + (j - a)] = p;
    u32 nr = prim + (b - a);
    if (round0 || b != a) rank[p] = nr;
    bool first = b == j;
    bool last = (j + 1 == m) || (B[j + 1] == j + 1);
    keep[j] = (first && last) ? 0u : 1u;
    A[j] = nr;
}

// idx = exclusive scan of keep (in place): element j kept iff its successor's index is larger.
extern "C" __global__ void sa_compact(const u32* __restrict__ val, const u32* __restrict__ idx, u32 m,
                                      u32 total, const u32* __restrict__ nrank, u32* out, u32* out_rank) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    u32 nx = j + 1 < m ? idx[j + 1] : total;
    if (nx > idx[j]) {
        out[idx[j]] = val[j];
        out_rank[idx[j]] = nrank[j];
    }
}

extern "C" __global__ void sa_keys(const u32* __restrict__ list, const u32* __restrict__ prim, u32 m,
                                   u32 h, u32 n, const u32* __restrict__ rank, u64* key, u32* val) {
    u32 k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    u32 p = list[k];
    u64 q = (u64)p + h;
    u32 r2 = q < n ? rank[q] + 1 : 0;
    key[k] = ((u64)prim[k] << 32) | r2;
    val[k] = p;
}

extern "C" __global__ void sa_isa(const u32* __restrict__ sa, u32 n, u32* isa, u32* err) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    u32 p = sa[r];
    if (p >= n) {
        atomicExch(err, 1u);
        return;
    }
    isa[p] = r;
}

// Exact O(n) check: SA is a permutation and adjacent suffixes are in order.
extern "C" __global__ void sa_check(const unsigned char* __restrict__ s, const u32* __restrict__ sa,
                                    const u32* __restrict__ isa, u32 n, u32* err) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    u32 b = sa[r];
    if (b >= n || isa[b] != r) {
        atomicExch(err, 2u);
        return;
    }
    if (r == 0) return;
    u32 a = sa[r - 1];
    if (a >= n) {
        atomicExch(err, 3u);
        return;
    }
    if (s[a] < s[b]) return;
    if (s[a] > s[b]) {
        atomicExch(err, 4u);
        return;
    }
    if (a + 1 == n) return;
    if (b + 1 == n || isa[a + 1] >= isa[b + 1]) atomicExch(err, 5u);
}

extern "C" __global__ void sa_bwt(const unsigned char* __restrict__ s, const u32* __restrict__ sa, u32 n,
                                  u32 seg, unsigned char* out, u32* idx) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    u32 p = sa[r];
    out[r] = p ? s[p - 1] : s[n - 1];
    if (p % seg == 0) idx[p / seg] = r + 1;
}

// ------------------------------------------------------------------ inverse BWT walk

// One thread per chain: chain t starts at matrix row rows[t] and emits text positions
// [t*seg, min((t+1)*seg, n)), following psi. fin[t] = the row after the last step, which
// the host checks against rows[t+1] (a corrupt walk cannot land there by accident).
extern "C" __global__ void ub_walk(const u32* __restrict__ psi, const u32* __restrict__ rows, u32 k, u32 seg,
                                   u32 n, const u32* __restrict__ c1g, unsigned char* out, u32* fin) {
    __shared__ u32 c1[257];
    for (u32 i = threadIdx.x; i < 257; i += blockDim.x) c1[i] = c1g[i];
    __syncthreads();
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= k) return;
    unsigned long long s0 = (unsigned long long)t * seg;
    if (s0 >= n) return;
    u32 start = (u32)s0;
    u32 end = min(start + seg, n);
    u32 r = rows[t];
    for (u32 i = start; i < end; i++) {
        u32 lo = 0;
        for (u32 step = 128; step; step >>= 1)
            if (c1[lo + step] <= r) lo += step;
        out[i] = (unsigned char)lo;
        r = psi[r];
    }
    fin[t] = r;
}

// Inverse BWT index on the device: psi[F-row] = L-row, a stable counting sort of the rows
// of L' by character. L'[0] = b[p1-1], L'[p1] = '$' (skipped), L'[r] = b[r-1] otherwise.
#define UB_T 2048
__device__ __forceinline__ u32 ub_lch(const unsigned char* b, u32 r, u32 p1) { return r ? b[r - 1] : b[p1 - 1]; }

// counts[c * tiles + t] = occurrences of c in tile t (UB_T rows each).
extern "C" __global__ void ub_count(const unsigned char* __restrict__ b, u32 rows, u32 p1, u32 tiles, u32* counts) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tiles) return;
    u32 c[256];
    for (int i = 0; i < 256; i++) c[i] = 0;
    u32 a = t * UB_T, z = min(a + UB_T, rows);
    for (u32 r = a; r < z; r++)
        if (r != p1) c[ub_lch(b, r, p1)]++;
    for (int i = 0; i < 256; i++) counts[i * tiles + t] = c[i];
}

// After an exclusive scan of counts: write psi (F rows start at 1; row 0 is '$').
extern "C" __global__ void ub_scatter(const unsigned char* __restrict__ b, u32 rows, u32 p1, u32 tiles,
                                      const u32* __restrict__ offs, u32* psi) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tiles) return;
    u32 o[256];
    for (int i = 0; i < 256; i++) o[i] = offs[i * tiles + t] + 1;
    u32 a = t * UB_T, z = min(a + UB_T, rows);
    for (u32 r = a; r < z; r++)
        if (r != p1) psi[o[ub_lch(b, r, p1)]++] = r;
    if (t == 0) psi[0] = p1;
}

// c1[c] = first F row of character c; c1[256] = rows.
extern "C" __global__ void ub_c1(const u32* __restrict__ offs, u32 tiles, u32 rows, u32* c1) {
    u32 c = threadIdx.x;
    c1[c] = offs[c * tiles] + 1;
    if (c == 0) c1[256] = rows;
}

// ------------------------------------------------------------------ LZ match finding

#define MF_MAXL 273
#define MF_K 24
#define MF_C 6

// lcp[r] = common prefix length of suffixes sa[r-1] and sa[r] (capped), lcp[0] = 0.
extern "C" __global__ void mf_lcp(const unsigned char* __restrict__ s, const u32* __restrict__ sa, u32 n,
                                  unsigned short* lcp) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    if (r == 0) {
        lcp[0] = 0;
        return;
    }
    u32 a = sa[r - 1], b = sa[r];
    u32 far = a > b ? a : b;
    u32 lim = n - far;
    if (lim > MF_MAXL) lim = MF_MAXL;
    u32 l = 0;
    while (l < lim && s[a + l] == s[b + l]) l++;
    lcp[r] = (unsigned short)l;
}

#define MF_LEVELS 6
__device__ __constant__ u32 MF_GRAM[MF_LEVELS] = {3, 4, 6, 8, 12, 16};

// Key of the gram of length L at p (exact up to 8 bytes, hashed above); positions too close
// to the end get a key of their own so they match nothing.
extern "C" __global__ void mf_keys(const unsigned char* __restrict__ s, u32 n, u32 L, u64* key, u32* val) {
    u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    // grams running past the end are zero-padded: false pairs are dropped by the length check
    u64 k;
    if (L <= 8) {
        k = 0;
        for (u32 j = 0; j < L; j++) k = (k << 8) | (p + j < n ? s[p + j] : 0);
    } else {
        u32 h = 2166136261u;
        for (u32 j = 0; j < L; j++) h = (h ^ (p + j < n ? s[p + j] : 0)) * 16777619u;
        k = h;
    }
    key[p] = k;
    val[p] = p;
}

// After a stable sort by key, each position's predecessor with the same key is its nearest
// earlier occurrence of that gram.
extern "C" __global__ void mf_prev(const u64* __restrict__ key, const u32* __restrict__ val, u32 n, u32* prev) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    prev[val[i]] = (i > 0 && key[i] == key[i - 1]) ? val[i - 1] : 0xFFFFFFFFu;
}

// For text positions [p0, p0 + cnt): earlier occurrences found among the MF_K nearest
// suffix-array neighbours on each side. Output per position: count, then up to MF_C
// (len, dist - 1) pairs with increasing length and increasing distance (each the nearest
// occurrence for its length range).
extern "C" __global__ void mf_find(const unsigned char* __restrict__ s, const u32* __restrict__ sa,
                                   const u32* __restrict__ isa, const unsigned short* __restrict__ lcp,
                                   const u32* __restrict__ prev, u32 n, u32 p0, u32 cnt, u32* out) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cnt) return;
    u32 p = p0 + t;
    u32 r = isa[p];
    u32 cl[2 * MF_K + MF_LEVELS], cd[2 * MF_K + MF_LEVELS];
    u32 nc = 0;
    // nearest earlier occurrence of each gram length
    u32 room = n - p;
    if (room > MF_MAXL) room = MF_MAXL;
    for (u32 v = 0; v < MF_LEVELS; v++) {
        u32 q = prev[(u64)v * n + p];
        if (q == 0xFFFFFFFFu) continue;
        u32 l = 0;
        while (l < room && s[q + l] == s[p + l]) l++;
        if (l >= 3) { cl[nc] = l; cd[nc] = p - q - 1; nc++; }
    }
    u32 m = MF_MAXL;
    for (u32 i = 1; i <= MF_K && i <= r; i++) {
        u32 l = lcp[r - i + 1];
        if (l < m) m = l;
        if (m < 3) break;
        u32 q = sa[r - i];
        if (q < p) { cl[nc] = m; cd[nc] = p - q - 1; nc++; }
    }
    m = MF_MAXL;
    for (u32 i = 1; i <= MF_K && r + i < n; i++) {
        u32 l = lcp[r + i];
        if (l < m) m = l;
        if (m < 3) break;
        u32 q = sa[r + i];
        if (q < p) { cl[nc] = m; cd[nc] = p - q - 1; nc++; }
    }
    // Pareto front: keep (len, dist) unless some candidate is at least as long and nearer.
    // Selection by repeatedly taking the longest remaining (ties: nearest).
    u32* o = out + (u64)t * (1 + 2 * MF_C);
    u32 kl[2 * MF_K + MF_LEVELS], kd[2 * MF_K + MF_LEVELS];
    u32 nk = 0;
    u32 best_d = 0xffffffffu;
    for (u32 it = 0; it < nc; it++) {
        u32 bi = 0xffffffffu;
        for (u32 j = 0; j < nc; j++) {
            if (cl[j] == 0) continue;
            if (bi == 0xffffffffu || cl[j] > cl[bi] || (cl[j] == cl[bi] && cd[j] < cd[bi])) bi = j;
        }
        if (bi == 0xffffffffu) break;
        if (cd[bi] < best_d) {
            best_d = cd[bi];
            kl[nk] = cl[bi];
            kd[nk] = cd[bi];
            nk++;
        }
        cl[bi] = 0;
    }
    // kl is decreasing; emit increasing, keeping the nearest and the MF_C - 1 longest
    u32 e = 0;
    if (nk > MF_C) {
        o[1 + 2 * e] = kl[nk - 1];
        o[2 + 2 * e] = kd[nk - 1];
        e++;
        for (u32 j = MF_C - 2; j < MF_C - 1; j--) {
            o[1 + 2 * e] = kl[j];
            o[2 + 2 * e] = kd[j];
            e++;
            if (j == 0) break;
        }
    } else {
        for (u32 j = nk; j > 0; j--) {
            o[1 + 2 * e] = kl[j - 1];
            o[2 + 2 * e] = kd[j - 1];
            e++;
        }
    }
    o[0] = e;
}

// Number of pairs each position keeps (0 inside skipped segments).
extern "C" __global__ void mf_count(const u32* __restrict__ out, const unsigned char* __restrict__ skip,
                                    u32 seg_bits, u32 p0, u32 cnt, u32* counts) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cnt) return;
    u32 c = out[(u64)t * (1 + 2 * MF_C)];
    counts[t] = skip[(p0 + t) >> seg_bits] ? 0u : (c < MF_C ? c : MF_C);
}

// Packs the kept pairs contiguously (offs = exclusive scan of counts).
extern "C" __global__ void mf_pack(const u32* __restrict__ out, const u32* __restrict__ offs,
                                   const u32* __restrict__ counts, u32 cnt, u32* packed) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cnt) return;
    const u32* o = out + (u64)t * (1 + 2 * MF_C);
    u32 k = counts[t], at = offs[t];
    for (u32 j = 0; j < k; j++) {
        packed[2 * (at + j)] = o[1 + 2 * j];
        packed[2 * (at + j) + 1] = o[2 + 2 * j];
    }
}

// ------------------------------------------------------------------ LZ optimal parse
//
// One warp parses one 64 KiB segment with a forward optimal parse (one arrival per
// position) priced from static tables. Lane k owns the target lengths k+1, k+33, ...:
// for each it takes the cheapest token ending there (literal / short rep for length 1,
// rep 0-3, the shortest listed match that reaches it), so every arrival is written by one
// lane only. Arrivals live in a shared-memory ring; back pointers go to global memory and
// lane 0 walks them at the end. Matches of LZ_NICE+ bytes are taken outright (as on the
// CPU), so relaxed lengths stay below LZ_NICE.

#define LZ_RING 128  // > the farthest relaxed target (lengths < LZ_NICE)
#define LZ_MAXL 273
#define LZ_NICE 64
#define LZ_SEGB 16
#define LZ_SEG (1u << LZ_SEGB)
#define LZ_INF 0xFFFFFFFFu
#define LZ_WARPS 8

// Static prices in 1/16 bit (layout shared with gpu.rs LzPrices).
struct LzP {
    u32 lit_flag[12][4];
    u32 match_flag[12][4];
    u32 rep_flag[12][2];
    u32 rep0[12][2];
    u32 rep0long[12][4][2];
    u32 rep1[12][2];
    u32 rep2[12][2];
    u32 mlen[4][272];
    u32 rlen[4][272];
    u32 slot[4][64];
    u32 small[4][128];
    u32 align[16];
};

struct LzE {
    u32 ps;  // price << 4 | state
    u32 r[4];
};

__device__ __forceinline__ u32 lz_next(u32 s, u32 kind) {  // kind: 0 lit, 1 short, 2 rep, 3 match
    if (kind == 0) return s < 4 ? 0 : (s < 10 ? s - 3 : s - 6);
    if (kind == 3) return s < 7 ? 7 : 10;
    if (kind == 2) return s < 7 ? 8 : 11;
    return s < 7 ? 9 : 11;
}

__device__ __forceinline__ u32 lz_dist(const LzP* __restrict__ P, u32 d, u32 len) {
    u32 ls = min(len - 2, 3u);
    if (d < 128) return P->small[ls][d];
    u32 nb = 31 - __clz(d);
    u32 s = (nb << 1) | ((d >> (nb - 1)) & 1);
    u32 footer = (s >> 1) - 1;
    return P->slot[ls][s] + (footer - 4) * 16 + P->align[d & 15];
}

__device__ __forceinline__ u32 lz_repcost(const LzP* __restrict__ P, u32 k, u32 st, u32 pst) {
    if (k == 0) return P->rep0[st][0] + P->rep0long[st][pst][1];
    if (k == 1) return P->rep0[st][1] + P->rep1[st][0];
    if (k == 2) return P->rep0[st][1] + P->rep1[st][1] + P->rep2[st][0];
    return P->rep0[st][1] + P->rep1[st][1] + P->rep2[st][1];
}

// Token word: len (9 bits) | kind << 9 | rep index << 11. `d` = match distance - 1, or the
// rep distance used (dist - 1) for reps and short reps.
extern "C" __global__ void lz_parse(const unsigned char* __restrict__ t, u32 n, u32 seg0, u32 nseg, u32 ib,
                                    const u32* __restrict__ idx, const u32* __restrict__ pairs,
                                    const unsigned char* __restrict__ dense, const LzP* __restrict__ P0, u32 pstep,
                                    const unsigned short* __restrict__ lit, u32* __restrict__ back_a,
                                    u32* __restrict__ back_d, u32* __restrict__ out_n) {
    __shared__ LzE ring[LZ_WARPS][LZ_RING];
    u32 w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    u32 sg = blockIdx.x * LZ_WARPS + w;
    if (sg >= nseg) return;
    u32 g = seg0 + sg;
    u32 s = g << LZ_SEGB;
    u32 e = min(s + LZ_SEG, n);
    if (s >= e || dense[g]) {
        if (lane == 0) out_n[sg] = LZ_INF;
        return;
    }
    const LzP* __restrict__ P = P0 + sg * pstep;  // one table, or one per segment
    LzE* R = ring[w];
    for (u32 k = lane; k < LZ_RING; k += 32) R[k].ps = LZ_INF;
    __syncwarp();
    if (lane == 0) {
        R[0].ps = 0;
        R[0].r[0] = R[0].r[1] = R[0].r[2] = R[0].r[3] = 0;
    }
    __syncwarp();
    u32 len_seg = e - s;
    u32 i = 0;
    while (i < len_seg) {
        LzE cur = R[i & (LZ_RING - 1)];
        __syncwarp();
        if (lane == 0) R[(i + LZ_NICE) & (LZ_RING - 1)].ps = LZ_INF;
        u32 p = s + i;
        u32 st = cur.ps & 15, price = cur.ps >> 4;
        u32 room = e - p, cap = min(room, (u32)LZ_MAXL), pst = p & 3;
        // rep match lengths: 32 bytes per step, first mismatch by ballot
        u32 rl[4];
        u32 maxl = 1, longest = 0, lk = 0;  // lk: 0-3 rep, 4.. match index + 4
#pragma unroll
        for (int k = 0; k < 4; k++) {
            u32 r = cur.r[k] + 1, L = 0;
            if (p >= r && cap >= 1) {
                for (u32 off = 0;; off += 32) {
                    u32 q = off + lane;
                    bool ne = q >= cap || t[p + q] != t[p - r + q];
                    u32 m = __ballot_sync(0xffffffffu, ne);
                    if (m) {
                        L = off + __ffs(m) - 1;
                        break;
                    }
                }
            }
            rl[k] = L;
            if (L >= 2 && L > longest) {
                longest = L;
                lk = k;
            }
            if (L >= 2) maxl = max(maxl, L);
        }
        // listed matches: kept when >= 3 and longer than the previous one
        u32 ml[6], md[6], mc = 0;
        u32 m0 = idx[p - ib], m1 = idx[p - ib + 1];
        for (u32 k = m0; k < m1 && k < m0 + 6; k++) {
            u32 L = min(pairs[2 * k], room);
            if (L >= 3 && (mc == 0 || ml[mc - 1] < L)) {
                ml[mc] = L;
                md[mc] = pairs[2 * k + 1];
                mc++;
            }
        }
        if (mc && ml[mc - 1] > longest) {
            longest = ml[mc - 1];
            lk = 4 + mc - 1;
        }
        if (mc) maxl = max(maxl, ml[mc - 1]);
        if (longest >= LZ_NICE) {
            // take it from here: the window restarts where it ends (as on the CPU)
            u32 j = i + longest;
            for (u32 k = lane; k < LZ_RING; k += 32) R[k].ps = LZ_INF;
            __syncwarp();
            if (lane == 0) {
                LzE nd;
                u32 a, d;
                if (lk < 4) {
                    u32 rd = cur.r[lk];
                    nd.r[0] = rd;
                    nd.r[1] = lk >= 1 ? cur.r[0] : cur.r[1];
                    nd.r[2] = lk >= 2 ? cur.r[1] : cur.r[2];
                    nd.r[3] = lk >= 3 ? cur.r[2] : cur.r[3];
                    nd.ps = (price << 4) | lz_next(st, 2);
                    a = longest | (2u << 9) | (lk << 11);
                    d = rd;
                } else {
                    d = md[lk - 4];
                    nd.r[0] = d;
                    nd.r[1] = cur.r[0];
                    nd.r[2] = cur.r[1];
                    nd.r[3] = cur.r[2];
                    nd.ps = (price << 4) | lz_next(st, 3);
                    a = longest | (3u << 9);
                }
                R[j & (LZ_RING - 1)] = nd;
                back_a[s + j - 1 - ib] = a;
                back_d[s + j - 1 - ib] = d;
            }
            __syncwarp();
            i = j;
            continue;
        }
        u32 mflag = price + P->match_flag[st][pst];
        u32 rbase = mflag + P->rep_flag[st][1];
        u32 mbase = mflag + P->rep_flag[st][0];
        for (u32 l = lane + 1; l <= maxl; l += 32) {
            u32 best = LZ_INF, a = 0, d = 0, kind = 0, rk = 0;
            if (l == 1) {
                best = price + P->lit_flag[st][pst] + lit[p - ib];
                a = 1;
                if (rl[0] >= 1) {
                    u32 c = rbase + P->rep0[st][0] + P->rep0long[st][pst][0];
                    if (c < best) {
                        best = c;
                        a = 1 | (1u << 9);
                        d = cur.r[0];
                        kind = 1;
                    }
                }
            } else {
#pragma unroll
                for (u32 k = 0; k < 4; k++) {
                    if (rl[k] >= l) {
                        u32 c = rbase + lz_repcost(P, k, st, pst) + P->rlen[pst][l - 2];
                        if (c < best) {
                            best = c;
                            a = l | (2u << 9) | (k << 11);
                            d = cur.r[k];
                            kind = 2;
                            rk = k;
                        }
                    }
                }
                if (l >= 3) {
                    for (u32 k = 0; k < mc; k++) {
                        if (ml[k] >= l) {
                            u32 c = mbase + P->mlen[pst][l - 2] + lz_dist(P, md[k], l);
                            if (c < best) {
                                best = c;
                                a = l | (3u << 9);
                                d = md[k];
                                kind = 3;
                            }
                            break;
                        }
                    }
                }
            }
            if (best == LZ_INF) continue;
            LzE* tg = &R[(i + l) & (LZ_RING - 1)];
            if (tg->ps != LZ_INF && (tg->ps >> 4) <= best) continue;
            LzE nd;
            if (kind == 3) {
                nd.r[0] = d; nd.r[1] = cur.r[0]; nd.r[2] = cur.r[1]; nd.r[3] = cur.r[2];
            } else if (kind == 2) {
                nd.r[0] = cur.r[rk];
                nd.r[1] = rk >= 1 ? cur.r[0] : cur.r[1];
                nd.r[2] = rk >= 2 ? cur.r[1] : cur.r[2];
                nd.r[3] = rk >= 3 ? cur.r[2] : cur.r[3];
            } else {
                nd.r[0] = cur.r[0]; nd.r[1] = cur.r[1]; nd.r[2] = cur.r[2]; nd.r[3] = cur.r[3];
            }
            nd.ps = (best << 4) | lz_next(st, kind);
            *tg = nd;
            back_a[p + l - 1 - ib] = a;
            back_d[p + l - 1 - ib] = d;
        }
        __syncwarp();
        i++;
    }
    // tokens stay in back_a/back_d at (end position - 1); the host walks them back
    if (lane == 0) out_n[sg] = 0;
}
