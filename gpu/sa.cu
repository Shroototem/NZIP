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
