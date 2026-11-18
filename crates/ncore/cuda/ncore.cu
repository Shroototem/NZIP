// ncore GPU kernels: one warp per lane, bit-exact with the Rust CPU reference
// (crates/ncore/src/{cpu,model,rans,lib}.rs). Integer arithmetic only.
//
// Per byte, each of the 32 threads owns 8 consecutive symbols (t*8 .. t*8+7):
//   weights -> quantise (warp sums/argmax) -> prefix sums (warp scan) -> code -> update.
// Decoding runs the whole lane inside one kernel launch: no host round-trips.

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define SCALE_BITS 16
#define TOTAL (1u << SCALE_BITS)
#define RANS_L (1u << 16)
#define FRAME (1u << 16)
#define O2_BITS 10
#define INC 32u
#define LIMIT 65000u
#define W2 4u
#define FULL 0xffffffffu

#define CNT1_SIZE (256u * 256u)
#define CNT2_SIZE ((1u << O2_BITS) * 256u)

__device__ __forceinline__ uint32_t o2_hash(uint32_t c2, uint32_t c1) {
    return (((c2 << 8) | c1) * 2654435761u) >> (32 - O2_BITS);
}

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// Exclusive scan across the warp.
__device__ __forceinline__ uint32_t warp_exscan(uint32_t v, int t) {
    uint32_t inc = v;
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t n = __shfl_up_sync(FULL, inc, o);
        if (t >= o) inc += n;
    }
    return inc - v;
}

struct Dist {
    uint32_t f[8];   // this thread's 8 frequencies
    uint32_t base;   // cumulative frequency before this thread's first symbol
};

// Compute the quantised distribution for the current context (mirrors cpu weights+quantize).
__device__ __forceinline__ void distribution(const uint16_t* __restrict__ row1, const uint16_t* __restrict__ row2, int t, Dist& d) {
    uint32_t r[8];
    // 16-byte vector loads: rows are 512 B aligned, each thread reads 8 x u16.
    uint4 a = reinterpret_cast<const uint4*>(row1)[t];
    uint4 b = reinterpret_cast<const uint4*>(row2)[t];
    const uint16_t* pa = reinterpret_cast<const uint16_t*>(&a);
    const uint16_t* pb = reinterpret_cast<const uint16_t*>(&b);
    uint32_t local = 0, best_v = 0, best_i = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        r[k] = (uint32_t)pa[k] + W2 * (uint32_t)pb[k];
        local += r[k];
        if (k == 0 || r[k] > best_v) { best_v = r[k]; best_i = t * 8 + k; }
    }
    uint32_t sum = warp_sum(local);  // < 2^32: 256 * 5 * 65000
    // argmax: larger value wins, ties -> lower index
    for (int o = 16; o > 0; o >>= 1) {
        uint32_t ov = __shfl_xor_sync(FULL, best_v, o);
        uint32_t oi = __shfl_xor_sync(FULL, best_i, o);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    uint32_t tot_local = 0;
    if (sum == 0) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) { d.f[k] = TOTAL / 256; }
        tot_local = 8 * (TOTAL / 256);
    } else {
        const uint64_t scale = ((uint64_t)(TOTAL - 256) << 32) / (uint64_t)sum;  // one division per byte
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            d.f[k] = 1u + (uint32_t)(((uint64_t)r[k] * scale) >> 32);
            tot_local += d.f[k];
        }
        uint32_t tot = warp_sum(tot_local);
        if ((best_i >> 3) == (uint32_t)t) {
            d.f[best_i & 7] += TOTAL - tot;
            tot_local += TOTAL - tot;
        }
    }
    d.base = warp_exscan(tot_local, t);
}

__device__ __forceinline__ void bump_row(uint16_t* row, uint32_t s, int t) {
    // Owner thread checks overflow; the whole warp halves the row if needed.
    uint32_t v = 0;
    if ((s >> 3) == (uint32_t)t) v = (uint32_t)row[s] + INC;
    uint32_t over = __ballot_sync(FULL, v > LIMIT);
    if (over) {
        __syncwarp();
        uint4 q = reinterpret_cast<uint4*>(row)[t];
        uint16_t* pq = reinterpret_cast<uint16_t*>(&q);
        #pragma unroll
        for (int k = 0; k < 8; ++k) pq[k] >>= 1;
        if ((s >> 3) == (uint32_t)t) pq[s & 7] = (uint16_t)(pq[s & 7] + INC / 2);
        reinterpret_cast<uint4*>(row)[t] = q;
    } else if ((s >> 3) == (uint32_t)t) {
        row[s] = (uint16_t)v;
    }
    __syncwarp();
}

// ---------------------------------------------------------------- encode
// out: per-lane region of `cap` words; frame words are emitted forward into a scratch
// region then copied reversed (decoder order), matching the CPU reference.
__global__ void encode_kernel(const uint8_t* __restrict__ data, uint64_t data_off, uint64_t total, uint32_t lane_len, uint32_t n_lanes,
                              uint32_t lane_base, uint32_t batch,
                              uint16_t* cnt1_all, uint16_t* cnt2_all, uint2* cf_all, uint16_t* tmp_all,
                              uint16_t* out_all, uint64_t cap, uint32_t* counts) {
    const int t = threadIdx.x & 31;
    const uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // index within batch
    const uint32_t lane = lane_base + j;
    if (j >= batch || lane >= n_lanes) return;
    const uint64_t start = (uint64_t)lane * lane_len;
    const uint64_t len = min((uint64_t)lane_len, total - start);
    const uint8_t* in = data + (start - data_off);
    uint16_t* cnt1 = cnt1_all + (uint64_t)j * CNT1_SIZE;
    uint16_t* cnt2 = cnt2_all + (uint64_t)j * CNT2_SIZE;
    uint2* cf = cf_all + (uint64_t)j * FRAME;
    uint16_t* tmp = tmp_all + (uint64_t)j * (FRAME + 2);
    uint16_t* out = out_all + (uint64_t)j * cap;
    uint64_t opos = 0;
    uint32_t c1 = 0, c2 = 0;
    Dist d;
    for (uint64_t fs = 0; fs < len; fs += FRAME) {
        const uint32_t fl = (uint32_t)min((uint64_t)FRAME, len - fs);
        for (uint32_t i = 0; i < fl; ++i) {
            const uint32_t b = in[fs + i];
            uint16_t* row1 = cnt1 + c1 * 256;
            uint16_t* row2 = cnt2 + o2_hash(c2, c1) * 256;
            distribution(row1, row2, t, d);
            if ((b >> 3) == (uint32_t)t) {
                uint32_t cum = d.base;
                for (uint32_t k = 0; k < (b & 7); ++k) cum += d.f[k];
                cf[i] = make_uint2(cum, d.f[b & 7]);
            }
            bump_row(row1, b, t);
            bump_row(row2, b, t);
            c2 = c1; c1 = b;
        }
        __syncwarp();
        // rANS backwards over the frame (single thread; cheap next to the model pass)
        uint32_t nw = 0;
        if (t == 0) {
            uint32_t x = RANS_L;
            for (int64_t i = (int64_t)fl - 1; i >= 0; --i) {
                const uint2 e = cf[i];
                if (x >= (e.y << 16)) { tmp[nw++] = (uint16_t)x; x >>= 16; }
                x = ((x / e.y) << SCALE_BITS) + (x % e.y) + e.x;
            }
            tmp[nw++] = (uint16_t)x;
            tmp[nw++] = (uint16_t)(x >> 16);
        }
        nw = __shfl_sync(FULL, nw, 0);
        __syncwarp();
        for (uint32_t k = t; k < nw; k += 32) out[opos + k] = tmp[nw - 1 - k];
        opos += nw;
        __syncwarp();
    }
    if (t == 0) counts[j] = (uint32_t)opos;
}

// ---------------------------------------------------------------- decode
// words/offs/out_all/status are the current batch's (offs rebased to its first word, out_all
// starting at its first lane).
__global__ void decode_kernel(const uint16_t* __restrict__ words, const uint64_t* __restrict__ offs, uint8_t* out_all,
                              uint64_t total, uint32_t lane_len, uint32_t n_lanes, uint32_t lane_base, uint32_t batch,
                              uint16_t* cnt1_all, uint16_t* cnt2_all, int* status) {
    const int t = threadIdx.x & 31;
    const uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = lane_base + j;
    if (j >= batch || lane >= n_lanes) return;
    const uint64_t start = (uint64_t)lane * lane_len;
    const uint64_t len = min((uint64_t)lane_len, total - start);
    uint8_t* out = out_all + (start - (uint64_t)lane_base * lane_len);
    const uint16_t* w = words + offs[j];
    const uint64_t nwords = offs[j + 1] - offs[j];
    uint16_t* cnt1 = cnt1_all + (uint64_t)j * CNT1_SIZE;
    uint16_t* cnt2 = cnt2_all + (uint64_t)j * CNT2_SIZE;
    uint64_t pos = 0;
    uint32_t c1 = 0, c2 = 0;
    int err = 0;
    Dist d;
    for (uint64_t fs = 0; fs < len && !err; fs += FRAME) {
        const uint32_t fl = (uint32_t)min((uint64_t)FRAME, len - fs);
        if (pos + 2 > nwords) { err = 1; break; }
        uint32_t x = ((uint32_t)w[pos] << 16) | w[pos + 1];
        pos += 2;
        for (uint32_t i = 0; i < fl; ++i) {
            uint16_t* row1 = cnt1 + c1 * 256;
            uint16_t* row2 = cnt2 + o2_hash(c2, c1) * 256;
            distribution(row1, row2, t, d);
            const uint32_t slot = x & (TOTAL - 1);
            uint32_t loc = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) loc += d.f[k];
            const uint32_t mine = __ballot_sync(FULL, slot >= d.base && slot < d.base + loc);
            const int owner = __ffs(mine) - 1;
            uint32_t s = 0, cum = 0, freq = 0;
            if (t == owner) {
                cum = d.base;
                int k = 0;
                while (cum + d.f[k] <= slot) { cum += d.f[k]; ++k; }
                s = (uint32_t)(t * 8 + k);
                freq = d.f[k];
            }
            s = __shfl_sync(FULL, s, owner);
            cum = __shfl_sync(FULL, cum, owner);
            freq = __shfl_sync(FULL, freq, owner);
            x = freq * (x >> SCALE_BITS) + slot - cum;
            if (x < RANS_L) {
                if (pos >= nwords) { err = 1; break; }
                x = (x << 16) | w[pos++];
            }
            if (t == 0) out[fs + i] = (uint8_t)s;
            bump_row(row1, s, t);
            bump_row(row2, s, t);
            c2 = c1; c1 = s;
        }
        if (!err && x != RANS_L) err = 2;
    }
    if (!err && pos != nwords) err = 3;
    if (t == 0) status[j] = err;
}

// ---------------------------------------------------------------- host API
static int check(cudaError_t e) { return e == cudaSuccess ? 0 : (int)e; }
#define CK(x) do { int _e = check(x); if (_e) { rc = _e; goto done; } } while (0)

static const int WARPS_PER_BLOCK = 4;
// Upper bound on lanes per launch; the actual batch is sized from free GPU memory.
static const uint32_t MAX_BATCH = 16384;
// GPU memory left untouched for the display, other processes and driver overhead.
static const size_t VRAM_RESERVE = 512u << 20;

// Lanes per batch that fit in free GPU memory, given the per-lane and fixed byte costs.
static uint32_t batch_lanes(uint32_t n_lanes, size_t per_lane, size_t fixed) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    // NZIP_VRAM_LIMIT (bytes) pretends the card has at most that much free memory, to
    // test small-GPU behaviour on a big one.
    if (const char* lim = getenv("NZIP_VRAM_LIMIT")) {
        // Parsed by hand: strtoull would bind to glibc 2.38's __isoc23_strtoull and stop
        // the binary from running on older distributions.
        unsigned long long l = 0;
        for (const char* c = lim; *c >= '0' && *c <= '9'; ++c) l = l * 10 + (unsigned long long)(*c - '0');
        if (l > 0 && l < free_b) free_b = (size_t)l;
    }
    size_t usable = free_b > VRAM_RESERVE + fixed ? free_b - VRAM_RESERVE - fixed : 0;
    size_t nb = usable / per_lane;
    if (nb > MAX_BATCH) nb = MAX_BATCH;
    if (nb > n_lanes) nb = n_lanes;
    return (uint32_t)nb;
}

extern "C" int ncore_gpu_available(void) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}


// Gather each lane's words (lane i: `counts[i]` words at out_all + i*cap) into one
// contiguous buffer at the exclusive prefix offsets `poff`. One block per lane.
__global__ void pack_kernel(const uint16_t* __restrict__ out_all, uint64_t cap, const uint32_t* __restrict__ counts,
                            const uint64_t* __restrict__ poff, uint16_t* __restrict__ packed) {
    const uint32_t i = blockIdx.x;
    const uint16_t* src = out_all + (uint64_t)i * cap;
    uint16_t* dst = packed + poff[i];
    for (uint32_t k = threadIdx.x; k < counts[i]; k += blockDim.x) dst[k] = src[k];
}

// Encode `total` bytes. Each lane's words go, in lane order and already packed, into
// the host buffer `packed` (capacity `packed_cap` words); `counts` gets each lane's word
// count and `*packed_words` the total. Only the packed words cross PCIe, in one copy
// per batch.
extern "C" int ncore_gpu_encode(const uint8_t* data, uint64_t total, uint32_t lane_len, uint64_t cap, uint32_t* counts,
                                uint16_t* packed, uint64_t packed_cap, uint64_t* packed_words, float* kernel_ms) {
    int rc = 0;
    const uint32_t n_lanes = (uint32_t)((total + lane_len - 1) / lane_len);
    // Per lane: counters, (cum,freq) frame buffer, frame scratch, output + packed words,
    // the lane's input bytes, count and offset.
    const size_t per_lane = (size_t)CNT1_SIZE * 2 + (size_t)CNT2_SIZE * 2 + (size_t)FRAME * sizeof(uint2)
                          + (size_t)(FRAME + 2) * 2 + 2 * (size_t)cap * 2 + lane_len + 4 + 8;
    const uint32_t nb = n_lanes ? batch_lanes(n_lanes, per_lane, 1u << 20) : 0;
    uint8_t* d_data = nullptr; uint16_t *d_c1 = nullptr, *d_c2 = nullptr, *d_tmp = nullptr, *d_out = nullptr, *d_packed = nullptr;
    uint2* d_cf = nullptr; uint32_t* d_counts = nullptr; uint64_t* d_poff = nullptr;
    uint64_t* poff = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    float total_ms = 0;
    uint64_t written = 0;
    *packed_words = 0;
    if (n_lanes == 0) return 0;
    if (nb == 0) return -4;  // not enough GPU memory for even one lane: caller uses the CPU
    poff = new uint64_t[nb + 1];
    CK(cudaMalloc(&d_data, (size_t)nb * lane_len));
    CK(cudaMalloc(&d_c1, (size_t)nb * CNT1_SIZE * 2));
    CK(cudaMalloc(&d_c2, (size_t)nb * CNT2_SIZE * 2));
    CK(cudaMalloc(&d_cf, (size_t)nb * FRAME * sizeof(uint2)));
    CK(cudaMalloc(&d_tmp, (size_t)nb * (FRAME + 2) * 2));
    CK(cudaMalloc(&d_out, (size_t)nb * cap * 2));
    CK(cudaMalloc(&d_packed, (size_t)nb * cap * 2));
    CK(cudaMalloc(&d_counts, (size_t)nb * 4));
    CK(cudaMalloc(&d_poff, (size_t)(nb + 1) * 8));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (uint32_t base = 0; base < n_lanes; base += nb) {
        const uint32_t batch = (n_lanes - base) < nb ? (n_lanes - base) : nb;
        const uint64_t d0 = (uint64_t)base * lane_len;
        const uint64_t d1 = (uint64_t)(base + batch) * lane_len < total ? (uint64_t)(base + batch) * lane_len : total;
        CK(cudaMemcpy(d_data, data + d0, d1 - d0, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_c1, 0, (size_t)batch * CNT1_SIZE * 2));
        CK(cudaMemset(d_c2, 0, (size_t)batch * CNT2_SIZE * 2));
        const int threads = 32 * WARPS_PER_BLOCK;
        const int blocks = (batch + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        cudaEventRecord(e0);
        encode_kernel<<<blocks, threads>>>(d_data, d0, total, lane_len, n_lanes, base, batch, d_c1, d_c2, d_cf, d_tmp, d_out, cap, d_counts);
        cudaEventRecord(e1);
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); total_ms += ms;
        CK(cudaMemcpy(counts + base, d_counts, (size_t)batch * 4, cudaMemcpyDeviceToHost));
        poff[0] = 0;
        for (uint32_t i = 0; i < batch; ++i) {
            if (counts[base + i] > cap) { rc = -2; goto done; }
            poff[i + 1] = poff[i] + counts[base + i];
        }
        if (written + poff[batch] > packed_cap) { rc = -3; goto done; }
        CK(cudaMemcpy(d_poff, poff, (size_t)(batch + 1) * 8, cudaMemcpyHostToDevice));
        pack_kernel<<<batch, 256>>>(d_out, cap, d_counts, d_poff, d_packed);
        CK(cudaGetLastError());
        CK(cudaMemcpy(packed + written, d_packed, (size_t)poff[batch] * 2, cudaMemcpyDeviceToHost));
        written += poff[batch];
    }
    *packed_words = written;
    if (kernel_ms) *kernel_ms = total_ms;
done:
    cudaFree(d_data); cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_cf); cudaFree(d_tmp); cudaFree(d_out);
    cudaFree(d_packed); cudaFree(d_counts); cudaFree(d_poff);
    delete[] poff;
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    return rc;
}

// Decode straight into the caller's `out` (exactly `total` bytes). `words` is the
// stream's packed little-endian word area (any alignment); `offs` has n_lanes + 1
// word offsets. `status` gets one code per lane.
extern "C" int ncore_gpu_decode(const void* words, uint64_t n_words, const uint64_t* offs, uint8_t* out, uint64_t total,
                                uint32_t lane_len, int* status, float* kernel_ms) {
    int rc = 0;
    const uint32_t n_lanes = (uint32_t)((total + lane_len - 1) / lane_len);
    // Per lane: counters, output bytes, worst-case words, offset and status. Words and
    // output move per batch, so GPU memory stays bounded for any frame size.
    const uint64_t max_lane_words = (uint64_t)lane_len + 2 * ((lane_len + FRAME - 1) / FRAME) + 2;
    const size_t per_lane = (size_t)CNT1_SIZE * 2 + (size_t)CNT2_SIZE * 2 + lane_len + (size_t)max_lane_words * 2 + 8 + 4;
    const uint32_t nb = n_lanes ? batch_lanes(n_lanes, per_lane, 1u << 20) : 0;
    uint16_t *d_w = nullptr, *d_c1 = nullptr, *d_c2 = nullptr; uint64_t* d_offs = nullptr; uint8_t* d_out = nullptr;
    int* d_status = nullptr;
    uint64_t* boffs = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    float total_ms = 0;
    const uint8_t* wbytes = (const uint8_t*)words;
    if (n_lanes == 0) return 0;
    if (nb == 0) return -4;
    boffs = new uint64_t[nb + 1];
    CK(cudaMalloc(&d_w, (size_t)nb * max_lane_words * 2));
    CK(cudaMalloc(&d_offs, (size_t)(nb + 1) * 8));
    CK(cudaMalloc(&d_out, (size_t)nb * lane_len));
    CK(cudaMalloc(&d_c1, (size_t)nb * CNT1_SIZE * 2));
    CK(cudaMalloc(&d_c2, (size_t)nb * CNT2_SIZE * 2));
    CK(cudaMalloc(&d_status, (size_t)nb * 4));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (uint32_t base = 0; base < n_lanes; base += nb) {
        const uint32_t batch = (n_lanes - base) < nb ? (n_lanes - base) : nb;
        const uint64_t w0 = offs[base], w1 = offs[base + batch];
        if (w1 < w0 || w1 > n_words) { rc = -2; goto done; }
        for (uint32_t i = 0; i <= batch; ++i) {
            boffs[i] = offs[base + i] - w0;
            if (i && (boffs[i] < boffs[i - 1] || boffs[i] - boffs[i - 1] > max_lane_words)) { rc = -2; goto done; }
        }
        if (w1 > w0) CK(cudaMemcpy(d_w, wbytes + w0 * 2, (size_t)(w1 - w0) * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_offs, boffs, (size_t)(batch + 1) * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_c1, 0, (size_t)batch * CNT1_SIZE * 2));
        CK(cudaMemset(d_c2, 0, (size_t)batch * CNT2_SIZE * 2));
        const int threads = 32 * WARPS_PER_BLOCK;
        const int blocks = (batch + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        cudaEventRecord(e0);
        decode_kernel<<<blocks, threads>>>(d_w, d_offs, d_out, total, lane_len, n_lanes, base, batch, d_c1, d_c2, d_status);
        cudaEventRecord(e1);
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); total_ms += ms;
        const uint64_t o0 = (uint64_t)base * lane_len;
        const uint64_t o1 = (uint64_t)(base + batch) * lane_len < total ? (uint64_t)(base + batch) * lane_len : total;
        CK(cudaMemcpy(status + base, d_status, (size_t)batch * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out + o0, d_out, o1 - o0, cudaMemcpyDeviceToHost));
    }
    if (kernel_ms) *kernel_ms = total_ms;
done:
    cudaFree(d_w); cudaFree(d_offs); cudaFree(d_out); cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_status);
    delete[] boffs;
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    return rc;
}

// Pay CUDA's one-time start-up costs now so a caller can overlap them with CPU work:
// driver + context creation, and loading the kernels (CUDA loads each kernel lazily at
// its first launch; cudaFuncGetAttributes forces the load). Returns 0 on success.
extern "C" int ncore_gpu_warmup(void) {
    cudaFuncAttributes a;
    if (cudaFree(0) != cudaSuccess) return 1;
    if (cudaFuncGetAttributes(&a, encode_kernel) != cudaSuccess) return 2;
    if (cudaFuncGetAttributes(&a, decode_kernel) != cudaSuccess) return 3;
    return 0;
}
