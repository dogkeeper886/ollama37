#include "mmid-k80.cuh"

// A block computes MMID_K80_ROWS output rows for up to MMID_K80_SLOTS (token, expert slot) pairs routed to
// one expert. Each step decodes one 32-value slice of its rows into shared memory and reuses it for every slot;
// each warp then multiplies a 32-row x 16-slot tile, 4 x 4 outputs per thread.
#define MMID_K80_ROWS    128
#define MMID_K80_SLOTS   16
#define MMID_K80_SLICE   32
#define MMID_K80_THREADS 128
#define MMID_K80_PLAN_THREADS 256

struct mmid_k80_tile { int expert, first, n; };

static __global__ void mmid_k80_count(const char * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1,
        const int n_used, const int n_tokens, int * __restrict__ counts) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_used*n_tokens) {
        return;
    }
    const int t = i / n_used, k = i % n_used;
    atomicAdd(&counts[*(const int32_t *) (ids + t*ids_nb1 + k*ids_nb0)], 1);
}

// One block: each expert's first slot and tiles, in expert order.
static __global__ void mmid_k80_plan(const int * __restrict__ counts, const int n_experts, int * __restrict__ cursor,
        mmid_k80_tile * __restrict__ tiles, int * __restrict__ n_tiles) {
    __shared__ int first_slot;
    __shared__ int first_tile;
    if (threadIdx.x == 0) {
        first_slot = 0;
        first_tile = 0;
    }
    __syncthreads();
    for (int e0 = 0; e0 < n_experts; e0 += MMID_K80_PLAN_THREADS) {
        const int e = e0 + threadIdx.x;
        const int c = e < n_experts ? counts[e] : 0;
        // serial scan by thread 0: n_experts is small (≤ a few hundred)
        __shared__ int slot_off[MMID_K80_PLAN_THREADS];
        __shared__ int tile_off[MMID_K80_PLAN_THREADS];
        if (threadIdx.x == 0) {
            int s = first_slot, t = first_tile;
            for (int j = 0; j < MMID_K80_PLAN_THREADS && e0 + j < n_experts; ++j) {
                const int cj = counts[e0 + j];
                slot_off[j] = s;
                tile_off[j] = t;
                s += cj;
                t += (cj + MMID_K80_SLOTS - 1) / MMID_K80_SLOTS;
            }
            first_slot = s;
            first_tile = t;
        }
        __syncthreads();
        if (e < n_experts) {
            cursor[e] = slot_off[threadIdx.x];
            for (int i = 0; i*MMID_K80_SLOTS < c; ++i) {
                tiles[tile_off[threadIdx.x] + i] = { e, slot_off[threadIdx.x] + i*MMID_K80_SLOTS, min(MMID_K80_SLOTS, c - i*MMID_K80_SLOTS) };
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *n_tiles = first_tile;
    }
}

static __global__ void mmid_k80_scatter(const char * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1,
        const int n_used, const int n_tokens, int * __restrict__ cursor, int * __restrict__ slots) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_used*n_tokens) {
        return;
    }
    const int t = i / n_used, k = i % n_used;
    const int e = *(const int32_t *) (ids + t*ids_nb1 + k*ids_nb0);
    slots[atomicAdd(&cursor[e], 1)] = i; // i = t*n_used + k
}

static __device__ __forceinline__ float mmid_k80_magic(const uint32_t bits) {
    return __int_as_float(0x4B000000u | bits) - 8388608.0f; // exact for bits < 2^23, no int->float conversion
}

// The raw bits of one 32-value slice of one row, loaded a slice ahead of their decode to hide load latency.
template <ggml_type type> struct mmid_k80_raw;

template <> struct mmid_k80_raw<GGML_TYPE_Q4_K> {
    uint32_t q[8];   // 32 bytes of quants; the slice is the low or high nibbles
    half2    dm;
    uint8_t  sc, m;  // the slice's 6-bit scale and min
    int      shift;

    __device__ __forceinline__ void load(const char * __restrict__ row, const int j) {
        const block_q4_K * b = (const block_q4_K *) row + j/8;
        const int sb = j % 8;
        const uint8_t * s = b->scales;
        if (sb < 4) {
            sc = s[sb] & 63;
            m  = s[sb + 4] & 63;
        } else {
            sc = (s[sb + 4] & 0xF) | ((s[sb - 4] >> 6) << 4);
            m  = (s[sb + 4] >>  4) | ((s[sb]     >> 6) << 4);
        }
        dm = b->dm;
        const uint32_t * qs = (const uint32_t *) (b->qs + (sb/2)*32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = qs[i];
        }
        shift = (sb % 2)*4;
    }

    __device__ __forceinline__ void decode(float * w) const {
        const float2 f = __half22float2(dm);
        const float d = f.x*sc, mn = f.y*m;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t v = (q[i] >> shift) & 0x0F0F0F0Fu;
            w[4*i + 0] = fmaf(d, mmid_k80_magic( v        & 0xFF), -mn);
            w[4*i + 1] = fmaf(d, mmid_k80_magic((v >>  8) & 0xFF), -mn);
            w[4*i + 2] = fmaf(d, mmid_k80_magic((v >> 16) & 0xFF), -mn);
            w[4*i + 3] = fmaf(d, mmid_k80_magic( v >> 24),         -mn);
        }
    }
};

template <> struct mmid_k80_raw<GGML_TYPE_Q6_K> {
    uint16_t ql[16], qh[16]; // 32 bytes each; block_q6_K is 2-byte aligned
    half     d;
    int8_t   sc0, sc1;       // scales of the slice's two 16-value halves
    int      lshift, hshift;

    __device__ __forceinline__ void load(const char * __restrict__ row, const int j) {
        const block_q6_K * b = (const block_q6_K *) row + j/8;
        const int sb = j % 8, n = sb / 4, qq = sb % 4;
        const uint16_t * l = (const uint16_t *) (b->ql + n*64 + (qq % 2)*32);
        const uint16_t * h = (const uint16_t *) (b->qh + n*32);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ql[i] = l[i];
            qh[i] = h[i];
        }
        d   = b->d;
        sc0 = b->scales[n*8 + 2*qq];
        sc1 = b->scales[n*8 + 2*qq + 1];
        lshift = (qq / 2)*4;
        hshift = 2*qq;
    }

    __device__ __forceinline__ void decode(float * w) const {
        const float df = __half2float(d);
        const float d0 = df*sc0, d1 = df*sc1;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint32_t lo = (ql[i/2] >> (8*(i % 2))) & 0xFF;
            const uint32_t hi = (qh[i/2] >> (8*(i % 2))) & 0xFF;
            const uint32_t v  = ((lo >> lshift) & 0xF) | (((hi >> hshift) & 3) << 4);
            w[i] = (i < 16 ? d0 : d1) * (mmid_k80_magic(v) - 32.0f);
        }
    }
};

template <ggml_type type>
static __global__ void __launch_bounds__(MMID_K80_THREADS) mmid_k80_gemm(
        const char * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int * __restrict__ slots, const mmid_k80_tile * __restrict__ tiles, const int * __restrict__ n_tiles,
        const int K, const int64_t row_nb, const int64_t expert_nb, const int n_used, const int ne11,
        const int64_t y_nb1, const int64_t y_nb2, const int64_t dst_nb1, const int64_t dst_nb2) {
    if ((int) blockIdx.y >= *n_tiles) {
        return;
    }
    const mmid_k80_tile tile = tiles[blockIdx.y];
    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x*MMID_K80_ROWS;

    __shared__ float ws[MMID_K80_SLICE][MMID_K80_ROWS];
    __shared__ float xs[MMID_K80_SLICE][MMID_K80_SLOTS];

    // decode: thread -> one row's whole slice; load: thread -> (slot, 4 values)
    const char * wrow = x + tile.expert*expert_nb + (row0 + tid)*row_nb;
    const int ls = tid / 8, lk = (tid % 8)*4;
    const float * yrow = nullptr;
    if (ls < tile.n) {
        const int s = slots[tile.first + ls];
        yrow = y + (s / n_used)*y_nb2 + ((s % n_used) % ne11)*y_nb1;
    }
    // compute: each warp a 32-row x 16-slot tile, each thread 4 rows x 4 slots
    const int lane = tid % WARP_SIZE;
    const int cr = (tid / WARP_SIZE)*32 + (lane % 8)*4, cs = (lane / 8)*4;
    float acc[4][4] = {};

    mmid_k80_raw<type> raw;
    raw.load(wrow, 0);
    float4 xv = yrow ? *(const float4 *) (yrow + lk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int kb = 0; kb < K; kb += MMID_K80_SLICE) {
        float w[MMID_K80_SLICE];
        raw.decode(w);
#pragma unroll
        for (int i = 0; i < MMID_K80_SLICE; ++i) {
            ws[i][tid] = w[i];
        }
        xs[lk + 0][ls] = xv.x;
        xs[lk + 1][ls] = xv.y;
        xs[lk + 2][ls] = xv.z;
        xs[lk + 3][ls] = xv.w;
        __syncthreads();
        if (kb + MMID_K80_SLICE < K) { // the next slice's loads overlap this slice's FMAs
            raw.load(wrow, kb/MMID_K80_SLICE + 1);
            xv = yrow ? *(const float4 *) (yrow + kb + MMID_K80_SLICE + lk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int k = 0; k < MMID_K80_SLICE; ++k) {
            const float4 a = *(const float4 *) &ws[k][cr];
            const float4 c = *(const float4 *) &xs[k][cs];
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float cv[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(av[i], cv[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (cs + j < tile.n) {
            const int s = slots[tile.first + cs + j];
            float * d = dst + (s / n_used)*dst_nb2 + (s % n_used)*dst_nb1 + row0 + cr;
            *(float4 *) d = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
        }
    }
}

bool ggml_cuda_should_use_mmid_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500 // Kepler
        && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32
        && dst->ne[2] >= (src0->type == GGML_TYPE_Q4_K ? 64 : 128) // below this ggml's per-expert mat-vecs win
        && src0->ne[3] == 1 && src1->ne[3] == 1
        && src0->ne[0] % QK_K == 0 && src0->ne[1] % MMID_K80_ROWS == 0
        && (src1->ne[1] == 1 || src1->ne[1] == ids->ne[0])
        && ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && (uintptr_t) dst->data % 16 == 0 // dst is written as float4
        && src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && (uintptr_t) src1->data % 16 == 0
        && (src0->type != GGML_TYPE_Q4_K || (src0->nb[1] % 4 == 0 && (uintptr_t) src0->data % 4 == 0)); // q4_K reads quants as uint32
}

void ggml_cuda_mul_mat_id_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    const int n_experts = src0->ne[2];
    const int n_used    = ids->ne[0];
    const int n_tokens  = ids->ne[1];
    const int n_slots   = n_used*n_tokens;
    const int max_tiles = (n_slots + MMID_K80_SLOTS - 1)/MMID_K80_SLOTS + n_experts;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int> counts(ctx.pool(), 2*n_experts + 1 + n_slots);
    ggml_cuda_pool_alloc<mmid_k80_tile> tiles(ctx.pool(), max_tiles);
    int * cursor  = counts.get() + n_experts;
    int * n_tiles = cursor + n_experts;
    int * slots   = n_tiles + 1;

    CUDA_CHECK(cudaMemsetAsync(counts.get(), 0, n_experts*sizeof(int), stream));
    const int nb = (n_slots + 255)/256;
    mmid_k80_count<<<nb, 256, 0, stream>>>((const char *) ids->data, ids->nb[0], ids->nb[1], n_used, n_tokens, counts.get());
    mmid_k80_plan<<<1, MMID_K80_PLAN_THREADS, 0, stream>>>(counts.get(), n_experts, cursor, tiles.get(), n_tiles);
    mmid_k80_scatter<<<nb, 256, 0, stream>>>((const char *) ids->data, ids->nb[0], ids->nb[1], n_used, n_tokens, cursor, slots);

    const dim3 grid(src0->ne[1]/MMID_K80_ROWS, max_tiles);
    const int64_t fs = sizeof(float);
    if (src0->type == GGML_TYPE_Q4_K) {
        mmid_k80_gemm<GGML_TYPE_Q4_K><<<grid, MMID_K80_THREADS, 0, stream>>>((const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, slots, tiles.get(), n_tiles, src0->ne[0], src0->nb[1], src0->nb[2], n_used, src1->ne[1],
            src1->nb[1]/fs, src1->nb[2]/fs, dst->nb[1]/fs, dst->nb[2]/fs);
    } else {
        mmid_k80_gemm<GGML_TYPE_Q6_K><<<grid, MMID_K80_THREADS, 0, stream>>>((const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, slots, tiles.get(), n_tiles, src0->ne[0], src0->nb[1], src0->nb[2], n_used, src1->ne[1],
            src1->nb[1]/fs, src1->nb[2]/fs, dst->nb[1]/fs, dst->nb[2]/fs);
    }
    CUDA_CHECK(cudaGetLastError());
}
