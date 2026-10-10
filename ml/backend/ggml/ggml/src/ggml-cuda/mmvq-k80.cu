#include "mmvq-k80.cuh"

static constexpr int MMVQ_K80_ROWS  = 4; // rows per warp: the warp loads src1 once for all of them
static constexpr int MMVQ_K80_WARPS = 4;

// Expert routing for MUL_MAT_ID (#572): block y is slot c; it reads expert ids[c] of src0, prepared input
// c % ny, and writes dst block c. A plain mat-vec passes ids = nullptr and a single slot.
struct mmvq_k80_channels {
    const int32_t * ids;
    size_t  x_stride;   // bytes between src0 experts
    int     ny;         // prepared inputs; 1 when every slot shares one
    int64_t y_stride;   // floats between prepared inputs
    int64_t dst_stride; // floats between dst slots
};

// ys[i] = y[i]. ysum16[g] = sum of y over the 16 values of group g, for the q4_K min term; skipped when
// ysum16 is null.
// Block y prepares input c: y + c*y_stride into ys + c*k (and ysum16 + c*k/16).
static __global__ void mmvq_k80_prep(const float * __restrict__ y, const int64_t y_stride, float * __restrict__ ys,
        float * __restrict__ ysum16, const int k) {
    const int c = blockIdx.y;
    y  += c*y_stride;
    ys += (int64_t) c*k;
    if (ysum16) {
        ysum16 += (int64_t) c*(k/16);
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < k ? y[i] : 0.0f;
    if (i < k) {
        ys[i] = v;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, o, 16);
    }
    if (ysum16 && (i & 15) == 0 && i < k) {
        ysum16[i >> 4] = v;
    }
}

// An integer below 2^23 placed in the mantissa of 2^23, minus 2^23: exact, with no int-to-float conversion.
static __device__ __forceinline__ float mmvq_k80_magic(const uint32_t bits) {
    return __int_as_float(0x4B000000u | bits) - 8388608.0f;
}

// Byte k of q placed in the mantissa of 2^23 by one PRMT (bytes: q.k, 0, 0, 0x4B), as a float: 2^23 + byte (#584).
// It replaces a mask and an OR per value.
template <int k>
static __device__ __forceinline__ float mmvq_k80_byte(const uint32_t q) {
    return __int_as_float(__byte_perm(q, 0x4B000000u, 0x7440 + k));
}

// One warp computes MMVQ_K80_ROWS rows. Lane t reads 16 qs bytes of block (b0 + t/8): chunk j = (t%8)/2,
// offset l = 16*(t%2). Low nibbles are values 64j+l.., sub-block 2j; high nibbles are 64j+32+l.., sub-block 2j+1.
static __global__ void __launch_bounds__(WARP_SIZE*MMVQ_K80_WARPS) mmvq_k80_q4_K(
        const char * __restrict__ x, const float * __restrict__ ys, const float * __restrict__ ysum16,
        float * __restrict__ dst, const int ncols, const int nrows, const size_t row_bytes, const mmvq_k80_channels ch) {
    const int c = blockIdx.y;
    x      += (size_t) (ch.ids ? ch.ids[c] : 0) * ch.x_stride;
    ys     += (c % ch.ny) * ch.y_stride;
    ysum16 += (c % ch.ny) * (ch.y_stride / 16);
    dst    += c * ch.dst_stride;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*MMVQ_K80_WARPS + threadIdx.y) * MMVQ_K80_ROWS;
    if (row0 >= nrows) {
        return;
    }
    const int nb   = ncols / QK_K;
    const int bo   = lane >> 3;
    const int j    = (lane & 7) >> 1;
    const int l    = (lane & 1) * 16;
    const int qoff = offsetof(block_q4_K, qs) + 32*j + l;
    const bool upper = j >= 2;    // sub-blocks 4..7 split their 6-bit scale and min across bytes
    const int  sh    = 16*(j & 1); // after the shift, byte 0 is sub-block 2j and byte 1 is 2j+1

    float acc[MMVQ_K80_ROWS] = {0.0f};

    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + bo;
        if (b >= nb) {
            break;
        }
        const int e = b*QK_K + 64*j + l;
        float4 yl[4], yh[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            yl[w] = __ldg((const float4 *) (ys + e) + w);
            yh[w] = __ldg((const float4 *) (ys + e + 32) + w);
        }
        const float sl = __ldg(ysum16 + (e >> 4));
        const float sh_sum = __ldg(ysum16 + ((e + 32) >> 4));

        uint4 hd[MMVQ_K80_ROWS], qv[MMVQ_K80_ROWS];
#pragma unroll
        for (int r = 0; r < MMVQ_K80_ROWS; ++r) {
            const char * bp = x + (size_t) min(row0 + r, nrows - 1)*row_bytes + (size_t) b*sizeof(block_q4_K);
            hd[r] = __ldg((const uint4 *) bp);
            qv[r] = __ldg((const uint4 *) (bp + qoff));
        }

#pragma unroll
        for (int r = 0; r < MMVQ_K80_ROWS; ++r) {
            const float d    = __half2float(__ushort_as_half((unsigned short) (hd[r].x & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) (hd[r].x >> 16)));
            const uint32_t a = hd[r].y >> sh, m = hd[r].z >> sh, c = hd[r].w >> sh;
            const uint32_t scw = upper ? (c & 0x0F0Fu) | ((a >> 2) & 0x3030u) : a & 0x3F3Fu;
            const uint32_t mw  = upper ? ((c >> 4) & 0x0F0Fu) | ((m >> 2) & 0x3030u) : m & 0x3F3Fu;

            const uint32_t q[4] = {qv[r].x, qv[r].y, qv[r].z, qv[r].w};
            float al = 0.0f, ah = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t lo = q[w] & 0x0F0F0F0Fu, hi = (q[w] >> 4) & 0x0F0F0F0Fu;
                al = fmaf(mmvq_k80_byte<0>(lo) - 8388608.0f, yl[w].x, al);
                al = fmaf(mmvq_k80_byte<1>(lo) - 8388608.0f, yl[w].y, al);
                al = fmaf(mmvq_k80_byte<2>(lo) - 8388608.0f, yl[w].z, al);
                al = fmaf(mmvq_k80_byte<3>(lo) - 8388608.0f, yl[w].w, al);
                ah = fmaf(mmvq_k80_byte<0>(hi) - 8388608.0f, yh[w].x, ah);
                ah = fmaf(mmvq_k80_byte<1>(hi) - 8388608.0f, yh[w].y, ah);
                ah = fmaf(mmvq_k80_byte<2>(hi) - 8388608.0f, yh[w].z, ah);
                ah = fmaf(mmvq_k80_byte<3>(hi) - 8388608.0f, yh[w].w, ah);
            }
            const float dot  = mmvq_k80_magic(scw & 0xFFu)*al + mmvq_k80_magic(scw >> 8)*ah;
            const float mins = mmvq_k80_magic(mw  & 0xFFu)*sl + mmvq_k80_magic(mw  >> 8)*sh_sum;
            acc[r] += d*dot - dmin*mins;
        }
    }

#pragma unroll
    for (int r = 0; r < MMVQ_K80_ROWS; ++r) {
        const float v = warp_reduce_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = v;
        }
    }
}

static constexpr int MMVQ_K80_Q6_K_ROWS = 2; // q6_K is arithmetic-bound here; 2 rows keeps registers low (#567)

// 4 bytes from a 2-byte-aligned address: a q6_K block is 210 bytes, so every other block is only 2-byte aligned.
static __device__ __forceinline__ uint32_t mmvq_k80_ld4(const char * p) {
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) __ldg(q) | ((uint32_t) __ldg(q + 1) << 16);
}

// One warp computes MMVQ_K80_Q6_K_ROWS rows, 2 blocks per step. Lane t decodes, in block (b0 + t/16) and
// half h = (t/8)%2, the values h*128 + g*32 + l0 + k for g, k in 0..3 with l0 = 4*(t%8): their low 4 bits
// come from ql and their high 2 bits from qh, rebuilt as the four bytes of q[g]. The "- 32" of each value
// folds into its byte's decode constant.
static __global__ void __launch_bounds__(WARP_SIZE*MMVQ_K80_WARPS) mmvq_k80_q6_K(
        const char * __restrict__ x, const float * __restrict__ ys,
        float * __restrict__ dst, const int ncols, const int nrows, const size_t row_bytes, const mmvq_k80_channels ch) {
    const int c = blockIdx.y;
    x   += (size_t) (ch.ids ? ch.ids[c] : 0) * ch.x_stride;
    ys  += (c % ch.ny) * ch.y_stride;
    dst += c * ch.dst_stride;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*MMVQ_K80_WARPS + threadIdx.y) * MMVQ_K80_Q6_K_ROWS;
    if (row0 >= nrows) {
        return;
    }
    const int nb = ncols / QK_K;
    const int bo = lane >> 4;
    const int h  = (lane >> 3) & 1;
    const int l0 = 4 * (lane & 7);
    const int is = l0 >> 4;
    const int ql_off = offsetof(block_q6_K, ql) + 64*h + l0;
    const int qh_off = offsetof(block_q6_K, qh) + 32*h + l0;
    const int sc_off = offsetof(block_q6_K, scales) + 8*h + is;

    float acc[MMVQ_K80_Q6_K_ROWS] = {0.0f};

    for (int b0 = 0; b0 < nb; b0 += 2) {
        const int b = b0 + bo;
        if (b >= nb) {
            break;
        }
        float4 yv[4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            yv[g] = __ldg((const float4 *) (ys + b*QK_K + 128*h + 32*g + l0));
        }

#pragma unroll
        for (int r = 0; r < MMVQ_K80_Q6_K_ROWS; ++r) {
            const char * bp = x + (size_t) min(row0 + r, nrows - 1)*row_bytes + (size_t) b*sizeof(block_q6_K);
            const uint32_t qa = mmvq_k80_ld4(bp + ql_off);
            const uint32_t qb = mmvq_k80_ld4(bp + ql_off + 32);
            const uint32_t qh = mmvq_k80_ld4(bp + qh_off);
            const float d = __half2float(__ushort_as_half(__ldg((const uint16_t *) (bp + offsetof(block_q6_K, d)))));
            const uint32_t q[4] = {
                ( qa       & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4),
                ( qb       & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2),
                ((qa >> 4) & 0x0F0F0F0Fu) |  (qh & 0x30303030u),
                ((qb >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u)};
            float s = 0.0f;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                // (2^23 + v) - (2^23 + 32) = v - 32, exact
                float dot = (mmvq_k80_byte<0>(q[g]) - (8388608.0f + 32.0f))*yv[g].x;
                dot = fmaf(mmvq_k80_byte<1>(q[g]) - (8388608.0f + 32.0f), yv[g].y, dot);
                dot = fmaf(mmvq_k80_byte<2>(q[g]) - (8388608.0f + 32.0f), yv[g].z, dot);
                dot = fmaf(mmvq_k80_byte<3>(q[g]) - (8388608.0f + 32.0f), yv[g].w, dot);
                // int8 scale: flip the sign bit, decode as unsigned, subtract the bias
                const float sc = mmvq_k80_magic((uint32_t) (uint8_t) __ldg(bp + sc_off + 2*g) ^ 0x80u) - 128.0f;
                s = fmaf(sc, dot, s);
            }
            acc[r] = fmaf(d, s, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < MMVQ_K80_Q6_K_ROWS; ++r) {
        const float v = warp_reduce_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = v;
        }
    }
}

bool ggml_cuda_should_use_mmvq_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500 // Kepler
        && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
        && (src0->type != GGML_TYPE_Q4_K || (uintptr_t) src0->data % 16 == 0); // the q4_K kernel reads src0 as uint4
}

// Prepares the ny inputs (input c at src1 + c*src1_stride floats) and runs the q4_K or q6_K kernel over nslots.
static void mmvq_k80_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const float * src1,
        const int64_t src1_stride, const int nslots, mmvq_k80_channels ch, float * dst) {
    const int ncols = src0->ne[0];
    const int nrows = src0->ne[1];
    cudaStream_t stream = ctx.stream();

    const bool q4_K = src0->type == GGML_TYPE_Q4_K;
    ggml_cuda_pool_alloc<float> ys(ctx.pool(), (size_t) ncols*ch.ny);
    ggml_cuda_pool_alloc<float> ysum16(ctx.pool()); // only q4_K's min term uses it
    if (q4_K) {
        ysum16.alloc((size_t) ncols/16*ch.ny);
    }
    mmvq_k80_prep<<<dim3((ncols + 255)/256, ch.ny), 256, 0, stream>>>(src1, src1_stride, ys.get(), ysum16.get(), ncols);
    ch.y_stride = ncols;

    const dim3 block_dims(WARP_SIZE, MMVQ_K80_WARPS, 1);
    if (!q4_K) {
        const int rows_per_block = MMVQ_K80_Q6_K_ROWS*MMVQ_K80_WARPS;
        mmvq_k80_q6_K<<<dim3((nrows + rows_per_block - 1)/rows_per_block, nslots), block_dims, 0, stream>>>(
            (const char *) src0->data, ys.get(), dst, ncols, nrows, src0->nb[1], ch);
    } else {
        const int rows_per_block = MMVQ_K80_ROWS*MMVQ_K80_WARPS;
        mmvq_k80_q4_K<<<dim3((nrows + rows_per_block - 1)/rows_per_block, nslots), block_dims, 0, stream>>>(
            (const char *) src0->data, ys.get(), ysum16.get(), dst, ncols, nrows, src0->nb[1], ch);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_vec_q_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const mmvq_k80_channels ch = { nullptr, 0, 1, 0, 0 };
    mmvq_k80_launch(ctx, src0, (const float *) src1->data, 0, 1, ch, (float *) dst->data);
}

bool ggml_cuda_should_use_mmvq_k80_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, int cc) {
    const int64_t n_used = ids->ne[0];
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500 // Kepler
        && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32
        && dst->ne[2] == 1 && src1->ne[2] == 1 && ids->ne[1] == 1 // one token
        && dst->ne[1] == n_used && (src1->ne[1] == 1 || src1->ne[1] == n_used)
        && src0->ne[3] == 1 && src0->nb[1] == ggml_row_size(src0->type, src0->ne[0]) && src0->nb[2] == src0->nb[1]*src0->ne[1]
        && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && ids->nb[0] == sizeof(int32_t)
        && (src0->type != GGML_TYPE_Q4_K || ((uintptr_t) src0->data % 16 == 0 && src0->nb[2] % 16 == 0)); // q4_K reads uint4
}

void ggml_cuda_mul_mat_vec_q_k80_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    const int n_used = ids->ne[0];
    const mmvq_k80_channels ch = { (const int32_t *) ids->data, src0->nb[2], (int) src1->ne[1], 0, (int64_t) (dst->nb[1] / sizeof(float)) };
    mmvq_k80_launch(ctx, src0, (const float *) src1->data, src1->nb[1] / sizeof(float), n_used, ch, (float *) dst->data);
}
