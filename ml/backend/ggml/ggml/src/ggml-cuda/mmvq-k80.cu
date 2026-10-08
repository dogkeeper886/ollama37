#include "mmvq-k80.cuh"

static constexpr int MMVQ_K80_ROWS  = 4; // rows per warp: the warp loads src1 once for all of them
static constexpr int MMVQ_K80_WARPS = 4;

// ys[i] = y[i] * {1, 2^-8, 2^-16, 1}[i % 4], undoing the bit position each nibble is decoded at.
// ysum16[g] = sum of y over the 16 values of group g, for the q4_K min term.
static __global__ void mmvq_k80_prep(const float * __restrict__ y, float * __restrict__ ys, float * __restrict__ ysum16, const int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < k ? y[i] : 0.0f;
    if (i < k) {
        const int p = i & 3;
        ys[i] = v * (p == 1 ? 1.0f/256.0f : p == 2 ? 1.0f/65536.0f : 1.0f);
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, o, 16);
    }
    if ((i & 15) == 0 && i < k) {
        ysum16[i >> 4] = v;
    }
}

// An integer below 2^23 placed in the mantissa of 2^23, minus 2^23: exact, with no int-to-float conversion.
static __device__ __forceinline__ float mmvq_k80_magic(const uint32_t bits) {
    return __int_as_float(0x4B000000u | bits) - 8388608.0f;
}

// One warp computes MMVQ_K80_ROWS rows. Lane t reads 16 qs bytes of block (b0 + t/8): chunk j = (t%8)/2,
// offset l = 16*(t%2). Low nibbles are values 64j+l.., sub-block 2j; high nibbles are 64j+32+l.., sub-block 2j+1.
static __global__ void __launch_bounds__(WARP_SIZE*MMVQ_K80_WARPS) mmvq_k80_q4_K(
        const char * __restrict__ x, const float * __restrict__ ys, const float * __restrict__ ysum16,
        float * __restrict__ dst, const int ncols, const int nrows, const size_t row_bytes) {
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
                al = fmaf(mmvq_k80_magic(lo & 0xFFu),     yl[w].x, al);
                al = fmaf(mmvq_k80_magic(lo & 0xFF00u),   yl[w].y, al);
                al = fmaf(mmvq_k80_magic(lo & 0xFF0000u), yl[w].z, al);
                al = fmaf(mmvq_k80_magic(lo >> 24),       yl[w].w, al);
                ah = fmaf(mmvq_k80_magic(hi & 0xFFu),     yh[w].x, ah);
                ah = fmaf(mmvq_k80_magic(hi & 0xFF00u),   yh[w].y, ah);
                ah = fmaf(mmvq_k80_magic(hi & 0xFF0000u), yh[w].z, ah);
                ah = fmaf(mmvq_k80_magic(hi >> 24),       yh[w].w, ah);
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

bool ggml_cuda_should_use_mmvq_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500 // Kepler
        && src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
        && (uintptr_t) src0->data % 16 == 0; // the kernel reads src0 as uint4
}

void ggml_cuda_mul_mat_vec_q_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols = src0->ne[0];
    const int nrows = src0->ne[1];
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> ys(ctx.pool(), ncols);
    ggml_cuda_pool_alloc<float> ysum16(ctx.pool(), ncols/16);
    mmvq_k80_prep<<<(ncols + 255)/256, 256, 0, stream>>>((const float *) src1->data, ys.get(), ysum16.get(), ncols);

    const int rows_per_block = MMVQ_K80_ROWS*MMVQ_K80_WARPS;
    const dim3 block_dims(WARP_SIZE, MMVQ_K80_WARPS, 1);
    mmvq_k80_q4_K<<<(nrows + rows_per_block - 1)/rows_per_block, block_dims, 0, stream>>>(
        (const char *) src0->data, ys.get(), ysum16.get(), (float *) dst->data, ncols, nrows, src0->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}
