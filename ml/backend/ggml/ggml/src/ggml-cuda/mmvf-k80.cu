#include "mmvf-k80.cuh"

static constexpr int MMVF_K80_THREADS = 128;
static constexpr int MMVF_K80_STEPS   = 4; // steps per block: amortizes the shared-memory src1 load

// dst[i12, i01] = sum_i00 src0[i00, i01, i02] * src1[i00, i12] with i02 = i12 / r2 (blockIdx.y = i02).
// L = ncols/16 lanes share a src0 row, 16 values each; the row is read and converted once for the
// up to gc src1 heads of its group that block z handles, so a large group can spread across blocks.
template <int ncols>
static __global__ void __launch_bounds__(MMVF_K80_THREADS) mul_mat_vec_f_k80(
        const half * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int r2, const int gc,
        const int64_t stride_row, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst) {
    constexpr int L   = ncols / 16;
    constexpr int CPW = WARP_SIZE / L; // rows per warp per step
    extern __shared__ float ys[];      // [gn][ncols]

    const int i02 = blockIdx.y;
    const int g0  = blockIdx.z * gc;
    const int gn  = min(gc, r2 - g0);
    for (int i = threadIdx.x; i < gn*ncols; i += blockDim.x) {
        ys[i] = y[(int64_t) (i02*r2 + g0 + i/ncols)*stride_channel_y + i % ncols];
    }
    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;
    const int sub  = lane % L;
    const int rw   = lane / L;
    const half * xc = x + i02*stride_channel_x + sub*16;

    for (int row0 = (blockIdx.x*nw + warp)*CPW; row0 < nrows; row0 += gridDim.x*nw*CPW) {
        const int row = row0 + rw;
        float xf[16];
        if (row < nrows) {
            const uint4 * p = (const uint4 *) (xc + row*stride_row);
            const uint4 a = __ldg(p);
            const uint4 b = __ldg(p + 1);
            const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 f = __half22float2(*(const half2 *) &w[i]);
                xf[2*i + 0] = f.x;
                xf[2*i + 1] = f.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                xf[i] = 0.0f;
            }
        }
        for (int g = 0; g < gn; ++g) {
            const float4 * y4 = (const float4 *) (ys + g*ncols + sub*16);
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = y4[i];
                s = fmaf(xf[4*i + 0], v.x, s);
                s = fmaf(xf[4*i + 1], v.y, s);
                s = fmaf(xf[4*i + 2], v.z, s);
                s = fmaf(xf[4*i + 3], v.w, s);
            }
#pragma unroll
            for (int o = L/2; o > 0; o >>= 1) {
                s += __shfl_xor_sync(0xFFFFFFFF, s, o, L);
            }
            if (sub == 0 && row < nrows) {
                dst[(int64_t) (i02*r2 + g0 + g)*stride_channel_dst + row] = s;
            }
        }
    }
}

bool ggml_cuda_should_use_mmvf_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    const int64_t ncols = src0->ne[0];
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500 // Kepler
        && src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && (ncols == 64 || ncols == 128 || ncols == 256 || ncols == 512) && src1->ne[0] == ncols
        && src1->ne[1] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1 && src1->ne[2] % src0->ne[2] == 0
        && src0->nb[0] == ggml_type_size(GGML_TYPE_F16) && src1->nb[0] == ggml_type_size(GGML_TYPE_F32)
        && ggml_is_contiguous(dst)
        // the kernel reads src0 rows as uint4
        && (uintptr_t) src0->data % 16 == 0 && src0->nb[1] % 16 == 0 && src0->nb[2] % 16 == 0;
}

template <int ncols>
static void launch_mul_mat_vec_f_k80(
        const half * x, const float * y, float * dst, const int nrows, const int nchannels_x, const int r2,
        const int64_t stride_row, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        cudaStream_t stream) {
    constexpr int rows_per_step = (MMVF_K80_THREADS/WARP_SIZE) * (WARP_SIZE/(ncols/16));
    const int gc = nchannels_x >= 4 ? r2 : min(r2, 4); // few src0 channels: split the group for parallelism
    const dim3 grid((nrows + rows_per_step*MMVF_K80_STEPS - 1)/(rows_per_step*MMVF_K80_STEPS), nchannels_x, (r2 + gc - 1)/gc);
    mul_mat_vec_f_k80<ncols><<<grid, MMVF_K80_THREADS, gc*ncols*sizeof(float), stream>>>(
        x, y, dst, nrows, r2, gc, stride_row, stride_channel_x, stride_channel_y, stride_channel_dst);
}

void ggml_cuda_mul_mat_vec_f_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half  * x = (const half  *) src0->data;
    const float * y = (const float *) src1->data;
    float     * d = (float       *) dst->data;
    const int nrows       = src0->ne[1];
    const int nchannels_x = src0->ne[2];
    const int r2          = src1->ne[2] / src0->ne[2];
    const int64_t stride_row         = src0->nb[1] / sizeof(half);
    const int64_t stride_channel_x   = src0->nb[2] / sizeof(half);
    const int64_t stride_channel_y   = src1->nb[2] / sizeof(float);
    const int64_t stride_channel_dst = dst->nb[2]  / sizeof(float);
    cudaStream_t stream = ctx.stream();

    switch (src0->ne[0]) {
        case 64:
            launch_mul_mat_vec_f_k80<64>(x, y, d, nrows, nchannels_x, r2, stride_row, stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            break;
        case 128:
            launch_mul_mat_vec_f_k80<128>(x, y, d, nrows, nchannels_x, r2, stride_row, stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            break;
        case 256:
            launch_mul_mat_vec_f_k80<256>(x, y, d, nrows, nchannels_x, r2, stride_row, stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            break;
        case 512:
            launch_mul_mat_vec_f_k80<512>(x, y, d, nrows, nchannels_x, r2, stride_row, stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            break;
        default:
            GGML_ABORT("unsupported ncols");
    }
    CUDA_CHECK(cudaGetLastError());
}
