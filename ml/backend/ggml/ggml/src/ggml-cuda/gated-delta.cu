#include "gated-delta.cuh"

// Block (h, seq), thread d0. The state is S[d1][d0] with d0 fastest, so for each d1 the block's threads read
// one contiguous row; each thread keeps its column of D values in registers and reads and writes it once:
//   S = g*S;  kv = S k;  delta = beta*(v - kv);  S = S + delta k^T;  out = S q
template <int D>
static __global__ void __launch_bounds__(D) gated_delta_step_f32(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ state,
        float * __restrict__ out, float * __restrict__ state_out, const int H, const float eps, const float scale) {
    constexpr int NW = D / WARP_SIZE;
    const int t  = blockIdx.y*H + blockIdx.x; // (seq, head)
    const int d0 = threadIdx.x;

    __shared__ float qs[D];
    __shared__ float ks[D];
    __shared__ float red[2][NW];

    // L2-normalize q and k per head, as ggml_l2_norm does, and scale q
    const float qv = q[(size_t) t*D + d0];
    const float kv0 = k[(size_t) t*D + d0];
    float sq = warp_reduce_sum(qv*qv);
    float sk = warp_reduce_sum(kv0*kv0);
    if (d0 % WARP_SIZE == 0) {
        red[0][d0 / WARP_SIZE] = sq;
        red[1][d0 / WARP_SIZE] = sk;
    }
    __syncthreads();
    sq = 0.0f;
    sk = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        sq += red[0][w];
        sk += red[1][w];
    }
    qs[d0] = qv  / fmaxf(sqrtf(sq), eps) * scale;
    ks[d0] = kv0 / fmaxf(sqrtf(sk), eps);
    __syncthreads();

    const float gh = expf(g[t]);
    const float bh = 1.0f / (1.0f + expf(-beta[t]));
    const float * S = state     + (size_t) t*D*D;
    float       * N = state_out + (size_t) t*D*D;

    float col[D];
    float kv = 0.0f;
#pragma unroll
    for (int d1 = 0; d1 < D; ++d1) {
        col[d1] = S[d1*D + d0] * gh;
        kv = fmaf(col[d1], ks[d1], kv);
    }
    const float delta = bh * (v[(size_t) t*D + d0] - kv);
    float o = 0.0f;
#pragma unroll
    for (int d1 = 0; d1 < D; ++d1) {
        const float s = fmaf(delta, ks[d1], col[d1]);
        N[d1*D + d0] = s;
        o = fmaf(s, qs[d1], o);
    }
    out[(size_t) t*D + d0] = o;
}

bool ggml_cuda_gated_delta_step_supported(const ggml_tensor * dst) {
    const int64_t D = dst->src[2]->ne[0];
    return D == 64 || D == 128;
}

void ggml_cuda_op_gated_delta_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int D      = v->ne[0];
    const int H      = v->ne[1];
    const int n_seqs = v->ne[3];

    float eps;
    float scale;
    memcpy(&eps,   (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&scale, (const float *) dst->op_params + 1, sizeof(float));

    float * out       = (float *) dst->data;
    float * state_out = out + (size_t) D*H*n_seqs;
    cudaStream_t stream = ctx.stream();

    const dim3 grid(H, n_seqs, 1);
    switch (D) {
        case 64:
            gated_delta_step_f32<64><<<grid, 64, 0, stream>>>(
                (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data,
                (const float *) beta->data, (const float *) state->data, out, state_out, H, eps, scale);
            break;
        case 128:
            gated_delta_step_f32<128><<<grid, 128, 0, stream>>>(
                (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data,
                (const float *) beta->data, (const float *) state->data, out, state_out, H, eps, scale);
            break;
        default:
            GGML_ABORT("unsupported head dim");
    }
    CUDA_CHECK(cudaGetLastError());
}
