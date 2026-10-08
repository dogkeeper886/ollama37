// Decode KQ on one K80 die: ggml's mul_mat_vec_f against a group-sharing kernel (#565).
//
// Usage: kq <head_dim> <q_heads> <kv_heads> <n_ctx> [iters]
//
// Build on the host against a build-local.sh tree (R = repo root):
//   nvcc -O3 -std=c++17 -ccbin /usr/local/bin/g++ -gencode arch=compute_37,code=sm_37 \
//     -I $R/ml/backend/ggml/ggml/include kq.cu -o kq \
//     -L $R/build/lib/ollama -lggml-base -lggml-cuda -Xlinker -rpath=$R/build/lib/ollama
// The "ggml" column is whatever that libggml-cuda.so dispatches: mul_mat_vec_f before #565,
// mul_mat_vec_f_k80 after it.
// K is the causal cache [head_dim, kv_heads, n_ctx] f16 viewed as [head_dim, n_ctx, kv_heads];
// Q is [head_dim, q_heads] f32 viewed as [head_dim, 1, q_heads], as ScaledDotProductAttention builds them.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// dst[g, c] = sum_d K[c, h, d] * Q[h*G + g, d] for each KV head h (blockIdx.y). L = D/16 lanes share a
// key row, 16 values each; the row is read and converted once for the up to gc query heads of
// its group that block z handles (gc per block), so a large group can spread across blocks.
template <int D>
__global__ void __launch_bounds__(128) kq_k80(const half * __restrict__ K, const float * __restrict__ Q,
    float * __restrict__ dst, const int n, const int G, const int gc, const int64_t s_c, const int64_t s_h,
    const int64_t s_q, const int64_t s_dst) {
  constexpr int L = D / 16;
  constexpr int CPW = 32 / L;  // cells per warp per step
  extern __shared__ float qs[];  // [gn][D]
  const int h = blockIdx.y, g0 = blockIdx.z * gc, gn = min(gc, G - g0);
  for (int i = threadIdx.x; i < gn * D; i += blockDim.x) qs[i] = Q[(int64_t) (h * G + g0 + i / D) * s_q + i % D];
  __syncthreads();

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int sub = lane % L, cw = lane / L;
  const half * kh = K + h * s_h + sub * 16;
  for (int c0 = (blockIdx.x * nw + warp) * CPW; c0 < n; c0 += gridDim.x * nw * CPW) {
    const int c = c0 + cw;
    float k[16];
    if (c < n) {
      const uint4 * p = reinterpret_cast<const uint4 *>(kh + c * s_c);
      const uint4 a = __ldg(p), b = __ldg(p + 1);
      const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float2 f = __half22float2(*reinterpret_cast<const half2 *>(&w[i]));
        k[2 * i] = f.x; k[2 * i + 1] = f.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) k[i] = 0.f;
    }
    for (int g = 0; g < gn; g++) {
      const float4 * q4 = reinterpret_cast<const float4 *>(qs + g * D + sub * 16);
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float4 q = q4[i];
        s = fmaf(k[4 * i], q.x, s); s = fmaf(k[4 * i + 1], q.y, s);
        s = fmaf(k[4 * i + 2], q.z, s); s = fmaf(k[4 * i + 3], q.w, s);
      }
#pragma unroll
      for (int o = L / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, L);
      if (sub == 0 && c < n) dst[(int64_t) (h * G + g0 + g) * s_dst + c] = s;
    }
  }
}

template <int D>
static void launch(const half * K, const float * Q, float * dst, int n, int nkv, int G, int64_t s_c, int64_t s_h,
    int64_t s_q, int64_t s_dst) {
  constexpr int CPB = 4 * (32 / (D / 16));  // cells per block per step
  const int steps = 4;                       // steps per block: amortizes the shared-memory Q load
  const int gc = nkv >= 4 ? G : min(G, 4);  // few KV heads: split the group for parallelism
  const dim3 grid((n + CPB * steps - 1) / (CPB * steps), nkv, (G + gc - 1) / gc);
  kq_k80<D><<<grid, 128, gc * D * sizeof(float)>>>(K, Q, dst, n, G, gc, s_c, s_h, s_q, s_dst);
}

int main(int argc, char ** argv) {
  if (argc < 5) { fprintf(stderr, "usage: kq <head_dim> <q_heads> <kv_heads> <n_ctx> [iters]\n"); return 2; }
  const int D = atoi(argv[1]), nq = atoi(argv[2]), nkv = atoi(argv[3]), n = atoi(argv[4]);
  const int iters = argc > 5 ? atoi(argv[5]) : 100, G = nq / nkv;

  ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
  ggml_context * ctx = ggml_init(ip);
  ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, nkv, n);
  ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, 1);
  ggml_tensor * k = ggml_permute(ctx, kc, 0, 2, 1, 3);
  ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor * kq = ggml_mul_mat(ctx, k, qp);
  ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
  ggml_cgraph * gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, kq);
  ggml_backend_t be = ggml_backend_cuda_init(0);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

  std::mt19937 rng(565);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<ggml_fp16_t> kh((size_t) D * nkv * n);
  std::vector<float> qh((size_t) D * nq);
  for (auto & e : kh) e = ggml_fp32_to_fp16(nd(rng));
  for (auto & e : qh) e = nd(rng);
  ggml_backend_tensor_set(kc, kh.data(), 0, ggml_nbytes(kc));
  ggml_backend_tensor_set(q, qh.data(), 0, ggml_nbytes(q));

  // Reference in double: out[qh][c].
  std::vector<double> ref((size_t) nq * n);
  for (int j = 0; j < nq; j++)
    for (int c = 0; c < n; c++) {
      double s = 0;
      for (int d = 0; d < D; d++) s += (double) ggml_fp16_to_fp32(kh[((size_t) c * nkv + j / G) * D + d]) * qh[(size_t) j * D + d];
      ref[(size_t) j * n + c] = s;
    }
  auto err = [&](const std::vector<float> & o) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) { num = fmax(num, fabs(o[i] - ref[i])); den = fmax(den, fabs(ref[i])); }
    return num / den;
  };

  ggml_backend_graph_compute(be, gf);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
  const double g_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::vector<float> og((size_t) nq * n);
  ggml_backend_tensor_get(kq, og.data(), 0, og.size() * sizeof(float));

  // The kernel on the same device buffers, with the strides of the permuted views.
  float * dout; CK(cudaMalloc(&dout, (size_t) nq * n * sizeof(float)));
  const half * K = (const half *) kc->data;
  const int64_t s_c = kc->nb[2] / 2, s_h = kc->nb[1] / 2, s_q = q->nb[1] / 4, s_dst = n;
  auto run = [&] {
    switch (D) {
      case 64:  launch<64>(K, (const float *) q->data, dout, n, nkv, G, s_c, s_h, s_q, s_dst); break;
      case 128: launch<128>(K, (const float *) q->data, dout, n, nkv, G, s_c, s_h, s_q, s_dst); break;
      case 256: launch<256>(K, (const float *) q->data, dout, n, nkv, G, s_c, s_h, s_q, s_dst); break;
      case 512: launch<512>(K, (const float *) q->data, dout, n, nkv, G, s_c, s_h, s_q, s_dst); break;
      default: fprintf(stderr, "head_dim %d unsupported\n", D); exit(2);
    }
  };
  run();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) run();
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaGetLastError());
  const double k_us = 1000.0 * ms / iters;
  std::vector<float> ok((size_t) nq * n);
  CK(cudaMemcpy(ok.data(), dout, ok.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const double bytes = (double) ggml_nbytes(kc);
  printf("D=%d q=%d kv=%d n=%d  K %.1f MB | ggml %8.1f us (wall) %6.1f GB/s err %.1e | k80 %7.1f us %6.1f GB/s err %.1e\n",
         D, nq, nkv, n, bytes / 1e6, g_us, bytes / g_us / 1e3, err(og), k_us, bytes / k_us / 1e3, err(ok));

  ggml_backend_buffer_free(buf);
  ggml_backend_free(be);
  ggml_free(ctx);
  return 0;
}
