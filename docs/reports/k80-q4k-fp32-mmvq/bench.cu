// q4_K mat-vec on one K80 die: ggml's CUDA path against an FP32-decode kernel (#563).
//
// Usage: [SWEEP=1] [ROOF=1] bench K N [iters]
//   K, N   weight shape: N rows of K values; K a multiple of 256
//   iters  timed launches per path, default 100
//   SWEEP  also time rows-per-warp x warps-per-block configurations of the FP32 kernel
//   ROOF   also time a plain read of the same bytes: the die's achievable bandwidth
//
// Build on the host against a build-local.sh tree (R = repo root):
//   nvcc -O3 -std=c++17 -ccbin /usr/local/bin/g++ -gencode arch=compute_37,code=sm_37 \
//     -I $R/ml/backend/ggml/ggml/include bench.cu -o bench \
//     -L $R/build/lib/ollama -lggml-base -lggml-cuda -Xlinker -rpath=$R/build/lib/ollama
// The "ggml" column is whatever that libggml-cuda.so dispatches: mul_mat_vec_q before #563,
// mmvq_k80_q4_K after it.
#include <cmath>
#include <cstdint>
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

static constexpr int QK = 256;   // values per q4_K block
static constexpr int BLK = 144;  // bytes per q4_K block: d, dmin (2+2), scales (12), qs (128)

// ys[i] = y[i] * {1, 2^-8, 2^-16, 1}[i % 4], matching the bit position each nibble is decoded at.
// ysum16[g] = sum of y over the 16 values of group g, for the q4_K min term.
__global__ void prep(const float * __restrict__ y, float * __restrict__ ys, float * __restrict__ ysum16, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = i < K ? y[i] : 0.f;
  if (i < K) {
    const int k = i & 3;
    ys[i] = v * (k == 1 ? 1.f / 256.f : k == 2 ? 1.f / 65536.f : 1.f);
  }
  for (int o = 8; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o, 16);
  if ((i & 15) == 0 && i < K) ysum16[i >> 4] = v;
}

// An integer below 2^23 placed in the mantissa of 2^23, minus 2^23: exact, no I2F (32/clk on Kepler).
__device__ __forceinline__ float magic(uint32_t bits) {
  return __int_as_float(0x4B000000u | bits) - 8388608.f;
}

// One warp computes R rows. Lane t reads 16 qs bytes of block (b0 + t/8): chunk j = (t%8)/2,
// offset l = 16*(t%2). Low nibbles are values 64j+l.., sub-block 2j; high nibbles 64j+32+l.., 2j+1.
template <int R, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) q4k_fp32(const uint8_t * __restrict__ W, const float * __restrict__ ys,
    const float * __restrict__ ysum16, float * __restrict__ out, int K, int N) {
  const int lane = threadIdx.x & 31;
  const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * R;
  if (row0 >= N) return;
  const int nb = K / QK;
  const size_t row_bytes = (size_t) nb * BLK;
  const int bo = lane >> 3;
  const int j = (lane & 7) >> 1;
  const int l = (lane & 1) * 16;
  const int qoff = 16 + 32 * j + l;
  const bool upper = j >= 2;      // sub-blocks 4..7 split their 6-bit scale and min across bytes
  const int sh = 16 * (j & 1);    // byte 0 of the shifted word = sub-block 2j, byte 1 = 2j+1

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;

  for (int b0 = 0; b0 < nb; b0 += 4) {
    const int b = b0 + bo;
    if (b >= nb) break;
    const int e = b * QK + 64 * j + l;
    float4 yl[4], yh[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
      yl[w] = __ldg(reinterpret_cast<const float4 *>(ys + e) + w);
      yh[w] = __ldg(reinterpret_cast<const float4 *>(ys + e + 32) + w);
    }
    const float sl = __ldg(ysum16 + (e >> 4));
    const float shs = __ldg(ysum16 + ((e + 32) >> 4));

    uint4 hd[R], qv[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint8_t * bp = W + (size_t) min(row0 + r, N - 1) * row_bytes + (size_t) b * BLK;
      hd[r] = __ldg(reinterpret_cast<const uint4 *>(bp));
      qv[r] = __ldg(reinterpret_cast<const uint4 *>(bp + qoff));
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
      const float d = __half2float(__ushort_as_half((unsigned short) (hd[r].x & 0xFFFF)));
      const float dmin = __half2float(__ushort_as_half((unsigned short) (hd[r].x >> 16)));
      const uint32_t a = hd[r].y >> sh, m = hd[r].z >> sh, c = hd[r].w >> sh;
      const uint32_t scw = upper ? (c & 0x0F0Fu) | ((a >> 2) & 0x3030u) : a & 0x3F3Fu;
      const uint32_t mw = upper ? ((c >> 4) & 0x0F0Fu) | ((m >> 2) & 0x3030u) : m & 0x3F3Fu;

      const uint32_t q[4] = {qv[r].x, qv[r].y, qv[r].z, qv[r].w};
      float al = 0.f, ah = 0.f;
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const uint32_t lo = q[w] & 0x0F0F0F0Fu, hi = (q[w] >> 4) & 0x0F0F0F0Fu;
        al = fmaf(magic(lo & 0xFFu), yl[w].x, al);
        al = fmaf(magic(lo & 0xFF00u), yl[w].y, al);
        al = fmaf(magic(lo & 0xFF0000u), yl[w].z, al);
        al = fmaf(magic(lo >> 24), yl[w].w, al);
        ah = fmaf(magic(hi & 0xFFu), yh[w].x, ah);
        ah = fmaf(magic(hi & 0xFF00u), yh[w].y, ah);
        ah = fmaf(magic(hi & 0xFF0000u), yh[w].z, ah);
        ah = fmaf(magic(hi >> 24), yh[w].w, ah);
      }
      const float dot = magic(scw & 0xFFu) * al + magic(scw >> 8) * ah;
      const float mins = magic(mw & 0xFFu) * sl + magic(mw >> 8) * shs;
      acc[r] += d * dot - dmin * mins;
    }
  }

#pragma unroll
  for (int r = 0; r < R; r++) {
    float v = acc[r];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0 && row0 + r < N) out[row0 + r] = v;
  }
}

// Times q4k_fp32<R, WARPS> alone (prep already run), in us per launch over iters back-to-back launches.
template <int R, int WARPS>
static double time_cfg(const uint8_t * dw, const float * dys, const float * dsum, float * dout, int K, int N, int iters) {
  const int grid = (N + R * WARPS - 1) / (R * WARPS);
  q4k_fp32<R, WARPS><<<grid, 32 * WARPS>>>(dw, dys, dsum, dout, K, N);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) q4k_fp32<R, WARPS><<<grid, 32 * WARPS>>>(dw, dys, dsum, dout, K, N);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaGetLastError());
  return 1000.0 * ms / iters;
}

// Bandwidth ceiling: each thread streams uint4s grid-stride and folds them into one word.
__global__ void read_roof(const uint4 * __restrict__ p, size_t n, uint32_t * __restrict__ sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
    const uint4 v = __ldg(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

static double max_rel_err(const std::vector<float> & got, const std::vector<double> & ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    num = fmax(num, fabs(got[i] - ref[i]));
    den = fmax(den, fabs(ref[i]));
  }
  return num / den;
}

int main(int argc, char ** argv) {
  if (argc < 3) { fprintf(stderr, "usage: bench K N [iters]\n"); return 2; }
  const int K = atoi(argv[1]), N = atoi(argv[2]), iters = argc > 3 ? atoi(argv[3]) : 100;
  if (K % QK) { fprintf(stderr, "K must be a multiple of %d\n", QK); return 2; }
  const size_t wbytes = (size_t) N * (K / QK) * BLK;

  std::mt19937 rng(563);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> wf((size_t) N * K), y(K);
  for (auto & v : wf) v = nd(rng) * 0.02f;
  for (auto & v : y) v = nd(rng);
  std::vector<uint8_t> wq(wbytes);
  if (ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), wq.data(), 0, N, K, nullptr) != wbytes) {
    fprintf(stderr, "unexpected q4_K size\n"); return 1;
  }

  // Reference: dequantize with ggml's own CPU routine, dot in double.
  const auto * tt = ggml_get_type_traits(GGML_TYPE_Q4_K);
  std::vector<double> ref(N);
  std::vector<float> row(K);
  for (int n = 0; n < N; n++) {
    tt->to_float(wq.data() + (size_t) n * (K / QK) * BLK, row.data(), K);
    double s = 0;
    for (int k = 0; k < K; k++) s += (double) row[k] * y[k];
    ref[n] = s;
  }

  // ggml's path: the same graph op the runner issues for a batch-1 decode.
  ggml_backend_t be = ggml_backend_cuda_init(0);
  ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
  ggml_context * ctx = ggml_init(ip);
  ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
  ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
  ggml_tensor * to = ggml_mul_mat(ctx, tw, tx);
  ggml_cgraph * gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, to);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  ggml_backend_tensor_set(tw, wq.data(), 0, wbytes);
  ggml_backend_tensor_set(tx, y.data(), 0, K * sizeof(float));
  ggml_backend_graph_compute(be, gf);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
  const double ggml_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::vector<float> og(N);
  ggml_backend_tensor_get(to, og.data(), 0, N * sizeof(float));

  // FP32-decode path.
  constexpr int R = 4, WARPS = 4;
  uint8_t * dw; float *dy, *dys, *dsum, *dout;
  CK(cudaMalloc(&dw, wbytes));
  CK(cudaMalloc(&dy, K * sizeof(float)));
  CK(cudaMalloc(&dys, K * sizeof(float)));
  CK(cudaMalloc(&dsum, K / 16 * sizeof(float)));
  CK(cudaMalloc(&dout, N * sizeof(float)));
  CK(cudaMemcpy(dw, wq.data(), wbytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy, y.data(), K * sizeof(float), cudaMemcpyHostToDevice));
  const int grid = (N + R * WARPS - 1) / (R * WARPS);
  auto run = [&] {
    prep<<<(K + 255) / 256, 256>>>(dy, dys, dsum, K);
    q4k_fp32<R, WARPS><<<grid, 32 * WARPS>>>(dw, dys, dsum, dout, K, N);
  };
  run();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) run();
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  const double fp32_us = 1000.0 * ms / iters;
  CK(cudaGetLastError());
  std::vector<float> of(N);
  CK(cudaMemcpy(of.data(), dout, N * sizeof(float), cudaMemcpyDeviceToHost));

  printf("K=%d N=%d weights=%.1f MB\n", K, N, wbytes / 1e6);
  printf("  ggml  %8.1f us/call (wall, incl. sync)  %6.1f GB/s  err %.2e\n", ggml_us, wbytes / ggml_us / 1e3, max_rel_err(og, ref));
  printf("  fp32  %8.1f us/call (events)            %6.1f GB/s  err %.2e\n", fp32_us, wbytes / fp32_us / 1e3, max_rel_err(of, ref));

  if (getenv("ROOF")) {
    uint32_t * sink; CK(cudaMalloc(&sink, 4));
    for (int blocks : {13 * 8, 13 * 16, 13 * 32, 13 * 64}) {
      read_roof<<<blocks, 256>>>(reinterpret_cast<const uint4 *>(dw), wbytes / 16, sink);
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0));
      for (int i = 0; i < iters; i++) read_roof<<<blocks, 256>>>(reinterpret_cast<const uint4 *>(dw), wbytes / 16, sink);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      printf("  read_roof blocks=%d  %7.1f us  %6.1f GB/s\n", blocks, 1000.0 * ms / iters, wbytes / (1000.0 * ms / iters) / 1e3);
    }
  }
  if (getenv("SWEEP")) {
#define CFG(r, w) { const double us = time_cfg<r, w>(dw, dys, dsum, dout, K, N, iters); \
    printf("  R=%d WARPS=%d  %7.1f us  %6.1f GB/s\n", r, w, us, wbytes / us / 1e3); }
    CFG(1, 4) CFG(1, 8) CFG(2, 2) CFG(2, 4) CFG(2, 8) CFG(4, 2) CFG(4, 4) CFG(4, 8) CFG(8, 2) CFG(8, 4)
  }

  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(be);
  return 0;
}
