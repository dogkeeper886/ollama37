// One decode token's non-FA attention, built the way ml/backend/ggml ScaledDotProductAttention
// builds it over the causal cache (PermutedV), for every layer of a llama3.1:8b-shaped model.
//
// Usage: attn <n_ctx> [iters] [layers]
//
// Build on the host against a build-local.sh tree (R = repo root):
//   nvcc -O3 -std=c++17 -ccbin /usr/local/bin/g++ -gencode arch=compute_37,code=sm_37 \
//     -I $R/ml/backend/ggml/ggml/include attn.cu -o attn \
//     -L $R/build/lib/ollama -lggml-base -lggml-cuda -Xlinker -rpath=$R/build/lib/ollama
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

int main(int argc, char ** argv) {
  if (argc < 2) { fprintf(stderr, "usage: attn <n_ctx> [iters] [layers]\n"); return 2; }
  const int n = atoi(argv[1]), iters = argc > 2 ? atoi(argv[2]) : 20, layers = argc > 3 ? atoi(argv[3]) : 32;
  const int hd = 128, nkv = 8, nq = 32;

  ggml_init_params ip = {ggml_tensor_overhead() * (16 * layers + 16) + ggml_graph_overhead_custom(16 * layers, false), nullptr, true};
  ggml_context * ctx = ggml_init(ip);
  ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16 * layers, false);
  std::vector<ggml_tensor *> fill;
  for (int l = 0; l < layers; l++) {
    ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, nkv, n);   // keys: [head_dim, kv_heads, cells]
    ggml_tensor * vc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n, hd, nkv);   // PermutedV values: [cells, head_dim, kv_heads]
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nq, 1);
    fill.push_back(kc); fill.push_back(vc); fill.push_back(q);
    ggml_tensor * k = ggml_permute(ctx, kc, 0, 2, 1, 3);
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, k, qp);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf(hd), 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, vc, kq);
    ggml_build_forward_expand(gf, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)));
  }

  ggml_backend_t be = ggml_backend_cuda_init(0);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  std::mt19937 rng(563);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (ggml_tensor * t : fill) {
    const int64_t ne = ggml_nelements(t);
    if (t->type == GGML_TYPE_F16) {
      std::vector<ggml_fp16_t> v(ne);
      for (auto & e : v) e = ggml_fp32_to_fp16(nd(rng));
      ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
      std::vector<float> v(ne);
      for (auto & e : v) e = nd(rng);
      ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
  }

  for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  const double kv = 2.0 * layers * n * nkv * hd * sizeof(ggml_fp16_t);
  printf("n_ctx %6d  %2d layers: %7.2f ms per token  KV %.0f MB  %.1f GB/s\n", n, layers, ms, kv / 1e6, kv / ms / 1e6);

  ggml_backend_buffer_free(buf);
  ggml_backend_free(be);
  ggml_free(ctx);
  return 0;
}
