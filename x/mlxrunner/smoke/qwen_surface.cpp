// K80 runner — Phase B forward-surface probe.
//
// This isn't a runner. It exercises the on-path ops a qwen3.5 inference pass
// actually uses (matmul, fast::rms_norm, fast::rope, softmax,
// fast::scaled_dot_product_attention, conv1d) on tiny random tensors. Two
// jobs: (1) the link surface — every op pulled into libmlx.a must resolve;
// (2) the runtime surface — every op's GPU primitive must actually dispatch
// without tripping a k80_runtime_stubs throw. A clean run means qwen3.5's
// inference compute path is wired end-to-end on the K80.
//
// What's NOT here on purpose:
//   - quantize() — the weight-conversion op. Dispatches to affine_quantize
//     which is still a throw-stub; inference never calls it.
//
// What IS here as of the dequant+matmul wiring landing: quantized_matmul()
// against hand-shaped pre-quantized inputs (uint8 packed + scales + biases),
// exercising the K80 fallback chain
//     QuantizedMatmul::eval_gpu -> dequant_then_matmul_k80
//       -> affine_dequantize  (Phase C kernel)
//       -> Matmul::eval_gpu   (Phase A cuBLAS path).
//
// Build via x/mlxrunner/CMakeLists.txt.

#include "mlx/mlx.h"

#include <iostream>

using namespace mlx::core;

int main() {
  // Force the CUDA backend (Device::gpu = DeviceType::gpu; default index 0).
  set_default_device(Device(Device::gpu));

  // Tiny shapes — values are irrelevant, only the symbols matter.
  const int B = 1, H = 1, T = 8, D = 64, K = 16;

  // --- core ops ---
  array a = random::uniform({B, T, D});
  array w = random::uniform({D, D});
  array c = matmul(a, w);
  array x = reshape(c, {B, H, T, D});

  // --- fast-fused ops ---
  array rw = random::uniform({D});
  array n = fast::rms_norm(c, rw, 1e-6f);
  array rq = fast::rope(x, D, /*traditional=*/false,
                        /*base=*/10000.0f, /*scale=*/1.0f, /*offset=*/0);

  array sm = softmax(n, -1);

  // --- attention (uses sdpa_vector on K80) ---
  array q = random::uniform({B, H, T, D});
  array kk = random::uniform({B, H, T, D});
  array vv = random::uniform({B, H, T, D});
  array o = fast::scaled_dot_product_attention(q, kk, vv, /*scale=*/0.125f);

  // --- 1d conv (DeltaNet short-conv on q/k/v gates) ---
  array x1d = random::uniform({B, T, K});
  array w1d = random::uniform({K, /*kw=*/3, K});
  array cv = conv1d(x1d, w1d, /*stride=*/1, /*padding=*/0,
                    /*dilation=*/1, /*groups=*/1);

  // --- quantized matmul (K80 fallback: dequant + cuBLAS) ---
  // Hand-shaped pre-quantized inputs for a 4-bit / group_size=64 weight
  // matrix matching qwen-style quants. We bypass quantize() (which calls
  // the still-stubbed affine_quantize); instead construct uint32 packed
  // weights + fp32 scales/biases directly, exactly the way an MLX
  // safetensors load would land them. The public API requires wq to be
  // uint32 (32/bits weights packed per uint32 — 8 weights/uint32 for
  // bits=4). Tiny shapes — only the dispatch surface matters.
  const int Nq = 32;      // out features
  const int Kq = 64;      // in features (one quant group)
  const int qbits = 4;
  const int qgroup = 64;
  array x_q = random::uniform({1, /*M=*/4, Kq});
  // wq packs 8 4-bit weights per uint32 -> (Nq, Kq/8) uint32.
  array wq = zeros({Nq, Kq / 8}, uint32);
  array scales = random::uniform({Nq, Kq / qgroup});
  array biases = random::uniform({Nq, Kq / qgroup});
  array qm = quantized_matmul(x_q, wq, scales, biases,
                              /*transpose=*/true,
                              /*group_size=*/qgroup,
                              /*bits=*/qbits);

  // Force evaluation so the backend dispatch -> kernel symbols get pulled in
  // and the actual GPU kernels execute.
  eval({c, n, sm, rq, o, cv, qm});

  std::cout << "qwen_smoke: forward-surface link OK\n";
  return 0;
}
