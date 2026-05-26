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
//   - quantize() / quantized_matmul() — those are weight-conversion ops that
//     dispatch to affine_quantize (currently a throw-stub). qwen3.5 loads
//     pre-quantized weights and only needs affine_dequantize at inference
//     time, which we'll exercise once task #13's dequant+cuBLAS reroute
//     lands.
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

  // Force evaluation so the backend dispatch -> kernel symbols get pulled in
  // and the actual GPU kernels execute.
  eval({c, n, sm, rq, o, cv});

  std::cout << "qwen_smoke: forward-surface link OK\n";
  return 0;
}
