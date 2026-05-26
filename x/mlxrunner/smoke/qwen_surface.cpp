// K80 runner — Phase B link-surface probe.
//
// This isn't a runner. It's the smallest C++ exe that *names* the symbols a
// qwen3.5 forward pass needs, so the linker enumerates what libmlx.a still
// can't satisfy. We don't care if the kernels run; we care which references
// the static archive can't resolve. Each undefined ref is a punch-list item
// for Phase B (reroute to stub / cuBLAS / dequant) before the real runner
// can link.
//
// Build via x/mlxrunner/CMakeLists.txt; expect link failures the first time.

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

  // --- quantized matmul dispatch (qmm impls deferred -> linker tells us) ---
  auto qpacks = quantize(w, /*group_size=*/64, /*bits=*/4);
  array qm = quantized_matmul(a, qpacks[0], qpacks[1], qpacks[2],
                              /*transpose=*/true, /*group_size=*/64,
                              /*bits=*/4);

  // Force evaluation so the backend dispatch -> kernel symbols get pulled in.
  eval({c, n, sm, rq, o, cv, qm});

  std::cout << "qwen_smoke: forward-surface link OK\n";
  return 0;
}
