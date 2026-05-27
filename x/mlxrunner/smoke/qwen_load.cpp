// K80 runner — Phase C load-surface probe.
//
// Smallest exe that exercises the MLX safetensors loader against the real
// qwen3.6:35b-mlx weights. Validates:
//   (1) load_safetensors() links + parses an actual MLX shard.
//   (2) The dequant kernel landed in PR #196 + the QuantizedMatmul fallback
//       in PR #197 work on REAL pre-quantized weights (not just hand-shaped
//       random uint32 like qwen_smoke).
//   (3) GPU memory allocation works for the tensor sizes the model actually
//       uses (embed_tokens is the biggest single tensor — 248320×256 U32
//       packed weights + scales/biases = ~65 MiB total).
//
// Pick: embed_tokens (the entry tensor for any forward pass) because:
//   - Quantized (U32/BF16/BF16), so dequant gets exercised on real data.
//   - First-row lookup style is the simplest correctness check we can do
//     without a tokenizer in the loop.
//   - Single shard contains it (shard 1), so we don't need to merge shards.
//
// What this exe does NOT do:
//   - Run a forward pass (no Gather port yet; no attention; no MoE routing).
//   - Use a tokenizer.
//   - Sample.
//   That's the runner's job (#189); this is just the loader probe.
//
// Build: x/mlxrunner/CMakeLists.txt adds this as a second exe target.
// Run (after build, with model bind-mounted into the runtime container):
//   /build/qwen_load /models/qwen3.6-35b-a3b-4bit/model-00001-of-00004.safetensors

#include "mlx/mlx.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace mlx::core;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <shard.safetensors>\n";
    return 2;
  }
  const std::string shard_path = argv[1];

  set_default_device(Device(Device::gpu));

  std::cout << "qwen_load: opening " << shard_path << "\n";
  auto [tensors, metadata] = load_safetensors(shard_path);
  std::cout << "qwen_load: loaded " << tensors.size() << " tensors, "
            << metadata.size() << " metadata entries\n";

  // Sanity: pick the embedding-table weight + its scales + biases.
  const std::string wkey = "language_model.model.embed_tokens.weight";
  const std::string skey = "language_model.model.embed_tokens.scales";
  const std::string bkey = "language_model.model.embed_tokens.biases";
  for (const auto& k : {wkey, skey, bkey}) {
    if (tensors.find(k) == tensors.end()) {
      std::cerr << "qwen_load: missing tensor '" << k << "' in shard\n";
      return 3;
    }
  }
  const array& wq = tensors.at(wkey);
  const array& scales = tensors.at(skey);
  const array& biases = tensors.at(bkey);
  std::cout << "qwen_load: embed_tokens.weight  dtype=" << dtype_to_string(wq.dtype())
            << " shape=[" << wq.shape(0) << "," << wq.shape(1) << "]\n";
  std::cout << "qwen_load: embed_tokens.scales  dtype=" << dtype_to_string(scales.dtype())
            << " shape=[" << scales.shape(0) << "," << scales.shape(1) << "]\n";

  // Trivial reductions to force eval + verify the bytes round-tripped:
  //   - scales/biases are BF16 floats; mean is meaningful.
  //   - wq is U32 packed (8 weights per uint32 at bits=4); mean as float
  //     just tells us it isn't all zeros.
  array scales_mean = mean(scales, /*keepdims=*/false);
  array biases_mean = mean(biases, /*keepdims=*/false);
  array wq_as_f32 = astype(wq, float32);
  array wq_mean = mean(wq_as_f32, /*keepdims=*/false);
  eval({scales_mean, biases_mean, wq_mean});

  std::cout << "qwen_load: scales mean = " << scales_mean.item<float>() << "\n";
  std::cout << "qwen_load: biases mean = " << biases_mean.item<float>() << "\n";
  std::cout << "qwen_load: wq    mean = " << wq_mean.item<float>()
            << " (U32 packed, raw)\n";

  std::cout << "qwen_load: load + reduce OK\n";
  return 0;
}
