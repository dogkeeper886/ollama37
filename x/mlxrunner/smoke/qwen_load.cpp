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
#include "mlx/dtype_utils.h"  // dtype_to_string (not pulled in by mlx.h)

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
  //
  // NOTE: array::item<float>() does `*data<float>()` — a raw pointer
  // dereference, NOT a dtype conversion. Reading a BF16 buffer as float*
  // yields ~1e-41 denormal garbage (bf16 bits in the low 2 bytes, high
  // 2 bytes whatever's next). astype(..., float32) first so item reads
  // an actual fp32 scalar.
  array scales_mean = astype(mean(scales, /*keepdims=*/false), float32);
  array biases_mean = astype(mean(biases, /*keepdims=*/false), float32);
  array wq_as_f32 = astype(wq, float32);
  array wq_mean = mean(wq_as_f32, /*keepdims=*/false);
  eval({scales_mean, biases_mean, wq_mean});

  std::cout << "qwen_load: scales mean = " << scales_mean.item<float>() << "\n";
  std::cout << "qwen_load: biases mean = " << biases_mean.item<float>() << "\n";
  std::cout << "qwen_load: wq    mean = " << wq_mean.item<float>()
            << " (U32 packed, raw)\n";

  // --- Gather (embed-lookup) probe ---
  // Look up a specific row of embed_tokens.scales via `take(scales, [42], axis=0)`.
  // Validates Gather::eval_gpu on real safetensors data.
  array token_id = array({42}, {1}, int32);
  array scales_row = take(scales, token_id, /*axis=*/0);  // shape [1, 32]
  array scales_row_f32 = astype(scales_row, float32);
  array scales_row_mean = mean(scales_row_f32, /*keepdims=*/false);
  eval({scales_row_f32, scales_row_mean});
  std::cout << "qwen_load: take(scales, [42], 0) shape=["
            << scales_row.shape(0) << "," << scales_row.shape(1) << "]"
            << " mean=" << scales_row_mean.item<float>() << "\n";
  std::cout << "qwen_load: take(scales, [42], 0)[0,0] = "
            << scales_row_f32.data<float>()[0] << "\n";

  // --- Dequantize (real packed weights) probe ---
  // Gather row 42 from weight + biases too, then dequantize. Exercises
  // affine_dequantize (PR #196) + fast::Quantize::eval_gpu on REAL q4
  // packed weights — not the hand-shaped uint32 zeros qwen_smoke uses.
  // For bits=4 group=64: wq [1, 256] uint32 -> w [1, 2048] bf16.
  array biases_row = take(biases, token_id, /*axis=*/0);  // [1, 32]
  array wq_row = take(wq, token_id, /*axis=*/0);          // [1, 256] uint32
  array w_full_row =
      dequantize(wq_row, scales_row, biases_row,
                 /*group_size=*/64, /*bits=*/4, /*mode=*/"affine");
  array w_full_row_f32 = astype(w_full_row, float32);
  array w_full_row_mean = mean(w_full_row_f32, /*keepdims=*/false);
  eval({w_full_row_f32, w_full_row_mean});
  std::cout << "qwen_load: dequantize(embed row 42) shape=["
            << w_full_row.shape(0) << "," << w_full_row.shape(1) << "]"
            << " mean=" << w_full_row_mean.item<float>() << "\n";
  std::cout << "qwen_load: dequant row 42 first 5: ";
  for (int i = 0; i < 5; i++) {
    std::cout << w_full_row_f32.data<float>()[i] << " ";
  }
  std::cout << "\n";

  // --- RMSNorm probe ---
  // Apply layer-0 input_layernorm to the dequant embed row. Exercises:
  //   - fast::rms_norm dispatch (in libmlx.a from Phase A)
  //   - the row-reduce path that PR #200 patched (sum-of-squares is a
  //     row reduce on the last dim with float accumulator AccT pattern).
  // The bf16 norm-gain weight + bf16 input both stress the precision
  // path that was the original BF16-mean bug.
  const std::string ln_key =
      "language_model.model.layers.0.input_layernorm.weight";
  if (tensors.find(ln_key) == tensors.end()) {
    std::cerr << "qwen_load: missing tensor '" << ln_key << "'\n";
    return 4;
  }
  const array& norm_gain = tensors.at(ln_key);  // bf16 [2048]
  array normed = fast::rms_norm(w_full_row, norm_gain, /*eps=*/1e-6f);
  array normed_f32 = astype(normed, float32);
  array normed_abs_mean = mean(abs(normed_f32), /*keepdims=*/false);
  eval({normed_f32, normed_abs_mean});
  std::cout << "qwen_load: rms_norm(dequant row 42) shape=["
            << normed.shape(0) << "," << normed.shape(1) << "]"
            << " abs_mean=" << normed_abs_mean.item<float>() << "\n";
  std::cout << "qwen_load: rms_norm first 5: ";
  for (int i = 0; i < 5; i++) {
    std::cout << normed_f32.data<float>()[i] << " ";
  }
  std::cout << "\n";

  std::cout << "qwen_load: load + reduce + take + dequant + rms_norm OK\n";
  return 0;
}
