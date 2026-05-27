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

  // --- quantized_matmul probe: real layer-0 DeltaNet in_proj_qkv ---
  // The next inference step for token 42 in layer 0 (a DeltaNet layer) is
  // projecting the rms-normed embed to the qkv stream. The weight is real
  // q4 packed: in_proj_qkv.weight U32 [8192, 256], dequant -> bf16
  // [8192, 2048]. matmul shape: (1, 2048) @ (2048, 8192) -> (1, 8192).
  //
  // Exercises the K80 dequant+cuBLAS fallback from PR #197 on REAL packed
  // weights for the first time (qwen_smoke uses hand-shaped zeros).
  auto get = [&](const std::string& k) -> const array& {
    auto it = tensors.find(k);
    if (it == tensors.end()) {
      std::cerr << "qwen_load: missing tensor '" << k << "'\n";
      std::exit(5);
    }
    return it->second;
  };
  const std::string qkv_pre = "language_model.model.layers.0.linear_attn.in_proj_qkv";
  const array& qkv_w = get(qkv_pre + ".weight");      // U32 [8192, 256]
  const array& qkv_s = get(qkv_pre + ".scales");      // BF16 [8192, 32]
  const array& qkv_b = get(qkv_pre + ".biases");      // BF16 [8192, 32]
  array qkv_proj = quantized_matmul(
      normed, qkv_w, qkv_s, qkv_b,
      /*transpose=*/true,
      /*group_size=*/64,
      /*bits=*/4,
      /*mode=*/"affine");
  array qkv_proj_f32 = astype(qkv_proj, float32);
  array qkv_proj_abs_mean = mean(abs(qkv_proj_f32), /*keepdims=*/false);
  eval({qkv_proj_f32, qkv_proj_abs_mean});
  std::cout << "qwen_load: in_proj_qkv(rms_normed) shape=["
            << qkv_proj.shape(0) << "," << qkv_proj.shape(1) << "]"
            << " abs_mean=" << qkv_proj_abs_mean.item<float>() << "\n";
  std::cout << "qwen_load: in_proj_qkv [0,0]=" << qkv_proj_f32.data<float>()[0]
            << " [0,1]=" << qkv_proj_f32.data<float>()[1] << "\n";

  // --- conv1d probe: real layer-0 DeltaNet conv1d ---
  // Depthwise conv1d (groups = C_out = 8192) with kw=4 over the q/k/v stream.
  // For an all-ones input [1, 4, 8192] with padding=0, output is [1, 1, 8192]
  // where out[0, 0, c] = sum_k weight[c, k, 0] (kernel sum per channel).
  // Exercises Phase A's gemm_conv path (PR #194 / commit 39abb387) on REAL
  // BF16 conv weights for the first time.
  const array& cv_w = get("language_model.model.layers.0.linear_attn.conv1d.weight");
  array cv_input = ones({1, 4, 8192}, bfloat16);
  array cv_out = conv1d(cv_input, cv_w,
                        /*stride=*/1, /*padding=*/0,
                        /*dilation=*/1, /*groups=*/8192);
  array cv_out_f32 = astype(cv_out, float32);
  array cv_abs_mean = mean(abs(cv_out_f32), /*keepdims=*/false);
  eval({cv_out_f32, cv_abs_mean});
  std::cout << "qwen_load: conv1d(ones, layer_0.conv1d.w) shape=["
            << cv_out.shape(0) << "," << cv_out.shape(1) << "," << cv_out.shape(2) << "]"
            << " abs_mean=" << cv_abs_mean.item<float>() << "\n";
  std::cout << "qwen_load: conv1d out [0,0,0..2] = "
            << cv_out_f32.data<float>()[0] << " "
            << cv_out_f32.data<float>()[1] << " "
            << cv_out_f32.data<float>()[2] << "\n";

  // --- silu probe: next op after conv1d on DeltaNet q/k/v gates ---
  // DeltaNet applies silu (a.k.a. swish: x * sigmoid(x)) to the conv1d
  // output before splitting into q/k/v streams. Exercises the elementwise
  // unary kernels in libmlx.a (sigmoid + multiply, both wired since Phase A).
  // Ground truth: silu(-0.061722) = -0.02991, silu(-0.045959) = -0.02245,
  // silu(0.067902) = 0.03510.
  array silu_out = sigmoid(cv_out) * cv_out;  // silu = sigmoid(x) * x
  array silu_out_f32 = astype(silu_out, float32);
  array silu_abs_mean = mean(abs(silu_out_f32), /*keepdims=*/false);
  eval({silu_out_f32, silu_abs_mean});
  std::cout << "qwen_load: silu(conv1d) abs_mean=" << silu_abs_mean.item<float>() << "\n";
  std::cout << "qwen_load: silu out [0,0,0..2] = "
            << silu_out_f32.data<float>()[0] << " "
            << silu_out_f32.data<float>()[1] << " "
            << silu_out_f32.data<float>()[2] << "\n";

  std::cout << "qwen_load: load + reduce + take + dequant + rms_norm + qproj + conv1d + silu OK\n";
  return 0;
}
