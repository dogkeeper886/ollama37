// Copyright © 2025 Apple Inc.
//
// K80 port: this file originally dispatched between a cuDNN flash-attention
// path (sdpa_cudnn / sdpa_backward_cudnn) and the manual vector kernel in
// scaled_dot_product_attention.cu. The K80 build has no cuDNN frontend
// (cuDNN 8.2 on CUDA 11.4 lacks fe::graph), so the entire cuDNN path —
// cudnn_utils.h include, SDPACacheKey/build_sdpa_graph, supports_sdpa_cudnn,
// sdpa_cudnn, sdpa_backward_cudnn, plus the cuDNN-only helpers
// (prepare_sdpa_sinks/malloc_with_same_layout/unslice_kv/use_cudnn_for_decoding,
// init_cudnn_sdpa_cache) — has been removed. Forward eval always goes through
// the manual sdpa_vector kernel. Backward (VJP) is training-only and falls
// back to CPU; eval_gpu is unreachable and throws if reached.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <stdexcept>

namespace mlx::core {

namespace {

// Return pointer alignment of |x|'s data. Lifted out of cudnn_utils.h (which
// the K80 strip removed) — generic ptr-alignment helper, no cuDNN dependency.
inline uint8_t get_alignment(const array& x) {
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(gpu_ptr<void>(x));
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

array prepare_sdpa_input(const array& x, Stream s) {
  // SDPA kernel's requirements on inputs:
  // 1. last dim's stride be 1;
  // 2. pointer be aligned.
  if (x.strides(-1) != 1 || get_alignment(x) < 16) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

} // namespace

// Defined in scaled_dot_product_attention.cu.
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp);
void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  // K80: only the manual vector kernel is available. If it can't handle this
  // shape/dtype, fall back to the CPU primitive.
  return !supports_sdpa_vector(q, k, v, has_arr_mask, output_logsumexp);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array& out = outputs[0];

  std::optional<array> sinks;
  if (has_sinks_) {
    sinks = inputs.back();
  }

  sdpa_vector(q, k, v, scale_, out, do_causal_, sinks, s);
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // K80: no cuDNN backward; always fall back to the CPU primitive.
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[ScaledDotProductAttentionVJP] No GPU backward on the K80 build; "
      "use_fallback returns true so this path should be unreachable.");
}

} // namespace fast

} // namespace mlx::core
