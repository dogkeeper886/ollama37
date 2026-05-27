// K80 port: focused Gather::eval_gpu impl for the embed-lookup case.
//
// Upstream's mlx/backend/cuda/indexing.cpp is JIT/NVRTC-based — it builds
// kernel names dynamically per (dtype, NIDX, IDX_NDIM, LocT) and invokes
// them through cu::JitModule. The K80 build deferred the JIT subsystem
// (`cu::get_jit_module` and `JitModule::get_kernel` are throw-stubs in
// k80_runtime_stubs.cpp), so upstream's Gather::eval_gpu can't link here.
//
// This file implements the *single* shape we need for embedding lookup
// (qwen3.6-A3B's first forward-pass op):
//
//   take(embed_table, token_ids, axis=0)
//   -> Gather(axes={0}, slice_sizes={1, embed_table.shape[1], ...})
//   -> out[i_flat, *trail] = embed_table[token_ids[i_flat], *trail]
//
// Conditions for the K80 fast path:
//   - inputs.size() == 2          (source + a single index array)
//   - axes_.size() == 1 && axes_[0] == 0
//   - slice_sizes_[0] == 1
//   - slice_sizes_[1..] == src.shape[1..]   (copy whole trailing dims)
//
// Other Gather shapes (multi-axis gather, partial slices, multi-index)
// throw clearly — they don't fire on the qwen3.6-A3B forward pass we're
// chasing, and porting more of upstream's general gather requires either
// JIT bring-up or a much broader static dispatch matrix.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <stdexcept>
#include <string>

namespace mlx::core {

namespace cu {

// out[i] = src[indices[i / row_size] * row_size + (i % row_size)]
// row_size = product of src trailing dims (after axis 0).
// N_total = n_indices * row_size = out.size().
template <typename T, typename IdxT>
__global__ void gather_take_axis0(
    const T* src,
    const IdxT* indices,
    T* out,
    int32_t row_size,
    int32_t src_rows,
    int64_t n_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_total) {
    return;
  }
  int64_t n = i / row_size;
  int32_t k = static_cast<int32_t>(i % row_size);
  // Indices are tokens / positions; treat as unsigned index into src rows.
  // Negative or out-of-range indices in MLX gather are undefined behavior;
  // upstream relies on the frontend to validate. We do the same.
  IdxT idx_raw = indices[n];
  int64_t src_offset =
      static_cast<int64_t>(static_cast<int64_t>(idx_raw)) *
          static_cast<int64_t>(row_size) +
      k;
  out[i] = src[src_offset];
}

} // namespace cu

// ----------------------------------------------------------------------------
// Helpers (this file only)
// ----------------------------------------------------------------------------
namespace {

bool is_embed_shape(
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    const array& src) {
  if (axes.size() != 1 || axes[0] != 0) return false;
  if (slice_sizes.empty() || slice_sizes[0] != 1) return false;
  if (static_cast<int>(slice_sizes.size()) != src.ndim()) return false;
  for (int d = 1; d < src.ndim(); ++d) {
    if (slice_sizes[d] != src.shape(d)) return false;
  }
  return true;
}

template <typename F>
void dispatch_idx_type(Dtype idx_dtype, F&& f) {
  switch (idx_dtype) {
    case int32:  f(std::integral_constant<int, 0>{}); return;  // tag -> int32_t
    case uint32: f(std::integral_constant<int, 1>{}); return;
    case int64:  f(std::integral_constant<int, 2>{}); return;
    case uint64: f(std::integral_constant<int, 3>{}); return;
    default:
      throw std::runtime_error(
          std::string("[K80 Gather] unsupported index dtype: ") +
          dtype_to_string(idx_dtype));
  }
}

template <int Tag> struct IdxTagToType;
template <> struct IdxTagToType<0> { using type = int32_t; };
template <> struct IdxTagToType<1> { using type = uint32_t; };
template <> struct IdxTagToType<2> { using type = int64_t; };
template <> struct IdxTagToType<3> { using type = uint64_t; };

} // namespace

// ----------------------------------------------------------------------------
// Gather::eval_gpu
// ----------------------------------------------------------------------------
void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu (K80 embed-take)");

  // We only handle the single-index, axis-0, whole-row case on K80. Throw
  // clearly for anything else so the runtime trip pinpoints what to port
  // next (rather than silently producing wrong output).
  if (inputs.size() != 2) {
    throw std::runtime_error(
        std::string("[K80 Gather] unsupported: ") +
        std::to_string(inputs.size() - 1) +
        " index arrays; this build handles only the single-index "
        "embed-lookup case.");
  }
  const array& src = inputs[0];
  const array& indices = inputs[1];
  if (!is_embed_shape(axes_, slice_sizes_, src)) {
    throw std::runtime_error(
        "[K80 Gather] unsupported shape: only the embed-style case "
        "(axes={0}, slice_sizes={1, *src.shape[1..]}) is implemented.");
  }

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

  int64_t row_size = 1;
  for (int d = 1; d < src.ndim(); ++d) {
    row_size *= src.shape(d);
  }
  if (row_size > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(
        "[K80 Gather] embed row_size exceeds int32; this build assumes "
        "rows fit in int32 elements.");
  }
  const int32_t src_rows = src.shape(0);
  const int64_t n_total = static_cast<int64_t>(out.size());

  // Make sure src + indices are row-contiguous so the linear src_offset math
  // matches the kernel. set_input_array does the bookkeeping; if the caller
  // provided a non-contiguous source, MLX's planner would have already
  // arranged a copy upstream (Reduce/SDPA/etc. follow the same convention).
  encoder.set_input_array(src);
  encoder.set_input_array(indices);
  encoder.set_output_array(out);

  // 1D launch: enough threads to cover n_total, capped by block dim.
  constexpr int kBlock = 256;
  const int64_t num_blocks = (n_total + kBlock - 1) / kBlock;

  dispatch_all_types(src.dtype(), [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_idx_type(indices.dtype(), [&](auto idx_tag) {
      using IdxT = typename IdxTagToType<idx_tag.value>::type;
      auto kernel = cu::gather_take_axis0<T, IdxT>;
      encoder.add_kernel_node(
          kernel,
          static_cast<uint32_t>(num_blocks),
          kBlock,
          gpu_ptr<T>(src),
          gpu_ptr<IdxT>(indices),
          gpu_ptr<T>(out),
          static_cast<int32_t>(row_size),
          src_rows,
          n_total);
    });
  });
}

} // namespace mlx::core
