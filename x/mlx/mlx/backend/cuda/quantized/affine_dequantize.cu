// Copyright © 2025 Apple Inc.
//
// K80 port: minimal dequant-only version of affine_quantize.cu. We only need
// affine_dequantize on the inference path (qwen3.6:35b-mlx is pre-quantized;
// weight quantization is a conversion-time op), so the affine_quantize kernel
// + host function are intentionally omitted — keeps us out of the half-arith
// + bit-packing mess that blocked the full file on CUDA 11.4.
//
// Three K80-specific patches vs. upstream:
//
//   1. `cg::this_grid().dim_blocks()` is a CUDA 11.6+ API; not in 11.4.
//      Replaced with `gridDim.x * blockDim.x` (identical value).
//
//   2. Arithmetic on `__half` / `__nv_bfloat16` is ambiguous on CUDA 11.4
//      whenever mlx's `complex64` operator overloads are in scope, because
//      both half types expose many `operator T() const` conversions
//      (int / uint / float / double / bool / …) that match the complex
//      operators equally with the built-in promotion. The kernel now does
//      *all* arithmetic in `float`, with a single `static_cast<T>` of the
//      final result. Avoids both half-half and half-int ambiguity in one
//      shot.
//
//   3. Same `static_cast<T>(int_expr)` int→half ambiguity (multiple ctor
//      overloads match int) is also gone for free since we never construct
//      `T` from an int — every value goes through float first.
//
// All bits (2/3/4/5/6/8) and group sizes (32/64/128) compile; qwen3.6:35b-mlx
// uses bits=4 group=64. The unused configurations stay in for upstream parity
// and cost only template instantiation time.

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {
namespace cu {

namespace cg = cooperative_groups;

template <typename T, int group_size, int bits>
__global__ void affine_dequantize(
    const uint8_t* w,
    const T* scales,
    const T* biases,
    T* out,
    size_t size) {
  auto block_size = blockDim;
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  // K80 patch (1): gridDim.x replaces cg::this_grid().dim_blocks().x (CUDA 11.6+).
  auto grid_dim_x = static_cast<size_t>(gridDim.x) * block_size.x;

  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);

  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  // K80 patch (2): hoist scale/bias to float once, do all arithmetic in float,
  // cast back to T at the very end. Avoids any operator on T (half / bfloat16).
  const float scale = static_cast<float>(scales[gindex]);
  const float bias = static_cast<float>(biases[gindex]);
  out += oindex;

  // Helper: pack one unsigned int sample → fp16/bf16 output via float.
  auto write = [&](int slot, unsigned int d) {
    out[slot] = static_cast<T>(scale * static_cast<float>(d) + bias);
  };

  if constexpr (bits == 3) {
    w += offset * bytes_per_pack;
    write(0, (w[0]) & 0x7);
    write(1, (w[0] >> 3) & 0x7);
    write(2, ((w[0] >> 6) & 0x3) | ((w[1] & 0x1) << 2));
    write(3, (w[1] >> 1) & 0x7);
    write(4, (w[1] >> 4) & 0x7);
    write(5, ((w[1] >> 7) & 0x1) | ((w[2] & 0x3) << 1));
    write(6, (w[2] >> 2) & 0x7);
    write(7, (w[2] >> 5) & 0x7);
  } else if constexpr (bits == 5) {
    w += offset * bytes_per_pack;
    write(0, (w[0]) & 0x1f);
    write(1, ((w[0] >> 5) & 0x7) | ((w[1] & 0x3) << 3));
    write(2, (w[1] >> 2) & 0x1f);
    write(3, ((w[1] >> 7) & 0x1) | ((w[2] & 0xf) << 1));
    write(4, ((w[2] >> 4) & 0xf) | ((w[3] & 0x1) << 4));
    write(5, (w[3] >> 1) & 0x1f);
    write(6, ((w[3] >> 6) & 0x3) | ((w[4] & 0x7) << 2));
    write(7, (w[4] >> 3) & 0x1f);
  } else if constexpr (bits == 6) {
    w += offset * bytes_per_pack;
    write(0, (w[0]) & 0x3f);
    write(1, ((w[0] >> 6) & 0x03) | ((w[1] & 0x0f) << 2));
    write(2, ((w[1] >> 4) & 0x0f) | ((w[2] & 0x03) << 4));
    write(3, (w[2] >> 2) & 0x3f);
  } else {
    // bits in {2, 4, 8}: one byte packs `pack_factor` weights uniformly.
    unsigned int val = w[offset];
#pragma unroll
    for (int i = 0; i < pack_factor; i++) {
      unsigned int d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else { // bits == 8
        d = val;
      }
      write(i, d);
    }
  }
}

} // namespace cu

// Local dispatch helpers (host-side; same shape as upstream's, kept local so
// this file is self-contained — affine_quantize.cu isn't built on K80).
namespace {

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32: f(std::integral_constant<int, 32>{}); break;
    case 64: f(std::integral_constant<int, 64>{}); break;
    case 128: f(std::integral_constant<int, 128>{}); break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 5: f(std::integral_constant<int, 5>{}); break;
    case 6: f(std::integral_constant<int, 6>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
  }
}

} // namespace

void affine_dequantize(
    const array& wq,
    const array& scales,
    const array& biases,
    array& w,
    int group_size_,
    int bits_,
    cu::CommandEncoder& enc,
    const Stream& /* s */) {
  // Same packs_per_int / grid_shape derivation as upstream (matches the kernel's
  // index math: each thread emits `pack_factor` output elements per uint8 read).
  constexpr int uint8_per_uint32 = 4;
  int packs_per_int;
  switch (bits_) {
    case 3:
    case 5: packs_per_int = 8; break;
    case 6: packs_per_int = 4; break;
    default: packs_per_int = 8 / bits_;
  }

  size_t size = w.size() / packs_per_int;
  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;

  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_input_array(biases);
  enc.set_output_array(w);
  dispatch_float_types(w.dtype(), "affine_dequantize", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        auto kernel = cu::affine_dequantize<T, group_size.value, bits.value>;
        auto [num_blocks, block_dims] =
            get_launch_args(size, grid_shape, w.strides(), large);
        enc.add_kernel_node(
            kernel,
            num_blocks,
            block_dims,
            gpu_ptr<uint8_t>(wq),
            gpu_ptr<T>(scales),
            gpu_ptr<T>(biases),
            gpu_ptr<T>(w),
            w.size());
      });
    });
  });
}

} // namespace mlx::core
