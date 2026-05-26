// Copyright © 2025 Apple Inc.
//
// K80 port: minimal dequant-only version of affine_quantize.cu. We only need
// affine_dequantize on the inference path (qwen3.6:35b-mlx is pre-quantized;
// weight quantization is a conversion-time op), so the affine_quantize kernel
// + host function are intentionally omitted — keeps us out of the half-arith
// + bit-packing mess that blocked the full file on CUDA 11.4.
//
// Two K80-specific patches vs. upstream:
//
//   1. `cg::this_grid().dim_blocks()` is a CUDA 11.6+ API; not in 11.4. Replaced
//      with `gridDim.x * blockDim.x` which computes the identical value.
//
//   2. `static_cast<T>(int_expr)` where T = __half / __nv_bfloat16 is ambiguous
//      on CUDA 11.4: both `T(float)` and `T(double)` overloads match an int
//      argument equally. Routed through a `to_t<T>(int)` helper that casts via
//      float first, which both half types accept unambiguously.
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

// K80 patch (2): unambiguous int → half/bfloat16 conversion via float.
template <typename T>
__device__ __forceinline__ T to_t(int x) {
  return static_cast<T>(static_cast<float>(x));
}

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
  T scale = scales[gindex];
  T bias = biases[gindex];
  out += oindex;

  if constexpr (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = to_t<T>(w[0] & 0x7) * scale + bias;
    out[1] = to_t<T>((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (to_t<T>((w[0] & 0xc0) >> 6) + to_t<T>((w[1] & 0x1) << 2)) *
            scale +
        bias;
    out[3] = to_t<T>((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = to_t<T>((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (to_t<T>((w[1] & 0x80) >> 7) + to_t<T>((w[2] & 0x3) << 1)) *
            scale +
        bias;
    out[6] = to_t<T>((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = to_t<T>((w[2] & 0xe0) >> 5) * scale + bias;
  } else if constexpr (bits == 5) {
    w += offset * bytes_per_pack;
    out[0] = to_t<T>(w[0] & 0x1f) * scale + bias;
    out[1] = (to_t<T>((w[0] & 0xe0) >> 5) + to_t<T>((w[1] & 0x3) << 3)) *
            scale +
        bias;
    out[2] = to_t<T>((w[1] & 0x7c) >> 2) * scale + bias;
    out[3] = (to_t<T>((w[1] & 0x80) >> 7) + to_t<T>((w[2] & 0xf) << 1)) *
            scale +
        bias;
    out[4] = (to_t<T>((w[2] & 0xf0) >> 4) + to_t<T>((w[3] & 0x1) << 4)) *
            scale +
        bias;
    out[5] = to_t<T>((w[3] & 0x3e) >> 1) * scale + bias;
    out[6] = (to_t<T>((w[3] & 0xc0) >> 6) + to_t<T>((w[4] & 0x7) << 2)) *
            scale +
        bias;
    out[7] = to_t<T>((w[4] & 0xf8) >> 3) * scale + bias;
  } else if constexpr (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = to_t<T>(w[0] & 0x3f) * scale + bias;
    out[1] = (to_t<T>((w[0] >> 6) & 0x03) + to_t<T>((w[1] & 0x0f) << 2)) *
            scale +
        bias;
    out[2] = (to_t<T>((w[1] >> 4) & 0x0f) + to_t<T>((w[2] & 0x03) << 4)) *
            scale +
        bias;
    out[3] = to_t<T>((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    // bits in {2, 4, 8}: one byte packs `pack_factor` weights uniformly.
    uint32_t val = w[offset];
#pragma unroll
    for (int i = 0; i < pack_factor; i++) {
      uint8_t d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * to_t<T>(d) + bias;
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
