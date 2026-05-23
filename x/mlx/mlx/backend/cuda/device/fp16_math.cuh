// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Binary ops for half types.
///////////////////////////////////////////////////////////////////////////////

// K80 port: __hmax/__hmin are sm_80+ half intrinsics, absent on sm_37. Compute the
// half/bf16 max/min via float (we only build sm_37).
#define MLX_DEFINE_BINARY_OP(NAME, FLOAT_OP)                       \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x, T y) {                 \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return __float2half(FLOAT_OP(__half2float(x), __half2float(y))); \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return __float2bfloat16(                                     \
          FLOAT_OP(__bfloat162float(x), __bfloat162float(y)));     \
    } else {                                                       \
      return ::NAME(x, y);                                         \
    }                                                              \
  }

MLX_DEFINE_BINARY_OP(max, ::fmaxf)
MLX_DEFINE_BINARY_OP(min, ::fminf)

#undef MLX_DEFINE_BINARY_OP

///////////////////////////////////////////////////////////////////////////////
// Additional C++ operator overrides between half types and native types.
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
constexpr bool is_integral_except =
    cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, U>;

template <typename T, typename U>
constexpr bool is_arithmetic_except =
    cuda::std::is_arithmetic_v<T> && !cuda::std::is_same_v<T, U>;

#define MLX_DEFINE_HALF_OP(HALF, HALF2FLOAT, FLOAT2HALF, OP)          \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(HALF x, T y) {          \
    return FLOAT2HALF(HALF2FLOAT(x) OP static_cast<float>(y));        \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(T x, HALF y) {          \
    return FLOAT2HALF(static_cast<float>(x) OP HALF2FLOAT(y));        \
  }

#define MLX_DEFINE_HALF_CMP(HALF, HALF2FLOAT, OP)                       \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(HALF x, T y) {            \
    return HALF2FLOAT(x) OP static_cast<float>(y);                      \
  }                                                                     \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(T x, HALF y) {            \
    return static_cast<float>(y) OP HALF2FLOAT(x);                      \
  }

MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, +)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, -)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, *)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, /)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, +)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, -)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, *)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, /)
MLX_DEFINE_HALF_CMP(__half, __half2float, <)
MLX_DEFINE_HALF_CMP(__half, __half2float, >)
MLX_DEFINE_HALF_CMP(__half, __half2float, <=)
MLX_DEFINE_HALF_CMP(__half, __half2float, >=)
MLX_DEFINE_HALF_CMP(__half, __half2float, ==)
MLX_DEFINE_HALF_CMP(__half, __half2float, !=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, ==)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, !=)

// K80 port: CUDA provides native __half/__nv_bfloat16 arithmetic & comparison
// operators only for __CUDA_ARCH__ >= 530. On sm_37 (Kepler) those are absent, so
// half-op-half expressions fall back to the many implicit half->builtin
// conversions and become ambiguous ("more than one conversion function ...").
// Define the half-op-half operators (computed via float) for the sub-530 device
// build. Guarded to device + sub-530 so sm_53+ keeps the native operators.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)

#define MLX_DEFINE_HALF_HALF_OP(HALF, HALF2FLOAT, FLOAT2HALF, OP) \
  __forceinline__ __device__ HALF operator OP(HALF x, HALF y) {   \
    return FLOAT2HALF(HALF2FLOAT(x) OP HALF2FLOAT(y));            \
  }
#define MLX_DEFINE_HALF_HALF_CMP(HALF, HALF2FLOAT, OP)            \
  __forceinline__ __device__ bool operator OP(HALF x, HALF y) {   \
    return HALF2FLOAT(x) OP HALF2FLOAT(y);                        \
  }

MLX_DEFINE_HALF_HALF_OP(__half, __half2float, __float2half, +)
MLX_DEFINE_HALF_HALF_OP(__half, __half2float, __float2half, -)
MLX_DEFINE_HALF_HALF_OP(__half, __half2float, __float2half, *)
MLX_DEFINE_HALF_HALF_OP(__half, __half2float, __float2half, /)
MLX_DEFINE_HALF_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, +)
MLX_DEFINE_HALF_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, -)
MLX_DEFINE_HALF_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, *)
MLX_DEFINE_HALF_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, /)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, <)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, >)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, <=)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, >=)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, ==)
MLX_DEFINE_HALF_HALF_CMP(__half, __half2float, !=)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, <)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, >)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, <=)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, >=)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, ==)
MLX_DEFINE_HALF_HALF_CMP(__nv_bfloat16, __bfloat162float, !=)

#undef MLX_DEFINE_HALF_HALF_OP
#undef MLX_DEFINE_HALF_HALF_CMP

#endif // sm_37 half-op-half operators

#undef MLX_DEFINE_HALF_OP
#undef MLX_DEFINE_HALF_CMP

} // namespace mlx::core::cu
