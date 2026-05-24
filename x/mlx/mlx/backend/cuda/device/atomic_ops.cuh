// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/complex.cuh"
#include "mlx/backend/cuda/device/fp16_math.cuh"

#include <cstdint>
#include <cstring>

// K80 port: libcu++ cuda::atomic_ref<T, thread_scope_device> hard-gates at sm_60.
// Implement the device atomics with native CUDA atomics + CAS loops that work on
// sm_37 (incl. 32-bit-word CAS for 16-bit half/bf16, which have no native atomic).

namespace mlx::core::cu {

// Read-modify-write a 16-bit value (half/bf16) via CAS on its containing 32-bit
// word. op(cur) -> new value.
template <typename Half, typename Op>
inline __device__ void atomic_rmw16(Half* out, Op op) {
  auto uaddr = reinterpret_cast<uintptr_t>(out);
  auto* base = reinterpret_cast<unsigned int*>(uaddr & ~uintptr_t(3));
  unsigned int shift = static_cast<unsigned int>(uaddr & 3u) * 8u;
  unsigned int old = *base, assumed;
  do {
    assumed = old;
    unsigned short cur_bits =
        static_cast<unsigned short>((assumed >> shift) & 0xffffu);
    Half cur;
    memcpy(&cur, &cur_bits, sizeof(Half));
    Half nxt = op(cur);
    unsigned short nxt_bits;
    memcpy(&nxt_bits, &nxt, sizeof(Half));
    unsigned int word = (assumed & ~(0xffffu << shift)) |
        (static_cast<unsigned int>(nxt_bits) << shift);
    old = atomicCAS(base, assumed, word);
  } while (assumed != old);
}

// Read-modify-write a 4- or 8-byte value via CAS (bit-cast through an integer).
template <typename T, typename Op>
inline __device__ void atomic_rmw_word(T* out, Op op) {
  if constexpr (sizeof(T) == 4) {
    auto* base = reinterpret_cast<unsigned int*>(out);
    unsigned int old = *base, assumed;
    do {
      assumed = old;
      T cur;
      memcpy(&cur, &assumed, 4);
      T nxt = op(cur);
      unsigned int nb;
      memcpy(&nb, &nxt, 4);
      old = atomicCAS(base, assumed, nb);
    } while (assumed != old);
  } else {
    auto* base = reinterpret_cast<unsigned long long*>(out);
    unsigned long long old = *base, assumed;
    do {
      assumed = old;
      T cur;
      memcpy(&cur, &assumed, 8);
      T nxt = op(cur);
      unsigned long long nb;
      memcpy(&nb, &nxt, 8);
      old = atomicCAS(base, assumed, nb);
    } while (assumed != old);
  }
}

template <typename T>
inline __device__ void atomic_add(T* out, T val) {
  if constexpr (
      cuda::std::is_same_v<T, __half> ||
      cuda::std::is_same_v<T, __nv_bfloat16>) {
    atomic_rmw16(out, [val](T c) { return c + val; });
  } else if constexpr (
      cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, int> ||
      cuda::std::is_same_v<T, unsigned int> ||
      cuda::std::is_same_v<T, unsigned long long>) {
    atomicAdd(out, val);
  } else {
    atomic_rmw_word(out, [val](T c) { return c + val; });
  }
}

template <typename T>
inline __device__ void atomic_prod(T* out, T val) {
  if constexpr (
      cuda::std::is_same_v<T, __half> ||
      cuda::std::is_same_v<T, __nv_bfloat16>) {
    atomic_rmw16(out, [val](T c) { return c * val; });
  } else {
    atomic_rmw_word(out, [val](T c) { return c * val; });
  }
}

template <typename T>
inline __device__ void atomic_max(T* out, T val) {
  if constexpr (
      cuda::std::is_same_v<T, __half> ||
      cuda::std::is_same_v<T, __nv_bfloat16>) {
    atomic_rmw16(out, [val](T c) { return max(c, val); });
  } else {
    atomic_rmw_word(out, [val](T c) { return max(c, val); });
  }
}

template <typename T>
inline __device__ void atomic_min(T* out, T val) {
  if constexpr (
      cuda::std::is_same_v<T, __half> ||
      cuda::std::is_same_v<T, __nv_bfloat16>) {
    atomic_rmw16(out, [val](T c) { return min(c, val); });
  } else {
    atomic_rmw_word(out, [val](T c) { return min(c, val); });
  }
}

template <typename T>
inline __device__ void atomic_add_general(T* out, T val) {
  atomic_add(out, val);
}

inline __device__ void atomic_add(complex64_t* out, complex64_t val) {
  // std::complex is layout-compatible with float[2] (real, imag).
  float* p = reinterpret_cast<float*>(out);
  atomicAdd(p, val.real());
  atomicAdd(p + 1, val.imag());
}

} // namespace mlx::core::cu
