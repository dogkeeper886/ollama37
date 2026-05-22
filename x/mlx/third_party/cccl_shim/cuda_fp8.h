// K80 port shim: cuda_fp8.h was added in CUDA 11.8; FP8 is an Ada/Hopper feature
// (sm_89+). Kepler (sm_37) has no FP8.
//
// MLX's CUDA backend includes this header and references __nv_fp8_e4m3 in dtype
// dispatch / cast ops (e.g. device/unary_ops.cuh FromFP8). Provide placeholder
// 8-bit storage types with float conversion + construction so the cast code
// compiles. These paths are DEAD on sm_37 — no FP8 model runs on K80 — so the
// conversions are stubs (return 0 / store 0), not real e4m3/e5m2 codecs.
#pragma once

#include <cstdint>

struct __nv_fp8_e4m3 {
  uint8_t __x;
  __host__ __device__ __nv_fp8_e4m3() : __x(0) {}
  __host__ __device__ __nv_fp8_e4m3(float) : __x(0) {} // encode stub (unused on sm_37)
  __host__ __device__ operator float() const { return 0.0f; } // decode stub
};

struct __nv_fp8_e5m2 {
  uint8_t __x;
  __host__ __device__ __nv_fp8_e5m2() : __x(0) {}
  __host__ __device__ __nv_fp8_e5m2(float) : __x(0) {}
  __host__ __device__ operator float() const { return 0.0f; }
};

typedef unsigned char __nv_fp8_storage_t;
