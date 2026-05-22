// K80 port shim: cuda_fp8.h was added in CUDA 11.8; FP8 is an Ada/Hopper feature
// (sm_89+). Kepler (sm_37) has no FP8.
//
// MLX's CUDA backend includes this header (via device/unary_ops.cuh) and references
// __nv_fp8_e4m3 as a storage type in dtype dispatch. Provide placeholder 8-bit
// storage types so the code compiles; the FP8 compute paths are never taken on sm_37.
#pragma once

#include <cstdint>

struct __nv_fp8_e4m3 {
  uint8_t __x;
};

struct __nv_fp8_e5m2 {
  uint8_t __x;
};

typedef unsigned char __nv_fp8_storage_t;
