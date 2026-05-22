#pragma once
// K80 port: minimal no-op stub for nvtx3.
//
// MLX uses NVTX only for profiler range annotations (`nvtx3::scoped_range`).
// NVTX3 headers aren't present in the CUDA 11.4 builder, and the annotations are
// non-essential (Nsight/nvprof markers). This stub satisfies the includes and the
// only API MLX uses, with zero runtime effect. Replace with real NVTX if profiling
// the MLX path becomes necessary.
#include <string>

namespace nvtx3 {

struct scoped_range {
  scoped_range() = default;
  explicit scoped_range(const char*) {}
  explicit scoped_range(const std::string&) {}
};

} // namespace nvtx3
