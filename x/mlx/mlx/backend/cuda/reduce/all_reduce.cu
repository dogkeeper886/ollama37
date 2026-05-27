// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename ReduceOp, int N = 4>
__global__ void all_reduce(T* in, U* out, size_t block_step, size_t size) {
  // TODO: Process multiple "rows" in each thread
  constexpr int M = 1;

  // K80 port: when U is half/bf16 (sizeof < 4), accumulating sums in U
  // loses precision catastrophically — BF16 has only 8 mantissa bits, so
  // a running sum over many small values saturates long before the end
  // (mean(248320×32 BF16 scales) returned 1.98e-41 vs. true 1.14e-05
  // before this fix). Accumulate in float for half/bf16; cast back to U
  // on the output write. Integral U was already promoted to int32 by
  // ReduceResult<Sum>, so the only case where sizeof(U)<4 is half/bf16.
  // Matches the softmax fp32-accumulator pattern (commit 36859c6c).
  using AccT = cuda::std::conditional_t<(sizeof(U) < 4), float, U>;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const AccT init = cast_to<AccT>(cu::ReduceInit<ReduceOp, T>::value());
  ReduceOp op;

  T vals[N];
  AccT accs[M];
  accs[0] = init;

  size_t start = mlx_block_rank() * block_step;
  size_t end = start + block_step;
  size_t check = min(end, size);

  size_t i = start;
  for (; i + block.size() * N <= check; i += block.size() * N) {
    cub::LoadDirectBlockedVectorized<T, N>(block.thread_rank(), in + i, vals);
    for (int j = 0; j < N; j++) {
      accs[0] = op(accs[0], cast_to<AccT>(vals[j]));
    }
  }

  if (i < check) {
    cub::LoadDirectBlocked(
        block.thread_rank(), in + i, vals, check - i, cast_to<T>(init));
    for (int i = 0; i < N; i++) {
      accs[0] = op(accs[0], cast_to<AccT>(vals[i]));
    }
  }

  // K80 port: __shared__ array of a non-trivially-constructible AccT
  // (complex; half/bf16 also need byte-buffer treatment from the earlier
  // K80 patch) triggers "initializer not allowed"; use an uninitialized
  // byte buffer. AccT is float for our half/bf16 case here, but keep the
  // pattern uniform so it works for U=complex too.
  __shared__ __align__(16) unsigned char shared_accumulators_bytes[32 * sizeof(AccT)];
  AccT* shared_accumulators = reinterpret_cast<AccT*>(shared_accumulators_bytes);
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[mlx_block_rank()] = cast_to<U>(accs[0]);
  }
}

} // namespace cu

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 8;

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto get_args = [](int size, int N) {
    int threads = std::min(512, (size + N - 1) / N);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int reductions_per_step = threads * N;
    size_t steps_needed =
        (size + reductions_per_step - 1) / reductions_per_step;

    int blocks;
    if (steps_needed < 32) {
      blocks = 1;
    } else if (steps_needed < 128) {
      blocks = 32;
    } else if (steps_needed < 512) {
      blocks = 128;
    } else if (steps_needed < 1024) {
      blocks = 512;
    } else {
      blocks = 1024;
    }

    size_t steps_per_block = (steps_needed + blocks - 1) / blocks;
    size_t block_step = steps_per_block * reductions_per_step;

    return std::make_tuple(blocks, threads, block_step);
  };

  int blocks, threads;
  size_t block_step;
  size_t insize = in.size();
  Dtype dt = in.dtype();
  void* indata = gpu_ptr<void>(in);

  // Large array so allocate an intermediate and accumulate there
  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  encoder.set_input_array(in);
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
    encoder.add_temporary(intermediate);
    encoder.set_output_array(intermediate);
    dispatch_all_types(dt, [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;
        auto kernel = cu::all_reduce<T, U, OP, N_READS>;
        encoder.add_kernel_node(
            kernel,
            blocks,
            threads,
            indata,
            gpu_ptr<U>(intermediate),
            block_step,
            insize);
      });
    });

    // Set the input for the next step and recalculate the blocks
    indata = gpu_ptr<void>(intermediate);
    dt = intermediate.dtype();
    insize = intermediate.size();
    std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
    encoder.set_input_array(intermediate);
  }

  encoder.set_output_array(out);
  dispatch_all_types(dt, [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;
      auto kernel = cu::all_reduce<T, U, OP, N_READS>;
      encoder.add_kernel_node(
          kernel, blocks, threads, indata, gpu_ptr<U>(out), block_step, insize);
    });
  });
}

} // namespace mlx::core
