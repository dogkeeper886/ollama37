// Copyright © 2025 Apple Inc.
//
// K80 port: this file originally dispatched between a cuDNN convolution
// path (build_conv_graph + cached DnnGraph runs) and a GEMM-based fallback
// (gemm_conv / gemm_grouped_conv in conv/). The K80 build has no cuDNN
// frontend (cuDNN 8.2 on CUDA 11.4 lacks fe::graph), so the entire cuDNN
// path — cudnn_utils.h include, ConvCacheKey/conv_cache, get_conv_settings,
// build_conv_graph, group_transpose, prepare_args, register_args,
// init_cudnn_conv_cache, plus the multi-backend cuDNN guessing — is removed.
// Convolution::eval_gpu always calls the GEMM-based gemm_conv kernel from
// conv/gemm_conv.cu, which uses our cublas_gemm under the hood (no CUTLASS).

#include "mlx/backend/cuda/conv/conv.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out_) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  if (out_.size() == 0) {
    return;
  }
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  array in = inputs[0];
  array wt = inputs[1];
  array& out = out_;
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  // K80: GEMM-based conv only (im2col-style unfold in conv/gemm_conv.cu
  // followed by cuBLAS GEMM). Handles forward conv1d/2d/3d incl. grouped.
  gemm_conv(
      encoder,
      in,
      wt,
      out,
      kernel_strides_,
      padding_lo_,
      kernel_dilation_,
      input_dilation_,
      groups_,
      flip_,
      s);
}

} // namespace mlx::core
