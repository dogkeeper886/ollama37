// K80 port: runtime throw-stubs for the off-path CUDA-backend symbols that
// libmlx.a's on-path objects reference but whose impls were deferred
// (CUTLASS qmm/gemm trio, block-mask helpers, JIT runtime, distributed
// collectives, FFT/Hadamard/Scatter/Gather/Scan/SliceUpdate/QQMatmul/FP8,
// affine + fp quant/dequant). The pattern mirrors backend/no_cpu/primitives.cpp.
//
// These exist purely to satisfy the linker so qwen_smoke (and later the real
// runner exe) can link against libmlx.a. Every symbol throws std::runtime_error
// if reached at runtime — that surfaces "qwen3.5 hit an off-path primitive"
// loudly instead of silently corrupting. The qwen3.5-relevant primitives
// (Gather/Scatter/SliceUpdate/Scan + the quantized matmul family) keep these
// throw stubs only as long as they aren't on the inference hot path; when one
// trips at runtime, port the real impl and remove its stub here.

#include "mlx/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/distributed/primitives.h"

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/gemms/block_mask.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/gemms/gather_gemm.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/jit_module.h"

#include <stdexcept>
#include <string>

namespace mlx::core {

namespace {
[[noreturn]] void k80_unimpl(const char* what) {
  throw std::runtime_error(
      std::string("K80 build: ") + what +
      " not implemented (off-path stub in k80_runtime_stubs.cpp)");
}
} // namespace

// ============================================================================
// UnaryPrimitive eval_gpu — on-path-but-unported primitives. Each one throws
// at runtime; if qwen3.5 trips one we port the real impl and drop the stub.
// ============================================================================

#define K80_UNARY_EVAL_GPU(Cls)                                      \
  void Cls::eval_gpu(const std::vector<array>&, array&) {            \
    k80_unimpl(#Cls "::eval_gpu");                                   \
  }

K80_UNARY_EVAL_GPU(FFT)
K80_UNARY_EVAL_GPU(Gather)
K80_UNARY_EVAL_GPU(GatherAxis)
K80_UNARY_EVAL_GPU(Hadamard)
K80_UNARY_EVAL_GPU(MaskedScatter)
K80_UNARY_EVAL_GPU(QQMatmul)
K80_UNARY_EVAL_GPU(Scan)
K80_UNARY_EVAL_GPU(Scatter)
K80_UNARY_EVAL_GPU(ScatterAxis)
K80_UNARY_EVAL_GPU(SliceUpdate)

#undef K80_UNARY_EVAL_GPU

// ============================================================================
// fast::ConvertFP8 (multi-output Primitive). FP8 isn't on any K80 path.
// ============================================================================

namespace fast {
void ConvertFP8::eval_gpu(
    const std::vector<array>&,
    std::vector<array>&) {
  k80_unimpl("fast::ConvertFP8::eval_gpu");
}
} // namespace fast

// ============================================================================
// distributed::* — no multi-GPU / multi-host on this single-K80 build.
// ============================================================================

namespace distributed {

#define K80_DIST_EVAL_GPU(Cls)                                            \
  void Cls::eval_gpu(                                                     \
      const std::vector<array>&, std::vector<array>&) {                   \
    k80_unimpl("distributed::" #Cls "::eval_gpu");                        \
  }
K80_DIST_EVAL_GPU(AllReduce)
K80_DIST_EVAL_GPU(AllGather)
K80_DIST_EVAL_GPU(ReduceScatter)
#undef K80_DIST_EVAL_GPU

#define K80_DIST_JVP(Cls)                                            \
  std::vector<array> Cls::jvp(                                       \
      const std::vector<array>&,                                     \
      const std::vector<array>&,                                     \
      const std::vector<int>&) {                                     \
    k80_unimpl("distributed::" #Cls "::jvp");                        \
  }
#define K80_DIST_VJP(Cls)                                            \
  std::vector<array> Cls::vjp(                                       \
      const std::vector<array>&,                                     \
      const std::vector<array>&,                                     \
      const std::vector<int>&,                                       \
      const std::vector<array>&) {                                   \
    k80_unimpl("distributed::" #Cls "::vjp");                        \
  }
#define K80_DIST_VMAP(Cls)                                           \
  std::pair<std::vector<array>, std::vector<int>> Cls::vmap(         \
      const std::vector<array>&, const std::vector<int>&) {          \
    k80_unimpl("distributed::" #Cls "::vmap");                       \
  }

K80_DIST_JVP(AllReduce)
K80_DIST_VJP(AllReduce)
K80_DIST_VMAP(AllReduce)
K80_DIST_JVP(AllGather)
K80_DIST_VJP(AllGather)
K80_DIST_VMAP(AllGather)
K80_DIST_VMAP(Send)

#undef K80_DIST_JVP
#undef K80_DIST_VJP
#undef K80_DIST_VMAP

} // namespace distributed

// ============================================================================
// Quantized matmul family — qmv/qmm_*/fp_*/gather_qmv (and supports_* guards).
// The supports_* guards return false so quantized.cpp's dispatch falls through;
// the actual qmm / qmv / etc. throw if reached. Real reroute (dequant + cuBLAS
// via affine_dequantize + matmul) is task #13.
// ============================================================================

#define K80_SUPPORTS_FALSE(name)                                      \
  bool name(                                                          \
      const array&, const array&, const array&,                       \
      const std::optional<array>&, const array&, bool,                \
      int, int, QuantizationMode, cu::Device&) {                      \
    return false;                                                     \
  }

K80_SUPPORTS_FALSE(supports_qmm_sm90)
K80_SUPPORTS_FALSE(supports_qmm_sm80)
K80_SUPPORTS_FALSE(supports_qmm_naive)
K80_SUPPORTS_FALSE(supports_fp_qmv)
K80_SUPPORTS_FALSE(supports_qmv)

#undef K80_SUPPORTS_FALSE

void qmm_sm90(
    const array&, const array&, const array&, const array&,
    array&, int, int, cu::CommandEncoder&, Stream) {
  k80_unimpl("qmm_sm90");
}

void qmm_sm80(
    const array&, const array&, const array&,
    const std::optional<array>&,
    const std::optional<array>&, const std::optional<array>&,
    array&, int, int, QuantizationMode, cu::CommandEncoder&) {
  k80_unimpl("qmm_sm80");
}

void qmm_naive(
    const array&, const array&, const array&,
    const std::optional<array>&,
    const std::optional<array>&, const std::optional<array>&,
    array&, bool, int, int, QuantizationMode, cu::CommandEncoder&) {
  k80_unimpl("qmm_naive");
}

void fp_qmv(
    const array&, const array&, const array&, array&,
    int, int, cu::CommandEncoder&, Stream) {
  k80_unimpl("fp_qmv");
}

void qmv(
    const array&, const array&, const array&,
    const std::optional<array>&, array&,
    int, int, QuantizationMode, cu::CommandEncoder&) {
  k80_unimpl("qmv");
}

void gather_qmv(
    const array&, const array&, const array&,
    const std::optional<array>&,
    const array&, const array&, array&,
    int, int, QuantizationMode, cu::CommandEncoder&) {
  k80_unimpl("gather_qmv");
}

// affine + fp quant/dequant (declared in quantized/quantized.h)
void affine_quantize(
    const array&, array&, array&, array&,
    int, int, cu::CommandEncoder&, const Stream&) {
  k80_unimpl("affine_quantize");
}
void affine_dequantize(
    const array&, const array&, const array&, array&,
    int, int, cu::CommandEncoder&, const Stream&) {
  k80_unimpl("affine_dequantize");
}
void fp_quantize(
    const array&, array&, array&,
    int, int, const std::optional<array>&,
    cu::CommandEncoder&, const Stream&) {
  k80_unimpl("fp_quantize");
}
void fp_dequantize(
    const array&, const array&, array&,
    int, int, const std::optional<array>&,
    cu::CommandEncoder&, const Stream&) {
  k80_unimpl("fp_dequantize");
}

// ============================================================================
// BlockMaskedMM helpers (block_mask.cu excluded from build)
// ============================================================================
void apply_block_mask(
    cu::CommandEncoder&, array&, const array&,
    int, int64_t, int64_t, int64_t) {
  k80_unimpl("apply_block_mask");
}
array copy_with_block_mask(
    cu::CommandEncoder&, const array&, const array&,
    int, int64_t, int64_t, int64_t) {
  k80_unimpl("copy_with_block_mask");
}

// ============================================================================
// CUTLASS gemm trio (CUTLASS impls excluded from build — sm_37 doesn't have
// tensor cores; the MoE / gather-mm paths fall back to per-expert cuBLAS,
// rerouted at exe link if/when qwen3.6 needs them).
// ============================================================================
void cutlass_gather_mm(
    bool, bool, const array&, const array&,
    const array&, const array&, array&,
    cu::CommandEncoder&) {
  k80_unimpl("cutlass_gather_mm");
}
void cutlass_grouped_gemm_unaligned(
    bool, int, bool, int, int,
    const array&, const array&, const array&, array&,
    cu::CommandEncoder&) {
  k80_unimpl("cutlass_grouped_gemm_unaligned");
}
void cutlass_segmented_mm(
    bool, int, bool, int, int, int, int,
    const array&, const array&, const array&,
    array&, cu::CommandEncoder&) {
  k80_unimpl("cutlass_segmented_mm");
}

// ============================================================================
// CublasGemm::run_batched (private member; called from matmul.cpp's batched
// path). The non-batched cuBLAS path works; batched stays a stub until a
// runtime trip motivates the cuBLASLt strided-batched port.
// ============================================================================
void CublasGemm::run_batched(
    cu::CommandEncoder&, array&,
    const array&, const array&,
    const Shape&, const Strides&, const Strides&,
    float) {
  k80_unimpl("CublasGemm::run_batched (a,b)");
}
void CublasGemm::run_batched(
    cu::CommandEncoder&, array&,
    const array&, const array&, const array&,
    const Shape&, const Strides&, const Strides&, const Strides&,
    float, float) {
  k80_unimpl("CublasGemm::run_batched (a,b,c)");
}

// ============================================================================
// JIT runtime — get_jit_module / JitModule::get_kernel.
// Called from slicing.cpp's compute_dynamic_offset; we defer the JIT subsystem
// (it needs a generated cuda_jit_sources.h). Runtime trip = port the JIT path.
// ============================================================================
namespace cu {

JitModule& get_jit_module(
    const mlx::core::Device&,
    const std::string&,
    const KernelBuilder&,
    bool) {
  k80_unimpl("cu::get_jit_module");
}

CUfunction JitModule::get_kernel(
    const std::string&,
    std::function<void(CUfunction)>) {
  k80_unimpl("cu::JitModule::get_kernel");
}

} // namespace cu

} // namespace mlx::core
