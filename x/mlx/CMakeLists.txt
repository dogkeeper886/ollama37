# K80 port of MLX — builds the VENDORED mlx/ source (x/mlx/mlx) with the existing
# builder toolchain: gcc-10 / nvcc 11.4, C++17, sm_37. No clang, no FetchContent,
# no builder/env changes — the source is ported to fit the toolchain (#187/#188).
#
# Built incrementally: core first (this commit), then backend/common + cpu, then
# the CUDA backend (CUTLASS/tensor-core paths rewritten to cuBLAS/SIMT).

cmake_minimum_required(VERSION 3.21)
project(mlx_k80 CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../MLX_VERSION" MLX_VER)
string(STRIP "${MLX_VER}" MLX_VER)
message(STATUS "MLX K80 port: vendored MLX @ ${MLX_VER}")

set(MLX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mlx)

# Core (host C++). Ported to C++17 — see device.h/stream.h (explicit ==/!=).
add_library(mlx STATIC
    ${MLX_SRC}/array.cpp
    ${MLX_SRC}/compile.cpp
    ${MLX_SRC}/device.cpp
    ${MLX_SRC}/dtype.cpp
    ${MLX_SRC}/dtype_utils.cpp
    ${MLX_SRC}/export.cpp
    ${MLX_SRC}/einsum.cpp
    ${MLX_SRC}/fast.cpp
    ${MLX_SRC}/fft.cpp
    ${MLX_SRC}/ops.cpp
    ${MLX_SRC}/graph_utils.cpp
    ${MLX_SRC}/primitives.cpp
    ${MLX_SRC}/random.cpp
    ${MLX_SRC}/scheduler.cpp
    ${MLX_SRC}/stream.cpp
    ${MLX_SRC}/transforms.cpp
    ${MLX_SRC}/utils.cpp
    ${MLX_SRC}/linalg.cpp
    ${MLX_SRC}/version.cpp
    # backend/common — device-agnostic backend helpers (host C++)
    ${MLX_SRC}/backend/common/broadcasting.cpp
    ${MLX_SRC}/backend/common/compiled.cpp
    ${MLX_SRC}/backend/common/common.cpp
    ${MLX_SRC}/backend/common/load.cpp
    ${MLX_SRC}/backend/common/reduce.cpp
    ${MLX_SRC}/backend/common/slicing.cpp
    ${MLX_SRC}/backend/common/utils.cpp
)

target_include_directories(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                       # "mlx/..." -> vendored tree
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt/include
)

# fmt header-only (vendored); FMT_UNICODE=0 mirrors MLX (Unicode fmt fails in .cu).
target_compile_definitions(mlx PRIVATE
    FMT_HEADER_ONLY=1
    FMT_UNICODE=0
    MLX_VERSION="${MLX_VER}"
)

###############################################################################
# CUDA backend (sm_37) — ON-PATH kernels only (see docs/traces/mlx-cuda-runtime-path.md)
#
# Build only the kernels the qwen3.5/3.6 inference path runs. Excluded entirely:
# fft, hadamard, distributed, cuDNN conv, FP8, and the CUTLASS qmm/gemm variants.
# Deferred (focused rewrites, added later): matmul.cpp + gemms (dense via cuBLAS),
# quantized (dequant+cuBLAS), conv1d (custom kernel), and the JIT subsystem
# (jit_module/indexing/custom_kernel need a generated cuda_jit_sources.h).
###############################################################################
find_package(CUDAToolkit REQUIRED)

set(CUDA_SRC ${MLX_SRC}/backend/cuda)

# Host-side (.cpp -> gcc): no device code per MLX convention.
set(MLX_CUDA_HOST
    ${CUDA_SRC}/allocator.cpp
    ${CUDA_SRC}/compiled.cpp
    ${CUDA_SRC}/cublas_utils.cpp
    ${CUDA_SRC}/device.cpp
    ${CUDA_SRC}/device_info.cpp
    ${CUDA_SRC}/eval.cpp
    ${CUDA_SRC}/fence.cpp
    ${CUDA_SRC}/load.cpp
    ${CUDA_SRC}/primitives.cpp
    ${CUDA_SRC}/slicing.cpp
    ${CUDA_SRC}/utils.cpp
    ${CUDA_SRC}/worker.cpp
    ${CUDA_SRC}/scaled_dot_product_attention.cpp
)

# Device-side (.cu -> nvcc).
file(GLOB MLX_CUDA_KERNELS
    ${CUDA_SRC}/binary/*.cu
    ${CUDA_SRC}/unary/*.cu
    ${CUDA_SRC}/reduce/*.cu
    ${CUDA_SRC}/copy/*.cu
)
list(APPEND MLX_CUDA_KERNELS
    ${CUDA_SRC}/event.cu
    ${CUDA_SRC}/kernel_utils.cu
    ${CUDA_SRC}/copy.cu
    ${CUDA_SRC}/arange.cu
    ${CUDA_SRC}/arg_reduce.cu
    ${CUDA_SRC}/binary_two.cu
    ${CUDA_SRC}/ternary.cu
    ${CUDA_SRC}/reduce.cu
    ${CUDA_SRC}/rms_norm.cu
    ${CUDA_SRC}/layer_norm.cu
    ${CUDA_SRC}/rope.cu
    ${CUDA_SRC}/scan.cu
    ${CUDA_SRC}/sort.cu
    ${CUDA_SRC}/softmax.cu
    ${CUDA_SRC}/logsumexp.cu
    ${CUDA_SRC}/random.cu
    ${CUDA_SRC}/scaled_dot_product_attention.cu
)

target_sources(mlx PRIVATE ${MLX_CUDA_HOST} ${MLX_CUDA_KERNELS})

set_target_properties(mlx PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 37
)

# nvcc 11.4 has no C++20; MLX device code needs relaxed-constexpr + extended-lambda.
# __grid_constant__ is a CUDA-11.7+ qualifier absent in 11.4 -> define it empty.
target_compile_options(mlx PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)
target_compile_definitions(mlx PRIVATE __grid_constant__=)

# cccl_shim must precede the toolkit include so the missing libcu++/fp8 headers
# resolve to our shims; nvtx_stub for the no-op nvtx3.
target_include_directories(mlx BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cccl_shim
)
target_include_directories(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nvtx_stub
    ${CUDAToolkit_INCLUDE_DIRS}   # host .cpp need <cuda.h>/<cuda_runtime.h> (nvcc adds these for .cu)
)

target_link_libraries(mlx PRIVATE CUDA::cudart CUDA::cublas)
