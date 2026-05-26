# K80 runner — scaffolding for Phase B. The goal is *not* to ship a full MLX
# runner here; it's to surface what symbols an MLX exe actually needs to link
# on top of libmlx.a (the on-path archive built from x/mlx). The first
# deliverable is `qwen_smoke` — a probe that touches qwen3.5's forward-pass
# op surface so the linker enumerates whatever's still missing.
#
# Build:
#   cmake -S x/mlxrunner -B /tmp/runner -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/runner -j$(nproc)
# Expect: link fails (initially) with undefined refs -> punch list for reroute.

cmake_minimum_required(VERSION 3.21)
project(mlx_runner CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Pull in the libmlx.a target from the sibling x/mlx tree.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../mlx mlx_build)

find_package(CUDAToolkit REQUIRED)

add_executable(qwen_smoke smoke/qwen_surface.cpp)

target_include_directories(qwen_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../mlx
)

# libmlx.a brings in the on-path CUDA backend; cudart + cuda_driver + cublas
# resolve the runtime + driver-API (cuLaunchKernel/cuGraphAddKernelNode) +
# GEMM symbols libmlx.a refers to.
target_link_libraries(qwen_smoke PRIVATE
    mlx
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)
