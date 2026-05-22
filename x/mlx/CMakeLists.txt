# K80 port of MLX — builds the VENDORED mlx/ source (x/mlx/mlx) with the existing
# builder toolchain: gcc-10 / nvcc 11.4, C++17, sm_37. No clang, no FetchContent,
# no builder/env changes — the source is ported to fit the toolchain (#187/#188).
#
# Built incrementally: core first (this commit), then backend/common + cpu, then
# the CUDA backend (CUTLASS/tensor-core paths rewritten to cuBLAS/SIMT).

cmake_minimum_required(VERSION 3.21)
project(mlx_k80 CXX)

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
