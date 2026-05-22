# Standalone MLX build for the K80 (sm_37) port — #188.
#
# This is intentionally NOT wired into the main Ollama CMake build yet: that build
# compiles GGML-CUDA with nvcc, but MLX requires C++20 device code and nvcc 11.4
# has no C++20. clang DOES support C++20 + sm_37 against the CUDA 11.4 toolkit
# (verified), so MLX must be built with clang as the CUDA compiler:
#
#   cmake -S x/mlx -B build/mlx -G Ninja \
#         -DCMAKE_CUDA_COMPILER=clang++ -DCMAKE_CUDA_ARCHITECTURES=37
#   cmake --build build/mlx
#
# Mixing nvcc (GGML) and clang (MLX) in one CMake project isn't possible, so the
# spike stays standalone until the runner integration (#191) decides how to ship it.

cmake_minimum_required(VERSION 3.25)
project(mlx_k80 CXX CUDA)

# Pin MLX core to the version upstream Ollama uses (shared file at repo root).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../MLX_VERSION" MLX_GIT_TAG)
string(STRIP "${MLX_GIT_TAG}" MLX_GIT_TAG)
message(STATUS "MLX K80 port: building MLX @ ${MLX_GIT_TAG}")

# CUDA backend only. No Metal (that's the macOS path); safetensors loader on so we
# can actually load model weights; GGUF loader off (we don't need MLX's GGUF).
set(MLX_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Target Kepler. Caller overrides with -DCMAKE_CUDA_ARCHITECTURES=37.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 37)
endif()
set(MLX_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

if(CMAKE_CUDA_COMPILER)
    set(MLX_BUILD_CUDA ON CACHE BOOL "Build MLX CUDA backend" FORCE)
else()
    message(FATAL_ERROR "MLX K80 build needs a CUDA compiler (pass -DCMAKE_CUDA_COMPILER=clang++)")
endif()

include(FetchContent)
FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG ${MLX_GIT_TAG}
)
FetchContent_MakeAvailable(mlx)

# MLX's CUDA kernels (CUTLASS-heavy) can each consume several GB while compiling.
# Cap concurrent CUDA compiles by a per-file RAM budget so the build host doesn't
# OOM. Ninja-only (JOB_POOLS has no effect under Make). Mirrors upstream Ollama.
if(CMAKE_GENERATOR STREQUAL "Ninja")
    file(GLOB_RECURSE _mlx_cu "${mlx_SOURCE_DIR}/mlx/backend/cuda/*.cu")
    if(_mlx_cu)
        set(MLX_CUDA_RAM_MB 22000 CACHE STRING "per-file RAM budget (MB) for cuda compile pool")
        cmake_host_system_information(RESULT _ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
        math(EXPR _pool "${_ram_mb} / ${MLX_CUDA_RAM_MB}")
        if(_pool LESS 2)
            set(_pool 2)
        endif()
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS cuda_compile=${_pool})
        foreach(f ${_mlx_cu})
            set_property(SOURCE "${f}" TARGET_DIRECTORY mlx PROPERTY JOB_POOL_COMPILE cuda_compile)
        endforeach()
        message(STATUS "MLX cuda_compile pool: ${_pool} (host RAM ${_ram_mb} MB)")
    endif()
endif()
