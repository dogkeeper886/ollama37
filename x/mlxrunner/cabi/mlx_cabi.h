// mlx_cabi.h — C ABI shim around libmlx.a so Go cgo can call into MLX.
//
// Phase D.2 step 1 (#189). Minimum surface to load a safetensors file
// from Go and report the tensor count. The C++ MLX API uses templates,
// std::optional, std::variant, and other things cgo can't see directly;
// every C-callable function here translates to plain C types and opaque
// pointers, with C++ exceptions caught at the boundary and reported as
// integer error codes.
//
// Ownership / lifetimes:
//   - Every constructor-shaped function returns a handle (pointer) the
//     caller must release via the matching mlx_*_release.
//   - Handles are opaque (forward-declared structs); no Go code peeks
//     inside.
//   - Functions returning int use 0 for success and negative for error
//     (-1 generic, more specific codes added as the API grows).
//
// Build: x/mlxrunner/CMakeLists.txt adds an mlx_cabi STATIC library that
// includes this header + mlx_cabi.cc and links libmlx.a transitively.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded safetensors file (name -> array map + metadata).
typedef struct mlx_safetensors_s* mlx_safetensors_t;

// Initialize MLX state and pin the default stream to the GPU device.
// Returns 0 on success, negative on error.
int mlx_init(void);

// Load a safetensors file at `path`. Returns NULL on failure (file not
// found, parse error, exception thrown). Caller must release via
// mlx_safetensors_release.
mlx_safetensors_t mlx_load_safetensors(const char* path);

// Number of tensors in a loaded safetensors handle. Returns -1 on a
// NULL handle.
int mlx_safetensors_count(mlx_safetensors_t st);

// Release a safetensors handle. Safe to call on NULL.
void mlx_safetensors_release(mlx_safetensors_t st);

#ifdef __cplusplus
}
#endif
