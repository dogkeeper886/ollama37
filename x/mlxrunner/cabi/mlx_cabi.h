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

// Opaque handle to a single mlx::core::array. Owned by the caller; release
// with mlx_array_release. Handles obtained from mlx_safetensors_get hold a
// copy of the underlying array's reference, so it's safe to release the
// safetensors handle before releasing arrays obtained from it.
typedef struct mlx_array_s* mlx_array_t;

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

// ----- mlx_array -----

// Look up a named tensor inside a loaded safetensors handle. Returns
// NULL if `name` isn't in the file (or `st` is NULL). Caller must
// release with mlx_array_release.
mlx_array_t mlx_safetensors_get(mlx_safetensors_t st, const char* name);

// Release an array handle. Safe to call on NULL.
void mlx_array_release(mlx_array_t arr);

// Dtype name as a static string ("bfloat16", "uint32", "float32", ...).
// Returns "unknown" for NULL or unrecognized dtype. The returned pointer
// has static storage; caller MUST NOT free it.
const char* mlx_array_dtype_name(mlx_array_t arr);

// Number of dimensions. Returns -1 on NULL.
int mlx_array_ndim(mlx_array_t arr);

// Size of dimension `axis` (0-indexed). Returns -1 on NULL or out-of-range
// axis.
long long mlx_array_dim(mlx_array_t arr, int axis);

// Total number of elements (product of dims). Returns -1 on NULL.
long long mlx_array_size(mlx_array_t arr);

// Construct a 1-D int32 array from Go-owned data; data is copied into a
// new mlx::core::array. Returns NULL on alloc failure.
mlx_array_t mlx_array_from_int32_1d(const int* data, int count);

// ----- ops -----

// take(arr, indices, axis) — gather rows along `axis`. Returns NULL on
// error (incompatible shapes, etc.). Caller owns the result.
mlx_array_t mlx_array_take(mlx_array_t arr, mlx_array_t indices, int axis);

// dequantize(wq, scales, biases, group_size, bits, mode="affine") — un-
// quantize a packed weight matrix. `biases` may be NULL for non-affine
// modes; pass mode = "affine" or "fp".
mlx_array_t mlx_array_dequantize(
    mlx_array_t wq, mlx_array_t scales, mlx_array_t biases,
    int group_size, int bits, const char* mode);

// Force evaluation of the given arrays (in-place). Returns 0 on success,
// negative on exception.
int mlx_eval(mlx_array_t* arrs, int count);

// astype(arr, float32). Returns NULL on error. Use this BEFORE
// mlx_array_copy_float — copying raw BF16/FP16 bytes through a float* lens
// reads denormal garbage (see PR #199 for the same bug on the C++ side).
mlx_array_t mlx_array_to_float32(mlx_array_t arr);

// Copy at most `max_count` float values from `arr`'s host buffer into
// `out`. The array MUST be (a) eval'd already and (b) dtype == float32.
// Returns number of values copied, or negative on error:
//   -1: NULL arg / max_count <= 0
//   -2: wrong dtype (the message goes to stderr via report_exc)
//   -3: other exception
int mlx_array_copy_float(mlx_array_t arr, float* out, int max_count);

// Elementwise absolute value. Returns NULL on error.
mlx_array_t mlx_array_abs(mlx_array_t arr);

// Reduce over all axes (no keepdims). Returns scalar array.
mlx_array_t mlx_array_mean_all(mlx_array_t arr);

// fast::rms_norm(x, weight, eps). `weight` may be NULL for no-gain norm.
mlx_array_t mlx_array_rms_norm(mlx_array_t x, mlx_array_t weight, float eps);

// quantized_matmul(x, wq, scales, biases, transpose, group_size, bits, mode).
// `biases` may be NULL for fp-mode quants. `transpose` is 0 or 1 (C-bool).
// `mode` is "affine" or "fp" (or others MLX supports).
mlx_array_t mlx_array_quantized_matmul(
    mlx_array_t x, mlx_array_t wq, mlx_array_t scales, mlx_array_t biases,
    int transpose, int group_size, int bits, const char* mode);

// conv1d(input, weight, stride, padding, dilation, groups). Input shape
// (B, L, C_in); weight shape (C_out, kw, C_in/groups). Returns NULL on
// shape mismatch / exception.
mlx_array_t mlx_array_conv1d(
    mlx_array_t input, mlx_array_t weight,
    int stride, int padding, int dilation, int groups);

// Elementwise sigmoid + multiply (compose for silu = sigmoid(x) * x).
mlx_array_t mlx_array_sigmoid(mlx_array_t arr);
mlx_array_t mlx_array_multiply(mlx_array_t a, mlx_array_t b);

// Construct an all-ones array. `dtype_name` accepts "float32" or "bfloat16"
// (the dtypes we exercise on the inference path); other names return NULL
// with a stderr message.
mlx_array_t mlx_array_ones(const long long* dims, int ndim,
                           const char* dtype_name);

#ifdef __cplusplus
}
#endif
