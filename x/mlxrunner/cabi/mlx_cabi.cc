// mlx_cabi.cc — C ABI implementation. See mlx_cabi.h for the contract.

#include "mlx_cabi.h"

#include "mlx/mlx.h"
#include "mlx/dtype_utils.h"  // dtype_to_string

#include <new>
#include <utility>

// Opaque struct definition; only visible inside this TU. The Go side
// only ever holds the pointer.
struct mlx_safetensors_s {
  // load_safetensors returns std::pair<unordered_map<string, array>,
  // unordered_map<string, string>>. Store as-is; lookups + metadata
  // access become trivial extensions later.
  mlx::core::SafetensorsLoad data;
};

// mlx::core::array is reference-counted internally, so we hold a copy
// per handle. Releasing the handle drops the ref; if it was the last
// ref the underlying buffer goes away.
struct mlx_array_s {
  mlx::core::array data;
};

int mlx_init(void) {
  try {
    mlx::core::set_default_device(
        mlx::core::Device(mlx::core::Device::gpu));
    return 0;
  } catch (...) {
    return -1;
  }
}

mlx_safetensors_t mlx_load_safetensors(const char* path) {
  if (!path) return nullptr;
  try {
    auto* st = new (std::nothrow) mlx_safetensors_s{
        mlx::core::load_safetensors(path)};
    return st;
  } catch (...) {
    return nullptr;
  }
}

int mlx_safetensors_count(mlx_safetensors_t st) {
  if (!st) return -1;
  return static_cast<int>(st->data.first.size());
}

void mlx_safetensors_release(mlx_safetensors_t st) {
  delete st;
}

// ----- mlx_array -----

mlx_array_t mlx_safetensors_get(mlx_safetensors_t st, const char* name) {
  if (!st || !name) return nullptr;
  try {
    auto it = st->data.first.find(name);
    if (it == st->data.first.end()) return nullptr;
    return new (std::nothrow) mlx_array_s{it->second};
  } catch (...) {
    return nullptr;
  }
}

void mlx_array_release(mlx_array_t arr) {
  delete arr;
}

const char* mlx_array_dtype_name(mlx_array_t arr) {
  if (!arr) return "unknown";
  try {
    return mlx::core::dtype_to_string(arr->data.dtype());
  } catch (...) {
    return "unknown";
  }
}

int mlx_array_ndim(mlx_array_t arr) {
  if (!arr) return -1;
  return static_cast<int>(arr->data.ndim());
}

long long mlx_array_dim(mlx_array_t arr, int axis) {
  if (!arr) return -1;
  if (axis < 0 || axis >= static_cast<int>(arr->data.ndim())) return -1;
  return static_cast<long long>(arr->data.shape(axis));
}
