// mlx_cabi.cc — C ABI implementation. See mlx_cabi.h for the contract.

#include "mlx_cabi.h"

#include "mlx/mlx.h"

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
