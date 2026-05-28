// mlx_cabi.cc — C ABI implementation. See mlx_cabi.h for the contract.

#include "mlx_cabi.h"

#include "mlx/allocator.h"
#include "mlx/mlx.h"
#include "mlx/dtype_utils.h"  // dtype_to_string

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Surface the exception what() to stderr so Go-side errors are debuggable.
// Returns the passed-through error code unchanged so callers don't need to
// change behavior.
static int report_exc(const char* where, int rc) {
  try { throw; }
  catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: %s: %s\n", where, e.what());
  } catch (...) {
    std::fprintf(stderr, "mlx_cabi: %s: <non-std exception>\n", where);
  }
  return rc;
}

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

long long mlx_array_size(mlx_array_t arr) {
  if (!arr) return -1;
  return static_cast<long long>(arr->data.size());
}

mlx_array_t mlx_array_from_int32_1d(const int* data, int count) {
  if (count < 0 || (count > 0 && !data)) return nullptr;
  try {
    // Allocate via MLX's allocator + memcpy + array(Buffer, Shape, Dtype)
    // produces a "leaf-with-data" array. The (Shape, Dtype, nullptr, {})
    // + set_data combo leaves the array in a state MLX's eval walker
    // rejects ("Attempting to eval an array without a primitive").
    const size_t nbytes = static_cast<size_t>(count) * sizeof(int32_t);
    auto buf = mlx::core::allocator::malloc(nbytes);
    if (count > 0) {
      std::memcpy(buf.raw_ptr(), data, nbytes);
    }
    mlx::core::array a(buf, mlx::core::Shape{count}, mlx::core::int32);
    return new (std::nothrow) mlx_array_s{std::move(a)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_from_int32_1d: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// ----- ops -----

mlx_array_t mlx_array_take(mlx_array_t arr, mlx_array_t indices, int axis) {
  if (!arr || !indices) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{
        mlx::core::take(arr->data, indices->data, axis)};
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_dequantize(
    mlx_array_t wq, mlx_array_t scales, mlx_array_t biases,
    int group_size, int bits, const char* mode) {
  if (!wq || !scales) return nullptr;
  try {
    std::optional<mlx::core::array> bs;
    if (biases) bs = biases->data;
    std::string m = mode ? mode : "affine";
    auto out = mlx::core::dequantize(
        wq->data, scales->data, bs, group_size, bits, m);
    return new (std::nothrow) mlx_array_s{std::move(out)};
  } catch (...) {
    return nullptr;
  }
}

int mlx_eval(mlx_array_t* arrs, int count) {
  if (count < 0 || (count > 0 && !arrs)) return -1;
  try {
    std::vector<mlx::core::array> to_eval;
    to_eval.reserve(count);
    for (int i = 0; i < count; i++) {
      if (!arrs[i]) return -2;
      to_eval.push_back(arrs[i]->data);
    }
    mlx::core::eval(std::move(to_eval));
    return 0;
  } catch (...) {
    return report_exc("mlx_eval", -3);
  }
}

mlx_array_t mlx_array_to_float32(mlx_array_t arr) {
  if (!arr) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{
        mlx::core::astype(arr->data, mlx::core::float32)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_to_float32: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

int mlx_array_copy_float(mlx_array_t arr, float* out, int max_count) {
  if (!arr || !out || max_count <= 0) return -1;
  if (arr->data.dtype() != mlx::core::float32) {
    std::fprintf(stderr,
        "mlx_cabi: mlx_array_copy_float: dtype must be float32; got %s\n",
        mlx::core::dtype_to_string(arr->data.dtype()));
    return -2;
  }
  try {
    int n = std::min(
        static_cast<int>(arr->data.size()),
        max_count);
    std::memcpy(out, arr->data.data<float>(), n * sizeof(float));
    return n;
  } catch (...) {
    return report_exc("mlx_array_copy_float", -3);
  }
}

mlx_array_t mlx_array_abs(mlx_array_t arr) {
  if (!arr) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{mlx::core::abs(arr->data)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_abs: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_mean_all(mlx_array_t arr) {
  if (!arr) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{
        mlx::core::mean(arr->data, /*keepdims=*/false)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_mean_all: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_rms_norm(mlx_array_t x, mlx_array_t weight, float eps) {
  if (!x) return nullptr;
  try {
    std::optional<mlx::core::array> w;
    if (weight) w = weight->data;
    return new (std::nothrow) mlx_array_s{
        mlx::core::fast::rms_norm(x->data, w, eps)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_rms_norm: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_quantized_matmul(
    mlx_array_t x, mlx_array_t wq, mlx_array_t scales, mlx_array_t biases,
    int transpose, int group_size, int bits, const char* mode) {
  if (!x || !wq || !scales) return nullptr;
  try {
    std::optional<mlx::core::array> bs;
    if (biases) bs = biases->data;
    std::string m = mode ? mode : "affine";
    auto out = mlx::core::quantized_matmul(
        x->data, wq->data, scales->data, bs,
        transpose != 0, group_size, bits, m);
    return new (std::nothrow) mlx_array_s{std::move(out)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_quantized_matmul: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_conv1d(
    mlx_array_t input, mlx_array_t weight,
    int stride, int padding, int dilation, int groups) {
  if (!input || !weight) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{mlx::core::conv1d(
        input->data, weight->data, stride, padding, dilation, groups)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_conv1d: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_sigmoid(mlx_array_t arr) {
  if (!arr) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{mlx::core::sigmoid(arr->data)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_sigmoid: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_multiply(mlx_array_t a, mlx_array_t b) {
  if (!a || !b) return nullptr;
  try {
    return new (std::nothrow) mlx_array_s{
        mlx::core::multiply(a->data, b->data)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_multiply: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array_t mlx_array_ones(const long long* dims, int ndim,
                           const char* dtype_name) {
  if (ndim < 0 || (ndim > 0 && !dims) || !dtype_name) return nullptr;
  mlx::core::Dtype dt = mlx::core::float32;
  std::string n = dtype_name;
  if (n == "float32") dt = mlx::core::float32;
  else if (n == "bfloat16") dt = mlx::core::bfloat16;
  else {
    std::fprintf(stderr,
        "mlx_cabi: mlx_array_ones: unsupported dtype '%s' "
        "(only float32 and bfloat16 currently)\n", dtype_name);
    return nullptr;
  }
  try {
    mlx::core::Shape shape;
    shape.reserve(ndim);
    for (int i = 0; i < ndim; i++) shape.push_back(static_cast<int>(dims[i]));
    return new (std::nothrow) mlx_array_s{mlx::core::ones(shape, dt)};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_cabi: mlx_array_ones: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}
