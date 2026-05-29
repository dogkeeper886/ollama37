package mlx

/*
#include <stdlib.h>
#include "mlx_cabi.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ----- construction -----

// FromInt32 wraps a Go []int32 as a 1-D int32 mlx::array. The data is
// copied (the C side owns its own buffer afterward).
func FromInt32(data []int32) (*Array, error) {
	var ptr *C.int
	if len(data) > 0 {
		ptr = (*C.int)(unsafe.Pointer(&data[0]))
	}
	h := C.mlx_array_from_int32_1d(ptr, C.int(len(data)))
	if h == nil {
		return nil, fmt.Errorf("mlx_array_from_int32_1d failed (len=%d)", len(data))
	}
	return wrapArray(h), nil
}

// Ones constructs an all-ones array of the requested shape and dtype.
// dtype is "float32" or "bfloat16" (the inference-path dtypes; other
// names return an error).
func Ones(shape []int64, dtype string) (*Array, error) {
	var ptr *C.longlong
	if len(shape) > 0 {
		ptr = (*C.longlong)(unsafe.Pointer(&shape[0]))
	}
	cdt := C.CString(dtype)
	defer C.free(unsafe.Pointer(cdt))
	h := C.mlx_array_ones(ptr, C.int(len(shape)), cdt)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_ones failed (shape=%v dtype=%s)", shape, dtype)
	}
	return wrapArray(h), nil
}

// ----- unary ops -----

// Sigmoid returns elementwise sigmoid(a). Combine with Multiply for silu.
func (a *Array) Sigmoid() (*Array, error) {
	if a == nil || a.h == nil {
		return nil, fmt.Errorf("Sigmoid: nil array")
	}
	h := C.mlx_array_sigmoid(a.h)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_sigmoid failed")
	}
	return wrapArray(h), nil
}

// Abs returns elementwise absolute value.
func (a *Array) Abs() (*Array, error) {
	if a == nil || a.h == nil {
		return nil, fmt.Errorf("Abs: nil array")
	}
	h := C.mlx_array_abs(a.h)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_abs failed")
	}
	return wrapArray(h), nil
}

// MeanAll reduces over all axes (no keepdims) and returns a scalar
// array. For per-axis mean, add an axis-aware mean op later.
func (a *Array) MeanAll() (*Array, error) {
	if a == nil || a.h == nil {
		return nil, fmt.Errorf("MeanAll: nil array")
	}
	h := C.mlx_array_mean_all(a.h)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_mean_all failed")
	}
	return wrapArray(h), nil
}

// ToFloat32 returns an astype'd copy in float32. Always call this
// before CopyFloat32 — reading BF16/FP16 bytes through a float lens
// gives denormal garbage (see PR #199).
func (a *Array) ToFloat32() (*Array, error) {
	if a == nil || a.h == nil {
		return nil, fmt.Errorf("ToFloat32: nil array")
	}
	h := C.mlx_array_to_float32(a.h)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_to_float32 failed")
	}
	return wrapArray(h), nil
}

// ----- binary ops -----

// Multiply returns elementwise a*b.
func (a *Array) Multiply(b *Array) (*Array, error) {
	if a == nil || a.h == nil || b == nil || b.h == nil {
		return nil, fmt.Errorf("Multiply: nil array")
	}
	h := C.mlx_array_multiply(a.h, b.h)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_multiply failed")
	}
	return wrapArray(h), nil
}

// Take gathers rows along `axis` using the indices array. Equivalent
// to numpy a[indices, ...] when axis=0.
func (a *Array) Take(indices *Array, axis int) (*Array, error) {
	if a == nil || a.h == nil || indices == nil || indices.h == nil {
		return nil, fmt.Errorf("Take: nil array")
	}
	h := C.mlx_array_take(a.h, indices.h, C.int(axis))
	if h == nil {
		return nil, fmt.Errorf("mlx_array_take failed")
	}
	return wrapArray(h), nil
}

// ----- multi-arg numeric ops -----

// Dequantize unpacks a quantized weight matrix. `biases` may be nil
// for non-affine modes. mode is typically "affine" or "fp".
func Dequantize(wq, scales, biases *Array, groupSize, bits int, mode string) (*Array, error) {
	if wq == nil || wq.h == nil || scales == nil || scales.h == nil {
		return nil, fmt.Errorf("Dequantize: wq and scales required")
	}
	var bh C.mlx_array_t
	if biases != nil {
		bh = biases.h
	}
	cmode := C.CString(mode)
	defer C.free(unsafe.Pointer(cmode))
	h := C.mlx_array_dequantize(wq.h, scales.h, bh, C.int(groupSize), C.int(bits), cmode)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_dequantize failed (gs=%d bits=%d mode=%s)",
			groupSize, bits, mode)
	}
	return wrapArray(h), nil
}

// RMSNorm runs MLX's fast::rms_norm. `weight` may be nil for no-gain
// (the implementation treats nil as a one-vector).
func RMSNorm(x, weight *Array, eps float32) (*Array, error) {
	if x == nil || x.h == nil {
		return nil, fmt.Errorf("RMSNorm: nil x")
	}
	var wh C.mlx_array_t
	if weight != nil {
		wh = weight.h
	}
	h := C.mlx_array_rms_norm(x.h, wh, C.float(eps))
	if h == nil {
		return nil, fmt.Errorf("mlx_array_rms_norm failed (eps=%g)", eps)
	}
	return wrapArray(h), nil
}

// QuantizedMatmul runs MLX's quantized_matmul. biases may be nil for
// non-affine modes; transpose toggles wq^T.
func QuantizedMatmul(x, wq, scales, biases *Array, transpose bool,
	groupSize, bits int, mode string) (*Array, error) {
	if x == nil || x.h == nil || wq == nil || wq.h == nil || scales == nil || scales.h == nil {
		return nil, fmt.Errorf("QuantizedMatmul: x, wq, scales required")
	}
	var bh C.mlx_array_t
	if biases != nil {
		bh = biases.h
	}
	cmode := C.CString(mode)
	defer C.free(unsafe.Pointer(cmode))
	var tr C.int
	if transpose {
		tr = 1
	}
	h := C.mlx_array_quantized_matmul(x.h, wq.h, scales.h, bh, tr,
		C.int(groupSize), C.int(bits), cmode)
	if h == nil {
		return nil, fmt.Errorf("mlx_array_quantized_matmul failed (gs=%d bits=%d mode=%s)",
			groupSize, bits, mode)
	}
	return wrapArray(h), nil
}

// Conv1d runs a 1-D convolution. Input shape (B, L, C_in); weight
// shape (C_out, kernel_w, C_in/groups). All ints follow MLX's
// convention (no auto-broadcast of group count, etc.).
func Conv1d(input, weight *Array, stride, padding, dilation, groups int) (*Array, error) {
	if input == nil || input.h == nil || weight == nil || weight.h == nil {
		return nil, fmt.Errorf("Conv1d: input and weight required")
	}
	h := C.mlx_array_conv1d(input.h, weight.h,
		C.int(stride), C.int(padding), C.int(dilation), C.int(groups))
	if h == nil {
		return nil, fmt.Errorf("mlx_array_conv1d failed")
	}
	return wrapArray(h), nil
}

// ----- evaluation + data extraction -----

// Eval forces evaluation of zero or more arrays in place. Required
// before reading data via CopyFloat32 (which only reads materialized
// host buffers).
func Eval(arrs ...*Array) error {
	if len(arrs) == 0 {
		return nil
	}
	handles := make([]C.mlx_array_t, len(arrs))
	for i, a := range arrs {
		if a == nil || a.h == nil {
			return fmt.Errorf("Eval: arr[%d] is nil", i)
		}
		handles[i] = a.h
	}
	rc := C.mlx_eval(&handles[0], C.int(len(handles)))
	if rc != 0 {
		return fmt.Errorf("mlx_eval failed: rc=%d", int(rc))
	}
	return nil
}

// CopyFloat32 copies up to len(out) float values from the array's
// host buffer into out. The array MUST be (a) eval'd already and (b)
// dtype == float32 (call ToFloat32 first if needed). Returns the
// number of values actually copied.
func (a *Array) CopyFloat32(out []float32) (int, error) {
	if a == nil || a.h == nil {
		return 0, fmt.Errorf("CopyFloat32: nil array")
	}
	if len(out) == 0 {
		return 0, nil
	}
	n := C.mlx_array_copy_float(a.h,
		(*C.float)(unsafe.Pointer(&out[0])), C.int(len(out)))
	if n < 0 {
		return 0, fmt.Errorf("mlx_array_copy_float failed: rc=%d (likely wrong dtype — astype to float32 first)", int(n))
	}
	return int(n), nil
}
