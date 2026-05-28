package mlx

/*
#include "mlx_cabi.h"
*/
import "C"

// Array wraps a single mlx::core::array, ref-counted on the C++ side.
// Each handle holds one reference; calling Release drops it. If you
// want to keep the array alive past the SafeTensors that produced it,
// the underlying ref-count makes that safe — but ownership of the
// Array handle is still yours.
type Array struct {
	h C.mlx_array_t
}

// wrapArray adopts a C-returned handle; not exported because raw
// handles aren't part of the Go-facing API.
func wrapArray(h C.mlx_array_t) *Array {
	if h == nil {
		return nil
	}
	return &Array{h: h}
}

// Release drops the array's reference. Safe to call on nil or on an
// already-released Array.
func (a *Array) Release() {
	if a == nil || a.h == nil {
		return
	}
	C.mlx_array_release(a.h)
	a.h = nil
}

// DType returns the dtype name ("bfloat16", "uint32", "float32", ...).
// Returns "unknown" if the array is nil or already released.
func (a *Array) DType() string {
	if a == nil || a.h == nil {
		return "unknown"
	}
	return C.GoString(C.mlx_array_dtype_name(a.h))
}

// Ndim returns the number of dimensions; -1 if released.
func (a *Array) Ndim() int {
	if a == nil || a.h == nil {
		return -1
	}
	return int(C.mlx_array_ndim(a.h))
}

// Dim returns the size of `axis`; -1 if released or out of range.
func (a *Array) Dim(axis int) int64 {
	if a == nil || a.h == nil {
		return -1
	}
	return int64(C.mlx_array_dim(a.h, C.int(axis)))
}

// Size returns the product of all dimensions; -1 if released.
func (a *Array) Size() int64 {
	if a == nil || a.h == nil {
		return -1
	}
	return int64(C.mlx_array_size(a.h))
}

// Shape returns the shape as a fresh slice. Convenient for logging
// (Printf("shape=%v", arr.Shape())).
func (a *Array) Shape() []int64 {
	n := a.Ndim()
	if n <= 0 {
		return nil
	}
	out := make([]int64, n)
	for i := 0; i < n; i++ {
		out[i] = a.Dim(i)
	}
	return out
}
