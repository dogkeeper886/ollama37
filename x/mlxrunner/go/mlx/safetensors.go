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

// SafeTensors is a loaded .safetensors file (name -> array map plus
// metadata). Release with .Release() when done. Array handles obtained
// via future Get calls keep their own references and stay valid even
// after the SafeTensors is released.
type SafeTensors struct {
	h C.mlx_safetensors_t
}

// LoadSafetensors parses a .safetensors file at `path` and pins all
// tensors into MLX-managed buffers. Errors out on missing files,
// parse errors, or any exception thrown across the C ABI boundary.
func LoadSafetensors(path string) (*SafeTensors, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	h := C.mlx_load_safetensors(cpath)
	if h == nil {
		return nil, fmt.Errorf("mlx_load_safetensors: failed to load %s", path)
	}
	return &SafeTensors{h: h}, nil
}

// Count returns the number of tensors in the file. Returns -1 if the
// handle has already been released (programming error — caller is
// expected to honor lifetime).
func (s *SafeTensors) Count() int {
	if s == nil || s.h == nil {
		return -1
	}
	return int(C.mlx_safetensors_count(s.h))
}

// Get looks up a tensor by name. Returns nil if the tensor isn't in
// the file. The returned Array holds its own reference and remains
// valid after the SafeTensors is released — caller still owns it and
// must Release it.
func (s *SafeTensors) Get(name string) *Array {
	if s == nil || s.h == nil {
		return nil
	}
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return wrapArray(C.mlx_safetensors_get(s.h, cname))
}

// Release frees the underlying safetensors handle. Safe to call on a
// nil receiver or one whose handle has already been released.
func (s *SafeTensors) Release() {
	if s == nil || s.h == nil {
		return
	}
	C.mlx_safetensors_release(s.h)
	s.h = nil
}
