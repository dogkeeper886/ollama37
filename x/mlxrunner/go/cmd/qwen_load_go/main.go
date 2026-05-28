// qwen_load_go — Phase D.2 step 1 (#189) cgo plumbing probe.
//
// Mirrors qwen_load.cpp's first line of output ("loaded N tensors") via
// the C ABI shim. Validates that Go can drive libmlx.a end-to-end:
//   - cgo headers find mlx_cabi.h
//   - linker resolves libmlx_cabi.a + libmlx.a + CUDA runtime libs
//   - C++ exceptions in libmlx.a don't cross the cgo boundary
//   - safetensors load actually works from a Go context
//
// Run via cicd/scripts/test-mlx-smoke.sh; the CI script bind-mounts the
// model dir into the runtime container and passes shard 1's path as
// argv[1].

package main

// cgo preamble notes (kept OUTSIDE the C block — every line in the C
// preamble is fed to the C compiler, including the human-readable text
// of `//` comments, so prose with `-lrt` and `C++` in it tripped the
// first build):
//
//   * LDFLAGS lists library NAMES only; -L paths come from CGO_LDFLAGS
//     env at build time (CI script supplies them).
//   * Link order: mlx_cabi -> mlx -> CUDA. Plus stdc++ (libmlx.a is C++)
//     and libmlx.a transitive deps (pthread, dl, m, rt).

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cabi
#cgo LDFLAGS: -lmlx_cabi -lmlx -lcublas -lcublasLt -lcudart -lcuda -lstdc++ -lpthread -ldl -lm -lrt
#include <stdlib.h>
#include "mlx_cabi.h"
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: qwen_load_go <shard.safetensors>")
		os.Exit(2)
	}

	if rc := C.mlx_init(); rc != 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: mlx_init failed: rc=%d\n", int(rc))
		os.Exit(3)
	}

	cpath := C.CString(os.Args[1])
	defer C.free(unsafe.Pointer(cpath))

	st := C.mlx_load_safetensors(cpath)
	if st == nil {
		fmt.Fprintf(os.Stderr, "qwen_load_go: mlx_load_safetensors failed for %s\n",
			os.Args[1])
		os.Exit(4)
	}
	defer C.mlx_safetensors_release(st)

	count := int(C.mlx_safetensors_count(st))
	fmt.Printf("qwen_load_go: loaded %d tensors\n", count)

	// Mirror qwen_load.cpp's "embed_tokens.weight ..." and ".scales ..." lines.
	// Validates: name lookup, dtype string accessor, ndim/dim accessors,
	// ownership pattern (get + release per array).
	for _, name := range []string{
		"language_model.model.embed_tokens.weight",
		"language_model.model.embed_tokens.scales",
	} {
		cname := C.CString(name)
		arr := C.mlx_safetensors_get(st, cname)
		C.free(unsafe.Pointer(cname))
		if arr == nil {
			fmt.Fprintf(os.Stderr, "qwen_load_go: missing tensor '%s'\n", name)
			os.Exit(5)
		}

		dtype := C.GoString(C.mlx_array_dtype_name(arr))
		ndim := int(C.mlx_array_ndim(arr))
		dims := make([]int64, ndim)
		for i := 0; i < ndim; i++ {
			dims[i] = int64(C.mlx_array_dim(arr, C.int(i)))
		}
		fmt.Printf("qwen_load_go: %s  dtype=%s shape=%v\n",
			shortName(name), dtype, dims)

		C.mlx_array_release(arr)
	}

	// --- take + dequantize chain on embed row 42 ---
	// Mirror qwen_load.cpp's take(scales, [42], 0) + dequantize lines.
	// Gets the packed weight row + scales/biases row, runs dequant on GPU,
	// reports the resulting shape (full bf16 [1, 2048] for a q4 group=64
	// embed row).
	wqArr := getTensor(st, "language_model.model.embed_tokens.weight")
	scalesArr := getTensor(st, "language_model.model.embed_tokens.scales")
	biasesArr := getTensor(st, "language_model.model.embed_tokens.biases")
	defer C.mlx_array_release(wqArr)
	defer C.mlx_array_release(scalesArr)
	defer C.mlx_array_release(biasesArr)

	tokenID := []int32{42}
	idx := C.mlx_array_from_int32_1d(
		(*C.int)(unsafe.Pointer(&tokenID[0])),
		C.int(len(tokenID)),
	)
	if idx == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: failed to build index array")
		os.Exit(6)
	}
	defer C.mlx_array_release(idx)

	wqRow := C.mlx_array_take(wqArr, idx, 0)
	scRow := C.mlx_array_take(scalesArr, idx, 0)
	bsRow := C.mlx_array_take(biasesArr, idx, 0)
	defer C.mlx_array_release(wqRow)
	defer C.mlx_array_release(scRow)
	defer C.mlx_array_release(bsRow)
	if wqRow == nil || scRow == nil || bsRow == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: take failed")
		os.Exit(7)
	}

	modeC := C.CString("affine")
	full := C.mlx_array_dequantize(wqRow, scRow, bsRow,
		C.int(64), C.int(4), modeC)
	C.free(unsafe.Pointer(modeC))
	if full == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: dequantize failed")
		os.Exit(8)
	}
	defer C.mlx_array_release(full)

	// Force eval — without this all the ops above are lazy and never run.
	toEval := []C.mlx_array_t{full}
	if rc := C.mlx_eval(&toEval[0], C.int(len(toEval))); rc != 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: eval failed: rc=%d\n", int(rc))
		os.Exit(9)
	}

	fullNdim := int(C.mlx_array_ndim(full))
	fullDims := make([]int64, fullNdim)
	for i := 0; i < fullNdim; i++ {
		fullDims[i] = int64(C.mlx_array_dim(full, C.int(i)))
	}
	fmt.Printf("qwen_load_go: dequantize(embed row 42)  dtype=%s shape=%v size=%d\n",
		C.GoString(C.mlx_array_dtype_name(full)), fullDims,
		int64(C.mlx_array_size(full)))

	// --- Materialize first 5 values to host for verification against numpy ---
	// astype-to-float32 BEFORE copying — copying raw BF16 bytes through a
	// float* lens gives denormal junk (same PR #199 trap on the C++ side).
	fullF32 := C.mlx_array_to_float32(full)
	if fullF32 == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: astype-float32 failed")
		os.Exit(10)
	}
	defer C.mlx_array_release(fullF32)

	// astype is lazy too; eval again.
	toEval2 := []C.mlx_array_t{fullF32}
	if rc := C.mlx_eval(&toEval2[0], C.int(len(toEval2))); rc != 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: eval(f32) failed: rc=%d\n", int(rc))
		os.Exit(11)
	}

	const N = 5
	vals := make([]float32, N)
	got := int(C.mlx_array_copy_float(
		fullF32, (*C.float)(unsafe.Pointer(&vals[0])), C.int(N)))
	if got < 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: copy_float failed: rc=%d\n", got)
		os.Exit(12)
	}
	fmt.Printf("qwen_load_go: dequant row 42 first %d = %v\n", got, vals[:got])

	fmt.Println("qwen_load_go: cgo OK")
}

func getTensor(st C.mlx_safetensors_t, name string) C.mlx_array_t {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	arr := C.mlx_safetensors_get(st, cname)
	if arr == nil {
		fmt.Fprintf(os.Stderr, "qwen_load_go: missing tensor '%s'\n", name)
		os.Exit(5)
	}
	return arr
}

// Strip the long "language_model.model." prefix for log readability.
func shortName(s string) string {
	const prefix = "language_model.model."
	if len(s) > len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):]
	}
	return s
}
