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

	// --- rms_norm probe (mirrors qwen_load.cpp PR #203) ---
	// Apply layer-0 input_layernorm to the dequant'd embed row. Bundles a
	// reduce path validation (sum-of-squares uses the row_reduce AccT
	// pattern from PR #200) AND the rms_norm dispatch on real BF16 input +
	// gain. Ground truth from qwen_load.cpp:
	//   first 5    = [0, 0, -1.09375, -1.00781, 0.710938]
	//   abs_mean   = 0.650743
	normGain := getTensor(st, "language_model.model.layers.0.input_layernorm.weight")
	defer C.mlx_array_release(normGain)

	normed := C.mlx_array_rms_norm(full, normGain, C.float(1e-6))
	if normed == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: rms_norm failed")
		os.Exit(13)
	}
	defer C.mlx_array_release(normed)

	first5OfArray(normed, "rms_norm(dequant row 42)")

	absMean := getAbsMeanScalar(normed)
	fmt.Printf("qwen_load_go: rms_norm abs_mean=%g\n", absMean)

	// --- quantized_matmul probe (mirrors qwen_load.cpp PR #204) ---
	// First Phase-D quantized linear from Go: project the rms-normed
	// embed via layer-0 in_proj_qkv. Bf16 [1, 2048] @ q4 [8192, 256] ->
	// bf16 [1, 8192]. Ground truth:
	//   out[0, 0] = 2.01562    out[0, 1] = -0.105469
	qkvW := getTensor(st, "language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
	qkvS := getTensor(st, "language_model.model.layers.0.linear_attn.in_proj_qkv.scales")
	qkvB := getTensor(st, "language_model.model.layers.0.linear_attn.in_proj_qkv.biases")
	defer C.mlx_array_release(qkvW)
	defer C.mlx_array_release(qkvS)
	defer C.mlx_array_release(qkvB)

	modeCqkv := C.CString("affine")
	qkvProj := C.mlx_array_quantized_matmul(
		normed, qkvW, qkvS, qkvB,
		C.int(1),     // transpose=true
		C.int(64),    // group_size
		C.int(4),     // bits
		modeCqkv)
	C.free(unsafe.Pointer(modeCqkv))
	if qkvProj == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: quantized_matmul failed")
		os.Exit(14)
	}
	defer C.mlx_array_release(qkvProj)

	// Print shape + first 2 values (the ones we have ground truth for).
	qkvDims := dimsOf(qkvProj)
	fmt.Printf("qwen_load_go: in_proj_qkv(rms_normed) shape=%v\n", qkvDims)
	first5OfArray(qkvProj, "in_proj_qkv")

	// --- conv1d probe (mirrors qwen_load.cpp PR #205) ---
	// Depthwise conv1d (groups=8192, kw=4) on a synthetic all-ones input.
	// For ones input + padding=0 the output is just the per-channel kernel
	// sum. Ground truth from PR #205: [-0.0617676, -0.0458984, 0.0678711].
	convW := getTensor(st, "language_model.model.layers.0.linear_attn.conv1d.weight")
	defer C.mlx_array_release(convW)

	convDims := []int64{1, 4, 8192}
	dtypeBF16 := C.CString("bfloat16")
	convIn := C.mlx_array_ones(
		(*C.longlong)(unsafe.Pointer(&convDims[0])),
		C.int(len(convDims)), dtypeBF16)
	C.free(unsafe.Pointer(dtypeBF16))
	if convIn == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: ones() failed")
		os.Exit(23)
	}
	defer C.mlx_array_release(convIn)

	convOut := C.mlx_array_conv1d(convIn, convW,
		C.int(1),    // stride
		C.int(0),    // padding
		C.int(1),    // dilation
		C.int(8192)) // groups (depthwise)
	if convOut == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: conv1d failed")
		os.Exit(24)
	}
	defer C.mlx_array_release(convOut)
	fmt.Printf("qwen_load_go: conv1d(ones, layer_0.conv1d.w) shape=%v\n",
		dimsOf(convOut))
	first5OfArray(convOut, "conv1d")

	// --- silu probe (mirrors qwen_load.cpp PR #208) ---
	// silu(x) = sigmoid(x) * x. Composed from two ops since MLX exposes
	// silu only via the higher-level fast namespace, not ops.h.
	sigOut := C.mlx_array_sigmoid(convOut)
	if sigOut == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: sigmoid failed")
		os.Exit(25)
	}
	defer C.mlx_array_release(sigOut)

	siluOut := C.mlx_array_multiply(sigOut, convOut)
	if siluOut == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: multiply (silu) failed")
		os.Exit(26)
	}
	defer C.mlx_array_release(siluOut)
	first5OfArray(siluOut, "silu(conv1d)")

	fmt.Println("qwen_load_go: cgo OK")
}

// dimsOf returns the shape of an array (uses mlx_array_ndim + mlx_array_dim).
func dimsOf(arr C.mlx_array_t) []int64 {
	n := int(C.mlx_array_ndim(arr))
	d := make([]int64, n)
	for i := 0; i < n; i++ {
		d[i] = int64(C.mlx_array_dim(arr, C.int(i)))
	}
	return d
}

// first5OfArray astype-fp32-then-copy + prints the first 5 elements. Common
// pattern; folding into a helper avoids three more boilerplate blocks.
func first5OfArray(arr C.mlx_array_t, label string) {
	f32 := C.mlx_array_to_float32(arr)
	if f32 == nil {
		fmt.Fprintf(os.Stderr, "qwen_load_go: %s astype failed\n", label)
		os.Exit(15)
	}
	defer C.mlx_array_release(f32)

	to := []C.mlx_array_t{f32}
	if rc := C.mlx_eval(&to[0], C.int(1)); rc != 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: %s eval failed: rc=%d\n", label, int(rc))
		os.Exit(16)
	}

	const N = 5
	vals := make([]float32, N)
	got := int(C.mlx_array_copy_float(
		f32, (*C.float)(unsafe.Pointer(&vals[0])), C.int(N)))
	if got < 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: %s copy failed: rc=%d\n", label, got)
		os.Exit(17)
	}
	fmt.Printf("qwen_load_go: %s first %d = %v\n", label, got, vals[:got])
}

// getAbsMeanScalar computes mean(abs(arr)) as fp32 and returns the scalar.
func getAbsMeanScalar(arr C.mlx_array_t) float32 {
	absA := C.mlx_array_abs(arr)
	if absA == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: abs failed")
		os.Exit(18)
	}
	defer C.mlx_array_release(absA)

	meanA := C.mlx_array_mean_all(absA)
	if meanA == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: mean failed")
		os.Exit(19)
	}
	defer C.mlx_array_release(meanA)

	f32 := C.mlx_array_to_float32(meanA)
	if f32 == nil {
		fmt.Fprintln(os.Stderr, "qwen_load_go: abs_mean astype failed")
		os.Exit(20)
	}
	defer C.mlx_array_release(f32)

	to := []C.mlx_array_t{f32}
	if rc := C.mlx_eval(&to[0], C.int(1)); rc != 0 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: abs_mean eval failed: rc=%d\n", int(rc))
		os.Exit(21)
	}

	var v float32
	if got := int(C.mlx_array_copy_float(
		f32, (*C.float)(unsafe.Pointer(&v)), C.int(1))); got != 1 {
		fmt.Fprintf(os.Stderr, "qwen_load_go: abs_mean copy got %d not 1\n", got)
		os.Exit(22)
	}
	return v
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
