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

// Library NAMES only here; -L paths supplied by CGO_LDFLAGS env at build
// time so the cgo pragma works whether the build dir is /build,
// /tmp/runner-X, or elsewhere. The CI script (test-mlx-smoke.sh) sets
// the right -L flags.
//
// Link order matters: mlx_cabi calls into mlx, mlx calls into CUDA libs.
// stdc++ is needed because libmlx.a is C++; -lrt -ldl -lpthread are
// libmlx.a transitive deps.
//
// #cgo CFLAGS: -I${SRCDIR}/../../../cabi
// #cgo LDFLAGS: -lmlx_cabi -lmlx -lcublas -lcublasLt -lcudart -lcuda -lstdc++ -lpthread -ldl -lm -lrt
// #include "mlx_cabi.h"
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
	fmt.Println("qwen_load_go: cgo OK")
}
