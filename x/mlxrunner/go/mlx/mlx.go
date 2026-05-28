// Package mlx wraps libmlx.a via the C ABI shim in x/mlxrunner/cabi.
// Phase D.3 step 2 (#189). Built out incrementally as the runner needs
// more of the cabi surface — start with init + safetensors loading.
//
// The cgo preamble lives in this single file (not duplicated across
// every .go file in the package); other files in the package see the
// C declarations through the shared cgo header.
package mlx

// cgo preamble notes:
//
//   * Path is package-relative: x/mlxrunner/go/mlx -> ../../cabi.
//   * LDFLAGS names only; -L paths come from CGO_LDFLAGS at build time
//     (CI script supplies them, mirroring qwen_load_go's setup).
//   * Link order: mlx_cabi -> mlx -> CUDA + C++ runtime deps.

/*
#cgo CFLAGS: -I${SRCDIR}/../../cabi
#cgo LDFLAGS: -lmlx_cabi -lmlx -lcublas -lcublasLt -lcudart -lcuda -lstdc++ -lpthread -ldl -lm -lrt
#include <stdlib.h>
#include "mlx_cabi.h"
*/
import "C"

import "fmt"

// Init pins the default MLX device to GPU and initializes runtime
// state. Calling more than once is safe but pointless.
func Init() error {
	if rc := C.mlx_init(); rc != 0 {
		return fmt.Errorf("mlx_init failed: rc=%d", int(rc))
	}
	return nil
}
