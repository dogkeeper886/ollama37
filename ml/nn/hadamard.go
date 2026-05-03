package nn

import (
	"math"

	"github.com/ollama/ollama/ml"
)

// hadamard64 is the normalized 64×64 Sylvester Hadamard matrix used by the
// MVP rotation integration. Computed once at package init so the per-call
// cost is just a small CPU→GPU upload, not a regenerate-then-upload.
var hadamard64 = HadamardMatrix(64)

// blockRotate applies the 64×64 Sylvester Hadamard `h` block-diagonally
// along the head dim of `t`. Input and output shapes are identical; the
// transform is exact in fp32 (orthogonal, dot-product preserving).
//
// Layout: t is [head_dim, heads, seq] in column-major order. Reshape to
// [64, k·heads, seq] (where k = head_dim/64) so the matmul applies H_64
// to each contiguous 64-element slice of the head dim independently — the
// block-diagonal effect comes for free from the reshape.
//
// Caller is responsible for the rotation gate (IsHadamardCompatible plus
// having a non-nil cache). This helper does no checks of its own.
func blockRotate(ctx ml.Context, t, h ml.Tensor) ml.Tensor {
	d := t.Dim(0)
	heads := t.Dim(1)
	seq := t.Dim(2)
	k := d / 64

	// h.Mulmat(reshaped) computes h^T · reshaped along the contracted dim
	// (the leading 64). Sylvester H is symmetric, so h^T = h, and the
	// result is the desired h · reshaped.
	reshaped := t.Reshape(ctx, 64, k*heads, seq)
	rotated := h.Mulmat(ctx, reshaped)
	return rotated.Reshape(ctx, d, heads, seq)
}

// hadamardTensor materializes the cached 64×64 Hadamard slice as a backend
// tensor in the supplied context. Uses ctx.Input() because the bare context
// during reserveWorstCaseGraph rejects FromFloats with "set Input or Layer
// before creating tensors"; matches the convention used by every other
// FromFloats caller in the codebase (e.g. model/models/gemma3/model.go:104,
// model/models/gemma4/model_audio.go:340). Cheap (16 KiB upload); not worth
// caching the tensor across calls because each call may target a different
// backend context with its own allocator.
func hadamardTensor(ctx ml.Context) ml.Tensor {
	return ctx.Input().FromFloats(hadamard64, 64, 64)
}

// HadamardMatrix returns a row-major normalized Sylvester Hadamard matrix of
// size n×n. n must be a power of 2 and ≥ 1. Every element is ±1/√n, the
// matrix is symmetric under the Sylvester recursion (not all Hadamard
// matrices are symmetric — Paley construction, for example, is not), and
// H·Hᵀ = I so it acts as an orthogonal rotation that preserves dot products.
//
// This is the rotation used by llama.cpp PR #21038 (commit 744c0c73) and
// Google DeepMind's TurboQuant (arXiv:2504.19874) to smooth coordinate
// distributions before KV cache quantization. See docs/design/kv-rotation.md
// for the integration plan.
func HadamardMatrix(n int) []float32 {
	if n < 1 || n&(n-1) != 0 {
		panic("nn.HadamardMatrix: n must be a power of 2")
	}
	h := make([]float32, n*n)
	h[0] = 1
	for size := 1; size < n; size *= 2 {
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				v := h[i*n+j]
				h[i*n+(j+size)] = v
				h[(i+size)*n+j] = v
				h[(i+size)*n+(j+size)] = -v
			}
		}
	}
	norm := float32(1.0 / math.Sqrt(float64(n)))
	for i := range h {
		h[i] *= norm
	}
	return h
}

// IsHadamardCompatible reports whether a head dimension supports KV cache
// rotation under the MVP integration: head dim is at least 64 and an exact
// multiple of 64. Head dims that satisfy this use a block-diagonal of 64×64
// Hadamard matrices, which is the conservative subset of upstream's choice
// (upstream uses the largest power-of-2 divisor of head_dim for Q/K and a
// fixed 64×64 for V; we unify on 64 for simplicity in the first cut).
//
// Models with head_dim ∈ {80, 96, 112} (some Llama-1/2 variants, MPT-7B)
// are silently excluded by this gate.
func IsHadamardCompatible(headDim int) bool {
	return headDim >= 64 && headDim%64 == 0
}
