package nn

import "math"

// HadamardMatrix returns a row-major normalized Sylvester Hadamard matrix of
// size n×n. n must be a power of 2 and ≥ 1. Every element is ±1/√n, the matrix
// is symmetric, and H·Hᵀ = I so it acts as an orthogonal rotation that
// preserves dot products.
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
func IsHadamardCompatible(headDim int) bool {
	return headDim >= 64 && headDim%64 == 0
}
