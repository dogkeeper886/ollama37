package nn

import (
	"math"
	"testing"
)

func TestHadamardMatrix_Sizes(t *testing.T) {
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64, 128} {
		h := HadamardMatrix(n)
		if len(h) != n*n {
			t.Errorf("n=%d: len=%d want %d", n, len(h), n*n)
		}
		// Length alone is guaranteed by make(); also verify orthogonality
		// at every size so a Sylvester recursion bug that only surfaces
		// at, say, n=128 does not slip through.
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				var sum float64
				for k := 0; k < n; k++ {
					sum += float64(h[i*n+k]) * float64(h[j*n+k])
				}
				want := 0.0
				if i == j {
					want = 1.0
				}
				if math.Abs(sum-want) > 1e-4 {
					t.Fatalf("n=%d: (H·Hᵀ)[%d,%d] = %v; want %v", n, i, j, sum, want)
				}
			}
		}
	}
}

func TestHadamardMatrix_Normalization(t *testing.T) {
	n := 64
	h := HadamardMatrix(n)
	expected := float32(1.0 / math.Sqrt(float64(n)))
	for i, v := range h {
		if math.Abs(float64(v)-float64(expected)) > 1e-6 && math.Abs(float64(v)+float64(expected)) > 1e-6 {
			t.Fatalf("h[%d] = %v; want ±%v", i, v, expected)
			return
		}
	}
}

func TestHadamardMatrix_Orthogonal(t *testing.T) {
	// H · Hᵀ should equal I (identity). Since Sylvester Hadamard is symmetric,
	// this is equivalent to H · H = I.
	n := 64
	h := HadamardMatrix(n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < n; k++ {
				sum += float64(h[i*n+k]) * float64(h[j*n+k])
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(sum-want) > 1e-5 {
				t.Fatalf("(H·Hᵀ)[%d,%d] = %v; want %v", i, j, sum, want)
			}
		}
	}
}

func TestHadamardMatrix_Symmetric(t *testing.T) {
	// Sylvester Hadamard matrices are symmetric: H[i,j] == H[j,i].
	n := 64
	h := HadamardMatrix(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if h[i*n+j] != h[j*n+i] {
				t.Fatalf("H[%d,%d]=%v != H[%d,%d]=%v", i, j, h[i*n+j], j, i, h[j*n+i])
			}
		}
	}
}

func TestHadamardMatrix_PowerOfTwoOnly(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-power-of-2 size")
		}
	}()
	HadamardMatrix(65)
}

func TestIsHadamardCompatible(t *testing.T) {
	cases := []struct {
		headDim int
		want    bool
	}{
		{32, false},   // below 64
		{63, false},   // below 64
		{64, true},    // exact
		{128, true},   // 2×64
		{192, true},   // 3×64
		{80, false},   // not divisible by 64
		{96, false},   // not divisible by 64
		{112, false},  // not divisible by 64
		{256, true},   // 4×64
	}
	for _, c := range cases {
		if got := IsHadamardCompatible(c.headDim); got != c.want {
			t.Errorf("IsHadamardCompatible(%d) = %v; want %v", c.headDim, got, c.want)
		}
	}
}
