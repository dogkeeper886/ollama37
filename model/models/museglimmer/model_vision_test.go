package museglimmer

import (
	"math"
	"slices"
	"testing"
)

func TestNewVisionInputs(t *testing.T) {
	// A 4x2 grid in 2x2 windows: two windows, left then right.
	in := newVisionInputs(4, 2, 2, 2)

	if want := []int32{0, 1, 4, 5, 2, 3, 6, 7}; !slices.Equal(in.spPerm, want) {
		t.Errorf("spPerm = %v, want %v", in.spPerm, want)
	}
	for i, orig := range in.spPerm {
		if in.invPerm[orig] != int32(i) {
			t.Errorf("invPerm[%d] = %d, want %d", orig, in.invPerm[orig], i)
		}
	}
	// 1-indexed column and row of each permuted patch.
	if want := []int32{1, 2, 1, 2, 3, 4, 3, 4}; !slices.Equal(in.posW, want) {
		t.Errorf("posW = %v, want %v", in.posW, want)
	}
	if want := []int32{1, 1, 2, 2, 1, 1, 2, 2}; !slices.Equal(in.posH, want) {
		t.Errorf("posH = %v, want %v", in.posH, want)
	}
	// Block-diagonal: patches attend within their own window only.
	n := 8
	for a := range n {
		for b := range n {
			same := a/4 == b/4
			if got := in.spMask[a*n+b]; same != (got == 0) || (!same && !math.IsInf(float64(got), -1)) {
				t.Fatalf("spMask[%d,%d] = %v, same window %v", a, b, got, same)
			}
		}
	}
	// Pixel shuffle: each output token gathers its 2x2 neighbours.
	if want := []int32{0, 1, 4, 5, 2, 3, 6, 7}; !slices.Equal(in.dsPerm, want) {
		t.Errorf("dsPerm = %v, want %v", in.dsPerm, want)
	}
}

func TestGridSize(t *testing.T) {
	cases := []struct {
		w, h, wantW, wantH int
	}{
		{56, 56, 56, 56},       // already 2x2 cells
		{100, 50, 112, 56},     // keeps 2:1
		{4000, 4000, 896, 896}, // capped at 32x32 cells (1024 tokens)
	}
	for _, c := range cases {
		gotW, gotH := gridSize(c.w, c.h, 28, 1024)
		if gotW != c.wantW || gotH != c.wantH {
			t.Errorf("gridSize(%d, %d) = %d x %d, want %d x %d", c.w, c.h, gotW, gotH, c.wantW, c.wantH)
		}
	}
}
