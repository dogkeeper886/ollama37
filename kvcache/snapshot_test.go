package kvcache

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func (t *testTensor) Bytes() []byte {
	b := make([]byte, 4*len(t.data))
	for i, f := range t.data {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func (c *testContext) FromBytes(dtype ml.DType, s []byte, shape ...int) ml.Tensor {
	t := c.Empty(dtype, shape...).(*testTensor)
	for i := range t.data {
		t.data[i] = math.Float32frombits(binary.LittleEndian.Uint32(s[4*i:]))
	}
	return t
}

// fill stores positions [from, to) of seq, each cell's two values pos*10 and pos*10+1.
func fill(t *testing.T, cache *Causal, seq int, from, to int32) {
	t.Helper()
	ctx := (&testBackend{}).NewContext()
	var pos []int32
	var seqs []int
	var vals []float32
	for p := from; p < to; p++ {
		pos, seqs, vals = append(pos, p), append(seqs, seq), append(vals, float32(p*10), float32(p*10+1))
	}
	if err := cache.StartForward(ctx, input.Batch{Positions: pos, Sequences: seqs}, false); err != nil {
		t.Fatal(err)
	}
	cache.SetLayer(0)
	x := ctx.FromFloats(vals, 2, 1, len(pos))
	cache.Put(ctx, x, x)
}

// held returns seq's key values in position order.
func held(cache *Causal, seq int) []float32 {
	var cells []int
	for i := range cache.cells {
		if slices.Contains(cache.cells[i].sequences, seq) {
			cells = append(cells, i)
		}
	}
	slices.SortFunc(cells, func(a, b int) int { return int(cache.cells[a].pos - cache.cells[b].pos) })
	data := cache.keys[0].(*testTensor).data
	var out []float32
	for _, i := range cells {
		out = append(out, data[2*i], data[2*i+1])
	}
	return out
}

// #608: a sequence saved to host memory loads back into another sequence with
// the same positions and values, even when its cells were split in two.
func TestCausalSnapshotRoundTrip(t *testing.T) {
	cache := NewCausalCache(nil)
	cache.Init(&testBackend{}, ml.DTypeF16, 2, 16, 4)

	fill(t, cache, 0, 0, 3)
	fill(t, cache, 1, 0, 2) // seq 1's cells split seq 0's
	fill(t, cache, 0, 3, 5)
	want := held(cache, 0)

	snap, err := cache.Snapshot(0)
	if err != nil {
		t.Fatal(err)
	}
	if snap.Len() != 5 {
		t.Fatalf("Len = %d, want 5", snap.Len())
	}
	if err := cache.RestoreSnapshot(1, snap); err != nil {
		t.Fatal(err)
	}
	if got := held(cache, 1); !slices.Equal(got, want) {
		t.Fatalf("restored %v, want %v", got, want)
	}
	if !cache.CanResume(1, 5) {
		t.Fatalf("CanResume(1, 5) = false after restore")
	}
}

// #608: a recurrent sequence saves its state with its attention cache and
// resumes live after a load.
func TestRecurrentSnapshotRoundTrip(t *testing.T) {
	cache := NewRecurrentCache(RecurrentConfig{ConvDim: 1, ConvChannels: 2, RecurrentStateSize: 3})
	cache.Init(&testBackend{}, ml.DTypeF16, 2, 16, 4)

	fill(t, cache.kv, 0, 0, 4)
	cache.slotForSeq[0], cache.refCount[0] = 0, 1
	cache.freeSlots = slices.DeleteFunc(cache.freeSlots, func(s int) bool { return s == 0 })
	cache.statePos[0] = 4
	copy(cache.convBuffer(0).(*testTensor).data, []float32{1, 2, 0, 0})
	copy(cache.recurrentBuffer(0).(*testTensor).data, []float32{3, 4, 5, 0, 0, 0})

	snap, err := cache.Snapshot(0)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.RestoreSnapshot(1, snap); err != nil {
		t.Fatal(err)
	}
	slot := cache.slotForSeq[1]
	conv := cache.convBuffer(0).(*testTensor).data[2*slot : 2*slot+2]
	rec := cache.recurrentBuffer(0).(*testTensor).data[3*slot : 3*slot+3]
	if !slices.Equal(conv, []float32{1, 2}) || !slices.Equal(rec, []float32{3, 4, 5}) {
		t.Fatalf("restored conv %v, recurrent %v", conv, rec)
	}
	if pos, ok := cache.PrepareRestore(1, 4); !ok || pos != 4 {
		t.Fatalf("PrepareRestore(1, 4) = %d, %v; want a live resume at 4", pos, ok)
	}

	// A state ahead of its attention cache is refused.
	cache.statePos[0] = 5
	if _, err := cache.Snapshot(0); err == nil {
		t.Fatalf("Snapshot took a state past its attention cache")
	}
}

// #608: a sliding-window snapshot resumes only where its window still fits.
func TestCausalSnapshotResumable(t *testing.T) {
	swa := &causalSnapshot{first: 100, n: 50, window: 10}
	for pos, want := range map[int32]bool{150: true, 110: true, 109: false, 151: false} {
		if got := swa.Resumable(pos); got != want {
			t.Errorf("SWA Resumable(%d) = %v, want %v", pos, got, want)
		}
	}
	full := &causalSnapshot{first: 0, n: 50, window: math.MaxInt32}
	if !full.Resumable(1) || !full.Resumable(50) || full.Resumable(51) {
		t.Errorf("full-attention snapshot resumes at 1..50 only")
	}
}
