package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/model/input"
)

// #608: the token that ended a reply goes through the cache in the next batch;
// the slot records it unless another prompt took the slot or a stop cut it.
func TestKeepFinishedInputs(t *testing.T) {
	tokens := func(n int) []*input.Input {
		in := make([]*input.Input, n)
		for i := range in {
			in[i] = &input.Input{Token: int32(i)}
		}
		return in
	}
	eos := &input.Input{Token: 99}
	batch := input.Batch{Sequences: []int{3, 1}, Positions: []int32{7, 5}}

	cases := []struct {
		name   string
		inUse  bool
		inputs int
		want   int
	}{
		{"free slot", false, 5, 6},
		{"slot taken by another prompt", true, 5, 5},
		{"stop sequence cut the inputs", false, 4, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			slot := &InputCacheSlot{Id: 1, Inputs: tokens(tc.inputs), InUse: tc.inUse}
			seq := &Sequence{cache: slot, pendingInputs: []*input.Input{eos}}
			keepFinishedInputs(batch, seq)
			if len(slot.Inputs) != tc.want {
				t.Fatalf("slot holds %d inputs, want %d", len(slot.Inputs), tc.want)
			}
			if tc.want == 6 && slot.Inputs[5] != eos {
				t.Fatalf("last input = %v, want the stop token", slot.Inputs[5])
			}
		})
	}
}
