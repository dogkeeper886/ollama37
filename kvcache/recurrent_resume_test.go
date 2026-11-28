package kvcache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/model/input"
)

// #608: a prompt that extends everything the sequence processed resumes on the
// live state, and keeps its checkpoints.
func TestCachePrepareRestoreLive(t *testing.T) {
	cache := newTestCache()
	cache.slotForSeq[1] = 0
	cache.refCount = []int{1}
	cache.statePos[1] = 20
	store := cache.checkpointStore(0)
	store.record(5)
	store.record(15)

	pos, ok := cache.PrepareRestore(1, 20)
	if !ok || pos != 20 {
		t.Fatalf("PrepareRestore(1, 20) = %d, %v; want 20, true", pos, ok)
	}
	if err := cache.Remove(1, pos, math.MaxInt32); err != nil {
		t.Fatalf("Remove after live resume: %v", err)
	}
	if cache.statePos[1] != 20 {
		t.Fatalf("statePos = %d, want 20", cache.statePos[1])
	}
	if _, best, ok := store.bestIndex(20); !ok || best != 15 {
		t.Fatalf("checkpoint 15 gone after live resume: best=%d ok=%v", best, ok)
	}

	// A prompt that diverges before the live end still restores a checkpoint.
	cache.statePos[1] = 20
	if pos, ok := cache.PrepareRestore(1, 18); !ok || pos != 16 {
		t.Fatalf("PrepareRestore(1, 18) = %d, %v; want 16, true", pos, ok)
	}
}

// #608: a sequence forked by CopyPrefix shares its source's state, so it resumes
// live only when its prompt extends all of the source's.
func TestCacheCopyPrefixStatePos(t *testing.T) {
	cache := newTestCache()
	cache.slotForSeq[0] = 0
	cache.refCount = []int{1, 0}
	cache.statePos[0] = 30
	cache.statePos[1] = 7
	cache.kv.cellRanges = map[int]cellRange{}

	cache.CopyPrefix(0, 1, 12)

	if cache.statePos[1] != 30 {
		t.Fatalf("forked statePos = %d, want the source's 30", cache.statePos[1])
	}
	if _, ok := cache.PrepareRestore(1, 12); ok {
		t.Fatalf("PrepareRestore(1, 12) resumed on a state that is at 30")
	}
}

// #608: CheckpointNext takes a checkpoint at the batch's end even inside the
// interval, once.
func TestPlanCheckpointsCheckpointNext(t *testing.T) {
	cache := newTestCache()
	cache.checkpointCount = 4
	cache.curSeqs = []int{1}
	cache.curSlots = []int{0}
	cache.checkpointStore(0).record(100)

	batch := input.Batch{Positions: []int32{101, 102, 103}, Sequences: []int{1, 1, 1}}

	cache.planCheckpoints(batch)
	if cache.curCheckpointPos[0] != -1 {
		t.Fatalf("checkpoint at %d inside the interval without a request", cache.curCheckpointPos[0])
	}

	cache.CheckpointNext(1)
	cache.planCheckpoints(batch)
	if cache.curCheckpointPos[0] != 103 || !cache.curCheckpointTail[0] {
		t.Fatalf("requested checkpoint: pos=%d tail=%v; want 103, true", cache.curCheckpointPos[0], cache.curCheckpointTail[0])
	}

	cache.planCheckpoints(batch)
	if cache.curCheckpointPos[0] != -1 {
		t.Fatalf("the request was not consumed: checkpoint at %d", cache.curCheckpointPos[0])
	}
}

// #608: prompt-end checkpoints reuse their own entries, so a run of requests
// leaves the interval checkpoints that reach further back.
func TestSlotCheckpointStoreRecordTail(t *testing.T) {
	store := newSlotCheckpointStore(4)
	store.record(1000)
	store.record(2664)
	for pos := int32(3000); pos < 3100; pos += 10 {
		store.recordTail(pos)
	}

	for _, want := range []int32{1000, 2664} {
		if _, got, ok := store.bestIndex(want + 1); !ok || got != want {
			t.Fatalf("interval checkpoint %d gone: best=%d ok=%v", want, got, ok)
		}
	}
	if _, got, ok := store.bestIndex(3095); !ok || got != 3090 {
		t.Fatalf("best for 3095 = %d, %v; want the latest tail 3090", got, ok)
	}
	if _, got, ok := store.bestIndex(3090); !ok || got != 3080 {
		t.Fatalf("best for 3090 = %d, %v; want the tail before it, 3080", got, ok)
	}
	if store.size != 4 {
		t.Fatalf("size = %d, want 4", store.size)
	}

	// An interval checkpoint that lands on a tail entry takes it over.
	idx := store.record(4800)
	for _, i := range store.tail {
		if i == idx {
			t.Fatalf("entry %d is both an interval and a tail checkpoint", idx)
		}
	}
}
