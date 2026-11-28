package ollamarunner

import (
	"testing"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model/input"
)

// fakeSnapshot resumes only at its end, like a recurrent one, unless anywhere is set.
type fakeSnapshot struct {
	n        int32
	anywhere bool
}

func (s fakeSnapshot) Len() int32               { return s.n }
func (s fakeSnapshot) Resumable(pos int32) bool { return pos == s.n || s.anywhere && pos < s.n }
func (s fakeSnapshot) Size() int64              { return int64(s.n) * 1000 }

// snapCache records the slots it saves and restores.
type snapCache struct {
	mockCache
	held            map[int]int32 // positions each slot holds
	saved, restored []int
	anywhere        bool // snapshots resume at any position they hold
}

func (c *snapCache) Snapshot(seq int) (kvcache.Snapshot, error) {
	c.saved = append(c.saved, seq)
	return fakeSnapshot{n: c.held[seq], anywhere: c.anywhere}, nil
}

func (c *snapCache) RestoreSnapshot(seq int, s kvcache.Snapshot) error {
	c.restored = append(c.restored, seq)
	c.held[seq] = s.Len()
	return nil
}

func run(from, n int32) []*input.Input {
	in := make([]*input.Input, n)
	for i := range in {
		in[i] = &input.Input{Token: from + int32(i)}
	}
	return in
}

// #608: a side request that takes the session's slot saves it to host memory,
// and the session's next turn loads it back instead of reading it again.
func TestHostCacheSideRequest(t *testing.T) {
	for _, multiUser := range []bool{false, true} {
		session := run(1, 2000)
		cache := &snapCache{held: map[int]int32{0: 2000}}
		c := InputCache{
			multiUserCache: multiUser,
			cache:          cache,
			hostBudget:     1 << 30,
			slots:          []InputCacheSlot{{Id: 0, Inputs: session, lastUsed: time.Now().Add(-time.Second)}},
		}

		side := run(5000, 40)
		slot, rest, err := c.LoadCacheSlot(side, true)
		if err != nil || slot.Id != 0 || len(rest) != 40 {
			t.Fatalf("multiUser=%v side request: slot %v, %d to read, %v", multiUser, slot.Id, len(rest), err)
		}
		if len(cache.saved) != 1 || len(c.host) != 1 {
			t.Fatalf("multiUser=%v: saved %v, host holds %d; want the session saved", multiUser, cache.saved, len(c.host))
		}
		slot.Inputs, slot.InUse = side, false
		cache.held[0] = 40

		next := append(append([]*input.Input{}, session...), run(9000, 10)...)
		slot, rest, err = c.LoadCacheSlot(next, true)
		if err != nil || len(rest) != 10 {
			t.Fatalf("multiUser=%v next turn: %d to read, %v; want 10", multiUser, len(rest), err)
		}
		if len(cache.restored) != 1 || cache.held[slot.Id] != 2000 {
			t.Fatalf("multiUser=%v: restored %v, slot holds %d; want the session loaded", multiUser, cache.restored, cache.held[slot.Id])
		}
		if len(cache.saved) != 1 {
			t.Fatalf("multiUser=%v: saved %v; the 40-token side request is not worth a save", multiUser, cache.saved)
		}
	}
}

// #608: saves past the budget drop the least recently used one.
func TestHostCacheBudget(t *testing.T) {
	cache := &snapCache{held: map[int]int32{0: 2000}}
	c := InputCache{cache: cache, hostBudget: 5_000_000}
	for i := range 3 {
		slot := &InputCacheSlot{Id: 0, Inputs: run(int32(10000*i), 2000)}
		c.saveSlot(slot)
	}
	if len(c.host) != 2 || c.host[0].inputs[0].Token != 10000 {
		t.Fatalf("host holds %d saves starting at token %d; want the 2 newest", len(c.host), c.host[0].inputs[0].Token)
	}
}

// #608: a saved cache that resumes anywhere loads for a prompt that diverges
// inside it; one that resumes only at its end does not.
func TestHostCachePartialPrefix(t *testing.T) {
	for _, anywhere := range []bool{true, false} {
		cache := &snapCache{held: map[int]int32{0: 2000}, anywhere: anywhere}
		c := InputCache{cache: cache, hostBudget: 1 << 30, slots: []InputCacheSlot{{Id: 0, Inputs: run(1, 2000)}}}
		c.saveSlot(&c.slots[0])
		c.slots[0].Inputs = run(5000, 40)

		next := append(run(1, 1900), run(9000, 30)...)
		_, rest, err := c.LoadCacheSlot(next, true)
		if err != nil {
			t.Fatal(err)
		}
		want := map[bool]int{true: 30, false: 1930}[anywhere]
		if len(rest) != want {
			t.Fatalf("anywhere=%v: %d to read, want %d", anywhere, len(rest), want)
		}
	}
}
