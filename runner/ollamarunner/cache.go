package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	cache kvcache.Cache

	// host holds slots' caches saved to host memory before another prompt took
	// the slot, up to hostBudget bytes (#608).
	host       []*hostEntry
	hostBudget int64
}

// hostEntry is a slot's cache in host memory, with the inputs it holds.
type hostEntry struct {
	inputs   []*input.Input
	snapshot kvcache.Snapshot
	lastUsed time.Time
}

const (
	// hostMinSave is the fewest tokens a slot must stand to lose before it is
	// saved: a short prompt reads again faster than it saves.
	hostMinSave = 1024
	// hostMinGain is the fewest tokens a saved cache must add over the slot's
	// own prefix before it is loaded.
	hostMinGain = 256
)

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, batchSize int, multiUserCache bool) (*InputCache, error) {
	numCtx := kvSize / int32(numSlots)

	if int(numCtx) < batchSize {
		return nil, fmt.Errorf("kv size must be at least as large as batch size * parallel (kv: %v batch: %v parallel: %v)", kvSize, batchSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

	cache := model.Config().Cache
	if cache != nil {
		cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, int(numCtx), batchSize)
	}

	var hostBudget int64
	if _, ok := cache.(kvcache.SnapshotCache); ok {
		hostBudget = int64(envconfig.CacheRAM()) << 20
	}

	return &InputCache{
		numCtx:         numCtx,
		enabled:        cache != nil,
		slots:          slots,
		multiUserCache: multiUserCache,
		cache:          cache,
		hostBudget:     hostBudget,
	}, nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	default:
		return ml.DTypeF16
	}
}

func (c *InputCache) Close() {
	if c != nil && c.cache != nil {
		c.cache.Close()
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []*input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []*input.Input, cachePrompt bool) (*InputCacheSlot, []*input.Input, error) {
	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if cachePrompt && c.hostBudget > 0 {
		if int32(len(slot.Inputs))-numPast >= hostMinSave {
			c.saveSlot(slot)
		}
		if e, n := c.bestEntry(prompt, numPast); e != nil && c.loadEntry(slot, e) {
			numPast = n
		}
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
		if numPast > 0 {
			// Recurrent caches use checkpoints to pick a safe resume position.
			if cc, ok := c.cache.(kvcache.CheckpointCache); ok {
				if restored, ok := cc.PrepareRestore(slot.Id, numPast); ok {
					numPast = restored
				} else {
					numPast = 0
				}
			} else if !c.cache.CanResume(slot.Id, numPast) {
				numPast = 0
			}
		}

		err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
		if err != nil {
			// Some models don't support partial erasure
			err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			if err != nil {
				return nil, nil, err
			}
			numPast = 0
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		if c.hostBudget > 0 {
			c.saveSlot(oldestSlot)
		}
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]*input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

// saveSlot copies slot's cache to host memory before another prompt takes the
// slot, dropping the least recently used saves that no longer fit.
func (c *InputCache) saveSlot(slot *InputCacheSlot) {
	n := int32(len(slot.Inputs))
	if n < hostMinSave {
		return
	}
	for _, e := range c.host {
		if countCommonPrefix(slot.Inputs, e.inputs) == n {
			e.lastUsed = time.Now()
			return
		}
	}

	start := time.Now()
	snap, err := c.cache.(kvcache.SnapshotCache).Snapshot(slot.Id)
	if err != nil {
		slog.Debug("host cache: slot not saved", "slot", slot.Id, "inputs", n, "error", err)
		return
	}
	// A cache that went past the slot's inputs, or stops short of them, would
	// load back onto positions the inputs do not describe.
	if snap.Len() != n || snap.Size() > c.hostBudget {
		slog.Debug("host cache: slot not saved", "slot", slot.Id, "inputs", n, "cache", snap.Len(), "bytes", snap.Size())
		return
	}

	// A save this one extends is redundant.
	c.host = slices.DeleteFunc(c.host, func(e *hostEntry) bool {
		return countCommonPrefix(e.inputs, slot.Inputs) == int32(len(e.inputs))
	})
	used := snap.Size()
	for _, e := range c.host {
		used += e.snapshot.Size()
	}
	for used > c.hostBudget && len(c.host) > 0 {
		oldest := 0
		for i, e := range c.host {
			if e.lastUsed.Before(c.host[oldest].lastUsed) {
				oldest = i
			}
		}
		used -= c.host[oldest].snapshot.Size()
		c.host = slices.Delete(c.host, oldest, oldest+1)
	}
	c.host = append(c.host, &hostEntry{inputs: slices.Clone(slot.Inputs), snapshot: snap, lastUsed: time.Now()})
	slog.Info("host cache: saved slot", "slot", slot.Id, "inputs", n, "MiB", snap.Size()>>20,
		"elapsed", time.Since(start), "saved", len(c.host), "used_MiB", used>>20)
}

// bestEntry returns the saved cache that shares the longest prefix with the
// prompt, and that prefix, when the cache can resume there and the prefix
// covers at least hostMinGain more tokens than the slot's own numPast.
func (c *InputCache) bestEntry(prompt []*input.Input, numPast int32) (*hostEntry, int32) {
	var best *hostEntry
	var bestN int32
	for _, e := range c.host {
		// Leave one input to sample, as LoadCacheSlot does.
		n := min(countCommonPrefix(e.inputs, prompt), int32(len(prompt))-1)
		if n < numPast+hostMinGain || n <= bestN || !e.snapshot.Resumable(n) {
			continue
		}
		best, bestN = e, n
	}
	return best, bestN
}

// loadEntry loads a saved cache into slot, which then holds the cache's inputs.
func (c *InputCache) loadEntry(slot *InputCacheSlot, e *hostEntry) bool {
	start := time.Now()
	if err := c.cache.(kvcache.SnapshotCache).RestoreSnapshot(slot.Id, e.snapshot); err != nil {
		slog.Warn("host cache: load failed, reading the prompt", "slot", slot.Id, "error", err)
		slot.Inputs = nil
		return false
	}
	slot.Inputs = slices.Clone(e.inputs)
	e.lastUsed = time.Now()
	slog.Info("host cache: loaded slot", "slot", slot.Id, "inputs", len(e.inputs), "elapsed", time.Since(start))
	return true
}

func countCommonPrefix(a []*input.Input, b []*input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// ShiftDiscard computes how many inputs can be discarded from the cache. Inputs in the same batch
// are discarded together.
func (c *InputCache) ShiftDiscard(inputs []*input.Input, numKeep int32) int32 {
	targetFree := max((c.numCtx-numKeep)/2, 1)
	currentFree := c.numCtx - int32(len(inputs))

	var discard, sameBatch int32
	for _, input := range inputs[numKeep:] {
		if sameBatch <= 0 && currentFree >= targetFree {
			break
		}

		sameBatch--
		currentFree++
		discard++

		if input.SameBatch > 0 {
			sameBatch = int32(input.SameBatch)
		}
	}

	return discard
}

type ErrReprocessInputs struct {
	Inputs []*input.Input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(slot.Inputs, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]*input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			slot.Inputs = []*input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
