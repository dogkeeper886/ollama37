package kvcache

import (
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// TODO(jmorganca): Add byte-serialized host-RAM checkpoints to reduce GPU
// memory usage while preserving prefix reuse for recurrent state.

type checkpointEntry struct {
	pos       int32
	conv      map[int]ml.Tensor
	recurrent map[int]ml.Tensor
}

type slotCheckpointStore struct {
	entries []checkpointEntry
	size    int
	next    int
	lastPos int32
	// tail holds the entries of the latest prompt-end checkpoints, oldest first.
	tail []int
}

type checkpointRestore struct {
	slot int
	idx  int
	pos  int32
	// live: the sequence's live state already ends at pos, so nothing is restored.
	live bool
}

func newSlotCheckpointStore(n int) *slotCheckpointStore {
	entries := make([]checkpointEntry, n)
	for i := range entries {
		entries[i].pos = -1
	}
	return &slotCheckpointStore{
		entries: entries,
		lastPos: -1,
	}
}

func (s *slotCheckpointStore) reset() {
	s.size = 0
	s.next = 0
	s.lastPos = -1
	s.tail = s.tail[:0]
	for i := range s.entries {
		s.entries[i].pos = -1
	}
}

func (s *slotCheckpointStore) record(pos int32) int {
	if len(s.entries) == 0 {
		return -1
	}
	idx := s.next
	s.next = (s.next + 1) % len(s.entries)
	if s.size < len(s.entries) {
		s.size++
	}
	s.entries[idx].pos = pos
	s.lastPos = pos
	s.tail = slices.DeleteFunc(s.tail, func(i int) bool { return i == idx })
	return idx
}

// recordTail records a prompt-end checkpoint. Past maxTailCheckpoints it reuses
// the oldest one's entry instead of the ring's next, so the interval checkpoints
// that reach further back stay.
func (s *slotCheckpointStore) recordTail(pos int32) int {
	if len(s.tail) < maxTailCheckpoints {
		idx := s.record(pos)
		if idx >= 0 {
			s.tail = append(s.tail, idx)
		}
		return idx
	}
	idx := s.tail[0]
	s.tail = append(s.tail[1:], idx)
	if s.entries[idx].pos < 0 {
		s.size++
	}
	s.entries[idx].pos = pos
	s.lastPos = pos
	return idx
}

func (s *slotCheckpointStore) bestIndex(targetPos int32) (int, int32, bool) {
	bestIdx := -1
	bestPos := int32(-1)
	for i := range s.entries {
		pos := s.entries[i].pos
		if pos < 0 || pos >= targetPos {
			continue
		}
		if pos > bestPos {
			bestPos = pos
			bestIdx = i
		}
	}
	if bestIdx < 0 {
		return -1, -1, false
	}
	return bestIdx, bestPos, true
}

func (s *slotCheckpointStore) pruneAfter(pos int32) {
	if len(s.entries) == 0 {
		s.size = 0
		s.next = 0
		s.lastPos = -1
		return
	}

	size := 0
	next := -1
	minPos := int32(math.MaxInt32)
	minIdx := 0
	for i := range s.entries {
		if s.entries[i].pos > pos {
			s.entries[i].pos = -1
		}
		if s.entries[i].pos >= 0 {
			size++
			if s.entries[i].pos < minPos {
				minPos = s.entries[i].pos
				minIdx = i
			}
		} else if next == -1 {
			next = i
		}
	}

	s.size = size
	if size == 0 {
		s.next = 0
		s.lastPos = -1
		return
	}
	if next != -1 {
		s.next = next
	} else {
		// Full ring: overwrite the oldest checkpoint next.
		s.next = minIdx
	}
	s.lastPos = pos
}

func (s *slotCheckpointStore) shiftRange(beginIndex, endIndex int32) {
	if len(s.entries) == 0 {
		s.size = 0
		s.next = 0
		s.lastPos = -1
		return
	}

	offset := beginIndex - endIndex

	size := 0
	next := -1
	minPos := int32(math.MaxInt32)
	maxPos := int32(-1)
	minIdx := 0

	for i := range s.entries {
		pos := s.entries[i].pos
		if pos >= 0 {
			if pos >= beginIndex && pos < endIndex {
				s.entries[i].pos = -1
			} else if pos >= endIndex {
				s.entries[i].pos = pos + offset
			}
		}

		pos = s.entries[i].pos
		if pos >= 0 {
			size++
			if pos < minPos {
				minPos = pos
				minIdx = i
			}
			if pos > maxPos {
				maxPos = pos
			}
		} else if next == -1 {
			next = i
		}
	}

	s.size = size
	if size == 0 {
		s.next = 0
		s.lastPos = -1
		return
	}

	if next != -1 {
		s.next = next
	} else {
		// Full ring: overwrite the oldest checkpoint next.
		s.next = minIdx
	}
	s.lastPos = maxPos
}

func (s *slotCheckpointStore) window() (size int, minPos, maxPos, lastPos int32) {
	minPos = int32(math.MaxInt32)
	maxPos = int32(-1)
	for i := range s.entries {
		pos := s.entries[i].pos
		if pos < 0 {
			continue
		}
		size++
		if pos < minPos {
			minPos = pos
		}
		if pos > maxPos {
			maxPos = pos
		}
	}
	if size == 0 {
		minPos = -1
		maxPos = -1
	}
	return size, minPos, maxPos, s.lastPos
}

func (c *Recurrent) checkpointTag() string {
	if c.logPrefix == "" {
		return "kvcache.recurrent"
	}
	return c.logPrefix
}

func (c *Recurrent) planCheckpoints(batch input.Batch) {
	if c.checkpointCount == 0 || len(c.curSeqs) == 0 {
		c.curCheckpointPos = c.curCheckpointPos[:0]
		c.curCheckpointTail = c.curCheckpointTail[:0]
		for k := range c.curCheckpointSlots {
			delete(c.curCheckpointSlots, k)
		}
		return
	}

	if cap(c.curCheckpointPos) < len(c.curSeqs) {
		c.curCheckpointPos = make([]int32, len(c.curSeqs))
	} else {
		c.curCheckpointPos = c.curCheckpointPos[:len(c.curSeqs)]
	}
	for i := range c.curCheckpointPos {
		c.curCheckpointPos[i] = -1
	}
	c.curCheckpointTail = slices.Grow(c.curCheckpointTail[:0], len(c.curSeqs))[:len(c.curSeqs)]
	for i := range c.curCheckpointTail {
		c.curCheckpointTail[i] = false
	}
	for k := range c.curCheckpointSlots {
		delete(c.curCheckpointSlots, k)
	}

	posMax := make(map[int]int32, len(c.curSeqs))
	for i, seq := range batch.Sequences {
		pos := batch.Positions[i]
		if cur, ok := posMax[seq]; !ok || pos > cur {
			posMax[seq] = pos
		}
	}

	for i, seq := range c.curSeqs {
		pos, ok := posMax[seq]
		if !ok {
			continue
		}
		tail := !c.reserveCheckpoints && c.checkpointNext[seq]
		if tail {
			delete(c.checkpointNext, seq)
		}
		if pos < c.checkpointMinPos {
			continue
		}
		slot := c.curSlots[i]
		store := c.checkpointStore(slot)
		lastPos := store.lastPos
		if tail || lastPos < 0 || pos-lastPos >= c.checkpointInterval {
			c.curCheckpointPos[i] = pos
			c.curCheckpointTail[i] = tail
		}
	}
}

func (c *Recurrent) CheckpointNext(seq int) {
	c.checkpointNext[seq] = true
}

func (c *Recurrent) checkpointStore(slot int) *slotCheckpointStore {
	store, ok := c.checkpoints[slot]
	if ok {
		return store
	}
	store = newSlotCheckpointStore(c.checkpointCount)
	c.checkpoints[slot] = store
	return store
}

func (c *Recurrent) checkpointIndexForSlot(slot int, pos int32, tail bool) int {
	if c.checkpointCount == 0 {
		return -1
	}
	if idx, ok := c.curCheckpointSlots[slot]; ok {
		return idx
	}
	store := c.checkpointStore(slot)
	var idx int
	if tail {
		idx = store.recordTail(pos)
	} else {
		idx = store.record(pos)
	}
	if idx >= 0 {
		c.curCheckpointSlots[slot] = idx
	}
	return idx
}

func (c *Recurrent) hasCheckpoint(seq int, pos int32) bool {
	if pos <= 0 {
		return false
	}
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return false
	}
	store, ok := c.checkpoints[slot]
	if !ok {
		return false
	}
	_, _, ok = store.bestIndex(pos)
	return ok
}

func (c *Recurrent) PrepareRestore(seq int, targetPos int32) (int32, bool) {
	if targetPos <= 0 {
		return 0, false
	}
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return 0, false
	}
	// The prompt extends everything the sequence has processed: resume on the
	// live state instead of re-reading from the checkpoint below it.
	if pos, ok := c.statePos[seq]; ok && pos == targetPos {
		c.pendingRestore[seq] = checkpointRestore{slot: slot, idx: -1, pos: targetPos - 1, live: true}
		slog.Debug(c.checkpointTag()+": live resume", "seq", seq, "slot", slot, "target", targetPos)
		return targetPos, true
	}
	store, ok := c.checkpoints[slot]
	if !ok {
		slog.Debug(c.checkpointTag()+": checkpoint miss", "seq", seq, "slot", slot, "target", targetPos, "size", 0)
		return 0, false
	}
	idx, pos, ok := store.bestIndex(targetPos)
	if !ok {
		size, minPos, maxPos, lastPos := store.window()
		slog.Debug(c.checkpointTag()+": checkpoint miss", "seq", seq, "slot", slot, "target", targetPos, "size", size,
			"min", minPos, "max", maxPos, "last", lastPos)
		return 0, false
	}
	c.pendingRestore[seq] = checkpointRestore{
		slot: slot,
		idx:  idx,
		pos:  pos,
	}
	slog.Debug(c.checkpointTag()+": checkpoint hit", "seq", seq, "slot", slot, "target", targetPos, "restore_pos", pos)
	return pos + 1, true
}

func (c *Recurrent) applyCheckpointRestore(restore checkpointRestore) error {
	if restore.live {
		return nil
	}
	entry, ok := c.restoreEntry(restore)
	if !ok {
		return ErrNotSupported
	}
	slog.Debug(c.checkpointTag()+": checkpoint restored", "slot", restore.slot, "pos", restore.pos,
		"conv_layers", len(entry.conv), "recurrent_layers", len(entry.recurrent))

	ctx := c.backend.NewContext()
	defer ctx.Close()

	slotIdx := ctx.Input().FromInts([]int32{int32(restore.slot)}, 1)
	for layer, src := range entry.conv {
		buf := c.convBuffer(layer)
		ctx.Forward(buf.SetRows(ctx, src, slotIdx))
	}
	for layer, src := range entry.recurrent {
		buf := c.recurrentBuffer(layer)
		ctx.Forward(buf.SetRows(ctx, src, slotIdx))
	}

	if len(entry.conv) > 0 || len(entry.recurrent) > 0 {
		ctx.Compute()
	}
	store := c.checkpoints[restore.slot]
	store.pruneAfter(restore.pos)
	return nil
}

func (c *Recurrent) restoreComplete(restore checkpointRestore) bool {
	if restore.live {
		return true
	}
	_, ok := c.restoreEntry(restore)
	return ok
}

func (c *Recurrent) restoreEntry(restore checkpointRestore) (*checkpointEntry, bool) {
	store, ok := c.checkpoints[restore.slot]
	if !ok || restore.idx < 0 || restore.idx >= len(store.entries) {
		return nil, false
	}
	entry := &store.entries[restore.idx]
	if entry.pos < 0 {
		return nil, false
	}
	if !c.entryComplete(entry) {
		return nil, false
	}
	return entry, true
}

func (c *Recurrent) entryComplete(entry *checkpointEntry) bool {
	for layer := range c.convStates {
		if entry.conv == nil || entry.conv[layer] == nil {
			return false
		}
	}
	for layer := range c.recurrentStates {
		if entry.recurrent == nil || entry.recurrent[layer] == nil {
			return false
		}
	}
	return true
}

func (c *Recurrent) clearCheckpoints(slot int) {
	if store, ok := c.checkpoints[slot]; ok {
		store.reset()
	}
}

func (c *Recurrent) shiftCheckpoints(slot int, beginIndex, endIndex int32) {
	if store, ok := c.checkpoints[slot]; ok {
		store.shiftRange(beginIndex, endIndex)
	}
}

func (c *Recurrent) copyCheckpoints(ctx ml.Context, srcSlot, dstSlot int) {
	if c.checkpointCount == 0 {
		return
	}
	srcStore, ok := c.checkpoints[srcSlot]
	if !ok || srcStore.size == 0 {
		return
	}
	dstStore := c.checkpointStore(dstSlot)
	dstStore.size = srcStore.size
	dstStore.next = srcStore.next
	dstStore.lastPos = srcStore.lastPos
	dstStore.tail = append(dstStore.tail[:0], srcStore.tail...)

	for i := range srcStore.entries {
		srcEntry := &srcStore.entries[i]
		dstEntry := &dstStore.entries[i]
		dstEntry.pos = srcEntry.pos
		if srcEntry.conv != nil {
			if dstEntry.conv == nil {
				dstEntry.conv = make(map[int]ml.Tensor)
			}
			for layer, src := range srcEntry.conv {
				dst := c.ensureCheckpointConv(layer, dstEntry)
				ctx.Forward(src.Copy(ctx, dst))
			}
		}
		if srcEntry.recurrent != nil {
			if dstEntry.recurrent == nil {
				dstEntry.recurrent = make(map[int]ml.Tensor)
			}
			for layer, src := range srcEntry.recurrent {
				dst := c.ensureCheckpointRecurrent(layer, dstEntry)
				ctx.Forward(src.Copy(ctx, dst))
			}
		}
	}
}

func (c *Recurrent) captureConvCheckpoint(ctx ml.Context, layer int, src ml.Tensor) {
	if c.checkpointCount == 0 {
		return
	}
	if c.reserveCheckpoints {
		c.reserveCheckpointConv(layer)
		return
	}
	if len(c.curCheckpointPos) == 0 {
		return
	}
	for i, pos := range c.curCheckpointPos {
		if pos < 0 {
			continue
		}
		slot := c.curSlots[i]
		idx := c.checkpointIndexForSlot(slot, pos, c.curCheckpointTail[i])
		if idx < 0 {
			continue
		}
		entry := &c.checkpoints[slot].entries[idx]
		dst := c.ensureCheckpointConv(layer, entry)
		seqSlice := src.Slice(ctx, 1, i, i+1, 1)
		ctx.Forward(seqSlice.Copy(ctx, dst))
	}
}

func (c *Recurrent) captureRecurrentCheckpoint(ctx ml.Context, layer int, src ml.Tensor) {
	if c.checkpointCount == 0 {
		return
	}
	if c.reserveCheckpoints {
		c.reserveCheckpointRecurrent(layer)
		return
	}
	if len(c.curCheckpointPos) == 0 {
		return
	}
	for i, pos := range c.curCheckpointPos {
		if pos < 0 {
			continue
		}
		slot := c.curSlots[i]
		idx := c.checkpointIndexForSlot(slot, pos, c.curCheckpointTail[i])
		if idx < 0 {
			continue
		}
		entry := &c.checkpoints[slot].entries[idx]
		dst := c.ensureCheckpointRecurrent(layer, entry)
		seqSlice := src.Slice(ctx, 1, i, i+1, 1)
		ctx.Forward(seqSlice.Copy(ctx, dst))
	}
}

func (c *Recurrent) ensureCheckpointConv(layer int, entry *checkpointEntry) ml.Tensor {
	if entry.conv == nil {
		entry.conv = make(map[int]ml.Tensor)
	}
	if t, ok := entry.conv[layer]; ok {
		return t
	}
	ctx, ok := c.checkpointConvCtxs[layer]
	if !ok {
		ctx = c.backend.NewContextSize(c.checkpointCtxSize).Layer(layer)
		c.checkpointConvCtxs[layer] = ctx
	}
	t := ctx.Zeros(ml.DTypeF32, c.convDim*c.convChannels, 1)
	entry.conv[layer] = t
	return t
}

func (c *Recurrent) ensureCheckpointRecurrent(layer int, entry *checkpointEntry) ml.Tensor {
	if entry.recurrent == nil {
		entry.recurrent = make(map[int]ml.Tensor)
	}
	if t, ok := entry.recurrent[layer]; ok {
		return t
	}
	ctx, ok := c.checkpointRecurCtxs[layer]
	if !ok {
		ctx = c.backend.NewContextSize(c.checkpointCtxSize).Layer(layer)
		c.checkpointRecurCtxs[layer] = ctx
	}
	t := ctx.Zeros(ml.DTypeF32, c.recurrentStateSize, 1)
	entry.recurrent[layer] = t
	return t
}

func (c *Recurrent) reserveCheckpointConv(layer int) {
	key := checkpointReserveKey(layer, 0)
	if _, ok := c.checkpointReserved[key]; ok {
		return
	}
	for slot := range c.maxSequences {
		store := c.checkpointStore(slot)
		for i := range store.entries {
			entry := &store.entries[i]
			_ = c.ensureCheckpointConv(layer, entry)
		}
	}
	c.checkpointReserved[key] = struct{}{}
}

func (c *Recurrent) reserveCheckpointRecurrent(layer int) {
	key := checkpointReserveKey(layer, 1)
	if _, ok := c.checkpointReserved[key]; ok {
		return
	}
	for slot := range c.maxSequences {
		store := c.checkpointStore(slot)
		for i := range store.entries {
			entry := &store.entries[i]
			_ = c.ensureCheckpointRecurrent(layer, entry)
		}
	}
	c.checkpointReserved[key] = struct{}{}
}

func checkpointReserveKey(layer int, kind int) int {
	return layer*2 + kind
}
