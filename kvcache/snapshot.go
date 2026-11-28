package kvcache

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
)

// Snapshot is one sequence's cache copied to host memory, so the runner can
// give its slot to another prompt and load it back later (#608).
type Snapshot interface {
	// Len is the number of positions the sequence held.
	Len() int32
	// Resumable reports whether a prompt that shares the first pos positions
	// can resume there once the snapshot is loaded.
	Resumable(pos int32) bool
	// Size is the host memory the snapshot takes, in bytes.
	Size() int64
}

// SnapshotCache optionally copies a sequence's cache to host memory and back.
type SnapshotCache interface {
	Snapshot(seq int) (Snapshot, error)
	// RestoreSnapshot replaces what seq holds with s.
	RestoreSnapshot(seq int, s Snapshot) error
}

var (
	_ SnapshotCache = (*Causal)(nil)
	_ SnapshotCache = (*Recurrent)(nil)
	_ SnapshotCache = (*WrapperCache)(nil)
)

// causalSnapshot holds positions [first, first+n) of every layer's keys and
// values, laid out as n consecutive cells.
type causalSnapshot struct {
	first  int32
	n      int
	window int32 // the sliding window, math.MaxInt32 without one
	k, v   map[int][]byte
	size   int64
}

func (s *causalSnapshot) Len() int32  { return s.first + int32(s.n) }
func (s *causalSnapshot) Size() int64 { return s.size }

// Resumable matches Causal.CanResume: the window ending at pos must lie in the
// positions held.
func (s *causalSnapshot) Resumable(pos int32) bool {
	return pos <= s.Len() && (s.window == math.MaxInt32 || max(0, pos-s.window) >= s.first)
}

// cellRun is a stretch of cells that hold consecutive positions of one sequence.
type cellRun struct {
	cell, off, n int
}

// seqRuns returns seq's cells as runs in position order, its first position and
// its position count. The positions must be consecutive.
func (c *Causal) seqRuns(seq int) (runs []cellRun, first int32, n int, err error) {
	var cells []int
	for i := range c.cells {
		if slices.Contains(c.cells[i].sequences, seq) {
			cells = append(cells, i)
		}
	}
	if len(cells) == 0 {
		return nil, 0, 0, ErrNotSupported
	}
	slices.SortFunc(cells, func(a, b int) int { return int(c.cells[a].pos - c.cells[b].pos) })
	first = c.cells[cells[0]].pos
	for i, cell := range cells {
		if c.cells[cell].pos != first+int32(i) {
			return nil, 0, 0, fmt.Errorf("%w: seq %d skips position %d", ErrNotSupported, seq, first+int32(i))
		}
		if i > 0 && cell == cells[i-1]+1 {
			runs[len(runs)-1].n++
		} else {
			runs = append(runs, cellRun{cell: cell, off: i, n: 1})
		}
	}
	return runs, first, len(cells), nil
}

// permutedVOK reports whether a permuted value tensor can be cut by cells: a
// quantized type packs several cells into one block.
func permutedVOK(dtype ml.DType) bool {
	return dtype == ml.DTypeF16 || dtype == ml.DTypeF32
}

func (c *Causal) Snapshot(seq int) (Snapshot, error) {
	runs, first, n, err := c.seqRuns(seq)
	if err != nil {
		return nil, err
	}
	s := &causalSnapshot{first: first, n: n, window: c.swaWindowSize, k: make(map[int][]byte), v: make(map[int][]byte)}

	for layer, key := range c.keys {
		value := c.values[layer]
		if key == nil || value == nil {
			continue
		}
		if c.config.PermutedV && !permutedVOK(value.DType()) {
			return nil, fmt.Errorf("%w: permuted %v values", ErrNotSupported, value.DType())
		}

		ctx := c.backend.NewContext()
		kHeadDim, numKVHeads := key.Dim(0), key.Dim(1)
		kRow := key.Stride(2)
		hostK := ctx.Input().Empty(key.DType(), kHeadDim*numKVHeads*n)
		var hostV ml.Tensor
		if c.config.PermutedV {
			vHeadDim, elem := value.Dim(1), value.Stride(0)
			hostV = ctx.Input().Empty(value.DType(), n, vHeadDim*numKVHeads)
			for _, r := range runs {
				ctx.Forward(
					key.View(ctx, kRow*r.cell, kHeadDim*numKVHeads*r.n).Copy(ctx, hostK.View(ctx, kRow*r.off, kHeadDim*numKVHeads*r.n)),
					value.View(ctx, elem*r.cell, r.n, len(c.cells)*elem, vHeadDim*numKVHeads).
						Copy(ctx, hostV.View(ctx, elem*r.off, r.n, n*elem, vHeadDim*numKVHeads)),
				)
			}
		} else {
			vHeadDim, vRow := value.Dim(0), value.Stride(2)
			hostV = ctx.Input().Empty(value.DType(), vHeadDim*numKVHeads*n)
			for _, r := range runs {
				ctx.Forward(
					key.View(ctx, kRow*r.cell, kHeadDim*numKVHeads*r.n).Copy(ctx, hostK.View(ctx, kRow*r.off, kHeadDim*numKVHeads*r.n)),
					value.View(ctx, vRow*r.cell, vHeadDim*numKVHeads*r.n).Copy(ctx, hostV.View(ctx, vRow*r.off, vHeadDim*numKVHeads*r.n)),
				)
			}
		}
		ctx.Compute(hostK, hostV)
		s.k[layer], s.v[layer] = hostK.Bytes(), hostV.Bytes()
		ctx.Close()
		s.size += int64(len(s.k[layer]) + len(s.v[layer]))
	}
	return s, nil
}

// freeBlock returns the first cell of n free consecutive cells.
func (c *Causal) freeBlock(n int) (int, bool) {
	var start, count int
	for i := range c.cells {
		if len(c.cells[i].sequences) != 0 {
			start, count = i+1, 0
			continue
		}
		if count++; count >= n {
			return start, true
		}
	}
	return 0, false
}

func (c *Causal) RestoreSnapshot(seq int, snap Snapshot) error {
	s, ok := snap.(*causalSnapshot)
	if !ok {
		return fmt.Errorf("%w: snapshot %T", ErrNotSupported, snap)
	}
	if err := c.Remove(seq, 0, math.MaxInt32); err != nil {
		return err
	}
	loc, ok := c.freeBlock(s.n)
	if !ok {
		c.defrag()
		if loc, ok = c.freeBlock(s.n); !ok {
			return fmt.Errorf("%w: no %d free cells", ErrKvCacheFull, s.n)
		}
	}

	for layer, k := range s.k {
		key, value := c.keys[layer], c.values[layer]
		if key == nil || value == nil {
			return fmt.Errorf("%w: layer %d has no cache", ErrNotSupported, layer)
		}
		ctx := c.backend.NewContext()
		kHeadDim, numKVHeads := key.Dim(0), key.Dim(1)
		ctx.Forward(ctx.Input().FromBytes(key.DType(), k, kHeadDim*numKVHeads*s.n).
			Copy(ctx, key.View(ctx, key.Stride(2)*loc, kHeadDim*numKVHeads*s.n)))
		if c.config.PermutedV {
			vHeadDim, elem := value.Dim(1), value.Stride(0)
			ctx.Forward(ctx.Input().FromBytes(value.DType(), s.v[layer], s.n, vHeadDim*numKVHeads).
				Copy(ctx, value.View(ctx, elem*loc, s.n, len(c.cells)*elem, vHeadDim*numKVHeads)))
		} else {
			vHeadDim := value.Dim(0)
			ctx.Forward(ctx.Input().FromBytes(value.DType(), s.v[layer], vHeadDim*numKVHeads*s.n).
				Copy(ctx, value.View(ctx, value.Stride(2)*loc, vHeadDim*numKVHeads*s.n)))
		}
		ctx.Compute()
		ctx.Close()
	}

	for i := range s.n {
		c.cells[loc+i] = cacheCell{pos: s.first + int32(i), sequences: []int{seq}}
	}
	c.cellRanges[seq] = cellRange{min: loc, max: loc + s.n - 1}
	return nil
}

// recurrentSnapshot adds each layer's conv and recurrent state to the
// attention layers' snapshot.
type recurrentSnapshot struct {
	kv              *causalSnapshot
	conv, recurrent map[int][]byte
	size            int64
}

func (s *recurrentSnapshot) Len() int32  { return s.kv.Len() }
func (s *recurrentSnapshot) Size() int64 { return s.size }

// Resumable: the state holds only its last position.
func (s *recurrentSnapshot) Resumable(pos int32) bool { return pos == s.Len() }

// stateRow is slot's row of a conv or recurrent state buffer.
func stateRow(ctx ml.Context, buf ml.Tensor, slot int) ml.Tensor {
	return buf.View(ctx, buf.Stride(1)*slot, buf.Dim(0))
}

func (c *Recurrent) Snapshot(seq int) (Snapshot, error) {
	slot, ok := c.slotForSeq[seq]
	if !ok || !c.validSlot(slot) {
		return nil, fmt.Errorf("%w: seq %d has no state", ErrNotSupported, seq)
	}
	kv, err := c.kv.Snapshot(seq)
	if err != nil {
		return nil, err
	}
	// The state must sit where the attention layers end, or it would resume
	// on positions they lack.
	if kvs := kv.(*causalSnapshot); kvs.first != 0 || c.statePos[seq] != kvs.Len() {
		return nil, fmt.Errorf("%w: state at %d, attention cache at [%d, %d)", ErrNotSupported, c.statePos[seq], kvs.first, kvs.Len())
	}
	s := &recurrentSnapshot{kv: kv.(*causalSnapshot), conv: make(map[int][]byte), recurrent: make(map[int][]byte), size: kv.Size()}

	ctx := c.backend.NewContext()
	defer ctx.Close()
	conv := make(map[int]ml.Tensor, len(c.convStates))
	recurrent := make(map[int]ml.Tensor, len(c.recurrentStates))
	var outs []ml.Tensor
	for layer, buf := range c.convStates {
		conv[layer] = ctx.Input().Empty(buf.DType(), buf.Dim(0))
		ctx.Forward(stateRow(ctx, buf, slot).Copy(ctx, conv[layer]))
		outs = append(outs, conv[layer])
	}
	for layer, buf := range c.recurrentStates {
		recurrent[layer] = ctx.Input().Empty(buf.DType(), buf.Dim(0))
		ctx.Forward(stateRow(ctx, buf, slot).Copy(ctx, recurrent[layer]))
		outs = append(outs, recurrent[layer])
	}
	ctx.Compute(outs...)
	for layer, t := range conv {
		s.conv[layer] = t.Bytes()
		s.size += int64(len(s.conv[layer]))
	}
	for layer, t := range recurrent {
		s.recurrent[layer] = t.Bytes()
		s.size += int64(len(s.recurrent[layer]))
	}
	return s, nil
}

func (c *Recurrent) RestoreSnapshot(seq int, snap Snapshot) error {
	s, ok := snap.(*recurrentSnapshot)
	if !ok {
		return fmt.Errorf("%w: snapshot %T", ErrNotSupported, snap)
	}
	if err := c.Remove(seq, 0, math.MaxInt32); err != nil {
		return err
	}
	if err := c.kv.RestoreSnapshot(seq, s.kv); err != nil {
		return err
	}
	slot, err := c.allocSlot()
	if err != nil {
		_ = c.kv.Remove(seq, 0, math.MaxInt32)
		return err
	}
	c.slotForSeq[seq] = slot
	c.refCount[slot] = 1
	c.clearCheckpoints(slot)

	ctx := c.backend.NewContext()
	defer ctx.Close()
	for layer, data := range s.conv {
		buf := c.convBuffer(layer)
		ctx.Forward(ctx.Input().FromBytes(buf.DType(), data, buf.Dim(0)).Copy(ctx, stateRow(ctx, buf, slot)))
	}
	for layer, data := range s.recurrent {
		buf := c.recurrentBuffer(layer)
		ctx.Forward(ctx.Input().FromBytes(buf.DType(), data, buf.Dim(0)).Copy(ctx, stateRow(ctx, buf, slot)))
	}
	ctx.Compute()
	c.statePos[seq] = s.Len()
	return nil
}

// wrapperSnapshot holds one snapshot per wrapped cache.
type wrapperSnapshot struct {
	parts []Snapshot
}

func (s *wrapperSnapshot) Len() int32 {
	n := int32(math.MaxInt32)
	for _, p := range s.parts {
		n = min(n, p.Len())
	}
	return n
}

func (s *wrapperSnapshot) Resumable(pos int32) bool {
	for _, p := range s.parts {
		if !p.Resumable(pos) {
			return false
		}
	}
	return true
}

func (s *wrapperSnapshot) Size() int64 {
	var size int64
	for _, p := range s.parts {
		size += p.Size()
	}
	return size
}

func (c *WrapperCache) Snapshot(seq int) (Snapshot, error) {
	s := &wrapperSnapshot{}
	for _, cache := range c.caches {
		sc, ok := cache.(SnapshotCache)
		if !ok {
			return nil, fmt.Errorf("%w: %T", ErrNotSupported, cache)
		}
		p, err := sc.Snapshot(seq)
		if err != nil {
			return nil, err
		}
		s.parts = append(s.parts, p)
	}
	return s, nil
}

func (c *WrapperCache) RestoreSnapshot(seq int, snap Snapshot) error {
	s, ok := snap.(*wrapperSnapshot)
	if !ok || len(s.parts) != len(c.caches) {
		return fmt.Errorf("%w: snapshot %T", ErrNotSupported, snap)
	}
	for i, cache := range c.caches {
		if err := cache.(SnapshotCache).RestoreSnapshot(seq, s.parts[i]); err != nil {
			for _, done := range c.caches[:i+1] {
				_ = done.Remove(seq, 0, math.MaxInt32)
			}
			return err
		}
	}
	return nil
}
