package qwen3_6_a3b

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/dogkeeper886/ollama37/x/mlxrunner/mlx"
)

// WeightIndex is the parsed model.safetensors.index.json: tensor name
// -> shard filename. The `metadata` block is preserved opaquely so
// future code can poke at fields like total_size without re-reading
// the file.
type WeightIndex struct {
	Metadata  map[string]any    `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`
}

// LoadIndex parses model.safetensors.index.json from the model dir.
// Accepts either the dir itself or a direct path to the index file.
func LoadIndex(path string) (*WeightIndex, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "model.safetensors.index.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read index: %w", err)
	}
	var idx WeightIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("parse index: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("index has empty weight_map")
	}
	return &idx, nil
}

// Shards returns the unique shard filenames referenced by the index,
// sorted lexicographically.
func (idx *WeightIndex) Shards() []string {
	seen := map[string]struct{}{}
	for _, s := range idx.WeightMap {
		seen[s] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// Weights ties the index to the actually-loaded shards. Get routes
// name lookups through the index to the right shard. Release frees
// every loaded shard.
type Weights struct {
	dir    string
	index  *WeightIndex
	shards map[string]*mlx.SafeTensors // keyed by shard filename
}

// LoadWeights parses index.json and loads every shard it references.
// Failure mid-way releases anything already loaded — leak-free.
func LoadWeights(modelDir string) (*Weights, error) {
	idx, err := LoadIndex(modelDir)
	if err != nil {
		return nil, err
	}
	shards := map[string]*mlx.SafeTensors{}
	for _, name := range idx.Shards() {
		st, err := mlx.LoadSafetensors(filepath.Join(modelDir, name))
		if err != nil {
			for _, loaded := range shards {
				loaded.Release()
			}
			return nil, fmt.Errorf("load shard %s: %w", name, err)
		}
		shards[name] = st
	}
	return &Weights{dir: modelDir, index: idx, shards: shards}, nil
}

// Get fetches a tensor by name via the index. Returns nil if the name
// isn't in the index or the resolved shard's Get returns nil (which
// shouldn't happen if the file is consistent with its index, but we
// don't crash on it). Caller owns the returned Array.
func (w *Weights) Get(name string) *mlx.Array {
	if w == nil {
		return nil
	}
	shardName, ok := w.index.WeightMap[name]
	if !ok {
		return nil
	}
	st, ok := w.shards[shardName]
	if !ok {
		return nil
	}
	return st.Get(name)
}

// Count returns the total tensor count across all shards.
func (w *Weights) Count() int {
	if w == nil {
		return 0
	}
	return len(w.index.WeightMap)
}

// NumShards returns how many distinct shard files were loaded.
func (w *Weights) NumShards() int {
	if w == nil {
		return 0
	}
	return len(w.shards)
}

// Names returns all tensor names, sorted. Useful for diagnostics; if
// you only need a few names, prefer Get directly.
func (w *Weights) Names() []string {
	if w == nil {
		return nil
	}
	names := make([]string, 0, len(w.index.WeightMap))
	for n := range w.index.WeightMap {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ShardOf returns which shard file contains `name`. Returns "" if not
// in the index.
func (w *Weights) ShardOf(name string) string {
	if w == nil {
		return ""
	}
	return w.index.WeightMap[name]
}

// Release frees every loaded shard. Safe to call on nil or an
// already-released Weights.
func (w *Weights) Release() {
	if w == nil {
		return
	}
	for _, st := range w.shards {
		st.Release()
	}
	w.shards = nil
}
