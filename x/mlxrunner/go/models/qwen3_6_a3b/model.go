package qwen3_6_a3b

import "fmt"

// Model bundles parsed config + loaded weights into one handle. The
// forward-pass methods (added in subsequent D.3 steps) hang off this
// type. Caller must Close to release the underlying shards.
type Model struct {
	Config  *Config
	Weights *Weights
}

// LoadModel reads config.json from modelDir, then loads every shard
// referenced by model.safetensors.index.json. Failure mid-way releases
// whatever already loaded — no leaks.
func LoadModel(modelDir string) (*Model, error) {
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		return nil, fmt.Errorf("LoadModel: %w", err)
	}
	w, err := LoadWeights(modelDir)
	if err != nil {
		return nil, fmt.Errorf("LoadModel: %w", err)
	}
	return &Model{Config: cfg, Weights: w}, nil
}

// Close releases the weights. Safe to call on a nil Model or one
// that's already been closed.
func (m *Model) Close() {
	if m == nil {
		return
	}
	m.Weights.Release()
	m.Weights = nil
}
