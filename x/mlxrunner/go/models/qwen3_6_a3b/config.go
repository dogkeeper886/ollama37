// Package qwen3_6_a3b assembles the qwen3.6-35b-a3b-4bit model (the
// MLX-quantized A3B / MoE+DeltaNet flavor) on top of the mlx_cabi
// cgo surface. Phase D.3 (#189).
//
// The model is multimodal in its config (vision_config present), but
// the runner here only consumes the language portion under
// text_config — the LLM forward pass is what the K80 path needs.
package qwen3_6_a3b

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config mirrors config.json with only the fields the runner actually
// reads. The full file has vision_config, transformers_version, per-
// tensor quantization overrides, and assorted special-token IDs — add
// fields here when the forward pass needs them, not before.
type Config struct {
	Architectures     []string     `json:"architectures"`
	ModelType         string       `json:"model_type"`
	TieWordEmbeddings bool         `json:"tie_word_embeddings"`
	Quantization      Quantization `json:"quantization"`
	TextConfig        TextConfig   `json:"text_config"`
}

// TextConfig is the language-model subtree under config.text_config.
// Top-level config holds vision + multimodal wiring; the LLM lives here.
type TextConfig struct {
	ModelType         string `json:"model_type"`
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	HiddenSize        int    `json:"hidden_size"`
	NumAttentionHeads int    `json:"num_attention_heads"`
	NumKeyValueHeads  int    `json:"num_key_value_heads"`
	VocabSize         int    `json:"vocab_size"`
	NumExperts        int    `json:"num_experts"`
	NumExpertsPerTok  int    `json:"num_experts_per_tok"`
}

// Quantization captures the file-wide default quant params. The full
// config also has per-tensor overrides (e.g. the 8-bit MoE router gate
// at language_model.model.layers.N.mlp.gate); those are read on demand
// by the weight loader, not parsed here.
type Quantization struct {
	GroupSize int    `json:"group_size"`
	Bits      int    `json:"bits"`
	Mode      string `json:"mode"`
}

// LoadConfig parses config.json from either a model directory or a
// direct file path. Passing the model directory is the common case;
// the file-path form is for tests with custom fixtures.
func LoadConfig(path string) (*Config, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "config.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &c, nil
}
