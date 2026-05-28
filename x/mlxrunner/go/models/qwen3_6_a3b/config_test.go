package qwen3_6_a3b

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleConfigJSON = `{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "tie_word_embeddings": true,
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
    "num_experts": 256,
    "num_experts_per_tok": 8
  }
}`

func TestLoadConfigFromDir(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(sampleConfigJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if got, want := c.ModelType, "qwen3_5_moe"; got != want {
		t.Errorf("ModelType = %q, want %q", got, want)
	}
	if got, want := c.TextConfig.NumHiddenLayers, 40; got != want {
		t.Errorf("NumHiddenLayers = %d, want %d", got, want)
	}
	if got, want := c.TextConfig.NumExperts, 256; got != want {
		t.Errorf("NumExperts = %d, want %d", got, want)
	}
	if got, want := c.Quantization.Bits, 4; got != want {
		t.Errorf("Quantization.Bits = %d, want %d", got, want)
	}
}

func TestLoadConfigMissing(t *testing.T) {
	if _, err := LoadConfig("/nonexistent/path/config.json"); err == nil {
		t.Error("expected error for missing file, got nil")
	}
}
