package qwen3_6_a3b

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

const sampleIndexJSON = `{
  "metadata": {"total_size": 12345},
  "weight_map": {
    "language_model.model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "language_model.model.embed_tokens.scales": "model-00001-of-00004.safetensors",
    "language_model.lm_head.weight":            "model-00004-of-00004.safetensors",
    "language_model.model.layers.7.mlp.gate":   "model-00002-of-00004.safetensors",
    "language_model.model.layers.7.mlp.up":     "model-00002-of-00004.safetensors",
    "language_model.model.layers.13.norm.w":    "model-00003-of-00004.safetensors"
  }
}`

func TestLoadIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"),
		[]byte(sampleIndexJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	idx, err := LoadIndex(dir)
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	if got, want := len(idx.WeightMap), 6; got != want {
		t.Errorf("WeightMap len = %d, want %d", got, want)
	}
	want := []string{
		"model-00001-of-00004.safetensors",
		"model-00002-of-00004.safetensors",
		"model-00003-of-00004.safetensors",
		"model-00004-of-00004.safetensors",
	}
	got := idx.Shards()
	sort.Strings(got)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Shards = %v, want %v", got, want)
	}
}

func TestLoadIndexEmptyMap(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"),
		[]byte(`{"metadata": {}, "weight_map": {}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadIndex(dir); err == nil {
		t.Error("expected error for empty weight_map, got nil")
	}
}

func TestLoadIndexMissing(t *testing.T) {
	if _, err := LoadIndex("/nonexistent/path"); err == nil {
		t.Error("expected error for missing file, got nil")
	}
}
