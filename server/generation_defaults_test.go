package server

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/types/model"
)

func TestGenerationDefaultsFromGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// The types ornith-1.5:35b stores: an integer top_k, float temp and top_p.
	if err := ggml.WriteGGUF(f, map[string]any{
		"general.architecture":   "qwen35moe",
		"general.sampling.top_k": uint32(20),
		"general.sampling.top_p": float32(0.5),
		"general.sampling.temp":  float32(1),
	}, nil); err != nil {
		t.Fatal(err)
	}
	f.Close()

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()

	got := generationDefaultsFromGGUF(gf)
	want := model.GenerationDefaults{"top_k": int64(20), "top_p": 0.5, "temperature": 1.0}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestModelOptionsLayering(t *testing.T) {
	m := &Model{
		GenerationDefaults: model.GenerationDefaults{"top_k": int64(20), "top_p": 0.5, "temperature": 1.0},
		Options:            map[string]any{"top_p": 0.75},
	}
	opts, err := modelOptions(m, map[string]any{"temperature": 0.0})
	if err != nil {
		t.Fatal(err)
	}
	// GGUF top_k, the params layer's top_p, the request's temperature.
	if opts.TopK != 20 || opts.TopP != 0.75 || opts.Temperature != 0 {
		t.Errorf("top_k=%d top_p=%v temperature=%v, want 20 0.75 0", opts.TopK, opts.TopP, opts.Temperature)
	}
}
