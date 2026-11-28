package gemma4

import (
	"fmt"
	"os"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/model"
)

// Gemma 4 ends a tool call with <|tool_response>, so it stops generation
// alongside <turn|> (#600).
func TestToolResponseStopsGeneration(t *testing.T) {
	tokens := make([]string, 107)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("<unused%d>", i)
	}
	tokens[1], tokens[2] = "<eos>", "<bos>"
	tokens[49], tokens[50] = "<tool_call|>", "<|tool_response>"
	tokens[106] = "<turn|>"

	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := fsggml.WriteGGUF(f, fsggml.KV{
		"general.architecture":        "gemma4",
		"tokenizer.ggml.tokens":       tokens,
		"tokenizer.ggml.scores":       make([]float32, len(tokens)),
		"tokenizer.ggml.token_type":   make([]int32, len(tokens)),
		"tokenizer.ggml.bos_token_id": uint32(2),
		"tokenizer.ggml.eos_token_id": uint32(106),
	}, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	meta, err := fsggml.Decode(f, -1)
	if err != nil {
		t.Fatal(err)
	}

	m, err := New(meta.KV())
	if err != nil {
		t.Fatal(err)
	}
	tp := m.(model.TextProcessor)

	for _, id := range []int32{50, 106} {
		if !tp.Is(id, model.SpecialEOS) {
			t.Errorf("token %d (%s) should stop generation", id, tokens[id])
		}
	}
	if tp.Is(49, model.SpecialEOS) {
		t.Errorf("token 49 (%s) should not stop generation", tokens[49])
	}
}
