package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

// Excerpts of the GGUF chat templates around the assistant's think block.
const (
	ornith15TemplateExcerpt = `        {%- set reasoning_content = reasoning_content|trim %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}`
	qwen35TemplateExcerpt = `        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}`
	qwen36TemplateExcerpt = `        {%- set reasoning_content = reasoning_content|trim %}
        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}`
)

func TestRendererForChatTemplate(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     string
	}{
		{"ornith-1.5", ornith15TemplateExcerpt, "ornith"},
		{"ornith-1.5 reindented", "{%- set reasoning_content = reasoning_content|trim %}{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}", "ornith"},
		{"qwen3.5", qwen35TemplateExcerpt, ""},
		{"qwen3.6", qwen36TemplateExcerpt, ""},
		{"no template", "", ""},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			kv := map[string]any{"general.architecture": "qwen35"}
			if tt.template != "" {
				kv["tokenizer.chat_template"] = tt.template
			}
			path := filepath.Join(t.TempDir(), "model.gguf")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := ggml.WriteGGUF(f, kv, nil); err != nil {
				t.Fatal(err)
			}
			f.Close()

			if got := rendererForChatTemplate(path); got != tt.want {
				t.Errorf("rendererForChatTemplate() = %q, want %q", got, tt.want)
			}
		})
	}

	if got := rendererForChatTemplate(filepath.Join(t.TempDir(), "missing.gguf")); got != "" {
		t.Errorf("missing file: got %q, want empty", got)
	}
}
