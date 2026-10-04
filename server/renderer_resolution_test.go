package server

import "testing"

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
			if got := rendererForChatTemplate(tt.template); got != tt.want {
				t.Errorf("rendererForChatTemplate() = %q, want %q", got, tt.want)
			}
		})
	}
}
