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

// Excerpts of the GGUF chat templates' generation prompts.
const (
	lfm25TemplateExcerpt = `{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n<think>" -}}
{%- endif -%}`
	qwen35GenerationPromptExcerpt = `{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}`
)

func TestRendererForChatTemplate(t *testing.T) {
	cases := []struct {
		name         string
		template     string
		wantRenderer string
		wantParser   string
	}{
		{"ornith-1.5", ornith15TemplateExcerpt, "ornith", "ornith"},
		{"ornith-1.5 reindented", "{%- set reasoning_content = reasoning_content|trim %}{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}", "ornith", "ornith"},
		{"lfm2.5-24b", lfm25TemplateExcerpt, "lfm2", "lfm2-thinking"},
		{"qwen3.5", qwen35TemplateExcerpt, "", ""},
		{"qwen3.5 generation prompt", qwen35GenerationPromptExcerpt, "", ""},
		{"qwen3.6", qwen36TemplateExcerpt, "", ""},
		{"no template", "", "", ""},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			renderer, parser := rendererForChatTemplate(tt.template)
			if renderer != tt.wantRenderer || parser != tt.wantParser {
				t.Errorf("rendererForChatTemplate() = %q, %q, want %q, %q", renderer, parser, tt.wantRenderer, tt.wantParser)
			}
		})
	}
}
