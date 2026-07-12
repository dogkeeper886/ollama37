package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestOrnithRendererMatchesAssistantHistoryThinkBlocks(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{Role: "assistant", Content: "Hello."},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith", msgs, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>

</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>
`
	if got != want {
		t.Fatalf("unexpected Ornith render\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

// ornith forces thinking on even when the request sets think:false, because the
// fork's parser cannot be told think:false and always collects a <think> span.
// So a think:false request renders identically to a thinking request (an open
// <think> prefill), never a closed empty block — keeping renderer and parser in sync.
func TestOrnithRendererForcesThinkingWhenDisabled(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "Say hello."},
		{
			Role:     "assistant",
			Thinking: "Keep it short.",
			Content:  "Hello.",
		},
		{Role: "user", Content: "Now say bye."},
	}

	got, err := RenderWithRenderer("ornith", msgs, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Say hello.<|im_end|>
<|im_start|>assistant
<think>
Keep it short.
</think>

Hello.<|im_end|>
<|im_start|>user
Now say bye.<|im_end|>
<|im_start|>assistant
<think>
`
	if got != want {
		t.Fatalf("unexpected Ornith render with think:false\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}
