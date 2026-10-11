package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// A re-rendered conversation must come out byte for byte the same, or the
// runner's KV cache prefix stops matching and the whole history is prefilled
// again (#555). Tool definitions and tool-call arguments are Go maps.
func TestRenderStableOrder(t *testing.T) {
	props := map[string]api.ToolProperty{}
	args := api.ToolCallFunctionArguments{}
	for _, name := range []string{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot"} {
		props[name] = api.ToolProperty{
			Type:        api.PropertyType{"string"},
			Description: "the " + name,
			Enum:        []any{"x", "y"},
			Items:       map[string]any{"type": "string"},
			Required:    []string{"x"},
		}
		args[name] = name + " value"
	}
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{
		Name:        "lookup",
		Description: "Look something up",
		Parameters:  api.ToolFunctionParameters{Type: "object", Properties: props, Required: []string{"alpha"}},
	}}}
	msgs := []api.Message{
		{Role: "user", Content: "look it up"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "lookup", Arguments: args}}}},
		{Role: "tool", Content: "found"},
		{Role: "user", Content: "thanks"},
	}

	for _, name := range []string{"qwen3-coder", "qwen3.8", "ornith", "functiongemma"} {
		t.Run(name, func(t *testing.T) {
			first, err := RenderWithRenderer(name, msgs, tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range 100 {
				got, err := RenderWithRenderer(name, msgs, tools, nil)
				if err != nil {
					t.Fatal(err)
				}
				if got != first {
					t.Fatalf("render %d differs from the first:\n%s\n---\n%s", i+2, got, first)
				}
			}
		})
	}
}
