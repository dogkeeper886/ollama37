package anthropic

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

// blockTypes fails the test unless the events open and close content blocks one
// at a time, at consecutive indexes, and close them all before message_delta.
// It returns the type of each block in order.
func blockTypes(t *testing.T, events []StreamEvent) []string {
	t.Helper()
	var types []string
	open := -1
	for _, e := range events {
		switch d := e.Data.(type) {
		case ContentBlockStartEvent:
			if open != -1 {
				t.Fatalf("block %d started while block %d is open", d.Index, open)
			}
			if d.Index != len(types) {
				t.Fatalf("block started at index %d, want %d", d.Index, len(types))
			}
			open = d.Index
			types = append(types, d.ContentBlock.Type)
		case ContentBlockDeltaEvent:
			if d.Index != open {
				t.Fatalf("delta for index %d, open block is %d", d.Index, open)
			}
		case ContentBlockStopEvent:
			if d.Index != open {
				t.Fatalf("stop for index %d, open block is %d", d.Index, open)
			}
			open = -1
		case MessageDeltaEvent:
			if open != -1 {
				t.Fatalf("message_delta while block %d is open", open)
			}
		}
	}
	return types
}

// TestStreamConverterBlockOrder covers #605: a thinking model that goes from
// thinking straight to a tool call must close the thinking block first.
func TestStreamConverterBlockOrder(t *testing.T) {
	thinking := func(s string) api.ChatResponse {
		return api.ChatResponse{Message: api.Message{Role: "assistant", Thinking: s}}
	}
	text := func(s string) api.ChatResponse {
		return api.ChatResponse{Message: api.Message{Role: "assistant", Content: s}}
	}
	toolCall := api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
		Function: api.ToolCallFunction{Name: "Bash", Arguments: api.ToolCallFunctionArguments{"command": "git pull"}},
	}}}}
	done := api.ChatResponse{Done: true, DoneReason: "stop"}
	toolCallDone := toolCall
	toolCallDone.Done, toolCallDone.DoneReason = true, "stop"

	cases := []struct {
		name   string
		chunks []api.ChatResponse
		want   []string
	}{
		{"thinking then tool call", []api.ChatResponse{thinking("Let me"), thinking(" run it."), toolCall, done}, []string{"thinking", "tool_use"}},
		{"thinking then tool call in the done chunk", []api.ChatResponse{thinking("Let me run it."), toolCallDone}, []string{"thinking", "tool_use"}},
		{"thinking, text, tool call", []api.ChatResponse{thinking("Let me run it."), text("Running."), toolCall, done}, []string{"thinking", "text", "tool_use"}},
		{"thinking then text", []api.ChatResponse{thinking("Easy."), text("Hi."), done}, []string{"thinking", "text"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewStreamConverter("msg_test", "test")
			var events []StreamEvent
			for _, chunk := range tc.chunks {
				events = append(events, c.Process(chunk)...)
			}
			if got := blockTypes(t, events); !slices.Equal(got, tc.want) {
				t.Fatalf("block types = %v, want %v", got, tc.want)
			}
		})
	}
}
