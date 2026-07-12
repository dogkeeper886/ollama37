package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// ornith reuses Qwen35Parser (thinking extraction + XML tool-call delegation to
// Qwen3CoderParser). These cover the always-thinking behaviour the fork's
// 2-arg Init produces for the "ornith" parser name.

func TestOrnithParserThinkingWithExplicitOpeningTag(t *testing.T) {
	parser := ParserForName("ornith")
	if parser == nil {
		t.Fatal("expected ornith parser")
	}

	parser.Init(nil, nil)
	content, thinking, calls, err := parser.Add("<think>\nLet me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestOrnithParserAssistantPrefillStartsInContent(t *testing.T) {
	parser := ParserForName("ornith")
	if parser == nil {
		t.Fatal("expected ornith parser")
	}

	last := &api.Message{Role: "assistant", Content: "Prefilled response start"}
	parser.Init(nil, last)

	content, thinking, calls, err := parser.Add(" and continued", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no thinking for assistant prefill continuation, got %q", thinking)
	}
	if content != " and continued" {
		t.Fatalf("expected content %q, got %q", " and continued", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestOrnithParserXMLToolCallEmittedInThinkingIsParsed(t *testing.T) {
	parser := ParserForName("ornith")
	if parser == nil {
		t.Fatal("expected ornith parser")
	}

	tools := []api.Tool{tool("get_weather", map[string]api.ToolProperty{
		"location": {Type: api.PropertyType{"string"}},
	})}

	parser.Init(tools, nil)
	input := "Need weather lookup<tool_call><function=get_weather><parameter=location>\nSF\n</parameter></function></tool_call>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}
	if loc := calls[0].Function.Arguments["location"]; loc != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", loc)
	}
}
