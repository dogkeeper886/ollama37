package renderers

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

func jsonString(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

type FunctionGemmaRenderer struct{}

func (r *FunctionGemmaRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// Collect system messages
	var system string
	for _, m := range messages {
		if m.Role == "system" {
			if system == "" {
				system = m.Content
			} else {
				system = system + "\n\n" + m.Content
			}
		}
	}

	// Developer turn: system message + tool declarations
	if len(tools) > 0 || system != "" {
		sb.WriteString("<start_of_turn>developer\n")
		if system != "" {
			sb.WriteString(system)
		}
		for _, tool := range tools {
			sb.WriteString("<start_function_declaration>declaration:")
			sb.WriteString(tool.Function.Name)
			sb.WriteString("{description:<escape>")
			sb.WriteString(tool.Function.Description)
			sb.WriteString("<escape>,parameters:{properties:{")

			first := true
			for _, name := range slices.Sorted(maps.Keys(tool.Function.Parameters.Properties)) {
				prop := tool.Function.Parameters.Properties[name]
				if !first {
					sb.WriteString(",")
				}
				first = false
				sb.WriteString(name)
				sb.WriteString(":{description:<escape>")
				sb.WriteString(prop.Description)
				sb.WriteString("<escape>,type:<escape>")
				if len(prop.Type) > 0 {
					sb.WriteString(strings.ToUpper(prop.Type[0]))
				}
				sb.WriteString("<escape>")
				if len(prop.Enum) > 0 {
					sb.WriteString(",enum:[")
					for i, e := range prop.Enum {
						if i > 0 {
							sb.WriteString(",")
						}
						sb.WriteString("<escape>")
						sb.WriteString(fmt.Sprintf("%v", e))
						sb.WriteString("<escape>")
					}
					sb.WriteString("]")
				}
				sb.WriteString("}")
			}
			sb.WriteString("},required:[")
			for i, req := range tool.Function.Parameters.Required {
				if i > 0 {
					sb.WriteString(",")
				}
				sb.WriteString("<escape>")
				sb.WriteString(req)
				sb.WriteString("<escape>")
			}
			sb.WriteString("],type:<escape>OBJECT<escape>}}<end_function_declaration>")
		}
		sb.WriteString("\n<end_of_turn>\n")
	}

	// Render messages
	for _, m := range messages {
		switch m.Role {
		case "system":
			continue
		case "user":
			sb.WriteString("<start_of_turn>user\n")
			sb.WriteString(m.Content)
			sb.WriteString("<end_of_turn>\n")
		case "assistant":
			sb.WriteString("<start_of_turn>model\n")
			if len(m.ToolCalls) > 0 {
				for _, tc := range m.ToolCalls {
					sb.WriteString("<start_function_call>call:")
					sb.WriteString(tc.Function.Name)
					sb.WriteString(jsonString(tc.Function.Arguments))
					sb.WriteString("<end_function_call>")
				}
			} else {
				sb.WriteString(m.Content)
			}
			sb.WriteString("<end_of_turn>\n")
		case "tool":
			sb.WriteString("<start_function_response>response:")
			sb.WriteString(m.ToolName)
			sb.WriteString(jsonString(m.Content))
			sb.WriteString("<end_function_response>")
		}
	}

	sb.WriteString("<start_of_turn>model\n")
	return sb.String(), nil
}
