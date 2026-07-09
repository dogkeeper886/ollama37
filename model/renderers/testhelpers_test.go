package renderers

import "github.com/ollama/ollama/api"

// testArgs creates ToolCallFunctionArguments from a map.
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.ToolCallFunctionArguments{}
	for k, v := range m {
		args[k] = v
	}
	return args
}
