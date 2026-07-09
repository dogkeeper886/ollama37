package parsers

import (
	"encoding/json"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// argsComparer compares ToolCallFunctionArguments by logical equality (same keys
// with same values), not by order.
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	if len(a) != len(b) {
		return false
	}
	for k, av := range a {
		bv, ok := b[k]
		if !ok {
			return false
		}
		aJSON, _ := json.Marshal(av)
		bJSON, _ := json.Marshal(bv)
		if string(aJSON) != string(bJSON) {
			return false
		}
	}
	return true
})

// toolsComparer compares tools/tool-calls using argsComparer for their arguments.
var toolsComparer = cmp.Options{argsComparer}

// testArgs creates ToolCallFunctionArguments from a map.
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.ToolCallFunctionArguments{}
	for k, v := range m {
		args[k] = v
	}
	return args
}
