package renderers

import "github.com/ollama/ollama/api"

type OrnithRenderer struct {
	Qwen35Renderer
}

func newOrnithRenderer() Renderer {
	return &OrnithRenderer{
		Qwen35Renderer: Qwen35Renderer{
			isThinking:                      true,
			alwaysRenderAssistantThinkBlock: true,
			useImgTags:                      RenderImgTags,
		},
	}
}

// Render forces thinking on. ornith is a thinking model, and the fork's Parser
// interface cannot be told think:false per request — Qwen35Parser always collects
// a <think>…</think> span. Keeping the renderer always-thinking matches the parser
// (and the fork's other thinking models, e.g. qwen3.5); otherwise a think:false
// request would emit a closed empty <think></think> in the prompt and the model's
// answer would be misrouted to the thinking channel, leaving content empty.
func (r *OrnithRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	return r.Qwen35Renderer.Render(messages, tools, &api.ThinkValue{Value: true})
}
