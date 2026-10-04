package model

// GenerationDefaults contains model-authored sampler defaults keyed by Ollama
// option names.
type GenerationDefaults map[string]any

type generationDefaultKind int

const (
	generationDefaultInt generationDefaultKind = iota
	generationDefaultFloat
)

type generationDefaultMapping struct {
	option  string
	ggufKey string
	kind    generationDefaultKind
}

var generationDefaultMappings = []generationDefaultMapping{
	{"top_k", "general.sampling.top_k", generationDefaultInt},
	{"top_p", "general.sampling.top_p", generationDefaultFloat},
	{"min_p", "general.sampling.min_p", generationDefaultFloat},
	{"typical_p", "general.sampling.typ_p", generationDefaultFloat},
	{"temperature", "general.sampling.temp", generationDefaultFloat},
	{"repeat_last_n", "general.sampling.penalty_last_n", generationDefaultInt},
	{"repeat_penalty", "general.sampling.penalty_repeat", generationDefaultFloat},
	{"presence_penalty", "general.sampling.penalty_present", generationDefaultFloat},
	{"frequency_penalty", "general.sampling.penalty_freq", generationDefaultFloat},
}

// ParseGGUFGenerationDefaults extracts sampler defaults from GGUF metadata.
func ParseGGUFGenerationDefaults(intValue func(string) (int64, bool), floatValue func(string) (float64, bool)) GenerationDefaults {
	defaults := GenerationDefaults{}
	for _, mapping := range generationDefaultMappings {
		switch mapping.kind {
		case generationDefaultInt:
			if value, ok := intValue(mapping.ggufKey); ok {
				defaults[mapping.option] = value
			}
		case generationDefaultFloat:
			if value, ok := floatValue(mapping.ggufKey); ok {
				defaults[mapping.option] = value
			}
		}
	}

	if len(defaults) == 0 {
		return nil
	}

	return defaults
}
