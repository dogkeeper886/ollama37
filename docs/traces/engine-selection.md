# Trace: Ollama engine selection (new vs llama.cpp)

**Date**: 2026-04-27
**Trigger**: Phase 1 audit doc (#122) for the FA-coverage cascade asserted "all 13 tested models use the new Ollama engine." Phase 2 (#123) empirical run on `deepseek-r1:14b` produced load-request log lines from `runner/llamarunner/` (the legacy engine), not `runner/ollamarunner/` (the new engine), contradicting the assertion. The error was assuming "package exists in `model/models/<arch>/`" implies "this arch uses the new engine at runtime." It does not.

## TL;DR

Two runners exist:

| Runner | Path | Identifier in logs |
|---|---|---|
| Legacy llama.cpp | `runner/llamarunner/runner.go` | `source=runner.go:950` (the `slog.Info("load", "request", req)` line) |
| New Ollama | `runner/ollamarunner/runner.go` | `source=runner.go:1264` (same log line, different file) |

Engine selection lives at `llm/server.go:153`:

```go
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    if len(projectors) == 0 {
        textProcessor, err = model.NewTextProcessor(modelPath)
    } else {
        err = errors.New("split vision models aren't supported")
    }
    if err != nil {
        slog.Debug("engine selected", "engine", "llama_compat", "arch", arch, "reason", err)
    } else {
        slog.Debug("engine selected", "engine", "ollama", "arch", arch)
    }
}
if textProcessor == nil {
    llamaModel, err = llama.LoadModelFromFile(modelPath, llama.ModelParams{VocabOnly: true})
    ...
}
```

Two conditions trigger the new-engine attempt: env var or per-arch flag.

## Decision flow

```
NewLlamaServer(...)
  │
  ├── if envconfig.NewEngine() || f.KV().OllamaEngineRequired():
  │   │
  │   ├── envconfig.NewEngine()           # OLLAMA_NEW_ENGINE env var (default false)
  │   ├── f.KV().OllamaEngineRequired()   # per-arch GGUF flag, see fs/ggml/ggml.go:241
  │   │   │
  │   │   └── slices.Contains([gemma3, gemma3n, gemma4, gptoss, gpt-oss,
  │   │                       llama4, mistral3, mllama, qwen25vl, qwen3,
  │   │                       qwen3moe, qwen3vl, qwen3vlmoe, qwen35], arch)
  │   │
  │   ├── try model.NewTextProcessor(modelPath)        # construct new-engine wrapper
  │   │   ├── on success → textProcessor != nil → new engine
  │   │   └── on error   → textProcessor still nil → fall through (e.g., split vision)
  │   │
  │   └── log "engine selected" debug line
  │
  └── if textProcessor == nil:                         # not in either condition above
      └── llama.LoadModelFromFile(...)                 # legacy llama.cpp engine
```

## Critical implication

**A `model/models/<arch>/` Go package can exist without that arch using the new engine at runtime.** Examples in our test lineup:

| Arch | Has `model/models/<arch>/` package? | In `OllamaEngineRequired` list? | Default engine (no env var) |
|---|---|---|---|
| gemma3 | yes | yes | new |
| qwen35 | yes | yes | new |
| qwen3vl / qwen3vlmoe | yes | yes | new |
| gptoss / gpt-oss | yes | yes | new |
| **qwen2** | yes (handles `qwen2`, `qwen3moe`-not-actually) | **NO** | **llama.cpp** |
| **llama** | yes | **NO** | **llama.cpp** |
| **deepseek2** | yes | **NO** | **llama.cpp** |

The audit error in Phase 1 was assuming the first column implied the third. It does not. `qwen2` (and friends) have new-engine packages but only use them when `OLLAMA_NEW_ENGINE=true` is explicitly set — which our CI runs do not.

## How to verify the engine for a running model

In container logs, look for either of these log paths:

- `level=DEBUG source=server.go:160 msg="engine selected" engine=ollama arch=...`
- `level=DEBUG source=server.go:160 msg="engine selected" engine=llama_compat arch=... reason=...`

Or distinguish by the load-request log:

- `source=runner.go:1264` (under `runner/ollamarunner/runner.go`) → new engine
- `source=runner.go:950` (under `runner/llamarunner/runner.go`) → legacy

## Why this matters for FA gating

The FA gate at `llm/server.go:251` calls `f.SupportsFlashAttention()` (legacy) and now also `f.SupportsFlashAttentionInNewEngine()` (post-#124). The deny list inside `SupportsFlashAttention` (`gemma2`, `qwen35`) was originally meant to redirect those arches to the new engine's "FA allowlist instead." That allowlist was aspirational until #124 implemented it.

For models on the legacy engine path (qwen2, llama, deepseek2 — when `OLLAMA_NEW_ENGINE=false`), the existing `SupportsFlashAttention` head-count check applies. They typically pass without changes.

For models on the new engine path (gemma3, gptoss, qwen35, qwen3vl, etc.), the new method adds an explicit allowlist that can bypass the deny list after empirical validation. See `docs/research/k80-fa-model-coverage.md` for the per-arch matrix.

## References

- Engine selection: `llm/server.go:147-180` (`NewLlamaServer`)
- `OllamaEngineRequired`: `fs/ggml/ggml.go:241-255`
- `envconfig.NewEngine`: `envconfig/config.go` (look for `OLLAMA_NEW_ENGINE`)
- New-engine model packages: `model/models/<arch>/`
- Legacy runner: `runner/llamarunner/runner.go`
- New runner: `runner/ollamarunner/runner.go`
- Sibling trace (consequence of the deny list, predates this trace): `docs/traces/qwen35-flash-attention-gate.md`
- FA coverage matrix using this knowledge: `docs/research/k80-fa-model-coverage.md`
