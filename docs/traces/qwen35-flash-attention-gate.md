# Trace: Why FA is silently disabled for qwen3.5 on K80

**Date**: 2026-04-27
**Trigger**: Benchmark run [24974247971](https://github.com/dogkeeper886/ollama37/actions/runs/24974247971) showed `OLLAMA_FLASH_ATTENTION=1` had no effect on `qwen3.5:9b` — load request had `FlashAttention:false` and tokens/sec was identical across `off-f16` / `on-f16` / `on-q8_0` configs.

## TL;DR

For models on the **new Ollama engine** (`model/models/qwen35/`), the per-model gate `f.SupportsFlashAttention()` returns false because of an explicit deny list at `fs/ggml/ggml.go:889` containing `gemma2` and `qwen35`. This blocks FA at `llm/server.go:247-250` before the load request is ever sent to the runner.

The new engine's SDPA backend path (`ml/backend/ggml/ggml.go:1773`) **does** support FA via `ggml_flash_attn_ext` when `t.b.flashAttention` is true. But that flag is plumbed from the load request's `FlashAttention` field, which is set to false by the gate. So the kernel is reachable but never invoked for these architectures.

The author comment at `ggml.go:887-888` says "flash attention is handled by the FlashAttention() allowlist instead" — suggesting an intent to bypass the gate for new-engine models. The code does not actually implement that bypass.

## Call flow

```
LLM server FA decision                                      # llm/server.go:236-267
  ├── fa := envconfig.FlashAttention(f.FlashAttention())     # :238
  │   │                                                      
  │   └── f.FlashAttention()                                 # fs/ggml/ggml.go:899-908
  │       └── slices.Contains(["gemma3","gpt-oss","qwen3",   # qwen35 IS in this list
  │                            "qwen35","qwen3vl",...], arch)
  │           => returns TRUE for qwen35
  │   => with OLLAMA_FLASH_ATTENTION=1, fa = true
  │
  ├── if fa && !ml.FlashAttentionSupported(gpus)             # :242 GPU gate
  │   └── ml/device.go:430 — K80 (compute 3.7) is in the     # post-PR #117
  │       supportsFA list, returns true
  │   => fa stays true
  │
  ├── if fa && !f.SupportsFlashAttention()                   # :247 model gate (THE BLOCKER)
  │   └── f.SupportsFlashAttention()                         # fs/ggml/ggml.go:879-897
  │       ├── if isEmbedding => false                        # :880-883 (n/a for qwen35)
  │       ├── if slices.Contains(["gemma2","qwen35"], arch)  # :889 DENY LIST — qwen35 hit
  │       │   => returns FALSE for qwen35
  │       └── (head-count check unreached for qwen35)
  │   => slog.Warn("flash attention enabled but not supported by model")
  │   => fa = false                                          # FA gate triggered
  │
  └── loadRequest.FlashAttention is never set to true        # :256 inside `if fa { ... }`
      (defaults to Go zero-value: false)

Runner subprocess (new engine)                              # runner/ollamarunner/runner.go
  └── load request received with FlashAttention: false      # :1286 - just a passthrough

Backend SDPA dispatch                                       # ml/backend/ggml/ggml.go:1773
  └── if t.b.flashAttention                                 # :1782 — false here
      ├── (true branch: ggml_flash_attn_ext)               # never reached for qwen35
      └── else branch                                       # :1791
          ├── kq = key.MulmatFullPrec(query)                # K·Qᵀ
          ├── kq = soft_max_ext(kq, mask, scale)            # softmax
          └── value.Mulmat(kq)                              # ·V
          => non-FA attention path runs
```

## Empirical confirmation

From workflow run 24974247971 (qwen3.5:9b, FA-on config), `container-logs.log`:

```
time=2026-04-27T02:53:08.568Z level=WARN source=server.go:248 msg="flash attention enabled but not supported by model"
time=2026-04-27T02:53:08.617Z level=INFO source=runner.go:1264 msg=load request="{... FlashAttention:false KvCacheType: ...}"
```

The warning fires from the deny list path. The load request goes to the new engine with FA off. KvCacheType also empty because the user-set q8_0 was rejected — V-cache quant requires FA, so when fa=false, kvct gets dropped silently at `llm/server.go:265-267`.

## Why qwen35 is in the deny list

qwen3.5 has a **hybrid architecture**: most layers are Mamba-like SSM (DeltaNet), some layers are attention. From `model/models/qwen35/model.go:516-517`:

```
slog.Debug("model config", "arch", "qwen35", "layers", numLayers,
           "recurrent", recurrentCount, "attention", numLayers-recurrentCount, "moe", isMoE)
```

The KV cache log line we saw initially (`source=model.go:520 msg="cache dimensions"`) reports both SSM (`conv_dim`, `conv_channels`, `delta_state_size`) and attention (`head_v_dim`, `num_v_heads`) parameters because the cache is hybrid.

Why the deny list specifically:
- `SupportsFlashAttention()` was originally written for the llama.cpp engine path. It checks GGUF metadata (`head_count_k == head_count_v`).
- Hybrid models like qwen35 don't fit that check cleanly — the head counts apply only to the attention layers, not the SSM ones.
- Author chose to deny these arches outright at the gate, with the comment that the new engine should handle FA itself.
- But the new engine doesn't have its own gate — it just consumes the `FlashAttention` field from the load request, which the server already cleared.

## What this means

| Question | Answer |
|---|---|
| Does qwen3.5:9b benefit from PR #117 (K80 FA enablement)? | **No** — silently rejected by the deny list |
| Does qwen3.5:9b benefit from `OLLAMA_KV_CACHE_TYPE=q8_0`? | **No** — V-cache quant requires FA, FA is denied |
| Could the new engine handle FA on the attention layers (and skip it on SSM layers)? | Yes, in principle — the SDPA backend at `ml/backend/ggml/ggml.go:1782` already gates per-call on `flashAttention`. SSM layers wouldn't go through SDPA at all. |
| Is the comment at `ggml.go:887-888` accurate? | Aspirational, not implemented. The "allowlist instead" mechanism doesn't exist as code. |

## Other models likely affected

`SupportsFlashAttention()` deny list currently contains:
- `gemma2` — full-transformer but with sliding-window+global hybrid attention; deny may be defensive
- `qwen35` — hybrid SSM+attention

Other potentially-affected models in our test lineup:
- `gemma4:e4b`, `gemma4:26b` — likely also hybrid (haven't verified architecture); not in deny list yet so might pass the gate, but kernel behavior unknown
- Anything else that uses the new engine and has cache_dimensions logged with SSM params

Worth a per-model FA audit (suggested in #116 as out of scope).

## Possible fixes

Out of scope for this trace, but options if/when someone wants to unlock FA for qwen35:

1. **Conditional bypass in `llm/server.go:247`**: skip `SupportsFlashAttention()` check when the model uses the new engine (textProcessor != nil, or check an architecture flag). Risk: assumes new-engine attention layers are FA-safe; needs verification.
2. **Remove qwen35 from deny list**: simpler but assumes the head-count check at `ggml.go:894-896` is meaningful for hybrid models. May produce wrong allow/deny for hybrid arches.
3. **Add a new method** `f.SupportsFlashAttentionInNewEngine()` — explicit allowlist for new-engine FA. More code surface but clearer intent.

Empirical validation needed for any of these — does qwen35's attention layer actually produce correct output with `t.b.flashAttention=true` on K80?

## References

- Deny list: `fs/ggml/ggml.go:887-891`
- FA preference allowlist: `fs/ggml/ggml.go:899-908`
- Server FA gate: `llm/server.go:236-267`
- New-engine SDPA dispatch: `ml/backend/ggml/ggml.go:1773-1804`
- New-engine load request: `runner/ollamarunner/runner.go:1286`
- qwen35 model implementation: `model/models/qwen35/`
- Trigger run: [24974247971](https://github.com/dogkeeper886/ollama37/actions/runs/24974247971)
