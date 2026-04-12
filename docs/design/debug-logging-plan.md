# Debug Logging Plan

## Problem

During qwen3.5 development (issues #57, #72, #73, traces in `docs/traces/qwen35-*.md`), we repeatedly hit blind spots:

### Evidence from git history and issues

**Engine confusion** (issue #72 comment):
> "engine: ollama (note: parser detected 'ollama' but this is actually llama.cpp path)"
> "Despite OllamaEngineRequired returning true, NewTextProcessor() fails, causing a silent fallback"

We couldn't tell which engine was running. The test framework guessed wrong. The fallback is silent — only a `slog.Debug` on the compatibility path, nothing on success.

**Memory opacity** (issue #72 comment):
> "the 3.5× graph safety multiplier inflates the estimate"
> "Unexplained: ~3.5 GiB on GPU0 — not reported by llama.cpp logging"

The memory estimator computed values but never logged them. We could only see the final nvidia-smi numbers, not how the system decided to use 4 GPUs.

**Tensor mapping** (commits 8e269e7, cb5d9a4, 9190c34, 162f4be, 7997180):
Five separate INSTRUMENT commits were needed to debug why `populateFields` wasn't mapping tensors. Each added ad-hoc `slog.Info("INSTRUMENT: ...")` that had to be removed later. If `populateFields` had debug logging, none of those commits would have been needed.

**Formula mismatch** (trace `qwen35-garbage-output.md`):
Model produced garbage — 10 bugs found. We had no way to know which layers were recurrent vs attention, or what formula was being applied, without reading C code.

**Shape mismatches** (issue #57, trace `qwen35-27b-shape-mismatch.md`):
> "tensor 'blk.0.attn_gate.weight' has wrong shape; expected 5120, 5120, got 5120, 6144"

No logging of expected vs actual dimensions during model construction.

### Current state

| Area | slog.Debug | slog.Info | Total |
|------|-----------|-----------|-------|
| `model/` (all models) | 4 | 9 | 13 |
| `kvcache/` | 3 | 0 | 3 |
| `runner/` | 16 | ~28 | ~44 |
| `ml/backend/` | 0 | ~10 | ~10 |

Zero metrics endpoints. No prometheus, no expvar.

## Design Principle

Log **decisions and state transitions**, not data dumps. Every log line should answer "why did the system do X?" not "here is a list of things".

## Logging Points

### Phase 1: Engine Selection (`llm/server.go:148-168`)

The single most impactful gap. Currently only logs on fallback.

```
DEBUG engine selected              engine=ollama arch=qwen35
DEBUG engine fallback              engine=llama_compat arch=qwen35 reason="model not yet supported"
```

**Why**: Issue #72 spent significant time confused about which engine was running. The test framework parsed "ollama" from the runner flags but the actual execution used llama.cpp.

### Phase 2: Model Construction (`model/models/*/model.go`)

Log config values that were read from GGUF. This replaces the 5 INSTRUMENT commits.

```
DEBUG model config                 arch=qwen35 layers=64 recurrent=48 attention=16 moe=false
DEBUG ssm config                   d_inner=6144 d_state=128 n_group=8 dt_rank=24 conv_kernel=5
DEBUG cache dimensions             type=HybridCache conv_dim=4 conv_channels=8192 delta_state_size=1572864
DEBUG rope config                  type=neox base=1000000 scale=1 mrope_sections=[24,24,16]
```

**Why**: During #57 (shape mismatch), the root cause was `n_embd != ssm_d_inner` for 27b. If SSM config was logged, we'd have seen the mismatch immediately instead of needing a trace doc.

### Phase 3: Tensor Population (`model/model.go:populateFields`)

This replaces commits cb5d9a4 and 9190c34.

```
DEBUG tensor mapped                layer=0 field=DeltaNet.SSMQKV name=blk.0.attn_qkv.weight
DEBUG tensor not found             layer=0 field=DeltaNet.SSMIn name=blk.0.ssm_in.weight
```

**Why**: Five commits were spent debugging tensor mapping. A permanent debug log here eliminates that entire class of debugging.

**Implementation note**: Only log at Debug level. Gate with `slog.Default().Enabled()` check since this runs for every tensor in every layer.

### Phase 4: GPU Layout Decision (`runner/ollamarunner/runner.go`)

The runner already logs `load request` at Info with the full layout. What's missing is a summary after the iterative fitting converges.

```
DEBUG gpu layout decided           gpus=2 split="35+30" total_vram_mib=21458
DEBUG gpu layout rejected          gpus=1 reason="insufficient vram" needed_mib=24000 available_mib=11200
```

**Why**: Issue #72's core problem — model used 4 GPUs because the llama engine's estimator with 3.5x multiplier inflated memory. We couldn't see the estimation logic.

### Phase 5: DeltaNet Path (`model/models/qwen35/deltanet.go:237`)

```
DEBUG deltanet forward             layer=0 path=autoregressive seq_tokens=1 n_seqs=1
DEBUG deltanet forward             layer=0 path=chunked seq_tokens=20 n_seqs=1 chunks=1
```

**Implementation note**: Only log on first occurrence of each path per model lifetime. Use a flag on the Model or GatedDeltaNet struct.

**Why**: During trace `qwen35-garbage-output.md`, 10 bugs were found in the DeltaNet path. Knowing which path executed would have narrowed debugging immediately.

### Phase 6: Recurrent Cache State (`kvcache/recurrent.go`)

```
DEBUG cache slot allocated         seq=0 slot=0 new=true
DEBUG cache slots zeroed           count=1 slots=[0]
DEBUG cache batch layout           type=single_sequence seq=0 tokens=1
```

Only log slot lifecycle events (allocate, zero, free), not per-layer Get/Put.

**Why**: The recurrent cache is new infrastructure with copy-on-write semantics. When multi-sequence batching is used, silent slot sharing bugs could cause state corruption with no trace.

### Phase 7: Checkpoints (`kvcache/recurrent_checkpoints.go`)

Already has Debug logging for checkpoint misses. Add for saves and restores.

```
DEBUG checkpoint saved             slot=0 pos=1664 index=0
DEBUG checkpoint restored          slot=0 from_pos=1664 layers=96
```

**Why**: Checkpoint restore failures cause full prompt reprocessing with no visible indication. Logging saves/restores makes the checkpoint system observable.

## What NOT to Log

- Per-tensor shapes during forward — use `/trace` skill for this
- Per-token logits or probabilities — too noisy, no debug value
- Every cache Get/Put per layer — 48+ calls per forward, use `/instrument` skill
- Tensor names during model loading — already logged at Info by `ggml.go:136`

## Scope

| Phase | Files | Lines | Priority | Replaces |
|-------|-------|-------|----------|----------|
| 1. Engine selection | `llm/server.go` | ~5 | **High** | Manual grep of runner logs |
| 2. Model config | `model/models/qwen35/model.go` | ~15 | **High** | 5 INSTRUMENT commits |
| 3. Tensor population | `model/model.go` | ~10 | **High** | Commits cb5d9a4, 9190c34 |
| 4. GPU layout | `runner/ollamarunner/runner.go` | ~10 | **High** | Trace doc analysis |
| 5. DeltaNet path | `model/models/qwen35/deltanet.go` | ~5 | Medium | — |
| 6. Cache slots | `kvcache/recurrent.go` | ~10 | Medium | — |
| 7. Checkpoints | `kvcache/recurrent_checkpoints.go` | ~5 | Low | Partial existing |

**Total**: ~60 lines of `slog.Debug`. All silent by default, visible with `OLLAMA_DEBUG=1`.

Phases 1-4 are highest impact — they answer "which engine, what config, why these GPUs, which tensors matched" and would have prevented most of the debugging effort on issues #57, #72, #73.

## Implementation Notes

- All new logs use `slog.Debug` — no new Info-level logs
- Phase 2 should apply to ALL models, not just qwen35 — consider adding to the `model.Model` interface or base
- For "first call only" logging (phases 4-5), use a flag on the struct
- Keep log keys consistent across phases: `arch`, `engine`, `layers`, `layer`, `gpus`, `path`, `seq`, `slot`
- Gate expensive operations: `if slog.Default().Enabled(nil, slog.LevelDebug) { ... }`
