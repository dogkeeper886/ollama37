# K80 Flash Attention Coverage by Model

**Issues**: [#122](https://github.com/dogkeeper886/ollama37/issues/122) (Phase 1) and [#123](https://github.com/dogkeeper886/ollama37/issues/123) (Phase 2) of [#121](https://github.com/dogkeeper886/ollama37/issues/121)
**Date**: 2026-04-27 (Phase 1 + Phase 2 results)
**Status**: Empirical verification complete for 5 architectures (Phase 2 representatives + qwen35 via #107). Phase 3 allowlist seeded and extended.

## TL;DR (after Phase 2)

Phase 2 corrected one Phase 1 assumption and confirmed three predictions:

- **Phase 1 wrong**: I claimed "all 13 models use the new Ollama engine." Empirical evidence shows `deepseek-r1:14b` (arch `qwen2`) actually runs on the **legacy llama.cpp engine** (`runner.go:950` from `runner/llamarunner/`), not the new engine (`runner.go:1264` from `runner/ollamarunner/`). Package existence in `model/models/<arch>/` does NOT imply the new engine is used at runtime — the engine selection has additional logic. **Phase 3 allowlist therefore only needs to cover models that actually use the new engine.**
- **Phase 1 right**: `qwen3.5:9b` and `qwen3.5:27b` (arch `qwen35`) silently lose FA via the deny list at `fs/ggml/ggml.go:889`. Confirmed both empirically (run 24974247971 baseline + 3 Phase 2 runs).
- **Phase 2 confirmed predictions**: `gpt-oss:20b` (new engine), `qwen3-vl:8b` (new engine), `deepseek-r1:14b` (llama.cpp engine) all enable FA correctly with `OLLAMA_FLASH_ATTENTION=1`. Output coherent across all three.

## Phase 2 empirical results

| Run | Model | Arch | Engine (verified) | FA enables? | Output coherent? | KV q8_0 reduction |
|---|---|---|---|---|---|---|
| [24960034243](https://github.com/dogkeeper886/ollama37/actions/runs/24960034243) | gemma3:4b | gemma3 | new (ollamarunner) | ✅ | ✅ bit-exact | 47% (254→135 MiB) |
| [24978297254](https://github.com/dogkeeper886/ollama37/actions/runs/24978297254) | gpt-oss:20b | gptoss | new (ollamarunner) | ✅ | ✅ bit-exact (off vs on); on-q8_0 minor word substitution | 47% (300→159 MiB) |
| [24978521066](https://github.com/dogkeeper886/ollama37/actions/runs/24978521066) | qwen3-vl:8b | qwen3vl | new (ollamarunner) | ✅ | ✅ thinking-model; coherent in `.thinking` field | 47% (~1184→611 MiB total across split GPUs) |
| [24978699810](https://github.com/dogkeeper886/ollama37/actions/runs/24978699810) | deepseek-r1:14b | qwen2 | **llama.cpp** (llamarunner) | ✅ | ✅ thinking-model; coherent | KV size not in logs (different log format on llama.cpp path) |

**Key finding**: 4 different architectures empirically validated. The new-engine models (gemma3, gptoss, qwen3vl) all produce correct output with FA. deepseek-r1's qwen2 path goes through llama.cpp and works via the existing head-count gate — not affected by Phase 3 changes.

## qwen35 follow-up validation (via #107, 2026-05-13)

Phase 2 deferred qwen35 with the note *"requires empirical validation of hybrid SSM+attention output correctness with FA enabled."* That validation was triggered while investigating #107 (asymmetric K-only KV quant), which was rendered obsolete if FA + symmetric Q8 KV could be made to work on qwen35.

Methodology was the existing `test-fa-k80.yml` benchmark mode (3 configs: off-f16 / on-f16 / on-q8_0), prefixed by a 1-line allowlist patch adding `"qwen35"` to `SupportsFlashAttentionInNewEngine`.

| Run | Model | GPUs used | FA enables? | Output coherent? | VRAM Δ (off→on-q8_0) | tok/s Δ |
|---|---|---|---|---|---|---|
| [25799484945](https://github.com/dogkeeper886/ollama37/actions/runs/25799484945) | qwen3.5:9b | 1 | ✅ | ✅ thinking-coherent across all 3 configs; on-f16 ≡ on-q8_0 same structure | -138 MiB (-1.8%) | -1.4% |
| [25799818450](https://github.com/dogkeeper886/ollama37/actions/runs/25799818450) | qwen3.5:27b | 2 (34+31 layer split) | ✅ | ✅ **bit-identical** thinking output across all 3 configs | -174 MiB (-1.6%) | -0.8% |

**VRAM savings smaller than Phase 2 baselines** (~1-2% here vs ~47% for gemma3). Expected: qwen35 is hybrid — only attention layers use KV cache (DeltaNet layers use a recurrent state, unaffected by FA or KV quant), and at 4k ctx the attention-layer KV is small. The fraction-of-a-small-thing pattern matches the architecture. FA benefit would compound at longer context.

**Tokens/sec essentially flat** (within ±1.5%). No speedup, no regression. The fewer-attention-layers structure means FA's compute savings have less surface area to apply.

**Resolution**: qwen35 added to `SupportsFlashAttentionInNewEngine` allowlist. This obsoletes the original framing of #107 (K-only Q8 was a workaround for the unavailable FA path — now the FA path is available).

## Workflow caveats discovered during Phase 2

1. **Per-GPU VRAM only**: the `vram-mib.txt` snapshot via `nvidia-smi --query-gpu=memory.used | head -1` only captures GPU0. For multi-GPU model splits (qwen3-vl:8b, deepseek-r1:14b spread across 2 K80s), the "Total VRAM" column in the comparison table is misleading. Worth fixing in a workflow follow-up — sum across all GPUs or report per-GPU.
2. **KV cache regex misses some formats**: deepseek-r1's llama.cpp engine doesn't emit the `"kv cache" device=... size="N MiB"` log line in the same format the new engine does. Coverage-doc's KV size column shows "unknown" for that run. Worth either adding a llama.cpp-format regex or accepting this is a new-engine-only metric.
3. **Per-device KV size, not total**: when a model splits across GPUs, the KV cache log shows ONE GPU's portion, not the total. Misleading until you account for the split.

These don't invalidate the FA enablement findings — just affect the memory accounting precision. Tokens/sec, FA dispatch confirmation, and output correctness are all reliable.

## Background

PR #117 productized K80 GPU-side FA support. The qwen3.5:9b benchmark (run [24974247971](https://github.com/dogkeeper886/ollama37/actions/runs/24974247971)) revealed a per-model gate that silently rejects some models. The trace at `docs/traces/qwen35-flash-attention-gate.md` walks the call flow from `OLLAMA_FLASH_ATTENTION=1` to the deny list at `fs/ggml/ggml.go:889`. This audit's purpose is to predict which models in our test lineup hit the deny list, so Phase 2 can verify the predictions empirically and Phase 3 can implement the bypass with a known-correct seeded allowlist.

## Methodology

For each model in `cicd/tests/testcases/models/TC-MODELS-*.yml`:

1. **Identify GGUF architecture** — usually inferable from the Ollama tag prefix (e.g., `gemma3:4b` → `gemma3`). Ambiguous tags (e.g., `deepseek-r1:14b` is a distill, not native DeepSeek) verified via Ollama library.
2. **Check engine path** — does `model/models/<arch>/` exist? If yes, new engine. None of the lineup falls through to llama.cpp.
3. **Read `model/models/<arch>/`** for hybrid SSM markers (`deltanet.go`, `ssm`, `recurrent`, `conv1d` outside vision/audio context).
4. **Check `fs/ggml/ggml.go:879-897`** (`SupportsFlashAttention`) — explicit deny list at line 889 for `["gemma2", "qwen35"]`, otherwise falls through to head-count check (`headCountK == headCountV`).
5. **Check `fs/ggml/ggml.go:899-908`** (`FlashAttention`) — model-default-on allowlist. Affects whether FA is on by default with no env var; doesn't change support.

## Coverage matrix

| Test ID | Model tag | Likely GGUF arch | Engine | Attention shape | In deny list? | In FA-default allowlist? | **Predicted FA-on-K80** |
|---|---|---|---|---|---|---|---|
| TC-MODELS-001 | `gpt-oss:20b` | `gptoss` | new | pure transformer | no | yes | ✅ enables |
| TC-MODELS-002 | `gemma3:27b` | `gemma3` | new | pure transformer + vision capable | no | yes | ✅ predicted (same arch as `:4b` validated) |
| TC-MODELS-003 | `deepseek-r1:14b` | `qwen2` (Qwen-2.5 distill, [confirmed via Ollama lib](https://ollama.com/library/deepseek-r1:14b)) | new | pure transformer | no | no | ⚠️ user-toggle only |
| TC-MODELS-004 | `qwen3.5:9b` | `qwen35` | new | **hybrid SSM (DeltaNet) + attention** | **YES** | yes | ❌ **silently denied** ([verified empirically](https://github.com/dogkeeper886/ollama37/actions/runs/24974247971)) |
| TC-MODELS-005 | `functiongemma:270m` | `gemma3` ([confirmed via Ollama lib](https://ollama.com/library/functiongemma:270m) — "built on the Gemma 3 270M model") | new | pure transformer | no | yes | ✅ predicted (gemma3 path validated for FA via gemma3:4b run #108) |
| TC-MODELS-006 | `gemma4:e4b` | `gemma4` | new | transformer + vision + audio | no | no | ⚠️ enables only if user sets `OLLAMA_FLASH_ATTENTION=1` (not default-on) |
| TC-MODELS-007 | `gemma4:26b` | `gemma4` | new | transformer + vision + audio | no | no | ⚠️ same as e4b |
| TC-MODELS-008 | `gemma3:4b` | `gemma3` | new | pure transformer + vision capable | no | yes | ✅ **verified empirically** ([run 24960034243](https://github.com/dogkeeper886/ollama37/actions/runs/24960034243)) |
| TC-MODELS-009 | `deepseek-r1:32b` | `qwen2` (Qwen-2.5 distill, predicted) | new | pure transformer | no | no | ⚠️ user-toggle only |
| TC-MODELS-010 | `qwen3.5:27b` | `qwen35` | new | hybrid SSM + attention | **YES** | yes | ❌ silently denied (same arch as `:9b`) |
| TC-MODELS-011 | `qwen3-vl:8b` | `qwen3vl` | new | pure transformer + vision | no | yes | ✅ enables (LM portion) |
| TC-MODELS-012 | `qwen3-vl:30b` | `qwen3vl` or `qwen3vlmoe` (likely MoE at 30b) | new | transformer + vision (+ MoE?) | no | yes | ✅ enables |
| TC-MODELS-013 | `ministral-3:3b` | `mistral3` | new | pure transformer + vision capable | no | no | ⚠️ user-toggle only |

### Status legend

- ✅ — `OLLAMA_FLASH_ATTENTION=1` enables FA; model is in either the allowlist (default-on) or passes the head-count check.
- ⚠️ — Same as ✅ but FA is **not** in the model's default — user must explicitly set the env var. Will work when set.
- ❌ — `OLLAMA_FLASH_ATTENTION=1` is silently overridden because the arch is in the deny list. Phase 3's job to fix.

## Per-architecture summary

| Architecture | Models in lineup | FA gate decision | Why |
|---|---|---|---|
| `gptoss` | 1 | passes | not in deny list; head-count check OK |
| `gemma3` | 3 (`:4b`, `:27b`, FunctionGemma) | passes | not in deny list; FA-default allowlist member |
| `gemma4` | 2 (`:e4b`, `:26b`) | passes | not in deny list; not FA-default but explicit env-var works |
| `qwen2` | 2 (deepseek-r1:14b/32b distills) | passes | not in deny list; user-toggle path |
| `qwen35` | 2 (`:9b`, `:27b`) | **denied** | explicit deny list at `ggml.go:889` (hybrid SSM model) |
| `qwen3vl` / `qwen3vlmoe` | 2 (`:8b`, `:30b`) | passes | FA-default allowlist member |
| `mistral3` | 1 (ministral-3:3b) | passes | user-toggle path |

**No model in the lineup uses the legacy llama.cpp engine path.** This is significant — it means the entire test lineup depends on the new engine's FA path working correctly. PR #117 closed the GPU-side gate; the per-model gate (and its qwen35 deny) is now the only remaining blocker for any tested model.

## Caveats / things this audit didn't verify

- ~~**GGUF arch values for distills.**~~ **Resolved in Phase 2**: `deepseek-r1:14b`'s arch is confirmed `qwen2` via the runner log line `architecture=qwen2` (run 24978699810).
- ~~**FunctionGemma.**~~ **Resolved in Phase 2**: confirmed Gemma 3-based via Ollama library page (NOT a custom Modelfile in our repo as I initially assumed — it's an Ollama-published model).
- **`qwen3-vl:30b` MoE detection.** The new-engine `qwen3vl` package registers both `qwen3vl` and `qwen3vlmoe` arches. The 30b tag is likely MoE, but the actual GGUF metadata wasn't read.
- **Vision model FA correctness.** Even if the FA gate passes for a vision model, FA is dispatched per attention call (`ml/backend/ggml/ggml.go:1782`). The vision projector / image encoder may use its own attention paths that haven't been validated on K80 with FA. Phase 2 verification on `qwen3-vl:8b` would surface any issues.
- **MoE on K80 generally.** The lineup includes potential MoE models (`gpt-oss:20b`, `qwen3-vl:30b` if MoE). MoE routing kernels weren't audited. K80 fitting for these is questionable; FA + KV quant may be moot if VRAM is the binding constraint.

## Phase 2 (#123) verification recommendations

Three representatives, chosen to cover the categories where the prediction is least certain or where empirical truth matters most:

### 1. `gpt-oss:20b` (pure transformer, large, FA-default allowlist)

**Why test**: confirms PR #117's productized FA path works for an arch we haven't empirically verified yet. Simple sanity check; if this fails the productize was incomplete.

**Expected**: ✅ FA enables, output bit-exact match between FA-off and FA-on, KV cache works with q8_0.

### 2. `qwen3-vl:8b` (vision-language)

**Why test**: confirms FA dispatch works on a multimodal model. The text-only LM portion should benefit; the vision encoder portion might use a different attention path. Failure mode would surface as garbage tokens or vision-grounding errors when given an image — but our text-only benchmark prompt would catch crash/numerical issues even without vision input.

**Expected**: ✅ FA enables on the LM, text output correct.

### 3. `deepseek-r1:14b` (qwen2 arch via distill, user-toggle path)

**Why test**: validates the model→arch mapping (we'll see "qwen2" in the runner logs if the prediction is right) and confirms a non-default-on model still enables FA when the env var is set. Also stress-tests the larger model on K80 with FA.

**Expected**: ✅ FA enables, response is reasonable (deepseek-r1 produces `<think>...</think>` chain-of-thought, may need to look at both `.thinking` and `.response` fields).

### Why NOT testing qwen3.5 in Phase 2

The deny list silently rejects it today. Validating qwen3.5 with FA enabled requires Phase 3's allowlist fix to land first. Phase 2's qwen3.5 work would be a non-FA baseline only — already covered by run 24974247971. Save the FA validation for after Phase 3 patches the gate.

### Why NOT testing all 13

Architecturally identical models (gemma3:4b vs gemma3:27b, qwen3.5:9b vs qwen3.5:27b) behave the same for FA purposes. Testing one per arch covers it. The 3 picks above cover 4 of the 7 unique arches in the lineup; the rest (gemma3 already done, gemma4, mistral3, qwen3vl-MoE) can be tested individually if Phase 3 lands and someone wants comprehensive coverage. Diminishing returns vs runner time after the first few representatives.

## What Phase 3 should seed the allowlist with (post-Phase 2)

Based on Phase 1 + Phase 2 empirical results, the initial `SupportsFlashAttentionInNewEngine()` allowlist should contain **only architectures that (a) empirically work on K80 with FA enabled AND (b) actually use the new engine**. After Phase 2 verification:

- ✅ **`gemma3`** — validated in #108 (gemma3:4b on new engine, bit-exact match)
- ✅ **`gptoss`** — Phase 2 confirmed (run 24978297254, bit-exact match between off-f16 and on-f16)
- ✅ **`qwen3vl`** — Phase 2 confirmed (run 24978521066, coherent thinking output, FA dispatched)
- ⚠️ **`qwen3vlmoe`** — predicted-equivalent to qwen3vl but not separately tested. Add with a `// TODO: empirically verify` comment, or pick the 30b variant for a follow-up bench.

**NOT in the new-engine allowlist** (different reasons):

- ❌ `qwen35` — currently in the deny list. Adding to the new-engine allowlist (which bypasses the deny list) requires its own targeted test. The qwen35 trace at `docs/traces/qwen35-flash-attention-gate.md` notes the SDPA backend supports FA in principle, but hybrid SSM+attention output correctness with `t.b.flashAttention=true` is unverified. Add as a separate post-Phase 3 follow-up issue.
- ❌ `qwen2` — empirically validated in Phase 2 BUT uses the **llama.cpp engine**, not the new engine. It already works via the existing head-count gate; no Phase 3 changes needed.
- ❌ `gemma4`, `mistral3`, `gemma3n`, `llama`, `llama4`, `mllama`, `deepseek2`, `gemma2` — not validated. Don't seed the new-engine allowlist with these even if a Go package exists.

### Concrete initial allowlist for Phase 3

```go
func (f GGML) SupportsFlashAttentionInNewEngine() bool {
    return slices.Contains([]string{
        "gemma3",   // validated #108 (gemma3:4b)
        "gptoss",   // validated #123 Phase 2 (gpt-oss:20b)
        "qwen3vl",  // validated #123 Phase 2 (qwen3-vl:8b)
        // qwen3vlmoe — likely safe (same package as qwen3vl) but unverified;
        //   either include with a TODO or run :30b in a Phase 2.5 sweep
        // qwen35 — needs targeted validation; see docs/traces/qwen35-flash-attention-gate.md
    }, f.KV().Architecture())
}
```

This mirrors the conservative-allowlist pattern from PR #117: empirically validate one model per arch, then enable.

## References

- Trace: `docs/traces/qwen35-flash-attention-gate.md`
- Deny list: `fs/ggml/ggml.go:887-891`
- FA-default allowlist: `fs/ggml/ggml.go:899-908`
- New-engine model registrations: grep for `model.Register(` in `model/models/`
- Tested models: `cicd/tests/testcases/models/TC-MODELS-*.yml`
- Sibling phase issues: [#123](https://github.com/dogkeeper886/ollama37/issues/123) (empirical verification), [#124](https://github.com/dogkeeper886/ollama37/issues/124) (allowlist implementation)
- Portal: [#121](https://github.com/dogkeeper886/ollama37/issues/121)
