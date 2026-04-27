# K80 Flash Attention Coverage by Model

**Issue**: [#122](https://github.com/dogkeeper886/ollama37/issues/122) (Phase 1 of [#121](https://github.com/dogkeeper886/ollama37/issues/121))
**Date**: 2026-04-27
**Status**: Code-study audit complete. Empirical verification of representatives is Phase 2 (#123).

## TL;DR

All 13 models in ollama37's tested lineup use the **new Ollama engine** (Go-native model packages in `model/models/<arch>/`). None go through the legacy llama.cpp engine path.

For each model's GGUF architecture, FA support today is determined entirely by `fs/ggml/ggml.go:879-897` (`SupportsFlashAttention`). The deny list there contains only `gemma2` and `qwen35`. Of our test lineup, **`qwen3.5:9b` and `qwen3.5:27b`** hit the deny list and silently lose FA. The other 11 models are predicted to enable FA correctly with `OLLAMA_FLASH_ATTENTION=1`. The two predicted-deny models are exactly the ones that motivated #121.

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
| TC-MODELS-008 | `gemma3:4b` | `gemma3` | new | pure transformer + vision capable | no | yes | ✅ **verified empirically** ([run 24960034243](https://github.com/dogkeeper886/ollama37/actions/runs/24960034243)) |
| TC-MODELS-002 | `gemma3:27b` | `gemma3` | new | pure transformer + vision capable | no | yes | ✅ predicted (same arch as `:4b`) |
| TC-MODELS-004 | `gemma4:e4b` | `gemma4` | new | transformer + vision + audio | no | no | ⚠️ enables only if user sets `OLLAMA_FLASH_ATTENTION=1` (not default-on) |
| TC-MODELS-003 | `gemma4:26b` | `gemma4` | new | transformer + vision + audio | no | no | ⚠️ same as e4b |
| TC-MODELS-005 | `deepseek-r1:14b` | `qwen2` (Qwen-2.5 distill, [confirmed via Ollama lib](https://ollama.com/library/deepseek-r1:14b)) | new | pure transformer | no | no | ⚠️ user-toggle only |
| TC-MODELS-009 | `deepseek-r1:32b` | `qwen2` (Qwen-2.5 distill, predicted) | new | pure transformer | no | no | ⚠️ user-toggle only |
| TC-MODELS-006 | `qwen3.5:9b` | `qwen35` | new | **hybrid SSM (DeltaNet) + attention** | **YES** | yes | ❌ **silently denied** ([verified empirically](https://github.com/dogkeeper886/ollama37/actions/runs/24974247971)) |
| TC-MODELS-011 | `qwen3.5:27b` | `qwen35` | new | hybrid SSM + attention | **YES** | yes | ❌ silently denied (same arch) |
| TC-MODELS-012 | `qwen3-vl:8b` | `qwen3vl` | new | pure transformer + vision | no | yes | ✅ enables (LM portion) |
| TC-MODELS-013 | `qwen3-vl:30b` | `qwen3vl` or `qwen3vlmoe` (likely MoE at 30b) | new | transformer + vision (+ MoE?) | no | yes | ✅ enables |
| TC-MODELS-010 | `ministral-3:3b` | `mistral3` | new | pure transformer + vision capable | no | no | ⚠️ user-toggle only |
| TC-MODELS-007 | FunctionGemma | `gemma3` (likely — Gemma3-based custom Modelfile) | new | pure transformer | no | yes | ✅ enables (predicted) |

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

- **GGUF arch values for distills.** `deepseek-r1:14b` and `:32b` are Qwen-2.5 distills per the Ollama library page, so arch should be `qwen2` — but the actual GGUF metadata wasn't read directly. Phase 2 verification of one of these models will surface the actual arch from the runner logs.
- **FunctionGemma.** This is a custom Modelfile in our project, presumably based on `gemma3`. Worth opening the Modelfile to confirm.
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

## What Phase 3 should seed the allowlist with

Based on this audit + Phase 2 verification, the initial `SupportsFlashAttentionInNewEngine()` allowlist should contain **only architectures empirically validated** to produce correct output on K80 with FA enabled. After Phase 2 the candidates are:

- `gemma3` (validated in #108)
- `gptoss` (Phase 2 verifies)
- `qwen3vl` and `qwen3vlmoe` (Phase 2 verifies)
- `qwen2` (Phase 2 verifies via deepseek-r1:14b)

NOT in the initial allowlist (require their own validation later):
- `qwen35` — even though the trace says the SDPA backend supports FA in principle, hybrid attention layers haven't been empirically validated for output correctness with `t.b.flashAttention=true`. Phase 3 should add it as a separate follow-up after a targeted test.
- `gemma4`, `mistral3`, `gemma3n`, `llama`, `llama4`, `mllama`, `deepseek2` — not in the lineup or not tested.

This mirrors the conservative-allowlist pattern from PR #117 itself: empirically validate, then enable.

## References

- Trace: `docs/traces/qwen35-flash-attention-gate.md`
- Deny list: `fs/ggml/ggml.go:887-891`
- FA-default allowlist: `fs/ggml/ggml.go:899-908`
- New-engine model registrations: grep for `model.Register(` in `model/models/`
- Tested models: `cicd/tests/testcases/models/TC-MODELS-*.yml`
- Sibling phase issues: [#123](https://github.com/dogkeeper886/ollama37/issues/123) (empirical verification), [#124](https://github.com/dogkeeper886/ollama37/issues/124) (allowlist implementation)
- Portal: [#121](https://github.com/dogkeeper886/ollama37/issues/121)
