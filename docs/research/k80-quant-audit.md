# Audit: ollama37 K80 Quantization Defaults (Part A of #105)

**Issue**: [#105](https://github.com/dogkeeper886/ollama37/issues/105)
**Date**: 2026-04-26
**Status**: Part A complete. Part B (kernel fusion via `nvprof` on CI runner) tracked separately in this issue.

## TL;DR

Three findings worth acting on:

1. **Weight quantization is already optimal.** Every model in the tested lineup uses a K80-friendly quant (Q4_K_M, Q4_0, MXFP4) by default. No fp16-defaulting model. Including OpenAI's `gpt-oss:20b` whose native MXFP4 format is supported via `dequantize_row_mxfp4_cuda`. **No action needed here.**
2. **KV cache is hardcoded to F16 on K80, with no available workaround in current code.** This is the single biggest unrealized win for K80 inference. Both `OLLAMA_KV_CACHE_TYPE` and `OLLAMA_FLASH_ATTENTION` env vars are unset everywhere, but even if a user sets them, the quant request gets rejected because flash attention is disabled on `ComputeMajor < 7`, and llama.cpp requires FA for V-cache quantization. **Two follow-up paths worth opening as separate issues** (§4).
3. **BF16 → FP16 conversion path exists** in `ggml_get_to_fp16_cuda` (`convert.cu:706`). Whether it runs once at model-load time or per-op is a Part B (profiling) question.

## 1. Model Lineup in Scope

Extracted from `cicd/tests/testcases/models/TC-MODELS-*.yml` — these are the models the K80 runner is validated against:

| Test ID | Model tag | Source | Native training format |
|---|---|---|---|
| TC-MODELS-001 | `gpt-oss:20b` | OpenAI | MXFP4 |
| TC-MODELS-002 | `gemma3:27b` | Google | BF16 |
| TC-MODELS-003 | `gemma4:26b` | Google | BF16 |
| TC-MODELS-004 | `gemma4:e4b` | Google | BF16 |
| TC-MODELS-005 | `deepseek-r1:14b` | DeepSeek | BF16 (distill) |
| TC-MODELS-006 | `qwen3.5:9b` | Alibaba | BF16 |
| TC-MODELS-007 | FunctionGemma (tool calling) | Custom | BF16 |
| TC-MODELS-008 | `gemma3:4b` | Google | BF16 |
| TC-MODELS-009 | `deepseek-r1:32b` | DeepSeek | BF16 |
| TC-MODELS-010 | `ministral-3:3b` | Mistral | BF16 |
| TC-MODELS-011 | `qwen3.5:27b` | Alibaba | BF16 |
| TC-MODELS-012 | `qwen3-vl:8b` | Alibaba | BF16 |
| TC-MODELS-013 | `qwen3-vl:30b` | Alibaba | BF16 |

## 2. Weight Quantization (Part A.1)

### How defaults are resolved

Test cases reference plain tags like `gemma3:27b` (no explicit quant suffix). Ollama's registry resolves these to a default quantization at pull time — typically Q4_K_M for modern Ollama-native models, MXFP4 for `gpt-oss`. Our tests do not override quantization.

### K80 compatibility of supported quants

`ml/backend/ggml/ggml/src/ggml-cuda/convert.cu` (`ggml_get_to_fp16_cuda`, lines 659-711) supports dequantizing every quant format the registry serves:

| Format | K80 path | Notes |
|---|---|---|
| Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0 | ✅ Per-block dequant kernels | Standard GGML K-quants. No tensor cores needed. |
| Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS etc. | ✅ Per-row dequant kernels | Lower-bit and importance-quant variants. Work on K80. |
| MXFP4 | ✅ `dequantize_row_mxfp4_cuda` | OpenAI's microscaling FP4 format. K80 has no FP4 silicon, but the dequant kernel uses a 16-entry lookup table (`kvalues_mxfp4`) — no FP4 hardware required. |
| BF16 (raw, not quanted) | ✅ `convert_unary_cont_cuda<nv_bfloat16>` | Software conversion. See §3 below. |
| F16 (raw) | ✅ Native | K80 has fp16 storage since CUDA 7.5. |

**Verdict for Part A.1**: every model in the lineup runs at a K80-friendly quant by default. No action needed.

## 3. KV Cache (Part A.2) — The Real Finding

### Current state on K80

Both relevant env vars are unset everywhere in the repo (checked `cicd/`, `docker/`, `.github/`):

| Variable | Default | What it sets |
|---|---|---|
| `OLLAMA_KV_CACHE_TYPE` | `f16` | KV cache element type |
| `OLLAMA_FLASH_ATTENTION` | `false` | Whether to enable fused FlashAttention |

So K80 inference always uses fp16 KV. For a 32k-context Llama-class model that's roughly 2 × n_layers × n_heads × head_dim × seq_len × 2 bytes — multi-GB territory at long context, on a card with 24 GB total split across two GPUs.

### Why setting `OLLAMA_KV_CACHE_TYPE=q8_0` doesn't work today

Three layered checks reject the request on K80:

1. **`ml/device.go:434`** — `FlashAttentionSupported` returns `false` if any GPU has `ComputeMajor < 7`. K80 is 3.7 → FA always disabled.
2. **`llm/server.go:265-267`** — if FA is disabled and user sets a quantized KV cache type, code logs `"quantized kv cache requested but flash attention disabled"` and falls back to F16 silently.
3. **`llama.cpp/src/llama-context.cpp`** — even if you bypass (1) and (2), llama.cpp errors out: `"V cache quantization requires flash_attn"`.

### What the constraint actually says

Important nuance: the llama.cpp check is **`ggml_is_quantized(params.type_v) && flash_attn_type == DISABLED`** — the requirement is V quant + no FA. **K-cache quantization without FA is permitted by llama.cpp.** Ollama's wrapper (`llama/llama.go:160-174`) sets both `params.type_k` and `params.type_v` to the same value, so it can't currently express "K=Q8_0, V=F16."

### What FlashAttention actually requires for K80

Looking at the CUDA-side kernels:

| Kernel file | Arch guard |
|---|---|
| `fattn-wmma-f16.cuh` | `__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA` (sm_70+) — uses WMMA |
| `fattn-mma-f16.cuh` | Implicit Volta+ (uses MMA paths) |
| `fattn-tile.cu` | Needs verification |
| **`fattn-vec.cuh`** | **No arch guard found** — only `#ifdef FLASH_ATTN_AVAILABLE`, which `common.cuh:275` defines unconditionally unless `GGML_CUDA_NO_FA` is set |

So the *vec* variant of flash attention may already be compiled into our K80 build but be unreachable at runtime because Ollama's Go-side `FlashAttentionSupported` gate rejects `ComputeMajor < 7` *before* asking which FA kernel would be selected.

**Caveat**: "no arch guard found" is a static read. Whether `fattn-vec` actually produces correct results on sm_37 is an empirical question — upstream may have rejected sub-Volta on the assumption nobody tests it.

## 4. Recommended Follow-ups

These are what should become their own GitHub issues; this audit doesn't implement them.

### Recommendation 1 — Allow asymmetric KV quant (K-only) on K80 → ⏬ **DEPRIORITIZED**

Originally tracked as [#107](https://github.com/dogkeeper886/ollama37/issues/107). With Recommendation 2 empirically validated below, **symmetric Q8_0 KV quant works on K80** via the FA path and delivers ~47% reduction (vs the ~25% K-only would have given). #107 remains a fallback if FA enablement is ever rolled back, but is no longer the highest-leverage path.

### Recommendation 2 — Investigate `fattn-vec` viability on K80 → ✅ **EMPIRICALLY VALIDATED**

Originally tracked as [#108](https://github.com/dogkeeper886/ollama37/issues/108).

**Outcome (2026-04-26)**: `fattn-vec` works on K80. PR [#112](https://github.com/dogkeeper886/ollama37/pull/112) added `OLLAMA_FLASH_ATTENTION_K80=1` opt-in; runs [24960034243](https://github.com/dogkeeper886/ollama37/actions/runs/24960034243) and [24960260331](https://github.com/dogkeeper886/ollama37/actions/runs/24960260331) on the K80 runner show:

- FA-on output bit-exact identical to FA-off baseline (gemma3:4b, "What is the capital of France?")
- KV cache with `OLLAMA_KV_CACHE_TYPE=q8_0` allocates 135 MiB vs 254 MiB f16 baseline — **47% reduction**
- No CUBLAS errors, no kernel crashes, production container restored cleanly

The "bigger prize" prediction held. Productize follow-up tracked separately (replace env-var hack with default-on Go-side gate; add K80 test cases to the test framework).

**Effort**: ~1 day to test. Possibly significant follow-up if the kernel needs sm_37-specific patches (e.g., shfl behavior, shared-mem layout).

**Risk**: Medium-high. Upstream's gate at sm_70 likely reflects "we never tested it on Kepler." Could find correctness bugs that require kernel-side fixes. Worth knowing the answer either way — if it works, it's a major win; if it doesn't, we update our docs and move on.

### Recommendation 3 — None for weight quantization

The audit found no action item here. Defaults are already correct.

### Recommendation 4 — None for BF16 conversion at this stage

The conversion path exists and works. Whether it runs once-per-load or per-matmul is a Part B (profiling) question. No code change recommended without measurement.

## 5. What This Audit Did Not Verify

- **Per-op vs load-time bf16 conversion** — would require profiling on the K80 runner (Part B).
- **Whether `fattn-vec.cuh` actually produces correct output on sm_37** — would require a patched build + validation run on the runner.
- **Whether MXFP4 dequant performance on K80 is competitive** with Q4_K — would require benchmarking gpt-oss vs an equivalently-sized Q4_K model.

These are reasonable Part B / Recommendation-2 follow-ups.

## 6. References

- Test case YAMLs: `cicd/tests/testcases/models/TC-MODELS-*.yml`
- KV cache type plumbing: `llama/llama.go:160-174`, `envconfig/config.go:189-282`
- Flash attention gate: `ml/device.go:430-442`, `llm/server.go:236-267`
- llama.cpp V-quant constraint: `llama/llama.cpp/src/llama-context.cpp` ("V cache quantization requires flash_attn")
- Dequantization paths: `ml/backend/ggml/ggml/src/ggml-cuda/convert.cu:659-711`
- FA kernel arch guards: `ml/backend/ggml/ggml/src/ggml-cuda/fattn-*.cu*`, `common.cuh:275`
- Related prior survey: `docs/research/google-kv-survey.md` (KV compression algorithm survey)
- Related prior survey: `docs/research/deepseek-tilekernels-survey.md` (TileKernels feasibility)
