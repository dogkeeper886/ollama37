# Audit: GGML CUDA Kernel Fusion Coverage on K80 (Part B of #105)

**Issue**: [#105](https://github.com/dogkeeper886/ollama37/issues/105)
**Date**: 2026-04-26
**Status**: Part B complete (static-analysis path; nvprof empirical path was attempted and abandoned — see "Method" below)

## TL;DR

GGML's CUDA backend already does substantially more graph-level fusion than I expected. Six fusion patterns are in place (RMS_NORM+MUL, RMS_NORM+MUL+ADD, n-way ADD chains, SCALE+TANH+SCALE softcap, two TopK-MoE variants), and the SwiGLU activation+gate is a single fused op via `GGML_OP_GLU`.

**For K80 specifically, the only fusion gap that matters is attention compute** (Q·Kᵀ → scale → softmax → ·V), which goes unfused on K80 because flash attention is gated to compute ≥7 — and that's tracked separately in **#108**. The other gaps I found (MUL_MAT+bias ADD, MUL_MAT+residual ADD, ROPE+KV-cache CPY) are real-but-small: each saves one global-memory round-trip per layer, but the projection matmuls themselves are far heavier than the boundary cost.

**Recommendation: don't open new fusion-implementation issues from this audit.** Focus implementation effort on #108 (fattn-vec on K80) which would close the single biggest gap. If #108 fails, revisit this audit's "smaller gaps" section.

## 1. Method

Originally Part B was supposed to be empirical: run inference under `nvprof` on the K80 runner, collect a kernel-by-kernel summary, identify unfused boundaries from kernel name patterns. Two false starts:

1. First nvprof workflow run failed because the profile image cherry-picked nvprof binaries but missed CUPTI injection libraries (`libaccinj64.so.11.4`). Fix would have been a heavier image (PR #110, closed).
2. The fall-back idea of "use `OLLAMA_DEBUG=1` + `GGML_CUDA_DEBUG=1` to get an op trace" turned out not to apply: those env vars don't enable per-op dispatch logging out of the box (only ~1 specific debug print in batched cuBLAS config).

After the second false start, switched to **static analysis**: read the GGML CUDA dispatch (`ggml_cuda_compute_forward`), the graph-level fusion logic (`ggml_cuda_can_fuse` and the loop in `evaluate_and_capture_cuda_graph`), and a representative transformer model graph (`llama-model.cpp` Llama4 block at line 6680) to enumerate what runs and what already fuses.

Trade-off: no kernel timings, no runtime selection details (e.g., MMQ vs MMA path). For "find unfused boundaries" the static read is sufficient — fusion is a graph-level decision and can be read from source. If we later need timings to prioritize implementation work, we can come back with proper nvprof setup.

## 2. Existing Fusion Patterns in GGML CUDA

From `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` (graph evaluation loop at line ~3215, can-fuse logic at line 3090):

| Pattern | Ops fused | Implementation | Disabled by |
|---|---|---|---|
| **RMS_NORM + MUL** | 2 | `ggml_cuda_op_rms_norm_fused` (`norm.cu`) | `GGML_CUDA_DISABLE_FUSION` |
| **RMS_NORM + MUL + ADD** | 3 | `ggml_cuda_op_rms_norm_fused_add` (`norm.cu`) | same |
| **N-way ADD chain** (up to 8) | 2-8 | `ggml_cuda_op_fused_add` (`binbcast.cu`) | same |
| **SCALE + UNARY(TANH) + SCALE** (softcap, used by Gemma) | 3 | `ggml_cuda_op_softcap` (`softcap.cu`) | same |
| **TopK-MoE without norm** (4 ops: softmax→argsort→view→get_rows) | 4 | `ggml_cuda_op_topk_moe` (`topk-moe.cu`) | same |
| **TopK-MoE with norm** (8 ops) | 8 | `ggml_cuda_op_topk_moe` with norm flag | same |

Plus single-op fused activations (not graph-level fusion, but functionally equivalent — they take both gate and up tensors and produce the GLU output in one kernel):

| Op | Implementation | Used by |
|---|---|---|
| **SwiGLU** | `ggml_cuda_op_swiglu` (`unary.cu:300`) | Llama, Gemma, Qwen, Mistral |
| **SwiGLU-OAI** | `ggml_cuda_op_swiglu_oai` (`unary.cu:343`) | OpenAI gpt-oss (with α/limit clamp) |
| **GeGLU**, **GeGLU-erf**, **GeGLU-quick**, **ReGLU** | `unary.cu:292-308` | various |

## 3. Typical Transformer Block on K80 — Op Sequence

From `llama/llama.cpp/src/llama-model.cpp` Llama4 block (around line 6680). Each `→` is a separate op, except `[fused]` blocks which dispatch as one kernel:

```
Per layer:
  RMS_NORM(input) → MUL(norm_weight)            [fused: RMS_NORM + MUL]
  → MUL_MAT(Wq) [+ bias ADD]                    [unfused if bias]
  → MUL_MAT(Wk) [+ bias ADD]                    [unfused if bias]
  → MUL_MAT(Wv) [+ bias ADD]                    [unfused if bias]
  → ROPE(Q) → ROPE(K)                           [unfused]
  → [optional Q-norm/K-norm: RMS_NORM × 2]      [unfused — single RMS_NORM, no MUL pair]
  → KV cache write (CPY)                        [unfused]
  → Attention compute:
     - non-FA path (K80): MUL_MAT(Q·Kᵀ) → SCALE → SOFT_MAX → MUL_MAT(·V)   [4 unfused ops] ⚠️
     - FA path (sm_70+):  ggml_flash_attn_ext()                             [single kernel]
  → MUL_MAT(Wo)
  → ADD(residual)                               [fused with next residuals via n-way ADD]
  → RMS_NORM + MUL                              [fused]
  → MUL_MAT(W_gate) → MUL_MAT(W_up)             [unfused — model parallel projections]
  → GLU(swiglu/swiglu_oai/etc.)                 [single fused activation]
  → MUL_MAT(W_down)
  → ADD(residual)                               [fused]
```

## 4. Unfused Boundaries (Ranked by K80 Impact)

### 4.1 ⚠️ Attention compute on non-FA path

**The single biggest gap on K80.** Without flash attention, attention is decomposed into 4 separate ops per layer per token:
- `MUL_MAT(Q · Kᵀ)` → produces `[seq_len, seq_len]` attention scores tensor (large for long context)
- `SCALE(1/√d_k)` → reads/writes scores
- `SOFT_MAX` → reads/writes scores
- `MUL_MAT(scores · V)` → reads scores

For a 32B model with 60 layers and a 4k-context, the unfused attention path adds roughly 60 × 3 = 180 extra global-memory passes over the intermediate scores tensor per forward pass. This is exactly what flash attention was designed to eliminate.

**Status**: gap is real and matters. **Already tracked in [#108](https://github.com/dogkeeper886/ollama37/issues/108)** (investigate `fattn-vec` viability on K80). Don't double-track here.

### 4.2 MUL_MAT + bias ADD

When a model uses bias in the projection layers (Q, K, V, output, FFN gate/up/down), each matmul is followed by a separate ADD. From `llama-model.cpp:6662-6666`:

```cpp
ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
if (model.layers[il].bk) {
    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
}
```

**Cost**: one global-memory read + write per biased projection per layer.
**Not currently fused** by any GGML CUDA pattern.
**K80 impact**: small. The ADD reads/writes the projection output tensor, which is `[batch, seq_len, n_embd]` — a few MB. The matmul itself is 100-1000× heavier. A fused matmul+bias would save ~1% per biased projection.
**Affected models in our lineup**: Most modern models *don't* use bias (Llama, Gemma, Qwen). Some older or specialized models do.

### 4.3 MUL_MAT + residual ADD

Pattern: `MUL_MAT(W_o)` → `ADD(residual)`. The n-way fused_add pattern handles consecutive ADDs but **not** matmul+add.

**Cost**: one extra read/write of the projection output.
**K80 impact**: same order as 4.2 — small relative to the matmul cost. Each transformer layer has 2 such boundaries (after attention output projection and after FFN down projection).
**Implementation cost**: would require a new "MUL_MAT + ADD" fused kernel, or extending cuBLAS's `gemm` with `beta != 0` for the residual. Doable but not trivial — the matmul kernels are highly tuned per quantization type.

### 4.4 ROPE + KV cache write

Pattern: `ROPE(K)` → `CPY(K → kv_cache)`. Two passes over the K tensor.

**Cost**: one extra read/write per K (per layer per token).
**K80 impact**: small. K is `[batch, seq_len, n_kv_head, head_dim]` — typically 1-4 MB. Saving one round-trip per layer adds up but isn't a top contributor.
**Implementation cost**: would need a fused "ROPE + write to cache slot" kernel. Architecturally awkward because RoPE and cache write are conceptually separate.

### 4.5 Q-norm / K-norm (when present)

For models that use `use_kq_norm` (e.g., Llama4): RMS_NORM is applied to Q and K **after** ROPE. These are bare `ggml_rms_norm` calls without an immediately-following `MUL`, so they don't trigger the `RMS_NORM + MUL` fusion path.

**Cost**: 2 unfused RMS_NORM kernels per layer.
**K80 impact**: tiny. RMS_NORM is cheap; the missing fusion target (MUL of a learnable weight) doesn't apply for L2-norm-style Q/K normalization.
**Note**: this is correct behavior for these models — there's no weight to multiply by, so there's nothing to fuse with.

## 5. Verdict

**Don't open new fusion-implementation issues from this audit.** The breakdown:

- **Gap §4.1 (attention compute)** is the only fusion gap with a meaningful K80 perf impact, and it's already tracked in **#108** under a different framing (FA viability). Implementing it would deliver 50-90% of the available fusion gains.
- **Gaps §4.2-§4.5** are real but each saves only ~1-3% per layer. Combined they might add 5-10% on the FFN / projection path, but at the cost of writing custom CUDA kernels for K80 — significant engineering for incremental gains, on a project where the bigger lever (FA enablement) is already prioritized.
- **What's already fused (§2)** covers the common transformer patterns reasonably well. The fusion infrastructure is there (`ggml_cuda_can_fuse`, `GGML_CUDA_DISABLE_FUSION` env var) and we benefit from upstream maintenance.

### Optional micro-action

**`GGML_CUDA_DISABLE_FUSION=1` env var should be exposed in our debug skill docs.** Useful for measuring "what does fusion buy us today on K80?" — run an inference with fusion enabled, run again with it disabled, compare token/s. If the delta is larger than expected, it means the fused kernels are doing real work for our workload. If smaller, it means there's headroom for new fusions. This is a 1-line update to `.claude/skills/debug/SKILL.md`. Not opening an issue for this — file as nice-to-have if anyone touches the debug skill.

### What this audit did not cover

- **Inside-kernel fusion in `ggml_cuda_op_mul_mat`** — the matmul kernels (mmq, mmf, mmvf, mmvq) have internal selection logic for the fastest implementation per quantization type. There may be sub-optimal selections on sm_37 specifically (e.g., MMQ chosen when scalar would be faster). Out of scope here.
- **CPU-side overhead** — the dispatch loop (`evaluate_and_capture_cuda_graph`) iterates over every node every forward pass. CUDA graphs are used to amortize this, but their effectiveness on K80 is unmeasured.
- **Empirical confirmation** — without runtime data, we don't know which fusion paths actually fire most often on our specific models. A future profiling run (with proper nvprof setup) could confirm this audit's predictions.

## 6. References

- Dispatch + fusion logic: `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:3090` (`ggml_cuda_can_fuse`), `:3215` (graph evaluation loop), `:3227` (`GGML_CUDA_DISABLE_FUSION`)
- Fused implementations: `binbcast.cu` (n-way ADD), `norm.cu` (RMS_NORM variants), `softcap.cu` (Gemma softcap), `topk-moe.cu` (MoE routing)
- GLU activations: `unary.cu:292-308` (reglu, geglu, swiglu, geglu_erf, geglu_quick), `:343` (swiglu_oai)
- Op dispatch: `ggml_cuda_compute_forward` switch at `:2466` (full op list)
- Typical model graph: `llama/llama.cpp/src/llama-model.cpp:6680` (Llama4 attention), `:6726` (FFN via `build_ffn`)
- Sibling tracked issues: [#108](https://github.com/dogkeeper886/ollama37/issues/108) (fattn-vec on K80 — closes §4.1), [#107](https://github.com/dogkeeper886/ollama37/issues/107) (K-only KV quant — different concern, KV memory not kernel fusion)
- Part A audit: `docs/research/k80-quant-audit.md`
