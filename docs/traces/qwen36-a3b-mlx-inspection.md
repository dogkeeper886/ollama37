# qwen3.6:35b-mlx (A3B) — model inspection trace

Captured 2026-05-26 from `mlx-community/Qwen3.6-35B-A3B-4bit` on Hugging Face, the
4-bit uniform MLX variant of the A3B (active-3B MoE) target for the MLX-on-K80
port (#187, #188).

## Source

| Field | Value |
|---|---|
| HF repo | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Format | MLX safetensors (4 shards) |
| Total size | 19.03 GB |
| Tokenizer | HF tokenizer.json (19 MB), vocab.json (6.4 MB) |
| Tensor count | 2090 across 4 shards |
| Local path | `/home/jack/models/qwen3.6-35b-a3b-4bit/` |

## Why HF instead of Ollama library

`ollama pull qwen3.6:35b-mlx` returns `412: this model requires macOS`. Ollama's
library enforces an OS gate on MLX models on the server side — our fork's whole
point is to break that assumption, but the gate is upstream. HF mirrors don't
have this restriction.

When the runner (#189) loads, it'll point at the local HF dir directly. We don't
need to round-trip through ollama's manifest system for Phase D.

## Architecture (from config.json)

- `architectures: ["Qwen3_5MoeForConditionalGeneration"]`
- `model_type: "qwen3_5_moe"`
- This is the MoE variant of the Qwen3_5 (DeltaNet hybrid) base. So **A3B = DeltaNet + MoE**, not pure-transformer + MoE.

## Quantization

```json
"quantization": {
  "group_size": 64,
  "bits": 4,
  "mode": "affine",
  // per-tensor overrides for sensitive MoE router weights:
  "language_model.model.layers.0.mlp.gate":         {"group_size": 64, "bits": 8},
  "language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
  // ... (40 layers × 2 overrides each)
}
```

- **Default**: `mode: "affine"`, `bits: 4`, `group_size: 64`. Matches our
  `affine_dequantize` (PR #196) directly — bits=4 is in the {2,4,8} regular
  case my K80 dequant fallback (#197) handles.
- **Per-tensor overrides**: every MoE router layer (`mlp.gate` and
  `mlp.shared_expert_gate`) uses `bits: 8` instead of 4. Still affine, still
  group_size=64 → still in my supported set.
- **Loader implication**: the runner has to read `config.json`'s quantization
  block per-tensor before constructing `QuantizedMatmul` primitives; can't
  assume uniform bits.

## Tensor patterns (canonicalized, layer-index collapsed to `*`)

### Language model — 40 layers split into two attention types

| Type | Count | Tensors per layer |
|---|---|---|
| **DeltaNet (linear_attn)** | 30 | `A_log`, `conv1d.weight`, `dt_bias`, `in_proj_a/b/qkv/z.{weight,scales,biases}`, `norm.weight`, `out_proj.{weight,scales,biases}` |
| **Self-attention** | 10 | `self_attn.{q,k,v,o}_proj.{weight,scales,biases}`, `self_attn.{q,k}_norm.weight` |

All layers also have:
- `input_layernorm.weight`
- `post_attention_layernorm.weight`

### MoE FFN (every layer)

| Component | Tensors |
|---|---|
| Router (8-bit) | `mlp.gate.{weight,scales,biases}` |
| Shared expert (always-on) | `mlp.shared_expert.{gate,up,down}_proj.{weight,scales,biases}` |
| Shared expert gate (8-bit) | `mlp.shared_expert_gate.{weight,scales,biases}` |
| Routed experts (switch_mlp) | `mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` |

The `switch_mlp.*` tensors are stacked across experts — single tensor shape
`(num_experts, ...)` per projection. Inference path: router output picks active
experts → gather their slices → batched GEMM. This is exactly what
`cutlass_grouped_gemm_unaligned` / `cutlass_gather_mm` would do; both are
currently throw-stubs in `k80_runtime_stubs.cpp` and will need real impls
(cuBLAS-per-expert or similar).

### Embed + LM head + final norm

- `language_model.model.embed_tokens.{weight,scales,biases}` — **quantized**.
  So embedding lookup goes `take` → quantized read → dequant first.
- `language_model.lm_head.{weight,scales,biases}` — quantized.
- `language_model.model.norm.weight` — final RMS norm.

### Vision tower (327 tensors, ignored for text inference)

27 vision_tower blocks (attn + mlp) + patch_embed + pos_embed + merger.
Multimodal model; we skip these for the text-only forward pass.

## Implication for the K80 punch list

Cross-referencing the tensor patterns against `k80_runtime_stubs.cpp`:

| What the model uses | Throw-stub status | Notes |
|---|---|---|
| Embedding lookup (`take` on quantized embed_tokens) | `Gather::eval_gpu` is stub | First call any forward pass makes. Will trip immediately. |
| Quantized linears (Q/K/V/O proj, FFN proj, lm_head) | **landed** PRs #196 + #197 | affine_dequantize + dequant→cuBLAS fallback wired |
| Standard attention (10 layers) | `Matmul`, `softmax`, `rms_norm`, `rope`, `sdpa` already in libmlx.a | should work |
| **DeltaNet recurrence (30 layers)** | `Scan::eval_gpu` is stub | Big port — state-space recurrence chunks; no upstream non-JIT version |
| **MoE expert routing** | `cutlass_grouped_gemm_unaligned` + `cutlass_gather_mm` stubs | Reroute to cuBLAS-per-expert; needs Gather/Scatter primitives in place first |
| KV cache reads/writes | `Gather`/`Scatter`/`SliceUpdate` stubs | Standard attention path |

## Why I'm not porting more kernels speculatively

Upstream `Gather::eval_gpu` (`mlx/backend/cuda/indexing.cpp`) is **deeply tied
to NVRTC + JitModule** — it generates kernel names dynamically per `dtype ×
NIDX × IDX_NDIM × LocT` combination and invokes them through the JIT cache.
Three options to port:

1. Bring up the JIT subsystem (NVRTC support, jit_module.cpp, the generated
   `cuda_jit_sources.h`). Large; not justified for one kernel.
2. Write a static-template version with pre-instantiated combinations
   covering only the shapes A3B needs. Tractable but requires data on what
   shapes the runner actually issues — which means having the runner first.
3. Skip until the runner trips it at runtime, then port the exact case
   that fires.

Same shape for `Scan`, `cutlass_grouped_gemm`, `Scatter`. Each port without
data wastes effort on cases that may never run. Option 3 is the discipline.

## Next-session priorities

1. **Runner skeleton (#189)** is now the critical path. Need Go cgo wrapper
   around `libmlx.a` + `x/models/qwen3_6_a3b` model definition + tokenizer
   binding + sampling. Big chunk — probably the largest single piece of
   remaining work.
2. As the runner exercises the forward pass, kernel stubs trip in priority
   order. Port real impls then, with concrete shape requirements known.
3. Memory topology: A3B 4-bit is 19 GB but **active params per token are
   ~3 B**. Sharded loading + per-token expert paging might let one K80 die
   (12 GB) hold enough hot weights, with cold experts on host RAM streamed
   in. This is the unproven part.
