# KV Cache Rotation — Design Notes

**Issue:** [#102](https://github.com/dogkeeper886/ollama37/issues/102)
**Status:** Phase 1 ollamarunner integration landed always-on for compatible head dims. llamarunner cherry-pick of upstream commit `744c0c73` is the next deliverable, tracked separately in #134.

## Goal

Apply a fixed orthogonal rotation (Sylvester Hadamard matrix, normalized by 1/√n) to Q/K/V activations before they enter the KV cache. This smooths per-coordinate distributions so that existing quantized KV types (`q4_0`, `q8_0`) recover most of the fp16-vs-quantized quality gap.

Mirrors ggml-org/llama.cpp#21038 (commit `744c0c73`, merged 2026-04-01), which showed the rotation alone closes the PPL gap on Qwen3.5-4B between `q4_0` KV and `fp16` to +0.22 %.

## What's landed

- `ml/nn/hadamard.go` — pure-Go normalized Hadamard matrix generator + `IsHadamardCompatible(headDim)` gate, package-level cached `hadamard64` slice, `blockRotate` helper, and `hadamardTensor(ctx)` to materialize the matrix on the active backend.
- `ml/nn/hadamard_test.go` — orthogonality (every n in 1..128), normalization, symmetry, power-of-2 panic, and `IsHadamardCompatible` cases.
- `ml/nn/attention.go` — `AttentionWithSinks` rotates Q/K/V before `cache.Put` and undoes the rotation on the attention output. Always-on when `cache != nil && IsHadamardCompatible(query.Dim(0))` — no env flag.
- `.github/workflows/test-kv-rotate.yml` — smoke workflow that runs the K80 killer combo (FA on + q8_0 KV) against a head_dim-256 model and asserts coherent output.

## Why no env flag

The env-gated approach was scaffolding that briefly let us merge the code without committing to flipping it on. After end-to-end smoke validation showed the rotation produces correct output, the flag became dead weight — same trajectory as `OLLAMA_FLASH_ATTENTION_K80` (PR #112 → productized in PR #117, flag deleted).

The transform is exact in fp32 (orthogonal: H·Hᵀ = I; symmetric: H = Hᵀ ⇒ H·H = I). When KV is fp16, rotation is small overhead with zero quality benefit — but mathematically it's still a no-op. The cost is bounded (one 64×64 matmul per Q/K/V/output per layer per forward pass), so paying it unconditionally is the simpler design.

## Tensor layout convention

`ml.Tensor` follows ggml column-major ordering. Shapes for attention inputs post-RoPE are:

- Q: `[head_dim, heads, seq_len_q]` — `Dim(0)` is head dim
- K: `[head_dim, kv_heads, seq_len_k]`
- V: `[head_dim, kv_heads, seq_len_k]` (or permuted via `PermutedV`; see `kvcache/causal.go:617,622,624`)

`blockRotate` rotates along `Dim(0)` (head dim).

## Ordering: rotation must happen AFTER RoPE

**Critical foot-gun.** Rotation must be applied *after* RoPE has been applied to Q and K. Applying the Hadamard before RoPE scrambles the position-dependent rotation RoPE introduces, and the resulting transform is not recoverable. Upstream llama.cpp commit `744c0c73` applies H after `ggml_rope`; we do the same — `AttentionWithSinks` runs after the model's RoPE step, so this ordering is enforced by where the function sits in the call graph.

## Math

```
Q' = Q · H   (rotated query for THIS turn)
K' = K · H   (rotated key, written to cache)
V' = V · H   (rotated value, written to cache)

Attention scores: Q' · (K')ᵀ
                = (Q·H) · (K·H)ᵀ
                = Q · H · Hᵀ · Kᵀ
                = Q · I · Kᵀ                 (orthogonality: H·Hᵀ = I)
                = Q · Kᵀ                     ← scores preserved exactly

Output:           softmax(Q'·(K')ᵀ / √d) · V'
                = softmax(Q·Kᵀ / √d) · V · H
                = (raw_output) · H

Undo:             out = (raw_output · H) · H
                       = raw_output · I       (symmetry: H·H = I for Sylvester)
```

Across turns, K/V from previous decode steps are stored rotated; the rotation gate is structural (depends only on head_dim and cache presence) so all writes use the same transform consistently.

## blockRotate — block-diagonal application

For head dim `d = 64·k`, applying `H_64` block-diagonally to a `[d, heads, seq]` tensor:

1. Reshape `[d, heads, seq]` → `[64, k·heads, seq]`.
2. `Mulmat` with `H_64`: result is `[64, k·heads, seq]`.
3. Reshape back to `[d, heads, seq]`.

This is the same trick upstream uses for V (fixed 64×64) applied to Q/K as well, at the cost of slightly less aggressive rotation than upstream's "largest pow-2 divisor" choice. The unified 64-wide rotation simplifies the tensor reshape bookkeeping. If MVP results show this matters, we can match upstream's per-head choice in a follow-up.

## Hadamard tensor materialization

The `hadamard64` float slice is computed once at package init. Per call, `hadamardTensor(ctx)` does a small CPU→GPU upload (16 KiB) — cheap enough not to bother caching the tensor across calls (each call may target a different backend context with its own allocator).

## Compatibility

- **Compatible head dims**: any multiple of 64 (≥ 64). Covers gemma3 (256), qwen3-vl (128), gpt-oss (64), most modern transformers.
- **Excluded head dims**: 80, 96, 112 — some Llama-1/2 variants, MPT-7B. Rotation is silently skipped for these (gate returns false). No warning is emitted because it's a structural property of the model, not a user request that we're failing to honor.
- **No cache**: rotation skipped (no quantization step to benefit from; rotating Q without rotating K would produce garbage).

## Performance

Per attention call, rotation adds:

- 1 small upload per attention call (16 KiB CPU→GPU)
- 4 matmuls: Q rotation, K rotation, V rotation, output un-rotation
- Each matmul is `[64, k·heads·seq] @ [64, 64]`

For decode (batch=1, seq=1) on a 32-layer model: ~100-200 µs per token added on K80 — negligible.
For prefill (seq=2048): maybe 50-100 ms total — noticeable but within the design-doc target (<10 % overhead vs un-rotated).

## llamarunner path

Out of scope for Phase 1 implementation. Tracked in #134. Either cherry-pick upstream commit `744c0c73` as `llama/patches/00NN-rotate-activations-for-better-quantization.patch` or bump the vendored `llama/llama.cpp/` past that commit. Cherry-pick is preferred unless there's a separate reason to bump the vendor tree.

## Testing

- **Unit (landed)**: Hadamard generator orthogonality, normalization, symmetry, power-of-2 panic. `hadamard_test.go`.
- **Integration smoke (landed)**: `.github/workflows/test-kv-rotate.yml` — FA on + q8_0 KV against gemma3:4b, asserts non-empty coherent response, no panics, no CUDA errors.
- **Perplexity (deferred)**: `q4_0` KV with rotation vs fp16 baseline on wikitext-2. Target: ≤ +0.3 % PPL delta. Needs offline tooling not yet in CI.
- **Overhead (deferred)**: decode tok/s comparison fp16 / q4_0 / q4_0+rotate on K80. Target: rotation overhead < 10 %.

## Out of scope for Phase 1

- TBQ3_0 / TBQ4_0 quant types (Phase 2, only if rotation + `q4_0` isn't sufficient).
- QJL residual correction.
- Asymmetric K/V type configuration (needs Cache API change — separate issue, #107).
- Per-head rotation selection (upstream picks largest pow-2 divisor of head_dim; we conservatively use 64).
- Bumping default KV cache type to `q4_0` — viable now that rotation is structural; tracked as a follow-up to PR #137.
