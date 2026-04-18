# KV Cache Rotation — Design Notes

**Issue:** [#102](https://github.com/dogkeeper886/ollama37/issues/102)
**Status:** In progress — scaffold committed, `attention.go` integration pending build-env availability.

## Goal

Apply a fixed orthogonal rotation (Sylvester Hadamard matrix, normalized by 1/√n) to Q/K/V activations before they enter the KV cache, so that existing quantized KV types (`q4_0`, `q8_0`) recover most of the fp16-vs-quantized quality gap.

Mirrors ggml-org/llama.cpp#21038 (commit `744c0c73`, merged 2026-04-01), which showed this alone closes the PPL gap on Qwen3.5-4B between `q4_0` KV and `fp16` to +0.22%.

## What's landed

- `envconfig.KVRotate` — `OLLAMA_KV_ROTATE=1` opt-in flag (default off).
- `ml/nn/hadamard.go` — pure-Go normalized Hadamard matrix generator + `IsHadamardCompatible(headDim)` gate.
- `ml/nn/hadamard_test.go` — orthogonality, normalization, symmetry, power-of-2 validation.

## What's pending (the actual integration)

### ollamarunner path — `ml/nn/attention.go`

The current `AttentionWithSinks` function (roughly):

```go
if key != nil && value != nil {
    cache.Put(ctx, key, value)
}
key, value, mask = cache.Get(ctx)
// ... attention math using query, key, value
```

Rotation wraps this:

```go
rotate := envconfig.KVRotate() && IsHadamardCompatible(query.Dim(0))

if rotate && key != nil && value != nil {
    H := hadamardTensor(ctx, 64)        // [64, 64] fp32, precomputed/cached
    query = blockRotate(ctx, query, H)  // applied along head dim, block-64 diagonal
    key   = blockRotate(ctx, key,   H)
    value = blockRotate(ctx, value, H)
    cache.Put(ctx, key, value)          // cache now stores rotated K, V
} else if key != nil && value != nil {
    cache.Put(ctx, key, value)
}

key, value, mask = cache.Get(ctx)
// attention math unchanged — dot products are preserved under orthogonal rotation

if rotate {
    // V rotation propagates to the attention output; undo on the way out.
    out = blockRotate(ctx, out, H)      // H is symmetric, so Hᵀ = H
}
```

### `blockRotate` helper

For head dim `d = 64·k`, applying `H_64` block-diagonally to a `[d, heads, seq]` tensor:

1. Reshape `[d, heads, seq]` → `[64, k·heads, seq]`.
2. Mulmat with `H_64`: result is `[64, k·heads, seq]`.
3. Reshape back to `[d, heads, seq]`.

This is the same trick upstream uses for V (fixed 64×64) applied to Q/K as well, at the cost of slightly less aggressive rotation than upstream's "largest pow-2 divisor" choice. The unified 64-wide rotation simplifies the tensor reshape bookkeeping.

### Caching the Hadamard tensor

Regenerating the matrix on every forward pass is wasteful. Options:

- **Per-context cache**: attach a `map[int]ml.Tensor` to the runner or a package-level `sync.Map`. Key by size (64 only in MVP).
- **Model-init-time**: add the Hadamard as a synthetic weight in the model structure. More invasive but matches upstream's "precomputed once per model" approach.

For MVP: per-backend-context cache (option 1). Upgrade later if it becomes a hot path.

### Gating

- Rotation applies only when **both** `OLLAMA_KV_ROTATE=1` **and** `IsHadamardCompatible(head_dim)` are true.
- Mismatched configurations (flag on, head_dim unsupported) log once at model load:
  ```
  slog.Info("kv rotation requested but head_dim is not a multiple of 64; disabling", "head_dim", d)
  ```
- Rotation state is sticky for the lifetime of a runner. Toggling the env var while a model is loaded does not mix rotated and unrotated cache entries.

### llamarunner path

Out of scope for Phase 1 implementation. Follow-up: either cherry-pick upstream commit `744c0c73` as `llama/patches/00NN-rotate-activations-for-better-quantization.patch` or bump the vendored `llama/llama.cpp/` past that commit. Track in a separate issue.

## Testing plan

### Unit (no runtime needed)

- [x] Hadamard generator: orthogonality, normalization, symmetry — landed in `hadamard_test.go`.
- [ ] `blockRotate` round-trip: `blockRotate(blockRotate(x, H), H) ≈ x` in fp32. Needs ml.Tensor test harness.

### Integration (K80, build env required)

- [ ] Model round-trip: generate with `OLLAMA_KV_ROTATE=0` and `=1`, compare perplexity on wikitext-2. Target: `q4_0` KV with rotation ≤ +0.3% PPL vs fp16.
- [ ] Overhead: decode tok/s for fp16 / q4_0 / q4_0+rotate. Target: rotation overhead < 10% on K80 `fattn-vec` path.
- [ ] Regression: `cicd/tests/testcases/` suites pass with rotation on.

### Validation against metrics

Once #98's `/api/metrics` endpoint lands, use the VRAM breakdown to verify KV cache size is unchanged (rotation doesn't affect quant bit-width — only quality). Useful sanity check.

## Out of scope for Phase 1

- TBQ3_0 / TBQ4_0 types (Phase 2, only if rotation + `q4_0` isn't sufficient).
- QJL residual correction.
- Asymmetric K/V type configuration (needs Cache API change — separate issue).
- Per-head rotation selection (upstream picks largest pow-2; we conservatively use 64).
