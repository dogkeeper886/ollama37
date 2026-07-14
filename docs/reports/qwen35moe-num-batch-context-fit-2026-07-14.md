# qwen35moe num_batch × context fit map — K80 (2026-07-14)

Measurement backing the batch-size solution for **#440** (qwen35moe long-context
CPU-spill / OOM). Establishes, per context length, the largest `num_batch` that keeps
`qwen3.6:35b` **100% on GPU** on the K80 — and shows a smaller batch *unlocks 256k*.

## Setup

- **Model:** `qwen3.6:35b` — arch `qwen35moe`, 36B (A3B: 256 experts / 8 active), 40 layers
  (10 full-attention + 30 GatedDeltaNet), Q4_K_M ~23 GB weights, native ctx 262144.
- **Rig:** `sm37` = 4× Tesla K80 dies, 11441 MiB each (~44.7 GB aggregate). FA off (sm_37 < Volta),
  f16 KV.
- **Vehicle:** `test-mcp.yml` `verify-live` with a Playwright distractor menu → a fixed
  **~7,796-token** prompt (a realistic long tool-menu). One run per cell → verdict (correctness),
  `total_s` (prefill+decode; prompt is fixed so it tracks prefill), and `eval_tps` (decode).
- **Isolation:** all cells on **one no-clamp image** (`dogkeeper886/ollama37:latest`, honors an
  explicit `num_batch`), via a `num_batch` knob threaded into the MCP host — so batch is the only
  variable, no build confound. MCP does not recreate the container, so the image is constant.
- **Mechanism:** with FA off, each full-attention layer materializes a `Q·Kᵀ` score buffer sized
  `n_ctx × n_batch × n_head × 4B`; it is reserved at load (worst-case `num_batch`), so it grows
  with both context and batch and drives the spill/OOM.

## Fit map — `eval_tps · total_s · verdict`   (✅ all-GPU · ⚠️ CPU spill · ❌ OOM)

| ctx | nb=512 | nb=256 | nb=128 |
|--|--|--|--|
| 8k (8192)     | 6.2 · 340 s · ✅ | *pending* | *pending* |
| 96k (98304)   | 6.7 · 340 s · ✅ | 7.02 · 391 s · ✅ | *pending* |
| 128k (131072) | **1.94 · 490 s · ⚠️** | 6.86 · 391 s · ✅ | *pending* |
| 192k (196608) | *pending (spill/OOM)* | 6.85 · 392 s · ✅ | *pending* |
| 256k (262144) | ❌ OOM† | **1.27 · 1184 s · ⚠️** | **5.53 · 437 s · ✅** |

All verdicts PASS (grounded tool call) where the model loaded. † `256k·512` OOM is from the
throughput sweep (run 29303371831); a matching MCP cell is pending. *pending* cells not yet run.

MCP runs: 29327864333 (8k·512), 29328422098 (96k·512), 29328815453 (256k·256),
29330053918 (256k·128), + batch 29330xxx (96k·256, 128k·512, 128k·256, 192k·256).

## Per-die VRAM (MiB) — from the throughput sweep (nvidia-smi; MCP does not capture it)

VRAM/fit is reservation-driven (independent of the actual prompt length), so throughput and MCP
agree on fit at the same ctx×batch.

| ctx·nb | d0 | d1 | d2 | d3 |
|--|--|--|--|--|
| 8k·512   | 8970 | 8406 | 8550 | — |
| 64k·512  | 8919 | 9039 | 9035 | 8500 |
| 128k·512 | 10381 | 10578 | 10955 | 11154 |
| 128k·256 | 9279 | 9529 | 9527 | 8862 |
| 128k·128 | 10673 | 10355 | 10259 | — |
| 256k·512 | ❌ OOM | | | |

(Per-die VRAM for the remaining ctx×batch cells is not yet captured — the MCP host does not import
`perf/gpu.ts`; adding `gpuInfo()`/`gpuOffload()` there would record it uniformly. Pending decision.)

## Findings

1. **Monotonic ladder** — each halving of `num_batch` ~doubles the max context that stays 100% on GPU:
   `512 → ≤96k`, `256 → ≤192k`, `128 → 256k`.
2. **256k is unlocked** — OOM at 512, crawls at 256 (1.27 tok/s), but **healthy at 128 (5.53 tok/s)**.
   This is the video's headline context, on the K80.
3. **Prefill cost is modest** — healthy `total_s` for the 7.8k prompt: 512 ≈ 340 s → 256 ≈ 391 s
   (**+15%**) → 128 ≈ 437 s (+29%). ~15% per batch halving (clean cross-context; the 8k same-context
   ladder is pending to confirm at one point).

## Solution (proposed) — context-tiered batch (option C)

```go
// qwen35moe on K80 (FA off): score buffer Q·Kᵀ = n_ctx × n_batch × n_head overflows
// VRAM as context grows. Cap num_batch to the largest that stays 100% on GPU at this
// context — measured here (4× K80 / 44.7 GB): ≤96k→512 · ≤192k→256 · 256k→128.
maxBatch := 512
if opts.NumCtx > 98304  { maxBatch = 256 }
if opts.NumCtx > 196608 { maxBatch = 128 }
if architecture == "qwen35moe" && opts.NumBatch > maxBatch {
    opts.NumBatch = maxBatch      // only reduces; a smaller user num_batch is kept
}
```
Placement: `llm/server.go`, after `opts.NumBatch = min(opts.NumBatch, opts.NumCtx)`.

## Caveats / pending

- Thresholds (96k / 192k) are calibrated on the **4-die rig**; a single-K80 (24 GB) tiers lower.
  A VRAM-derived threshold would be portable (larger change).
- MCP does not capture per-die VRAM; the table above uses `eval_tps` as the fit proxy plus the
  throughput VRAM rows.
- Pending cells: 8k·{256,128}, 96k·128, 128k·128, 192k·{512,128}, 256k·512.
- Effect scope is `qwen35moe` only (arch string); the dense `qwen35` (27b/9b) is untested here.
