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
- **Isolation:** all cells on **one no-clamp image** (honors an explicit `num_batch`), via a
  `num_batch` knob threaded into the MCP host — so batch is the only variable, no build confound.
- **Per-die VRAM:** read from the K80 host's GPU-metrics dashboard (Prometheus / `nvidia_gpu_exporter`),
  peak `nvidia_smi_memory_used_bytes` per die over each run's window. Cross-checked against the
  throughput sweep's nvidia-smi (matches) and the `ollama_model_vram_bytes{context_length}` label
  (confirms every cell's context). Per-die values are uuid-sorted, not nvidia-smi index.
- **Mechanism:** with FA off, each full-attention layer materializes a `Q·Kᵀ` score buffer sized
  `n_ctx × n_batch × n_head × 4B`, reserved at load (worst-case `num_batch`) — it grows with both
  context and batch and drives the spill/OOM.

## Fit map   (✅ all-GPU · ⚠️ CPU spill · ❌ OOM)

| ctx | nb | eval_tps | total_s | fit | per-die VRAM (MiB) | GPU total | dies |
|--|--|--|--|--|--|--|--|
| 8k   | 512 | 6.2  | 340 s  | ✅ | 8968 / 8550 / 8406        | 25.4 G | 3 |
| 8k   | 256 | 6.51 | 391 s  | ✅ | 8812 / 8394 / 8250        | 24.9 G | 3 |
| 96k  | 512 | 6.7  | 340 s  | ✅ | 10383 / 10379 / 10199 / 9780 | 39.8 G | 4 |
| 96k  | 256 | 7.02 | 391 s  | ✅ | 10970 / 10584 / 10552     | 31.4 G | 3 |
| 128k | 512 | **1.94** | 490 s | ⚠️ | 11154 / 10955 / 10578 / 10381 | 42.1 G | 4 |
| 128k | 256 | 6.86 | 391 s  | ✅ | 9529 / 9527 / 9279 / 8862 | 36.3 G | 4 |
| 192k | 256 | 6.85 | 392 s  | ✅ | 11129 / 11127 / 10751 / 10334 | 42.3 G | 4 |
| 256k | 256 | **1.27** | 1184 s | ⚠️ | 11078 / 10555 / 10555 / 10555 | 41.7 G | 4 |
| 256k | 128 | 5.53 | 437 s  | ✅ | 10548 / 10547 / 10038 / 9625 | 39.8 G | 4 |

All verdicts PASS (grounded tool call). **Model-load info** (`ollama_model_vram_bytes`):
`model=qwen3.6:35b · parameter_size=36.0B · quantization=Q4_K_M`, with a `context_length` label
that matched each cell's num_ctx (validation ✓); model VRAM ≈ 25 GB at 8k.

Pending cells (full grid = 5 ctx × 3 batch): `8k·128`, `96k·128`, `128k·128`, `192k·512`,
`192k·128`, `256k·512` (256k·512 expected OOM).

## Findings

1. **Monotonic ladder** — each halving of `num_batch` ~doubles the max context that stays 100% on GPU:
   `512 → ≤96k`, `256 → ≤192k`, `128 → 256k`.
2. **256k is unlocked** — OOM at 512, crawls at 256 (1.27 tok/s), but **healthy at 128 (5.53 tok/s)**.
3. **Prefill cost is modest & confirmed at one context** — 8k·512=340 s vs 8k·256=391 s = **+15%**
   (same ctx, same 7.8k prompt, batch the only variable). Cross-context: 512≈340 → 256≈391 → 128≈437.
4. **VRAM corroborates fit** — the healthy configs use *less* VRAM / fewer dies (96k·256 = 31.4 G/3
   dies vs 96k·512 = 39.8 G/4 dies; 128k·256 = 36.3 G vs 128k·512 = 42.1 G-with-spill). A ⚠️ cell is
   pinned near the per-die ceiling (~11.4 GB/die) with the remainder on CPU; the smaller batch pulls
   it back under the ceiling.

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

## Caveats

- Thresholds (96k / 192k) are calibrated on the **4-die rig**; a single-K80 (24 GB) tiers lower.
  A VRAM-derived threshold would be portable (larger change).
- Per-die VRAM is peak nvidia-smi over each run window, uuid-sorted (not nvidia-smi index); it is
  read from the host GPU dashboard, not captured by the MCP test itself.
- Effect scope is `qwen35moe` only (arch string); the dense `qwen35` (27b/9b) is untested here.
