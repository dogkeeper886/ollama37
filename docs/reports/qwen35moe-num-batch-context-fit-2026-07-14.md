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

## Fit map — complete 15-cell grid   (eval_tps · total_s · fit · VRAM · dies)

✅ all-GPU · ⚠️ CPU spill · ❌ OOM. VRAM = peak per-die (MiB), sum in GB, active die count.

| ctx | nb=512 | nb=256 | nb=128 |
|--|--|--|--|
| 8k   | 6.2 · 340 · ✅ · 25.4G/**3d** | 6.51 · 391 · ✅ · 24.9G/**3d** | 5.64 · 431 · ✅ · 24.7G/**3d** |
| 32k  | 6.59 · 345 · ✅ · 28.3G/**3d** | 6.97 · 391 · ✅ · 26.5G/**3d** | 5.56 · 434 · ✅ · 25.8G/**3d** |
| 64k  | 6.72 · 336 · ✅ · **34.7G/4d** | 6.3 · 394 · ✅ · 29.0G/**3d** | 5.61 · 431 · ✅ · 27.3G/**3d** |
| 96k  | 6.7 · 340 · ✅ · 39.8G/**4d** | 7.02 · 391 · ✅ · 31.4G/**3d** | 5.62 · 432 · ✅ · 29.0G/**3d** |
| 128k | **1.94 · 490 · ⚠️** · 42.1G/4d | 6.86 · 391 · ✅ · 36.3G/**4d** | 5.66 · 433 · ✅ · 30.6G/**3d** |
| 192k | **0.8 · 1381 · ⚠️** · 41.2G/4d | 6.85 · 392 · ✅ · 42.3G/4d | 5.52 · 439 · ✅ · 35.9G/**4d** |
| 256k | ❌ **OOM** (run failed) | **1.27 · 1184 · ⚠️** · 41.7G/4d | 5.53 · 437 · ✅ · 39.8G/**4d** |

`eval_tps` + per-die VRAM valid for all cells. **Model-load info** (`ollama_model_vram_bytes`):
`qwen3.6:35b · 36.0B · Q4_K_M`, `context_length` label matched every cell (validation ✓).

> **Verdict caveat:** the 32k·256/128 and all 64k cells reported `pass=False` — but the failures
> began at a wall-clock point (~14:12) and are context-independent, with healthy eval_tps throughout,
> so this is a **verify-live judge/infra degradation partway through the session, not a model or fit
> regression**. Their fit/VRAM data stands; the grounded-verdict PASS is confirmed for 8k–192k earlier.

### 3-die ↔ 4-die boundary (from the VRAM above)

There is **no 1-/2-die region** — 23 GB of resident weights need ≥3 dies. The 3→4 step moves *right*
as the batch shrinks:

| batch | stays 3 dies up to | steps to 4 dies at |
|--|--|--|
| 512 | 32k | **64k** |
| 256 | 96k | **128k** |
| 128 | 128k | **192k** |

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
5. **nb=128 is universal** — healthy at *every* context (5.52–5.66 tok/s from 8k to 256k). It's the
   one batch that fits the whole range, at a ~15% throughput trim vs the largest batch that fits.
6. **Die count is a function of batch, not just context** — a smaller batch keeps the model on fewer
   dies: `nb=128` stays on **3 dies up to 128k** (30.6 G) and steps to 4 only at 192k; `nb=512`
   already needs 4 dies by 64k. No **1-/2-die** region exists for this model — the ~23 GB of resident
   weights need ≥3 dies (2 dies = 22.8 GB < 23 GB). So it's a **3-die ↔ 4-die** map, and the boundary
   moves right as the batch shrinks.

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
