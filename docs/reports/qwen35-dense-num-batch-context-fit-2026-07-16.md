# qwen35 (dense) num_batch × context fit map — K80 (2026-07-16)

Measurement backing the `num_batch` clamp for the **dense** `qwen35` arch — the 27b (#452)
and 9b (#455) siblings of `qwen35moe` (#440). Establishes, per context length, the largest
`num_batch` that keeps each model **100% on GPU** on the K80.

## Setup

- **Models:** `qwen3.6:27b` / `qwen3.5:27b` (arch `qwen35`, 27.8B, Q4_K_M ~17.4 GB, 64 blocks)
  and `qwen3.5:9b` (arch `qwen35`, 9.7B, ~6.6 GB, 32 blocks). GatedDeltaNet hybrid: full
  attention every 4th layer (`full_attention_interval=4`), the rest recurrent (fixed-state).
- **Rig:** `sm37` = 4× Tesla K80 dies, 11441 MiB each (~44.7 GB aggregate). FA off (sm_37 < Volta), f16 KV.
- **Vehicle:** throughput bench (short prompt) + fit-map (MCP) for the num_batch grid — per-die
  VRAM + offload% via `perf/gpu.ts`, no external dashboard (STORY-022).
- **Mechanism:** with FA off, each full-attention layer materializes a `Q·Kᵀ` score buffer sized
  `n_ctx × n_batch × n_head × 4B`, reserved worst-case at load — it grows with both context and
  batch and drives the spill/OOM. The recurrent layers carry fixed-size state (context-independent).

## Full-attention structure (the spill driver)

`model/models/qwen35/model.go:inferRecurrentLayers` + GGUF `full_attention_interval`:

| Model | blocks | full-attn layers | Q heads | score-buffer weight (layers × heads) |
|--|--|--|--|--|
| qwen3.6:35b (`qwen35moe`) | 40 | 10 | 16 | 160 |
| **qwen3.6:27b** (`qwen35`) | 64 | **16** | **24** | **384** ← 2.4× the 35b |
| qwen3.5:9b (`qwen35`) | 32 | 8 | 16 | 128 |

The 27b materializes ~2.4× the 35b's score buffer per token **despite being the smaller model** —
which is why it spills far earlier and why batch-halving can't reach 256k for it.

## Fit maps (fit · per-die VRAM · dies · tok/s)

### qwen3.6:27b — full grid (run 29413759899)

| ctx | nb=512 | nb=256 | nb=128 |
|--|--|--|--|
| 8k | ✂️ SAT¹ 25.4G/3d | ✂️ SAT¹ | ✂️ SAT¹ |
| 64k | ✅ 34.7G/4d | ✅ 29.6G/3d | ✅ 27.1G/3d |
| 96k | ⚠️ 91% 43.1G/4d | ✅ 36.7G/4d | ✅ 30.3G/3d |
| 128k | ⚠️ 77% 43.4G/4d | ✅ 42.0G/4d | ✅ 35.7G/4d |
| 192k | ❌ OOM | ⚠️ 78% 42.6G/4d | ⚠️ 94% 41.8G/4d |
| 256k | ❌ OOM | ⚠️ 64% 42.3G/4d | ⚠️ 78% 41.6G/4d |

¹ 8k prompt ≈ the 8k window → KV saturation; VRAM valid, tok/s invalid (STORY-022 #449).

### qwen3.5:9b — high context (runs 29416450389, 29467635730, 29468072044)

| ctx | nb=512 | nb=256 | nb=128 |
|--|--|--|--|
| 128k | ✅ 20.5G/2d, 8.2 t/s | — | — |
| 192k | ✅ 41.5G/4d, 8.5 t/s | — | — |
| 256k | ❌ 0% (CPU fallback), 0.32 t/s | ✅ 30.4G/3d, 4.9 t/s | ✅ 21.0G/2d, 5.4 t/s |

## Solution — context-tiered clamp per size (`llm/server.go`)

`num_batch` capped to the largest that stays 100% on GPU, gated by arch + `BlockCount`:

| Class | Gate | Tiers |
|--|--|--|
| 35b | `qwen35moe` | ≤96k→512 · ≤192k→256 · >192k→128 |
| 27b | `qwen35` & `BlockCount≥48` | ≤64k→512 · ≤128k→256 · >128k→128 |
| 9b | `qwen35` & `BlockCount<48` | ≤192k→512 · >192k→256 |

## Findings

1. **27b — recovered ≤128k, 256k out of reach by batch.** The clamp pulls 96k/128k from CPU
   spill to 100% GPU (verified, run 29466751889). 256k spills at every batch (78% even at nb=128) —
   the 2.4× score buffer needs KV-quant/FA, not batch. Out of scope.
2. **9b — fits 256k at nb≤256.** Fits at the default 512 through 192k; only 256k@512 overflows to
   CPU. `nb=256` restores 100% GPU (30.4 G / 3 dies); `nb=128` is roomier still (21.0 G / 2 dies).
3. **Not a bug.** The `fs/ggml/ggml.go` estimate bounds the recurrent layers to fixed state
   correctly; the 256k@512 fallback is the real score buffer overflowing, and it is batch-fixable.
4. **Size-specific thresholds.** The 27b spills at 96k, the 9b only at 256k — same arch, different
   boundaries — so a single threshold set can't serve both; `BlockCount` splits them.
