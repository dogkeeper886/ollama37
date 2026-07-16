# gpt-oss:20b num_batch × context fit map — K80 (2026-07-16)

Measurement backing the `num_batch` clamp for arch `gptoss` (#453).

## Setup

- **Model:** `gpt-oss:20b` — arch `gptoss`, 20.9B MoE (32 experts), **MXFP4 ~13.8 GB**, 24 blocks,
  heads 64 / KV 8 (GQA), key_length 64, sliding_window 128, **native ctx 131072 (128k)**.
- **Rig:** `sm37` = 4× Tesla K80 dies, 11441 MiB each (~44.7 GB aggregate). FA off (sm_37 < Volta), f16 KV.
- **Mechanism:** with FA off, the full-attention (odd) layers materialize a `Q·Kᵀ` score buffer reserved
  at load — ~16 GiB at 128k/nb=512, which **exceeds a single 11.4 GiB die**, so nothing fits on GPU and
  the model runs 0% on GPU (CPU, ~0.3 tok/s). `GraphSize` already bounds the SWA (even) layers' KV
  correctly, so KV is small (~3 GiB); the score buffer, which scales with `num_batch`, is the driver —
  making it clampable (unlike gemma4, which is KV-bound and FA-ceiling, #451 wontfix).

## Fit map — offload% (direct loads; 100% = fully on GPU)

| ctx | nb=512 | nb=256 | nb=128 |
|--|--|--|--|
| 32k | ✅ 100% | ✅ 100% | ✅ 100% |
| 64k | ⚠️ 86% | ✅ 100% | ✅ 100% |
| 96k | ❌ 0% | ✅ 100% | ✅ 100% |
| 128k | ❌ 0% | ⚠️ 85% | ✅ 100% |

## Clamp-selected cells — per-die VRAM + throughput (100% GPU)

The clamp picks the largest `num_batch` that stays 100% on GPU at each context:

| clamp cell | decode tok/s | prefill tok/s | per-die used (MiB) | dies / total |
|--|--|--|--|--|
| 32k @nb=512 | 14.93 | 45.1 | 10796, 10970, 7, 7 | 2 / 21.3 G |
| 64k @nb=256 | 14.62 | 45.0 | 9091, 9211, 8685, 7 | 3 / 26.4 G |
| 96k @nb=256 | 14.32 | 45.4 | 9892, 9890, 10356, 10531 | 4 / 39.7 G |
| 128k @nb=128 | 14.68 | 44.9 | 9207, 9671, 9848, 7 | 3 / 28.1 G |

## Solution — clamp (`llm/server.go`)

`case "gptoss", "gpt-oss": clampBatch(32768, 98304)` → **≤32k→512 · ≤96k→256 · >96k→128**.

## Findings

1. **Spills earliest of the K80 families** — 512 fits only ≤32k (the 16 GiB score buffer vs 11.4 GiB dies).
   But `nb=128` is 100% at every context including native 128k, so the clamp fully rescues it.
2. **Clamping is free on decode** — decode is flat ~14.3–14.9 tok/s across the clamp; the only alternative
   at high ctx is a CPU spill at ~0.3 tok/s (≈50× worse). So max-batch-at-100%-GPU costs ~nothing.
3. **Margin-thinnest tier: 96k@256** — 4 dies near the ceiling (peak 10.5 / 11.4 G). Measured 100% and
   stable (score buffer fixed at load, KV bounded by ctx), but the tightest cell in the map.
