# num_batch clamp re-tune — fewest-dies criterion — K80 (2026-07-17)

Supersedes the "largest-batch" tiers in `qwen35moe-num-batch-context-fit-2026-07-14.md`,
`qwen35-dense-num-batch-context-fit-2026-07-16.md`, and `gptoss-num-batch-context-fit-2026-07-16.md`.
Backs the re-tune in #458 (`llm/server.go` `clampBatch`).

## The correction

The earlier clamps capped `num_batch` to the **largest** batch that stays 100% on GPU, on the
stated assumption that *decode is flat across batch*. Measured on the real MCP workload, that
assumption is **false**.

With flash attention hard-off on sm_37, each full-attention layer materializes a `Q·Kᵀ` score
buffer sized `n_ctx × n_batch × n_head`, reserved at load. A **larger batch inflates that buffer**,
which spreads the model across **more K80 dies**. The four dies talk over **PCIe**, so each extra
die costs **decode tok/s** — and VRAM, the scarce resource on this rig (a MoE keeps all experts
resident, so dies are what you run out of).

**New criterion:** cap `num_batch` to the **fewest-die batch that stays 100% on GPU**. Break a die
tie by measured tok/s (usually the larger batch — faster prefill). Take a larger, more-die batch
**only where a real number shows it decodes faster**.

## Setup

- **Rig:** `sm37` = 4× Tesla K80 dies, 11441 MiB each (~44.7 GB aggregate). FA off (sm_37 < Volta), f16 KV.
- **Vehicle:** MCP fit-map (`test-report-sweep.yml`, `suite=none`) — per-die VRAM + `activeDies` +
  `eval_tps` per (model, ctx, batch) cell (STORY-022).
- **Runs:** 29482994550 (gpt-oss 64k/96k), 29484930437 (35b/27b/9b `{256,128}`), 29530127306 (512
  low-tier), 29550110578 (35b @64k re-measure), 29550117574 (9b @256k @128).

## Measured — dies / decode tok-s per batch (✅ = 100% GPU)

### qwen3.6:35b (`qwen35moe`)
| ctx | nb=512 | nb=256 | nb=128 | pick |
|--|--|--|--|--|
| 32k | ✅ 3d / 6.98 | ✅ 3d / 6.83 | ✅ 3d / 5.64 | **512** (min dies, largest) |
| 64k | ✅ 4d / 6.80 | ✅ **3d / 7.06** | ✅ 3d / 5.64 | **256** (fewer dies & faster) |
| 96k | ✅ 4d / 6.47 | ✅ **3d / 6.93** | ✅ 3d / 5.67 | **256** (fewer dies & faster) |
| 128k | ⚠️ | ✅ **4d / 6.33** | ✅ 3d / 5.65 | **256** (more-die, but 6.33 > 5.65 measured) |
| 192k | ⚠️ | ✅ **4d / 6.62** | ✅ 4d / 5.58 | **256** (die tie → faster) |
| 256k | ❌ | ✅ 4d / 5.54 | ✅ **4d / 5.65** | **128** (die tie → faster) |

→ `clampBatch(32768, 196608)` — ≤32k:512 · ≤192k:256 · >192k:128

### qwen3.6:27b (`qwen35`, BlockCount ≥ 48)
| ctx | nb=512 | nb=256 | nb=128 | pick |
|--|--|--|--|--|
| 32k | ✅ 3d / 2.25 | ✅ 3d / 2.26 | ✅ 3d / 2.26 | **512** (min dies, largest) |
| 64k | ✅ 4d / 2.15 | ✅ **3d / 2.24** | ✅ 3d / 2.26 | **256** (fewer dies) |
| 96k | ⚠️/256 | ✅ 4d / 2.15 | ✅ **3d / 2.27** | **128** (fewer dies & faster) |
| 128k | ⚠️ | ✅ 4d / 2.15 | ✅ **4d / 2.16** | **128** (die tie → faster) |
| 192k / 256k | ❌ | ⚠️ spill | ⚠️ spill | out of scope (spills at every batch) |

→ `clampBatch(32768, 65536)` — ≤32k:512 · ≤64k:256 · >64k:128

### qwen3.5:9b (`qwen35`, BlockCount < 48)
| ctx | nb=512 | nb=256 | nb=128 | pick |
|--|--|--|--|--|
| 32k | ✅ 1d / 6.45 | ✅ 1d / 6.43 | ✅ 1d / 6.45 | **512** (min dies, largest) |
| 64k | ✅ **2d** / 5.50 | ✅ **1d** / 6.45 | ✅ 1d / 6.46 | **256** (fewer dies & faster) |
| 96k | ✅ 2d / 5.34 | ✅ 2d / 5.35 | ✅ 2d / 5.38 | **256** (2d tie → prefill; Δ within noise) |
| 128k | ✅ 2d | ✅ 2d / 5.32 | ✅ 2d / 5.38 | **256** (2d tie → prefill; Δ within noise) |
| 192k | 4d (old) | ✅ 2d / 5.23 | ✅ **2d / 5.36** | **128** (faster at equal 2d, avg of 2 runs) |
| 256k | ❌ | ✅ 3d / 5.10 | ✅ **2d** / 5.35 | **128** (fewer dies & faster) |

→ `clampBatch(32768, 131072)` — ≤32k:512 · ≤128k:256 · >128k:128

### gpt-oss:20b (`gptoss`, native 128k)
| ctx | nb=512 | nb=256 | nb=128 | pick |
|--|--|--|--|--|
| 32k | ✅ 2d / — | ✅ 2d | ✅ 2d | **512** (2-die weight floor, largest) |
| 64k | ⚠️ 86% | ✅ 3d / 8.14 | ✅ **2d / 8.72** | **128** (fewer dies & +7%) |
| 96k | ❌ | ✅ 4d / 8.14 | ✅ **2d / 8.86** | **128** (fewer dies & +9%) |
| 128k | ❌ | ⚠️ 85% | ✅ 3d | **128** (only 100%-GPU batch) |

→ `clampBatch(32768, 32768)` — ≤32k:512 · >32k:128

## Notes

1. **The 9b VRAM pattern is non-linear** (1 die ≤64k, 2 dies to 192k, 3 at 256k@256), and `nb=512`
   costs a die at 64k (2d vs 256's 1d) and 192k (4d vs 2d). From 96k–192k all of 256/128 sit at
   2 dies, so the pick is a pure decode-tiebreak: 128 measured ≥ 256 throughout and clearly faster
   at 192k (5.36 vs 5.23, avg of runs 29484930437/29557518952), so the clamp drops to 128 by 192k;
   96k/128k stay at 256, where the delta is within noise and the larger batch prefills faster.
2. **The 35b @64k cell was re-measured** (29550110578) after an outlier (256=6.23) in the first pass;
   the clean value (256 = 3d / 7.06 vs 512 = 4d / 6.80) makes 256 the unambiguous pick — fewer dies
   *and* faster.
3. **The 128k tier on the 35b is the one place a more-die batch wins** (256 = 4d / 6.33 beats
   128 = 3d / 5.65) — allowed because a real number shows it decodes faster.
4. **27b ≥192k stays out of scope** — its 2.4× score buffer spills at every batch; needs KV-quant/FA,
   not num_batch (unchanged from #452).
