# K80 VRAM & Throughput by Model Family

How models across families place on the Tesla K80 — per-die VRAM, GPU vs CPU
placement, and throughput — swept over context length. Run with the #352 fix live
(`graphSafetyMultiplier` default 2.5). For tool-call capability see the companion
report `k80-mcp-by-family.md`.

## Setup

| | |
|---|---|
| Hardware | 4 × Tesla K80 · 11441 MiB/die (~45.8 GB pooled) |
| Engine | per-model (legacy llama.cpp or new ollama engine) |
| Flash attention | off (sm_37) |
| KV cache | f16 |
| `graphSafetyMultiplier` | 2.5 (default, #352) |
| Throughput | `num_predict=16` |
| Context sweep | 2k / 4k / 8k / 16k |
| Build | ollama `d83061de` · branch `main` — lfm2 (LFM2 MoE) support merged (#383, STORY-016) |
| Swept | 2026-07-10 — full 14-model re-sweep (13 baseline + lfm2.5:8b), one model per run |
| Version caveat | the binary self-reports `2.1.0`; the commit is recovered from the deploy pipeline (the build does not yet embed its git SHA — tracked follow-up) |

## Reading the metrics

- **GPU%** — layers on GPU (100% = fully on GPU; lower = CPU offload).
- **d0–d3** — per-die VRAM used (MiB). Shows the layer distribution across the four dies.
- **tok/s** — generation throughput.
- **Context (2k→16k)** — footprint = weights + KV(∝ctx) + graph(∝ctx)×2.5. The sweep maps
  each model's **fit cliff** (the context where it tips from full GPU to CPU offload). Cross-read
  with `k80-mcp-by-family.md`: an MCP fail there at high ctx while GPU% here is 100% is a
  capability/truncation result, **not** a speed/offload artifact — this column disambiguates.

> **No-harm vs `c282ba37`:** 49/51 shared (model × ctx) rows are within ±3 % on tok/s — the
> `lfm2` merge did not regress the other families. The two exceptions are **gemma4:26b** @8k
> (−8.8 %) and @16k (−3.2 %): on this build it now spreads across **all four dies** (was two),
> so cross-die traffic costs ~9 % — a gemma4-family placement shift from the gemma4/KV-graph
> changes merged since `c282ba37`, **not** lfm2 (which touches no GPU layer distribution).

---

## gemma4

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `gemma4:e2b`       | 2k  | 100% | 7352 | 7 | 7 | 7 | 24.85 |
|                    | 4k  | 100% | 7388 | 7 | 7 | 7 | 24.71 |
|                    | 8k  | 100% | 7418 | 7 | 7 | 7 | 25.15 |
|                    | 16k | 100% | 7495 | 7 | 7 | 7 | 25.03 |
| `gemma4:e4b`       | 2k  | 100% | 9796 | 7 | 7 | 7 | 15.02 |
|                    | 4k  | 100% | 9908 | 7 | 7 | 7 | 15.09 |
|                    | 8k  | 100% | 9992 | 7 | 7 | 7 | 15.09 |
|                    | 16k | 100% | 10155 | 7 | 7 | 7 | 15.04 |
| `gemma4:12b`       | 2k  | 100% | 4575 | 8735 | 7 | 7 | 6.45 |
|                    | 4k  | 100% | 4693 | 8791 | 7 | 7 | 6.29 |
|                    | 8k  | 100% | 4935 | 8903 | 7 | 7 | 6.23 |
|                    | 16k | 100% | 4155 | 3662 | 7983 | 7 | 6.42 |
| `gemma4:26b`       | 2k  | 100% | 9295 | 9288 | 7 | 7 | 14.37 |
|                    | 4k  | 100% | 9625 | 9472 | 7 | 7 | 14.19 |
|                    | 8k  | 100% | 5229 | 4919 | 5021 | 5422 | 13.24 ⚠️ |
|                    | 16k | 100% | 5416 | 5255 | 5325 | 5726 | 13.81 ⚠️ |
| `gemma4:31b`       | 2k  | 100% | 7657 | 7769 | 7922 | 7 | 3.04 |
|                    | 4k  | 100% | 8449 | 8577 | 8354 | 7 | 3.03 |
|                    | 8k  | 100% | 8962 | 8793 | 9058 | 7 | 3.07 |
|                    | 16k | 100% | 9682 | 9577 | 9776 | 7 | 3.08 |

> ⚠️ `gemma4:26b` @8k/16k now spreads across **all four dies** (was two dies on `c282ba37`) and
> drops from ~14.5 to ~13.2–13.8 tok/s (−9 % / −3 %). Cross-die traffic on the K80's slow links is
> the cost. Attributable to gemma4/KV-graph changes merged since `c282ba37`, not lfm2 — flagged for
> a follow-up.

---

## qwen3.6

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `qwen3.6:27b`      | 2k  | 100% | 10634 | 10514 | 7 | 7 | 2.94 |
|                    | 4k  | 100% | 10798 | 10678 | 7 | 7 | 2.94 |
|                    | 8k  | 100% | 11126 | 11004 | 7 | 7 | 2.95 |
|                    | 16k | 100% | 8344 | 8251 | 8222 | 7 | 3.02 |
| `qwen3.6:35b`      | 2k  | 100% | 8156 | 8720 | 8308 | 7 | 10.59 |
|                    | 4k  | 100% | 8236 | 8804 | 8390 | 7 | 10.71 |
|                    | 8k  | 100% | 8968 | 8404 | 8550 | 7 | 10.55 |
|                    | 16k | 100% | 9288 | 8740 | 8868 | 7 | 10.75 |

---

## deepseek-r1

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `deepseek-r1:1.5b` | 2k  | 100% | 2350 | 7 | 7 | 7 | 34.38 |
|                    | 4k  | 100% | 2406 | 7 | 7 | 7 | 34.07 |
|                    | 8k  | 100% | 2518 | 7 | 7 | 7 | 34.16 |
|                    | 16k | 100% | 2871 | 7 | 7 | 7 | 34.29 |
| `deepseek-r1:7b`   | 2k  | 100% | 6831 | 7 | 7 | 7 | 11.38 |
|                    | 4k  | 100% | 6943 | 7 | 7 | 7 | 11.11 |
|                    | 8k  | 100% | 7358 | 7 | 7 | 7 | 11.15 |
|                    | 16k | 100% | 8270 | 7 | 7 | 7 | 11.12 |
| `deepseek-r1:8b`   | 2k  | 100% | 5459 | 7 | 7 | 7 | 11.3 |
|                    | 4k  | 100% | 5879 | 7 | 7 | 7 | 11.12 |
|                    | 8k  | 100% | 6719 | 7 | 7 | 7 | 11.11 |
|                    | 16k | 100% | 8399 | 7 | 7 | 7 | 11.15 |
| `deepseek-r1:14b`  | 2k  | 100% | 4829 | 7974 | 7 | 7 | 5.86 |
|                    | 4k  | 100% | 5205 | 8227 | 7 | 7 | 5.87 |
|                    | 8k  | 100% | 5957 | 8947 | 7 | 7 | 5.76 |
|                    | 16k | 100% | 7461 | 10387 | 7 | 7 | 5.78 |
| `deepseek-r1:32b`  | 2k  | 100% | 7346 | 7242 | 9993 | 7 | 2.85 |
|                    | 4k  | 100% | 7698 | 7594 | 10222 | 7 | 2.84 |
|                    | 8k  | 100% | 8402 | 8298 | 10894 | 7 | 2.85 |
|                    | 16k | 100% | 8146 | 7644 | 7644 | 10574 | 2.76 |

---

## gpt-oss

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `gpt-oss:20b`      | 2k  | 100% | 7382 | 7556 | 7 | 7 | 16.69 |
|                    | 4k  | 100% | 7382 | 7556 | 7 | 7 | 16.72 |
|                    | 8k  | 100% | 7382 | 7556 | 7 | 7 | 16.52 |
|                    | 16k | 100% | 8520 | 8694 | 7 | 7 | 16.51 |

---

## lfm2

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `lfm2.5:8b`        | 2k  | 100% | 6358 | 7 | 7 | 7 | 33.02 |
|                    | 4k  | 100% | 6406 | 7 | 7 | 7 | 33.14 |
|                    | 8k  | 100% | 6722 | 7 | 7 | 7 | 32.96 |
|                    | 16k | 100% | 7346 | 7 | 7 | 7 | 32.97 |

> `lfm2.5:8b` (LFM2 MoE, 8B total / ~1B active) — new in this build (#383). Fully on a single die
> (~6.4 GB), flat ~33 tok/s across contexts — the fastest dense-feeling model in the fleet after the
> deepseek-r1:1.5b (34 tok/s), and ~5× gemma4:12b at similar footprint.

---
