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
| Swept | 2026-06-23 — one model per run |

## Reading the metrics

- **GPU%** — layers on GPU (100% = fully on GPU; lower = CPU offload).
- **d0–d3** — per-die VRAM used (MiB). Shows the layer distribution across the four dies.
- **tok/s** — generation throughput.
- **Context (2k→16k)** — footprint = weights + KV(∝ctx) + graph(∝ctx)×2.5. The sweep maps
  each model's **fit cliff** (the context where it tips from full GPU to CPU offload). Cross-read
  with `k80-mcp-by-family.md`: an MCP fail there at high ctx while GPU% here is 100% is a
  capability/truncation result, **not** a speed/offload artifact — this column disambiguates.

---

## gemma4

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `gemma4:e2b` | 2k  | 100% | 7352 | 7 | 7 | 100 | 14.71 |
|              | 4k  | 100% | 7388 | 7 | 7 | 7 | 24.92 |
|              | 8k  | 100% | 7418 | 7 | 7 | 7 | 25.11 |
|              | 16k | 100% | 7495 | 7 | 7 | 7 | 25.1 |
| `gemma4:e4b` | 2k  | 100% | 9796 | 7 | 7 | 7 | 15.11 |
|              | 4k  | 100% | 9908 | 7 | 7 | 7 | 15.21 |
|              | 8k  | 100% | 9992 | 7 | 7 | 7 | 15.19 |
|              | 16k | 100% | 10155 | 7 | 7 | 7 | 15.24 |
| `gemma4:12b` | 2k  | 100% | 4575 | 8735 | 7 | 7 | 6.32 |
|              | 4k  | 100% | 4693 | 8791 | 7 | 7 | 6.18 |
|              | 8k  | 100% | 4935 | 8903 | 7 | 7 | 6.18 |
|              | 16k | 100% | 4155 | 3662 | 7983 | 7 | 6.38 |
| `gemma4:26b` | 2k  | 100% | 9295 | 9288 | 7 | 7 | 14.39 |
|              | 4k  | 100% | 9623 | 9472 | 7 | 7 | 14.32 |
|              | 8k  | 100% | 9852 | 9558 | 7 | 7 | 14.43 |
|              | 16k | 100% | 10188 | 9814 | 7 | 7 | 14.31 |
| `gemma4:31b` | 2k  | 100% | 7659 | 7769 | 7922 | 7 | 3.08 |
|              | 4k  | 100% | 8449 | 8577 | 8354 | 102 | 2.8 |
|              | 8k  | 100% | 8962 | 8793 | 9056 | 7 | 3.08 |
|              | 16k | 100% | 9682 | 9577 | 9778 | 7 | 3.07 |

---

## qwen3.6

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `qwen3.6:27b` | 2k  | 100% | 10634 | 10514 | 7 | 7 | 2.96 |
|               | 4k  | 100% | 10798 | 10678 | 7 | 7 | 2.98 |
|               | 8k  | 100% | 11126 | 11004 | 7 | 7 | 2.97 |
|               | 16k | 100% | 8344 | 8251 | 8222 | 7 | 3.04 |
| `qwen3.6:35b` | 2k  | 100% | 8156 | 8720 | 8308 | 7 | 10.71 |
|               | 4k  | 100% | 8236 | 8804 | 8390 | 7 | 10.41 |
|               | 8k  | 100% | 8968 | 8404 | 8548 | 7 | 10.56 |
|               | 16k | 100% | 9288 | 8740 | 8868 | 7 | 10.54 |

---

## deepseek-r1

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `deepseek-r1:1.5b` | 2k  | 100% | 2350 | 7 | 7 | 7 | 34.28 |
|                    | 4k  | 100% | 2406 | 7 | 7 | 7 | 34.28 |
|                    | 8k  | 100% | 2518 | 7 | 7 | 7 | 34.02 |
|                    | 16k | 100% | 2871 | 7 | 7 | 7 | 34.18 |
| `deepseek-r1:7b` | 2k  | 100% | 6831 | 7 | 7 | 7 | 11.25 |
|                  | 4k  | 100% | 6943 | 7 | 7 | 7 | 11.36 |
|                  | 8k  | 100% | 7358 | 7 | 7 | 7 | 11.26 |
|                  | 16k | 100% | 8270 | 7 | 7 | 7 | 11.35 |
| `deepseek-r1:8b` | 2k  | 100% | 5459 | 7 | 7 | 7 | 11.37 |
|                  | 4k  | 100% | 5879 | 7 | 7 | 7 | 11.37 |
|                  | 8k  | 100% | 6719 | 7 | 7 | 7 | 11.42 |
|                  | 16k | 100% | 8399 | 7 | 7 | 7 | 11.34 |
| `deepseek-r1:14b` | 2k  | 100% | 4829 | 7974 | 7 | 7 | 5.81 |
|                   | 4k  | 100% | 5205 | 8227 | 7 | 7 | 5.9 |
|                   | 8k  | 100% | 5957 | 8947 | 7 | 7 | 5.88 |
|                   | 16k | 100% | 7461 | 10387 | 7 | 7 | 5.94 |
| `deepseek-r1:32b` | 2k  | 100% | 7346 | 7242 | 9993 | 7 | 2.85 |
|                   | 4k  | 100% | 7698 | 7594 | 10222 | 7 | 2.86 |
|                   | 8k  | 100% | 8402 | 8298 | 10894 | 7 | 2.87 |
|                   | 16k | 100% | 8146 | 7644 | 7644 | 10574 | 2.77 |

---

## gpt-oss

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |
|---|---|---|--:|--:|--:|--:|--:|
| `gpt-oss:20b` | 2k  | 100% | 7382 | 7556 | 7 | 7 | 16.73 |
|               | 4k  | 100% | 7382 | 7556 | 7 | 7 | 16.57 |
|               | 8k  | 100% | 7382 | 7556 | 7 | 7 | 16.68 |
|               | 16k | 100% | 8520 | 8694 | 7 | 7 | 16.95 |
