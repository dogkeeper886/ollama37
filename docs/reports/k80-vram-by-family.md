# K80 VRAM & Capability by Model Family

How models across families place on the Tesla K80 — per-die VRAM, GPU vs CPU
placement, throughput, and MCP tool-call capability — swept over context length.
Run with the #352 fix live (`graphSafetyMultiplier` default 2.5).

## Setup

| | |
|---|---|
| Hardware | 4 × Tesla K80 · 11441 MiB/die (~45.8 GB pooled) |
| Engine | per-model (legacy llama.cpp or new ollama engine) |
| Flash attention | off (sm_37) |
| KV cache | f16 |
| `graphSafetyMultiplier` | 2.5 (default, #352) |
| Throughput | `num_predict=16` |
| MCP | tool-call capability test |
| Context sweep | 2k / 4k / 8k / 16k |

## Reading the metrics

- **GPU%** — layers on GPU (100% = fully on GPU; lower = CPU offload).
- **d0–d3** — per-die VRAM used (MiB). Shows the layer distribution across the four dies.
- **tok/s** — generation throughput.
- **MCP** — tool-call capability (✅ pass · ❌ fail · — not run).
- **Context (2k→16k)** — footprint = weights + KV(∝ctx) + graph(∝ctx)×2.5. The sweep maps
  each model's **fit cliff** (the context where it tips from full GPU to CPU offload) and whether
  MCP holds as context grows. Key cross-read: an **MCP fail at high ctx with a low GPU%** is a
  speed/timeout artifact (it offloaded), **not** a capability gap — the GPU% column disambiguates.

---

## gemma4

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s | MCP |
|---|---|---|--:|--:|--:|--:|--:|:--:|
| `gemma4:e2b` | 2k  | | | | | | | |
|              | 4k  | | | | | | | |
|              | 8k  | | | | | | | |
|              | 16k | | | | | | | |
| `gemma4:e4b` | 2k  | | | | | | | |
|              | 4k  | | | | | | | |
|              | 8k  | | | | | | | |
|              | 16k | | | | | | | |
| `gemma4:12b` | 2k  | | | | | | | |
|              | 4k  | | | | | | | |
|              | 8k  | | | | | | | |
|              | 16k | | | | | | | |
| `gemma4:26b` | 2k  | | | | | | | |
|              | 4k  | | | | | | | |
|              | 8k  | | | | | | | |
|              | 16k | | | | | | | |
| `gemma4:31b` | 2k  | | | | | | | |
|              | 4k  | | | | | | | |
|              | 8k  | | | | | | | |
|              | 16k | | | | | | | |

---

## qwen3.6

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s | MCP |
|---|---|---|--:|--:|--:|--:|--:|:--:|
| `qwen3.6:27b` | 2k  | | | | | | | |
|               | 4k  | | | | | | | |
|               | 8k  | | | | | | | |
|               | 16k | | | | | | | |
| `qwen3.6:35b` | 2k  | | | | | | | |
|               | 4k  | | | | | | | |
|               | 8k  | | | | | | | |
|               | 16k | | | | | | | |

---

## deepseek-r1

_16k rows seeded from the #352 throughput runs (multiplier 2.5); 2k/4k/8k pending._

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s | MCP |
|---|---|---|--:|--:|--:|--:|--:|:--:|
| `deepseek-r1:1.5b` | 2k  | | | | | | | |
|                    | 4k  | | | | | | | |
|                    | 8k  | | | | | | | |
|                    | 16k | 100% | 2871 | 7 | 7 | 7 | 34.34 | — |
| `deepseek-r1:7b`   | 2k  | | | | | | | |
|                    | 4k  | | | | | | | |
|                    | 8k  | | | | | | | |
|                    | 16k | 100% | 8270 | 7 | 7 | 7 | 11.34 | — |
| `deepseek-r1:8b`   | 2k  | | | | | | | |
|                    | 4k  | | | | | | | |
|                    | 8k  | | | | | | | |
|                    | 16k | 100% | 8399 | 7 | 7 | 7 | 11.37 | — |
| `deepseek-r1:14b`  | 2k  | | | | | | | |
|                    | 4k  | | | | | | | |
|                    | 8k  | | | | | | | |
|                    | 16k | 100% | 7461 | 10387 | 7 | 7 | 5.85 | — |
| `deepseek-r1:32b`  | 2k  | | | | | | | |
|                    | 4k  | | | | | | | |
|                    | 8k  | | | | | | | |
|                    | 16k | 100% | 8146 | 7644 | 7644 | 10574 | 2.77 | — |

---

## gpt-oss

| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s | MCP |
|---|---|---|--:|--:|--:|--:|--:|:--:|
| `gpt-oss:20b` | 2k  | | | | | | | |
|               | 4k  | | | | | | | |
|               | 8k  | | | | | | | |
|               | 16k | | | | | | | |
