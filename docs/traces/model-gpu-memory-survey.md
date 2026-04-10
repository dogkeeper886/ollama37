# Model GPU Memory Survey

**Date**: 2026-04-10
**CI run**: https://github.com/dogkeeper886/ollama37/actions/runs/24251137369
**Environment**: 4× Tesla K80 (11.2 GiB each), FlashAttention=disabled, BatchSize=512, KvSize=4096

## nvidia-smi Per-GPU Memory (MiB)

| Model | File | GPUs | GPU0 | GPU1 | GPU2 | GPU3 | Total | Ratio |
|---|---|---|---|---|---|---|---|---|
| gpt-oss:20b | 13 GiB | 2 | 7375 | 7549 | — | — | 14924 | 1.1× |
| ministral-3:3b | 3 GiB | 2 | 9961 | 10228 | — | — | 20189 | 6.7× |
| gemma3:27b | 17 GiB | 4 | 5195 | 8217 | 93 | 93 | 13598 | 0.8× |
| deepseek-r1:14b | 9 GiB | 2 | 10489 | 10355 | — | — | 20844 | 2.3× |
| qwen3-vl:30b | 19 GiB | 1 | 7608 | — | — | — | 7608 | 0.4× |
| qwen3.5:27b | 17 GiB | 4 | 10149 | 4325 | 4303 | 4434 | 23211 | 1.4× |
| gemma3:4b | 3 GiB | 3 | 1042 | 1776 | 2641 | — | 5459 | 1.8× |
| gemma4:26b | 17 GiB | 2 | 9614 | 9460 | — | — | 19074 | 1.1× |
| gemma4:e4b | ~6 GiB | 1 | 9899 | — | — | — | 9899 | 1.7× |
| functiongemma:270m | 0.3 GiB | 1 | 517 | — | — | — | 517 | 1.7× |
| qwen3.5:9b | 6.6 GiB | 4 | 9509 | 93 | 93 | 93 | 9788 | 1.5× |

Ratio = Total VRAM / File Size. Anything above 2× indicates excessive overhead.

## Memory Profile (from test framework log parser)

| Model | Engine | Layers | Layer Split | Total |
|---|---|---|---|---|
| gpt-oss:20b | ollama | ? | ? | ? |
| ministral-3:3b | ollama | 27/27 | ? | ? |
| gemma3:27b | ollama | 63/63 | GPU0:36, GPU1:27 | 20.5 GiB |
| deepseek-r1:14b | ollama | 49/49 | GPU0:25, GPU1:24 | ? |
| qwen3.5:27b | llama.cpp | 64/65 | 16/16/16/16 | ? |
| gemma3:4b | ollama | ? | ? | ? |
| gemma4:26b | ollama | ? | ? | ? |
| gemma4:e4b | ollama | ? | ? | ? |
| qwen3.5:9b | ollama | ? | ? | ? |

## Observations

1. **Vision models are worst**: ministral-3 (6.7×), gemma4:e4b (1.7×) — vision reservation allocates large compute buffers even for text-only inference
2. **Non-flash attention overhead**: BatchSize=512 × KvSize=4096 × heads creates ~192 MiB attention matrices per layer
3. **GPU imbalance**: One GPU consistently uses much more than others (GPU0 for old engine, last GPU for new engine)
4. **Small models on many GPUs**: gemma3:4b on 3 GPUs, qwen3.5:9b on 4 GPUs — the compute buffer exceeds single-GPU capacity

## Issues

- #72 — qwen3.5:27b GPU0 10 GiB imbalance
- #73 — Small models using too many GPUs
- #63 — ministral-3 vision reservation (fixed in PR #71)
