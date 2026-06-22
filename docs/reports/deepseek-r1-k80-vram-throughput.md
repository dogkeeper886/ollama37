# DeepSeek-R1 on Tesla K80 — VRAM & Throughput, before/after the graph-multiplier change

**Issue:** [#352](https://github.com/dogkeeper886/ollama37/issues/352) — large models offload to CPU with VRAM free
**Branch:** `issue-352-plan-a-graph-multiplier`
**Status:** _data pending — tables below are populated from K80 runs_

## Purpose

Measure whether lowering the fork-only `graphSafetyMultiplier` (the 3.5× compute-graph
over-reservation in `llm/memory.go`) moves large legacy-engine DeepSeek-R1 models **off CPU
onto the K80's GPUs** — *without* breaking small models or reintroducing the
`failed to allocate compute buffers` OOM the 3.5× was added to guard.

#352 is a **per-die accounting** bug, not a capacity bug: the model fits in pooled VRAM, but the
3.5× graph reservation is charged to *every* die, so layers stop fitting per-die and spill to CPU
**while pooled VRAM sits free**. Lowering the multiplier should reclaim that wrongly-reserved VRAM.

## Hardware & configuration

| | |
|---|---|
| Runner | Tesla K80 — 4 × 12 GB dies (~45.8 GB pooled) |
| Engine | legacy llama.cpp (every size below is `qwen2`/`llama` arch → legacy path, the one with the static estimator) |
| Flash attention | off (forced off on sm_37) |
| KV cache | f16 |
| Context (`num_ctx`) | **16384** (the #352 repro — where the graph reservation is largest) |
| `num_predict` | 128 |
| Judge | simple |
| **Only variable between BEFORE/AFTER** | `OLLAMA_GRAPH_SAFETY_MULTIPLIER` |

Columns come straight from `cli.ts bench-throughput`: **Gen tok/s**, **GPU%** (higher = more layers
on GPU), **VRAM used (MiB)**. A model that offloads shows *low GPU% + low tok/s + low VRAM* despite
free pooled memory.

## Models under test

| Tag | Arch | ~Weights (Q4_K_M) | Single-die or multi-die |
|---|---|---|---|
| `deepseek-r1:1.5b` | qwen2 | ~1.1 GB | 1 die |
| `deepseek-r1:7b` | qwen2 | ~4.7 GB | 1 die |
| `deepseek-r1:8b` | llama | ~4.9 GB | 1 die |
| `deepseek-r1:14b` | qwen2 | ~9 GB | 1–2 dies |
| `deepseek-r1:32b` | qwen2 | ~20 GB | 3–4 dies |

## BEFORE — `OLLAMA_GRAPH_SAFETY_MULTIPLIER` unset (= 3.5, default)

| Model | Check | Gen tok/s | GPU% | VRAM used (MiB) |
|---|---|---|---|---|
| `deepseek-r1:1.5b` | _pending_ | | | |
| `deepseek-r1:7b`   | _pending_ | | | |
| `deepseek-r1:8b`   | _pending_ | | | |
| `deepseek-r1:14b`  | _pending_ | | | |
| `deepseek-r1:32b`  | _pending_ | | | |

## AFTER — `OLLAMA_GRAPH_SAFETY_MULTIPLIER` = _&lt;calibrated&gt;_

| Model | Check | Gen tok/s | GPU% | VRAM used (MiB) |
|---|---|---|---|---|
| `deepseek-r1:1.5b` | _pending_ | | | |
| `deepseek-r1:7b`   | _pending_ | | | |
| `deepseek-r1:8b`   | _pending_ | | | |
| `deepseek-r1:14b`  | _pending_ | | | |
| `deepseek-r1:32b`  | _pending_ | | | |

## Analysis

_Filled once both passes are in:_
- Which sizes offloaded BEFORE (low GPU%/tok/s with free VRAM) and recovered AFTER.
- OOM check: did 14b / 32b load without `failed to allocate compute buffers` at the AFTER value?
- Whether a **single** multiplier satisfies all sizes, or the data shows a single constant can't
  (large wants it low, the OOM-guard wants it high) → conditional fix needed.
- **Chosen value:** the lowest multiplier keeping large models on GPU without OOM.

## How to reproduce

The throughput workflow measures whatever is **deployed** on the runner; it does not set the
multiplier. So each pass is *deploy → run*:

**BEFORE** (default 3.5×, current deployment):
```
gh workflow run test-throughput.yml \
  -f models="deepseek-r1:1.5b deepseek-r1:7b deepseek-r1:8b deepseek-r1:14b deepseek-r1:32b" \
  -f context_size=16384
```

**AFTER** (on the runner): build & deploy the `issue-352-plan-a-graph-multiplier` image, set the
value in `docker/.env`, restart, then run the same workflow:
```
echo "OLLAMA_GRAPH_SAFETY_MULTIPLIER=1.5" >> docker/.env   # try 2.0 / 1.5 / ...
docker compose up -d --force-recreate
# then the same `gh workflow run test-throughput.yml ...` as above
```
