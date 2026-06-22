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
| `num_predict` | 16 — low on purpose: the broken (CPU-offloaded) 32b runs at ~0.3 tok/s, so 128 tokens exceeds the HTTP timeout. 16 lets it finish; offload%/VRAM are exact, tok/s is directional. Same value used for AFTER (apples-to-apples). |
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
| `deepseek-r1:1.5b` | pass | 34.37 | 100% | 2 892 — die0 only |
| `deepseek-r1:7b`   | pass | 11.32 | 100% | 8 270 — die0 |
| `deepseek-r1:8b`   | pass | 11.33 | 100% | 8 399 — die0 |
| `deepseek-r1:14b`  | pass | 6.05 | 100% | 19 838 — dies 0–2 |
| `deepseek-r1:32b`  | pass | **0.43** | **93%** | 29 868 — 4 dies, **~15.9 GB free** |

_Run [27952933017](https://github.com/dogkeeper886/ollama37/actions/runs/27952933017), git `f86abb1`,
`num_predict=16`, `num_ctx=16384`. (`Check=pass` only means the response was non-empty — 32b
"passes" but at 0.43 tok/s it is effectively unusable.)_

**Headline:** only **32b** reproduces #352 — 7% of layers stranded on CPU with **~15.9 GB VRAM free**,
collapsing throughput to **0.43 tok/s**. **14b is 100% on GPU** (6.05 tok/s) and 1.5b trivially so —
so at 16k, the per-die over-reservation only bites the 32b's footprint. Calibration target = **32b**;
**14b = the OOM-guard** (currently healthy at 3.5×, must stay healthy when the multiplier drops).

## AFTER — `OLLAMA_GRAPH_SAFETY_MULTIPLIER` = **2.5** (calibrated, now the default)

| Model | Check | Gen tok/s | GPU% | VRAM used (MiB) |
|---|---|---|---|---|
| `deepseek-r1:1.5b` | pass | 34.34 | 100% | 2 892 — die0 |
| `deepseek-r1:7b`   | pass | 11.34 | 100% | 8 270 — die0 |
| `deepseek-r1:8b`   | pass | 11.37 | 100% | 8 399 — die0 |
| `deepseek-r1:14b`  | pass | 5.85 | 100% | 17 862 — dies 0–1 |
| `deepseek-r1:32b`  | pass | **2.77** | **100%** | 34 008 — 4 dies (uses the once-stranded VRAM) |

### The multiplier sweep (32b @ 16k — finding the value)

| Multiplier | 32b GPU% | 32b tok/s | verdict |
|---|---|---|---|
| 3.5 (original) | 93% | 0.43 | broken — #352 |
| 3.0 | 99% | 1.40 | nearly — 1 layer on CPU still tanks it |
| **2.5 (chosen)** | **100%** | **2.77** | **fixed — highest value with full GPU placement** |
| 2.0 | 100% | 2.79 | fixed (less OOM margin) |

## Analysis

- **BEFORE:** only `deepseek-r1:32b` reproduced #352 — 93% GPU / 0.43 tok/s with ~15.9 GB free.
  `14b`, `8b`, `7b`, and `1.5b` were already 100% on GPU. The bug needs the 32b's multi-die footprint.
- **Small models are multiplier-neutral:** `7b`/`8b` (single-die) read **100% GPU at both 3.5 and
  2.5** (~11.3 tok/s, flat) — the fix changes nothing for models that already fit one die.
- **AFTER @ 2.5:** 32b → **100% GPU, 2.77 tok/s** (6.4× faster), now using all four dies. `14b`
  stays 100% on GPU with **no `failed to allocate compute buffers`** — the OOM-guard passes.
- **A single constant suffices** here — no conditional fix needed. 2.5 satisfies both the fit (32b)
  and the OOM-guard (14b).
- **Why 2.5, not 2.0:** the failure modes are asymmetric — too high merely offloads (slow but
  works), too low fails to allocate (won't load). So we pick the **highest** value that still puts
  32b 100% on GPU, maximizing the margin against the OOM the multiplier guards. 2.5 is that value
  (3.0 already drops to 99%); 2.0 also works but sits further from the safety margin.

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
