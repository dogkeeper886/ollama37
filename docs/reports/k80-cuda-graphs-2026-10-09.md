# CUDA graphs on the K80 (#573)

| | |
|---|---|
| Build | branch `study-573-k80-graphs` on `0a05ff64` (#563, #565, #567, #571, #572 merged), version 2.3.3-573 |
| Baseline | CI run 37906763009 (#572's code, the same as `main`), same host |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470.256.02 |
| Measured | 2026-10-09 |

ggml turns CUDA graphs off below Ampere by architecture alone, so on the K80 every op of a decode token
launched its own kernel: a few hundred per token on a dense model, ~2,000 on `qwen3.6:35b` before #571 and
#572. CUDA 11.4 captures and replays graphs on sm_37. Kepler now keeps them. ggml's own checks still turn
graphs off where they don't fit: four consecutive updates, split buffers, batch above one. Maxwell to
Turing keep ggml's gate.

## Host A/B, same build

`ollama.service` with `GGML_CUDA_DISABLE_GRAPHS` set and unset, the repo's `speech-rewrite` prompt,
200 decode tokens, temperature 0.

| Model | Graphs off | Graphs on | Δ | Reply |
|---|--:|--:|--:|---|
| `qwen3-vl:30b` | 18.61 | 21.97 | +18 % | identical |
| `qwen3.6:35b` | 19.26 | 21.75 | +13 % | identical |
| `gemma3:270m` | 91.96 | 101.54 | +10 % | identical |
| `qwen3-vl:2b` | 52.50 | 55.52 | +6 % | identical |
| `gemma3:4b` | 27.53 | 27.99 | +2 % | identical |
| `lfm2.5:8b` | 58.06 | 59.00 | +2 % | identical |
| `llama3.1:8b` | 17.60 | 17.85 | +1 % | identical |
| `qwen3.5:9b` | 15.62 | 15.65 | +0 % | identical |

A lab-only log, removed before commit, showed which models replay graphs:

- **Most models capture once and replay:** ~17 re-captures per 512 launches, as the KV window grows.
- **`lfm2.5:8b` runs three graphs per token on one die** (10, 1,269 and 1 nodes). ggml caches one graph per
  device, so each call replaces the last; after four updates ggml turns graphs off for that model. It
  runs as before.

## Fusions

ggml-cuda's fusions have no architecture gate and already run on the K80 path: top-k MoE routing (with
and without the norm), RMS norm with mul (and add), and scale-tanh-scale. No change.

## Decode in CI

Pipeline [37917497279](https://github.com/dogkeeper886/ollama37/actions/runs/37917497279): build,
deploy and canary pass; the models test judges all 29 replies across 22 models, 0 failed.

`test-throughput.yml`, the standard `speech-rewrite` run on all 25 models: branch run
[37924854227](https://github.com/dogkeeper886/ollama37/actions/runs/37924854227) against
[37906763009](https://github.com/dogkeeper886/ollama37/actions/runs/37906763009). All 25 replies pass the
check and the judge, and **all 25 are word for word identical** to the baseline's.

| Models | Decode Δ |
|---|--:|
| MoE: `qwen3.6:35b`, both `ornith` 35B, `qwen3-vl:30b` | +13 % |
| `gemma3:270m` | +13 % |
| `gemma3n:e2b`, `deepseek-r1:1.5b`, `gemma4:26b` | +9 % to +11 % |
| `qwen3-vl:2b`, `gemma4:e2b`, `gemma4:12b` | +6 % to +7 % |
| the other dense models, `gpt-oss:20b` | +1 % to +4 % |
| `lfm2.5:8b`, `lfm2.5-thinking:1.2b` | −0.5 % to −0.2 % |

- **Every model but the two `lfm2.5` gains, +5.8 % on average across all 25.** The gain follows how short
  each kernel is: fast models and MoE models, with many small kernels per token, gain most.
- **`lfm2.5` stays flat:** ggml turns its graphs off, as above.
- **Prefill: within ±1 %**, except `gemma3:270m` at +42 % in this one run. Its prompt runs at batch above
  one, where graphs stay off, so this run alone can't explain it.

## Decision

**Keep.** Speed rises on 23 of 25 models and the replies don't change.

## Limits

- **One CI run per row.**
- **Models that run several graphs per die get no gain:** a per-graph cache, as later ggml versions keep,
  would cover `lfm2.5`.
- **Not measured:** wall time against GPU time per token for each model (issue step 1). The graphs-off to
  graphs-on gain stands in for the share of launch cost that graphs remove.
