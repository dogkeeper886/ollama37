# Decode KQ over the f16 key cache on the K80 (#565)

| | |
|---|---|
| Build | branch `study-565-k80-kq-attention` on `d8ac96ec`, built with `build-local.sh`, version 2.3.3-565 |
| Baseline | `d8ac96ec` (#563 merged), same host, same day |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470.256.02; lab runs on die 1 (die 3 is power-throttled) |
| Measured | 2026-10-08 |

With flash attention off on the K80 (#337), decode attention computes KQ with `mul_mat_vec_f` over
the f16 key cache. A key row holds only 64–512 values; `mul_mat_vec_f` spends a thread block per
row, once per query head, so the query heads that share a KV head each read it again. It read the
cache at 2–15 GB/s, and every 1,000 tokens of context added ~14.5 ms to each `llama3.1:8b` token.
`mul_mat_vec_f_k80` has head_dim/16 lanes share a row, converts it once, and applies it to every
query head of its group. It takes F16 × F32 at batch 1 with rows of 64, 128, 256 or 512 on Kepler
only; every other case keeps ggml's path.

The f16-to-FP32 conversion was not the cost. A die delivers ~7.5 f16 values per clock per SM,
and Kepler converts 32, so the kernel keeps the conversion instruction; a bit-operation decode
would spend more shifts than it saves.

## Decode against context, end to end

`/api/generate` on `ollama.service`, 64 tokens, temperature 0, one run per row
([`ctx.py`](./k80-kq-attention/ctx.py)). Head dim and query heads per KV head in brackets.

| Model | Prompt tokens | Baseline tok/s | #565 tok/s | Δ |
|---|--:|--:|--:|--:|
| `llama3.1:8b` (128, 4) | 19 | 18.69 | 18.82 | +1 % |
| | 3,363 | 9.94 | 15.43 | +55 % |
| | 7,277 | 6.24 | 12.93 | +107 % |
| `gpt-oss:20b` (64, 8) | 71 | 18.45 | 18.71 | +1 % |
| | 3,239 | 13.28 | 17.77 | +34 % |
| | 6,947 | 9.86 | 16.32 | +66 % |
| `gemma4:12b` (512 global / 256 local, up to 16) | 27 | 10.85 | 11.44 | +5 % |
| | 6,363 | 6.55 | 9.58 | +46 % |
| | 13,779 | 4.55 | 7.43 | +63 % |
| `gemma3:4b` (256, 2) | 17 | 26.45 | 26.04 | −2 % |
| | 3,185 | 21.49 | 24.00 | +12 % |
| | 6,893 | 20.08 | 23.32 | +16 % |
| `qwen3.5:9b` (256, 4) | 19 | 12.30 | 12.13 | −1 % |
| | 3,363 | 10.79 | 11.40 | +6 % |
| | 7,277 | 7.67 | 8.04 | +5 % |

- **The gain tracks how much attention grows with context.** `gemma3:4b` keeps most layers on a
  sliding window, and `qwen3.5:9b` has few attention layers, so their slopes were small to begin
  with; something other than KQ still slows `qwen3.5:9b` at 7k.
- **Replies read the same as the baseline's.**
- **`gemma4:12b` crashes on the baseline too.** Its runner exits with status 2 on the first
  request after a context-size reload; a retry succeeds. Not caused by this change.

## Decode at long context in CI

`test-throughput.yml` with `context=6000` (prompts of 7,100–8,050 tokens) on the K80 runner, the
same 8 models before and after: `main`'s image (#563, run
[37750376116](https://github.com/dogkeeper886/ollama37/actions/runs/37750376116)) against the
branch image after the pipeline deployed it (`8cc376c6`, run
[37763537673](https://github.com/dogkeeper886/ollama37/actions/runs/37763537673)).

| Model | Prompt tokens | Decode before | Decode after | Δ |
|---|--:|--:|--:|--:|
| `ministral-3:3b` | 8,048 | 5.12 | 16.59 | +224 % |
| `deepseek-r1:8b` | 7,399 | 4.60 | 11.66 | +153 % |
| `qwen3-vl:30b` | 7,407 | 4.35 | 10.54 | +142 % |
| `llama3.1:8b` | 7,403 | 6.01 | 13.03 | +117 % |
| `gpt-oss:20b` | 7,726 | 8.97 | 14.91 | +66 % |
| `gemma3:27b` | 7,141 | 3.47 | 4.25 | +22 % |
| `qwen3.6:27b` | 7,408 | 2.36 | 2.60 | +10 % |
| `gemma4:12b` | — | runner crashed | runner crashed | |

- **Prefill: unchanged** (within ±3 %), as expected: batched KQ never takes the new path.
- **Both runs fail the same two models,** so neither failure comes from this change.
  `gemma4:12b`'s runner stops (the crash above). `ministral-3:3b`'s prompt fills its whole
  8,048-token window and loses the planted fact the check looks for; its decode numbers are still
  real.
- **`qwen3.6:27b` gains least:** its shape (head dim 256, 6 query heads per KV head) is one of the
  kernel's slower ones, and it is a hybrid with few attention layers.
- **Every model that answered recalls the planted fact on both sides:** 5 of 5 end with
  `LAUNCH CODE: 7492`, which sits 30 % deep in the prompt, so attention over the whole cache is
  computed correctly. `llama3.1:8b` answers word for word as before; the others say the same in
  different words. `qwen3.6:27b` spends all 1,024 tokens thinking on both sides.

## Decode on the short prompt in CI

`test-throughput.yml` with the default `speech-rewrite` prompt (~870 tokens in, 100 out), on 8
models that cover every head shape: branch run
[37770510257](https://github.com/dogkeeper886/ollama37/actions/runs/37770510257) against #563's
run [37735178243](https://github.com/dogkeeper886/ollama37/actions/runs/37735178243), whose code
matches `main`. All 8 replies pass the check and the judge.

| Model | Head dim, group | Decode before | Decode after | Δ |
|---|---|--:|--:|--:|
| `llama3.1:8b` | 128, 4 | 14.34 | 17.15 | +20 % |
| `deepseek-r1:1.5b` | 128, 6 | 41.88 | 49.01 | +17 % |
| `gemma4:12b` | 512 / 256, 16 | 8.07 | 9.28 | +15 % |
| `gpt-oss:20b` | 64, 8 | 14.90 | 16.44 | +10 % |
| `gemma3:4b` | 256, 2 | 22.80 | 24.76 | +9 % |
| `muse-glimmer:30b` | 128, 16 | 4.97 | 5.13 | +3 % |
| `qwen3.6:27b` | 256, 6 | 4.10 | 4.20 | +2 % |
| `qwen3.6:35b` | 256, 8 | 11.38 | 11.64 | +2 % |

- **No regression; short prompts gain too.** At ~900 tokens of context KQ already cost
  `llama3.1:8b` about 20 ms per token (632 µs per layer in the lab).
- **7 of 8 replies are word for word identical to #563's.** `gemma4:12b` differs from word 10:
  it lists the same task constraints, numbered differently.
- **Prefill: within ±3 %,** except `deepseek-r1:1.5b` (+11 %) and `gemma3:4b` (+5 %), small
  models where run-to-run variance is larger.

## Attention replica, one die

[`attn.cu`](./k80-kq-attention/attn.cu) builds the non-FA attention graph the way
`ScaledDotProductAttention` builds it over the PermutedV causal cache: 32 layers, head dim 128,
32 query heads over 8 KV heads.

| Context | Baseline ms/token | #565 ms/token |
|--:|--:|--:|
| 4,096 | 60.2 | 16.1 |

At 4k the baseline spent 84 % of attention in KQ (`mul_mat_vec_f`, 1.73 ms per layer). After the
change KQ takes 0.12 ms per layer, and KQV (`mul_mat_vec_f` over V, 0.24–0.34 ms per layer) is the
largest part.

## KQ kernel, one die

[`kq.cu`](./k80-kq-attention/kq.cu): the KQ mat-vec alone, both paths on the same device buffers,
against a double-precision reference. ggml's time is wall clock per graph run, which includes
~20 µs of synchronization.

| Head dim, q / kv heads, context | Like | K | ggml | #565 | Speedup |
|---|---|--:|--:|--:|--:|
| 128, 32 / 8, 1k | llama3.1, deepseek-r1:8b | 2.1 MB | 632 µs | 50 µs | 12.7× |
| 128, 32 / 8, 4k | | 8.4 MB | 2,378 µs | 119 µs | 20.0× |
| 128, 32 / 8, 8k | | 16.8 MB | 3,968 µs | 165 µs | 24.1× |
| 64, 64 / 8, 4k | gpt-oss | 4.2 MB | 2,173 µs | 63 µs | 34.5× |
| 256, 24 / 4, 4k | qwen3.6:27b, qwen3.8 | 8.4 MB | 1,652 µs | 206 µs | 8.0× |
| 256, 16 / 2, 4k | qwen3.6:35b, ornith | 4.2 MB | 1,085 µs | 142 µs | 7.6× |
| 256, 8 / 4, 4k | gemma3:4b | 8.4 MB | 548 µs | 82 µs | 6.7× |
| 512, 16 / 1, 4k | gemma4:12b, e2b global | 4.2 MB | 2,221 µs | 285 µs | 7.8× |
| 512, 32 / 4, 4k | gemma4:31b global | 16.8 MB | 4,433 µs | 637 µs | 7.0× |
| 512, 32 / 16, 4k | | 67.1 MB | 4,434 µs | 579 µs | 7.7× |
| 128, 32 / 2, 4k | muse-glimmer | 2.1 MB | 1,605 µs | 89 µs | 18.1× |
| 128, 12 / 2, 4k | deepseek-r1:1.5b | 2.1 MB | 612 µs | 43 µs | 14.2× |

Max relative error is 0.9–1.6e-7 on both paths.

- **Few KV heads, large groups are the weakest shapes** (15–30 GB/s): each converted row serves up
  to 16 query heads in turn. With fewer than 4 KV heads the group is split across blocks, at most
  4 query heads each, which halved the 512 / 16 / 1 case.

## Limits of this measurement

- **One run per row** on the host and in CI; decode varies a few percent between runs.
