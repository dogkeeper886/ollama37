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

## Correctness

Two checks, each of which holds the model fixed, so they test the kernel rather than what a
model can do:

- **Against an exact reference.** On every head shape below, the kernel's KQ matches a
  double-precision computation to a max relative error of 0.9–1.6e-7, the same as
  `mul_mat_vec_f`.
- **Against the old build.** The same model and prompt at temperature 0 and a fixed seed, `main`
  against the branch, compared word for word:

  | Run | Identical replies | The rest: words before they part |
  |---|---|---|
  | CI short prompt, 25 models | 19 | 10–64 words (`gemma4:12b` 10, `ornith-1.5:35b` 16, `gemma3:270m` 37, `gemma4:26b` 51, `qwen3-vl:30b` 56, `qwen3-vl:2b` 64) |
  | CI long context, 7 models | `llama3.1:8b` | 22–126 words (`qwen3.6:27b` 22, `ministral-3:3b` 27, `gemma3:27b` 48, `qwen3-vl:30b` 61, `deepseek-r1:8b` 71, `gpt-oss:20b` 126) |

  Replies that part stay on task and read as fluently as the old ones; the judge passes every
  one. The new kernel adds its products in a different order than `mul_mat_vec_f`, so a
  near-tie between two next tokens can fall the other way, and the texts part from there.

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
  `gemma4:12b`'s runner stops while tokenizing the prompt. `ministral-3:3b`'s prompt fills its whole
  8,048-token window and loses the planted fact the check looks for; its decode numbers are still
  real.
- **`qwen3.6:27b` gains least:** its shape (head dim 256, 6 query heads per KV head) is one of the
  kernel's slower ones, and it is a hybrid with few attention layers.
- **That prompt is gone:** #566 replaced the generated filler with a real speech
  (`farewell-summary`); these numbers stand as recorded.
- **The planted fact, as an observation only:** the same 5 models write `LAUNCH CODE: 7492` on
  both sides. Recalling it measures the model as much as the kernel, so it is not counted as a
  correctness check; see [Correctness](#correctness).

## Decode on the short prompt in CI

`test-throughput.yml` as always: the default `speech-rewrite` prompt (~870 tokens in, 100 out) on
all 25 models in the runner's list. Branch run
[37773830995](https://github.com/dogkeeper886/ollama37/actions/runs/37773830995) against #563's
run [37735178243](https://github.com/dogkeeper886/ollama37/actions/runs/37735178243), whose code
matches `main`. All 25 replies pass the check and the judge.

Head dim is global / local for gemma4; its KV-head count varies by layer, so its group is a range.

| Model | Head dim, group | Decode before | Decode after | Δ |
|---|---|--:|--:|--:|
| `ministral-3:3b` | 128, 4 | 21.31 | 29.54 | +39 % |
| `qwen3-vl:30b` | 128, 8 | 12.08 | 15.68 | +30 % |
| `gemma4:26b` | 512 / 256, 2–8 | 11.78 | 14.70 | +25 % |
| `qwen3-vl:2b` | 128, 2 | 39.37 | 48.04 | +22 % |
| `deepseek-r1:8b` | 128, 4 | 13.68 | 16.20 | +18 % |
| `gemma4:12b` | 512 / 256, 2–16 | 8.07 | 9.51 | +18 % |
| `deepseek-r1:1.5b` | 128, 6 | 41.88 | 49.11 | +17 % |
| `llama3.1:8b` | 128, 4 | 14.34 | 16.53 | +15 % |
| `gemma4:31b` | 512 / 256, 2–8 | 3.55 | 4.03 | +14 % |
| `gemma4:e2b` | 512 / 256, 8 | 28.33 | 32.23 | +14 % |
| `lfm2.5-thinking:1.2b` | 64, 4 | 77.03 | 87.42 | +13 % |
| `gpt-oss:20b` | 64, 8 | 14.90 | 16.54 | +11 % |
| `gemma3n:e2b` | —, 4 | 25.17 | 27.32 | +9 % |
| `gemma3:4b` | 256, 2 | 22.80 | 24.72 | +8 % |
| `gemma3:270m` | 256, 4 | 83.86 | 89.37 | +7 % |
| `lfm2.5:8b` | 64, 4 | 40.24 | 43.09 | +7 % |
| `gemma3:27b` | 128, 2 | 4.75 | 5.04 | +6 % |
| `muse-glimmer:30b` | 128, 16 | 4.97 | 5.16 | +4 % |
| `ornith:35b` | 256, 8 | 11.33 | 11.65 | +3 % |
| `ornith-1.5:35b` | 256, 8 | 11.38 | 11.68 | +3 % |
| `qwen3.6:27b` | 256, 6 | 4.10 | 4.21 | +3 % |
| `ornith-1.5:9b` | 256, 4 | 11.30 | 11.52 | +2 % |
| `qwen3.5:9b` | 256, 4 | 11.61 | 11.87 | +2 % |
| `qwen3.6:35b` | 256, 8 | 11.38 | 11.65 | +2 % |
| `qwen3.8:27b` | 256, 6 | 4.10 | 4.20 | +2 % |

- **No regression; every model gains.** At ~900 tokens of context KQ already cost `llama3.1:8b`
  about 20 ms per token (632 µs per layer in the lab). The hybrid `qwen35` family (`qwen3.5`,
  `qwen3.6`, `qwen3.8`, `ornith`) gains least: few of its layers are attention.
- **19 of 25 replies are word for word identical to #563's.** The other 6 part after 10–64 words
  and say the same in other words ("negatives into affirmatives" against "into positives",
  `"no," "not,"` against `"not," "no,"`); `gemma3:270m` swaps one sentence of its rewrite.
- **Prefill: within ±4 %.**
- **An earlier 8-model subset** (run
  [37770510257](https://github.com/dogkeeper886/ollama37/actions/runs/37770510257)) agrees: +2 % to
  +20 %, 7 of 8 replies identical.

## Long context on a real speech in CI: crash check

`test-throughput.yml` with `prompt=farewell-summary` (#566: Washington's Farewell Address, 37,091
characters, `num_ctx` 12288) on all 25 models, run
[37780450973](https://github.com/dogkeeper886/ollama37/actions/runs/37780450973). Its purpose is to
find crashes or abnormal output at long context; it has no baseline yet.

| Check | Result |
|---|---|
| Runner crashes or errors | none |
| Spill to CPU | none; every model 100 % on GPU |
| Broken text (judge) | none; all 25 judged not broken |
| Loops | none; no 3-word phrase repeats more than twice |
| Window | 24 models used 7,132–8,685 of 12,288 tokens; **`gemma4:12b` filled all 12,288 and was cut** |

- **`gemma4:12b`'s overflow is its own bug, not this change's:** its siblings read the same text in
  ~7,140 tokens. Tracked in #568, with its long-line crash and a VRAM estimate 1.79× its real use.

## Attention replica, one die

[`attn.cu`](https://github.com/dogkeeper886/ollama37/blob/3ac65491/docs/reports/k80-kq-attention/attn.cu) (removed after this study) builds the non-FA attention graph the way
`ScaledDotProductAttention` builds it over the PermutedV causal cache: 32 layers, head dim 128,
32 query heads over 8 KV heads.

| Context | Baseline ms/token | #565 ms/token |
|--:|--:|--:|
| 4,096 | 60.2 | 16.1 |

At 4k the baseline spent 84 % of attention in KQ (`mul_mat_vec_f`, 1.73 ms per layer). After the
change KQ takes 0.12 ms per layer, and KQV (`mul_mat_vec_f` over V, 0.24–0.34 ms per layer) is the
largest part.

## KQ kernel, one die

[`kq.cu`](https://github.com/dogkeeper886/ollama37/blob/3ac65491/docs/reports/k80-kq-attention/kq.cu) (removed after this study): the KQ mat-vec alone, both paths on the same device buffers,
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
| 512, 16 / 1, 4k | gemma4:12b global | 4.2 MB | 2,221 µs | 285 µs | 7.8× |
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
