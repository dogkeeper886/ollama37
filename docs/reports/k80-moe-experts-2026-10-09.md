# MoE expert mat-vecs on the K80 kernels (#572)

| | |
|---|---|
| Build | branch `study-572-k80-moe-experts` on `991a5c7f` (#563, #565, #567, #571 merged), version 2.3.3-572 |
| Baseline | CI run 37876494396 (#571's code, the same kernels as `main`), same host |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470.256.02; lab runs on die 1 |
| Measured | 2026-10-09 |

At one token, `MUL_MAT_ID` (the MoE expert path) went straight to ggml's `mul_mat_vec_q` with expert ids,
so MoE models never reached #563's q4_K or #567's q6_K kernels. On `qwen3.6:35b` the expert mat-vecs took
~18 % of a decode token, and the down-projection read at 28 GB/s. Both K80 kernels now take an expert
routing argument: block y is the slot, which reads expert `ids[slot]`, prepared input `slot % ne11` (one
shared input for gate/up, one per expert for down) and writes the slot's output. `ggml_cuda_mul_mat_id`
uses them for q4_K and q6_K experts at batch 1 on Kepler; every other case keeps ggml's path, and the
dense path passes one slot and no ids, unchanged.

## Lab, one die

`ggml_mul_mat_id` at `qwen3.6:35b`'s expert shapes, 8 of 256 experts, `nvprof` kernel times; the same lab
program on `main`'s library and on the branch's.

| Expert mat-vec | Reads | `mul_mat_vec_q` | K80 kernel | Speedup | Max relative error |
|---|--:|--:|--:|--:|---|
| q4_K gate/up, 2048 → 512, shared input | 4.7 MB | 132 µs | 70.5 µs | 1.9× | 6.2e-3 → 2.2e-7 |
| q4_K down, 512 → 2048, 8 inputs | 4.7 MB | 327 µs | 129 µs | 2.5× | 6.2e-3 → 2.2e-7 |
| q6_K down, 512 → 2048, 8 inputs | 6.9 MB | 329 µs | 135 µs | 2.4× | 6.1e-3 → 9.6e-8 |

Errors are against a double-precision reference. The dense paths through the same kernels keep their
#563 and #567 results (q4_K 2.4e-7, q6_K 1.3–1.4e-7).

## Decode in CI

Pipeline [37899521950](https://github.com/dogkeeper886/ollama37/actions/runs/37899521950): build,
deploy and canary pass; the models test judges all 29 replies across 22 models, 0 failed.

`test-throughput.yml`, the standard `speech-rewrite` run on all 25 models: branch run
[37906763009](https://github.com/dogkeeper886/ollama37/actions/runs/37906763009) against
[37876494396](https://github.com/dogkeeper886/ollama37/actions/runs/37876494396). All 25 replies pass the
check and the judge.

| Model (experts) | Decode before | Decode after | Δ |
|---|--:|--:|--:|
| `qwen3-vl:30b` (q4_K, q6_K) | 15.60 | 20.72 | +33 % |
| `lfm2.5:8b` (q4_K, q6_K) | 44.73 | 59.21 | +32 % |
| `ornith:35b` (q4_K, q6_K) | 16.40 | 19.33 | +18 % |
| `qwen3.6:35b` (q4_K, q6_K) | 16.60 | 19.39 | +17 % |
| `ornith-1.5:35b` (q4_K, q6_K) | 16.63 | 19.42 | +17 % |
| `gemma4:26b` (q4_K, q5_1, q8_0) | 13.74 | 15.04 | +9 % |
| `gpt-oss:20b` (MXFP4) | 16.45 | 16.65 | +1 % |
| the other 18 models (dense) | | | −2 % to +3 % |

- **MoE models with q4_K and q6_K experts gain +23 % on average** (+17 % to +33 %). `gemma4:26b` gains
  less: only its q4_K experts take the new path.
- **`gpt-oss:20b`'s MXFP4 experts and the dense models keep their paths;** their change is noise.
- **Prefill: within ±2 %.**
- **23 of 25 replies are word for word identical** to the baseline's. `lfm2.5:8b` and `gemma4:26b` part
  after 68 and 34 words and keep the same plan in other words.

## Limits

- **One run per row** in CI.
- **Not covered:** q5_1, q8_0 and MXFP4 experts, and batch 2–8.
- **q4_K experts with 512 inputs use half of each warp:** #563's kernel covers 4 blocks per warp step and
  a 512-value row has 2. It still runs 2.5× faster than `mul_mat_vec_q`; a layout for short rows could take
  more.
