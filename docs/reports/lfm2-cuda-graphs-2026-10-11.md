# lfm2.5: one GPU graph per token, so CUDA graphs replay (#578)

| | |
|---|---|
| Build | branch `study-578-lfm2-graphs` on `64d09130`, version 2.3.3-578 |
| Baseline | `main`'s build on the same host; CI run 38061659277 |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470.256.02 |
| Measured | 2026-10-11 |

#573 let the K80 keep ggml's CUDA graphs, and decode rose on 23 of 25 CI models. The two lfm2.5 models
stayed flat, because each lfm2.5 token ran as three GPU graphs on one die. ggml-cuda caches one CUDA graph
per device, so every call changed it, and after four updates in a row it turned graphs off. Two placements
in llama.cpp's lfm2 code caused the split. With both fixed, a token is one CPU split and one GPU graph,
graphs replay, and decode rises 9–12 % on the host with replies unchanged.

## Why lfm2.5 split

`GGML_SCHED_DEBUG=1` on `ollama.service`, lfm2.5:8b (`lfm2moe`, llama.cpp engine), one decode token on `main`:

| Split | Backend | Nodes |
|---|---|---|
| 0 | CUDA0 | the recurrent-state copies of `cache_r_l0` (10 nodes) |
| 1 | CPU | `GET_ROWS` of `token_embd`, the token lookup |
| 2 | CUDA0 | the 24 layers and the final `RMS_NORM` (1,269 nodes) |
| 3 | CPU | `MUL` by `token_embd_norm.weight`, the final norm's weight |
| 4 | CUDA0 | `lm_head`, the output `MUL_MAT` |

- **The final norm on the CPU (splits 3, 4):** lfm2 stores its final norm as `token_embd_norm`.
  llama.cpp loads it as `LLM_TENSOR_TOKEN_EMBD_NORM`, an input-layer tensor, and input-layer tensors stay
  on the CPU. So the scheduler ran its `MUL` on the CPU, between the layers and `lm_head`.
- **The token lookup after the state copies (split 0):** `build_inp_mem_hybrid` expands the recurrent-state
  copies into the graph when it is called. The token lookup enters the graph only at the final expand, so
  it comes after them, and the CPU lookup cuts the GPU work once more.
- **Not a CUDA-graph problem:** the nodes inside each graph never changed between calls. Only the graph
  ggml-cuda was handed alternated (10, 1,269 and 1 nodes).

## The change

In `llama/llama.cpp/src`:

- **`LLM_TENSOR_OUTPUT_NORM_LFM2`:** a tensor kind named `token_embd_norm` for LFM2 and LFM2MOE, in the
  output layer. lfm2's `tok_norm` loads through it, so it sits on the GPU next to `lm_head`.
- **`llm_build_lfm2`:** the token lookup is expanded right after it is built, ahead of the state copies.

A token is now two splits: the CPU lookup, then one CUDA0 graph with every other node. The change is in
llama.cpp's model code, not in ggml-cuda, and changes no arithmetic.

## On the host

`ollama.service`, the `speech-rewrite` prompt as `test-throughput.yml` sends it, 100 tokens, two rounds
each. Graphs off is `GGML_CUDA_DISABLE_GRAPHS=1`:

| Model | `main`, graphs on | `main`, graphs off | branch, graphs on | branch, graphs off |
|---|--:|--:|--:|--:|
| lfm2.5:8b | 60.21, 61.88 | 60.11, 61.07 | **68.80, 68.50** | 62.25, 62.00 |
| lfm2.5-thinking:1.2b | 99.34, 99.17 | 99.11, 99.02 | **108.23, 107.73** | 99.83, 99.51 |

- **Graphs now pay:** with graphs on, decode is +12.5 % on lfm2.5:8b (61.05 → 68.65) and +8.8 % on
  lfm2.5-thinking:1.2b (99.26 → 107.98), against `main`.
- **Without graphs:** +1.6 % and +0.5 %, from the two CPU round trips the final norm no longer makes.
- **Prefill:** unchanged, 265–270 and 742–748 tok/s.
- **Replies:** both are word for word identical to `main`'s.

## CI

Pipeline [38109275331](https://github.com/dogkeeper886/ollama37/actions/runs/38109275331): build, deploy and
canary pass; the models test judges all 29 replies across 22 models, 0 failed.

`test-throughput.yml`, the standard `speech-rewrite` run on all 25 models: branch
[38113164152](https://github.com/dogkeeper886/ollama37/actions/runs/38113164152) against
[38061659277](https://github.com/dogkeeper886/ollama37/actions/runs/38061659277) (`main`'s code).

- **All 25 replies pass the check and the judge, and all 25 are word for word identical.**
- **lfm2.5:8b decode +11.0 %** (61.19 → 67.90 tok/s), **lfm2.5-thinking:1.2b +9.4 %** (97.67 → 106.88).
- **The other 23 models run no changed code:** decode moves −3.9 % to +4.4 % (mean −0.3 %), prefill −3.8 %
  to +2.5 %. The extremes are gemma3:27b −3.9 % and muse-glimmer:30b +4.4 %, which is this run's noise.

## Decision

**Keep.** lfm2.5 decode rises 9–11 % in CI and 9–12 % on the host, with replies unchanged and no other
model's code path touched.

## Limits

- **One CI run per row.**
- **Other llama.cpp-engine models were not checked for splits:** gemma4:12b gained from graphs in #573, so
  it runs as one GPU graph. Any other model the scheduler cuts into several GPU graphs on one die still
  loses CUDA graphs. ggml-cuda still caches one graph per device.
