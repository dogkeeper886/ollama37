---
id: TS-04
title: Models — per-model regression on K80
namespace: ollama37
story: STORY-005
story_hash: 8dc577f7876df4962321b6b7aff6e5ccd37e0f12d1c2590b08062eae9342b523
status: green
---

## Why this scenario exists

The build/runtime/inference suites prove the *stack* works; this scenario proves each supported
**model** actually runs on K80 (compute 3.7) — coherent output, real GPU memory, the expected
GPU count, and no `CUBLAS_STATUS` / `CUDA error`, with the model unloaded afterwards to free VRAM
for the next. It is the per-model regression half of [STORY-005](../stories/STORY-005.md). Each
case is one model; `Script:` carries the real `TC-MODELS-*.yml` (the YAML ids have gaps where
cases were retired and run to `024`).

Models are chosen **one tag per code path** — engine (Ollama vs llama.cpp) × architecture ×
GGUF layout (vision inline or a split projector) × any size-gated branch — using the smallest tag
that reaches the path. A new model gets a case only when it reaches a path no case covers yet.

**The two judges assert different things.** The simple judge asks whether the model loaded
and the request survived — an `{"error":…}` body, a missing `done`, an empty body or a CUDA
error is a failure, and the reply's text is not its business. The agent judge reads the reply
and asks whether it is language a person could read; **a wrong answer is a pass**, because a
270M model getting arithmetic wrong says nothing about the K80. That is why the suite defaults
to `judge_mode: dual` — under `simple` alone nothing checks the output at all.

Two kinds of reply are broken whatever they say, and the step flags them itself rather than
leave them to the agent judge: **`REPLY_NO_TEXT`** (no letter or digit in any script — empty,
whitespace, punctuation) and **`REPLY_REPEAT`** (one short unit over and over, e.g. `4 4 4 …`).
Either fails the simple judge. **The agent judge is not run on a test the simple judge already
failed** — the verdict is already FAIL — which also keeps repeated text away from it: it loops
when it quotes repetition back. A loop that starts anyway is cancelled by its loop guard (a
3-word phrase seen 100 times in 30 seconds) and recorded as FAIL.

The prompt asks for a sentence rather than a bare number, so there is language to judge.

Every case runs the same four steps unless noted, at `temperature` 0 with a fixed
`seed` so the run is reproducible:

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-01: gpt-oss:20b

- **Objective:** gpt-oss:20b (~20B params) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-001.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-02: gemma3:27b

- **Objective:** gemma3:27b (~27B params) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-002.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-03: deepseek-r1:1.5b (llama.cpp qwen2)

- **Objective:** deepseek-r1:1.5b runs on K80 compute 3.7 — the llama.cpp `qwen2` path (7b/14b/32b run the same builder).
- **Script:** cicd/tests/testcases/models/TC-MODELS-003.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-04: qwen3.5:9b (DeltaNet)

- **Objective:** qwen3.5:9b (DeltaNet architecture) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-004.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-06: gemma4:e2b (per-layer embeddings + audio)

- **Objective:** gemma4:e2b runs on K80 compute 3.7 — the Ollama-engine gemma4 path with per-layer embeddings, shared KV layers and the audio tower (same code as e4b).
- **Script:** cicd/tests/testcases/models/TC-MODELS-006.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-07: gemma4:31b (multi-GPU)

- **Objective:** gemma4:31b runs on K80 compute 3.7 (multi-GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-007.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-08: gemma3:270m (text-only gemma3)

- **Objective:** gemma3:270m runs on K80 compute 3.7 — the Ollama-engine gemma3 path with no vision tower.
- **Script:** cicd/tests/testcases/models/TC-MODELS-008.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-10: qwen3-vl:2b (qwen3vl dense)

- **Objective:** qwen3-vl:2b runs on K80 compute 3.7 — the Ollama-engine `qwen3vl` dense path (4b/8b/32b run the same code).
- **Script:** cicd/tests/testcases/models/TC-MODELS-011.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-11: qwen3-vl:30b (multi-GPU)

- **Objective:** qwen3-vl:30b runs on K80 compute 3.7 (multi-GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-012.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-12: ministral-3:3b (mistral3)

- **Objective:** ministral-3:3b runs on K80 compute 3.7 — the Ollama-engine `mistral3` path (8b/14b run the same code).
- **Script:** cicd/tests/testcases/models/TC-MODELS-013.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-13: qwen3.8:27b (qwen3.8 renderer + parser)

- **Objective:** qwen3.8:27b runs on K80 compute 3.7 and its tool calls parse. On the model side it is identical to `qwen3.6:27b` — Ollama engine, `qwen35` graph, split `qwen3vl_merger` projector, the 27B `num_batch` branch — but its chat path is not: the `qwen3.8` renderer (`Qwen35Renderer`, variant 38, always renders the think block) prompts for an XML tool format that only `Qwen35Parser` reads, and it stops on `<|im_end|>` alone. `qwen3.6:27b`, which this case covered before [#492](https://github.com/dogkeeper886/ollama37/issues/492), left the suite: its engine and size branch are covered here, and its `Qwen3VL` renderer and parser by TC-04 and TC-14.
- **Script:** cicd/tests/testcases/models/TC-MODELS-014.yml

Five steps — the tool call is what this case exists for:

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Tool call | `TOOL_OK` — a no-arg `get_current_time` call comes back in `message.tool_calls`, parsed by `Qwen35Parser` |
| 3 | Check GPU memory | reports non-zero `MiB` in use |
| 4 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 5 | Unload model | `Model unloaded` |

### TC-14: qwen3.6:35b MoE (qwen35moe arch)

- **Objective:** qwen3.6:35b (35B-A3B) runs on K80 compute 3.7 through the MoE path (GGUF arch `qwen35moe`).
- **Script:** cicd/tests/testcases/models/TC-MODELS-015.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-15: gemma4:12b (split-vision gemma4)

- **Objective:** gemma4:12b — the only gemma4 packaged as a **split-vision** model (a separate `projector` blob, vs the embedded projector in e4b/26b) — loads on the **new engine** and runs on K80 compute 3.7. Regression for [#367](https://github.com/dogkeeper886/ollama37/issues/367): today the fork's `NewLlamaServer` refuses split-vision models (`reason="split vision models aren't supported"`) and dead-ends in legacy llama.cpp (`unknown model architecture: 'gemma4'`); green once [#370](https://github.com/dogkeeper886/ollama37/issues/370) lets a new-engine arch with a projector use the new engine (upstream already dropped this limitation). The text inference step is the acceptance gate.
- **Script:** cicd/tests/testcases/models/TC-MODELS-016.yml

Six steps, not four — it is the only case that exercises image and audio input:

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Warm up | `WARM_OK` — absorbs the cold-load retry under suite GPU pressure |
| 2 | Text inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true; no `unknown model architecture: 'gemma4'` fallback, no `CUBLAS_STATUS` / `CUDA error` |
| 3 | GPU offload check | reports non-zero `MiB` in use |
| 4 | Vision inference | `VISION_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — the request reached the vision path and returned; no refusal (`cannot see`, `does not support`). The agent judge reads the `content` and asks whether it describes something seen |
| 5 | Audio inference | `AUDIO_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — `espeak-ng` present, the request returned, no refusal. Transcription accuracy is not gated: synthetic speech is marginal |
| 6 | Unload model | `Model unloaded` |

### TC-16: lfm2.5:8b (Liquid LFM2 MoE)

- **Objective:** lfm2.5:8b — the text LFM2 **MoE** (8B total / ~1B active, ~5.2 GB) — runs on K80 compute 3.7 (single GPU). The per-model regression half of [STORY-016](../stories/STORY-016.md): the fork already vendors the `lfm2`/`lfm2moe` llama.cpp arch, so this case gates that the GGUF loads and generates coherently once STORY-016's Go parser/renderer port lands — **red until then**. The inference step pins `"think": false` so the deterministic answer isn't buried inside a `<think>` span (lfm2.5 has a thinking mode). The suite runs **dual** by default, so "coherent, not fluent garbage" is checked on every run.
- **Script:** cicd/tests/testcases/models/TC-MODELS-017.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-17: deepseek-r1:8b (qwen3)

- **Objective:** deepseek-r1:8b runs on K80 compute 3.7 — the Ollama-engine `qwen3` path (the only deepseek-r1 tag on the `qwen3` arch).
- **Script:** cicd/tests/testcases/models/TC-MODELS-022.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-18: llama3.1:8b (llama.cpp llama)

- **Objective:** llama3.1:8b runs on K80 compute 3.7 — the llama.cpp `llama` path and its `GraphSize` "llama" estimate.
- **Script:** cicd/tests/testcases/models/TC-MODELS-023.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-19: gemma3n:e2b (gemma3n)

- **Objective:** gemma3n:e2b runs on K80 compute 3.7 — the Ollama-engine `gemma3n` path.
- **Script:** cicd/tests/testcases/models/TC-MODELS-024.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-20: ornith:35b (ornith renderer + parser)

- **Objective:** ornith:35b (35B-A3B) runs on K80 compute 3.7 through the ornith renderer and parser ported for [STORY-020](../stories/STORY-020.md) — a Qwen3.5-family model with its own chat template and thinking forced on. At ~22.8 GB it is ~99.5% of two dies, too tight for KV growth, so a **3-die split is expected**: its die check only flags waste when the model would fit in one die fewer with ~10% headroom to spare.
- **Script:** cicd/tests/testcases/models/TC-MODELS-019.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK`, judged with ~10% headroom per die. One overshoot reloads the model once, and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |

### TC-21: lfm2.5-thinking:1.2b (LFM2 thinking)

- **Objective:** lfm2.5-thinking:1.2b runs on K80 compute 3.7 — the thinking variant of LFM2 from [STORY-019](../stories/STORY-019.md), which ships without the embedding norm the other LFM2 GGUFs carry ([#438](https://github.com/dogkeeper886/ollama37/issues/438)).
- **Script:** cicd/tests/testcases/models/TC-MODELS-021.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | `LOAD_OK`, no `REPLY_NO_TEXT` / `REPLY_REPEAT` — no `{"error":…}`, `done` true, no `CUBLAS_STATUS` / `CUDA error`. The agent judge reads the `response` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`). One overshoot reloads the model once — placement is not deterministic — and a second overshoot fails |
| 4 | Unload model | `Model unloaded` |
