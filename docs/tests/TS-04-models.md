---
id: TS-04
title: Models — per-model regression on K80
namespace: ollama37
story: STORY-005
story_hash: 0598aff09de9ddd319d9c168f10a059e16496ea8a26a0d7dcfd1f39f12d9e0f3
status: green
---

## Why this scenario exists

The build/runtime/inference suites prove the *stack* works; this scenario proves each supported
**model** actually runs on K80 (compute 3.7) — coherent output, real GPU memory, the expected
GPU count, and no `CUBLAS_STATUS` / `CUDA error`, with the model unloaded afterwards to free VRAM
for the next. It is the per-model regression half of [STORY-005](../stories/STORY-005.md). Each
case is one model; `Script:` carries the real `TC-MODELS-*.yml` (the YAML ids skip `010` and run
to `015`).

Every case runs the same four steps unless noted:

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-01: gpt-oss:20b

- **Objective:** gpt-oss:20b (~20B params) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-001.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-02: gemma3:27b

- **Objective:** gemma3:27b (~27B params) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-002.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-03: deepseek-r1:14b

- **Objective:** deepseek-r1:14b (~14B params) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-003.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-04: qwen3.5:9b (DeltaNet)

- **Objective:** qwen3.5:9b (DeltaNet architecture) runs on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-004.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-05: FunctionGemma (tool calling)

- **Objective:** FunctionGemma generates a valid tool call on K80 compute 3.7.
- **Script:** cicd/tests/testcases/models/TC-MODELS-005.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test tool calling via the chat API | emits a `get_weather` tool call for `San Francisco`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-06: gemma4:e4b (single GPU)

- **Objective:** gemma4:e4b runs on K80 compute 3.7 (single GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-006.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-07: gemma4:26b (multi-GPU)

- **Objective:** gemma4:26b runs on K80 compute 3.7 (multi-GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-007.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-08: gemma3:4b (single GPU)

- **Objective:** gemma3:4b runs on K80 compute 3.7 (single GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-008.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-09: deepseek-r1:32b (multi-GPU)

- **Objective:** deepseek-r1:32b runs on K80 compute 3.7 (multi-GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-009.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-10: qwen3-vl:8b (single GPU)

- **Objective:** qwen3-vl:8b runs on K80 compute 3.7 (single GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-011.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-11: qwen3-vl:30b (multi-GPU)

- **Objective:** qwen3-vl:30b runs on K80 compute 3.7 (multi-GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-012.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-12: ministral-3:3b (single GPU)

- **Objective:** ministral-3:3b runs on K80 compute 3.7 (single GPU).
- **Script:** cicd/tests/testcases/models/TC-MODELS-013.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-13: qwen3.6:27b (qwen35 arch)

- **Objective:** qwen3.6:27b runs on K80 compute 3.7 (reuses the Qwen3.5 architecture — GGUF arch `qwen35`).
- **Script:** cicd/tests/testcases/models/TC-MODELS-014.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |

### TC-14: qwen3.6:35b MoE (qwen35moe arch)

- **Objective:** qwen3.6:35b (35B-A3B) runs on K80 compute 3.7 through the MoE path (GGUF arch `qwen35moe`).
- **Script:** cicd/tests/testcases/models/TC-MODELS-015.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test inference | returns a `response`; no `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory | reports non-zero `MiB` in use |
| 3 | Check GPU count | `GPU_COUNT_OK` (not `GPU_COUNT_EXCEEDED`) |
| 4 | Unload model | `Model unloaded` |
