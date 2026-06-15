---
id: TS-03
title: Inference — model pull and GPU inference smoke on K80
namespace: ollama37
story: STORY-005
story_hash: 8dc577f7876df4962321b6b7aff6e5ccd37e0f12d1c2590b08062eae9342b523
status: green
---

## Why this scenario exists

The point of the build is to *infer* on K80. This scenario proves a model pulls into the
container and the REST API generates a real response using GPU memory — without a CUBLAS/CUDA
error and without silently falling back to CPU. It is the inference half of
[STORY-005](../stories/STORY-005.md).

### TC-01: Pull the test model

- **Objective:** the `gemma3:4b` test model pulls into the container.
- **Script:** cicd/tests/testcases/inference/TC-INFERENCE-001.yml
- **Preconditions:** the runtime container is up and healthy.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Pull the test model | reports `success` / `already exists` / `pulling`; no `error pulling` / `Error:` |
| 2 | Verify the model is available | `gemma3:4b` listed |

### TC-02: API inference allocates GPU memory

- **Objective:** the generate endpoint returns a response using GPU memory, with no CUDA/CUBLAS error.
- **Script:** cicd/tests/testcases/inference/TC-INFERENCE-002.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Call the generate endpoint | returns a `response`; no `error` / `CUBLAS_STATUS` / `CUDA error` |
| 2 | Check GPU memory usage | reports non-zero `MiB` in use |
| 3 | Unload the model after the test | `Model unload requested` |
