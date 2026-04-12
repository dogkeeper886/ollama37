# Trace: Ghost 93 MiB GPU Allocations

**Date**: 2026-04-11
**Issue**: #73
**Run**: 24271631817 (full suite)

## Problem

Some models show 93 MiB CUDA primary context overhead on GPUs they don't use.
Each model runs in its own subprocess (different PIDs), so these are NOT
residual from previous models.

## Full Suite Data

| # | Model | GPU0 | GPU1 | GPU2 | GPU3 | Ghosts |
|---|-------|------|------|------|------|--------|
| 1 | gpt-oss:20b | 7375 | 7549 | — | — | 0 |
| 3 | gemma3:27b | 9961 | 10228 | — | — | 0 |
| 4 | deepseek-r1:14b | 5195 | 8217 | **93** | **93** | 2 |
| 5 | qwen3-vl:30b | 10487 | 10355 | **93** | — | 1 |
| 6 | qwen3-vl:8b | 7610 | — | — | — | 0 |
| 7 | qwen3.5:27b | 10151 | 4327 | 4305 | 4436 | 0 |
| 8 | deepseek-r1:32b | 7689 | 7585 | 10213 | **93** | 1 |
| 9 | gemma3:4b | 4818 | — | — | — | 0 |
| 10 | gemma4:26b | 9614 | 9462 | — | — | 0 |
| 11 | gemma4:e4b | 9899 | — | — | — | 0 |
| 12 | functiongemma | 517 | — | — | — | 0 |
| 13 | qwen3.5:9b | 9509 | **93** | **93** | **93** | 3 |

## Root Cause: Two Code Paths

### Path 1: llama engine — pipeline parallelism events (confirmed)

**Affected**: deepseek-r1:14b, deepseek-r1:32b, qwen3-vl:30b

The llama engine runner calls `llama_init_from_model()` which creates a
llama.cpp context. At `llama-context.cpp:240-244`:

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&           // ALWAYS 4 (all GPUs added)
    model.params.n_gpu_layers > n_layer // true when fully offloaded
    ...
```

When `pipeline_parallel=true`, the scheduler creates CUDA events on
**all 4 GPU backends** (`ggml-backend.cpp:1692-1696`):

```cpp
if (sched->n_copies > 1) {
    for (int c = 0; c < sched->n_copies; c++) {
        sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
    }
}
```

`ggml_backend_event_new` calls `cudaSetDevice(device)` +
`cudaEventCreateWithFlags` (`ggml-cuda.cu:4109-4112`), which creates a
CUDA primary context (~93 MiB) on each device.

**Why `model.n_devices() = 4`**: At `llama.cpp:188-248`, ALL visible
GPUs are added to `model->devices` regardless of which ones have layers.
Only `LLAMA_SPLIT_MODE_NONE` prunes the list, but the default is
`LLAMA_SPLIT_MODE_LAYER`.

### Path 2: ollama engine — needs instrumentation (unconfirmed)

**Affected**: qwen3.5:9b

The ollama engine passes `parallel=false` to the scheduler (`ggml.go:388`),
so no events are created. The GPU filter at `ggml.go:364` correctly
excludes unused GPUs from the scheduler.

The ghost source for qwen3.5:9b is still unconfirmed. Most likely the
`/info` endpoint is called on the runner subprocess during the load
process, creating a dummy model with `AllocMemory=false` that queries
all GPUs via `ggml_backend_dev_memory()` → `cudaMemGetInfo`.

## Fix

### For llama engine (Path 1)

The fix is in `llama.cpp:188-248`: only add GPUs that actually have
layers to `model->devices`. Or add a check in the context init to only
create backends/events for devices that have tensors.

**Option A**: Filter `model->devices` based on `tensor_split` — only
include GPUs with non-zero split.

**Option B**: Change `pipeline_parallel` condition to check if layers are
actually distributed across multiple devices, not just that multiple
devices exist.

### For ollama engine (Path 2)

Add instrumentation to confirm, then either:
- Make `BackendDevices()` always skip GPUs not in `schedBackends`
- Or restrict the `/info` dummy model to CPU-only

## Code References

- `model->devices` population: `llama.cpp:188-248`
- Pipeline parallel decision: `llama-context.cpp:240-244`
- Event creation: `ggml-backend.cpp:1692-1696`
- CUDA event → context: `ggml-cuda.cu:4109-4112`
- Ollama scheduler parallel flag: `ggml.go:388` (`false`)
- Ollama GPU filter: `ggml.go:364-367`
