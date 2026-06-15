---
id: TS-02
title: Runtime — container, GPU detection, health, metrics on K80
namespace: ollama37
story: STORY-005
story_hash: 8dc577f7876df4962321b6b7aff6e5ccd37e0f12d1c2590b08062eae9342b523
status: green
---

## Why this scenario exists

A built image is worthless if the runtime doesn't actually come up on K80: the container must
start healthy, the Tesla K80 must be detected by both `nvidia-smi` and Ollama's CUDA runtime
(not fall back to CPU), the health/API must respond, and the `/api/metrics` endpoint must
report a well-formed schema. This scenario is the runtime half of
[STORY-005](../stories/STORY-005.md).

### TC-01: Container starts with GPU passthrough and reports healthy

- **Objective:** the ollama37 container starts with GPU passthrough and reaches healthy status.
- **Script:** cicd/tests/testcases/runtime/TC-RUNTIME-001.yml
- **Preconditions:** the `ollama37:latest` image is built; the host has a Tesla K80 + driver.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Clean up any old container | prior container removed (idempotent; no failure if absent) |
| 2 | Start the container | comes up with no `error` / `Error` in the logs |
| 3 | Wait for container health | reports `Container is healthy` within the timeout |
| 4 | Check container status | shows `ollama37` `Up` |

### TC-02: Tesla K80 detected by nvidia-smi and the CUDA runtime

- **Objective:** the K80 is detected by `nvidia-smi` and by Ollama's CUDA runtime — no CPU fallback.
- **Script:** cicd/tests/testcases/runtime/TC-RUNTIME-002.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check `nvidia-smi` inside the container | lists `Tesla K80` on driver `470.`; no "NVIDIA-SMI has failed" / "No devices were found" |
| 2 | Check the CUDA libraries | `cuda` libraries present |
| 3 | Ensure the UVM device exists | `nvidia-uvm` device present |
| 4 | Check Ollama GPU detection in the logs | `library=CUDA`, `compute=3.7`; never `library=cpu` |

### TC-03: Health check and API responsiveness

- **Objective:** the server reports healthy and the API responds.
- **Script:** cicd/tests/testcases/runtime/TC-RUNTIME-003.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check container health status | `healthy` |
| 2 | Test the API endpoint | returns a `models` payload |
| 3 | Check the Ollama version | reports `ollama` |

### TC-04: /api/metrics returns a well-formed schema

- **Objective:** `GET /api/metrics` returns HTTP 200 and JSON conforming to the documented schema.
- **Script:** cicd/tests/testcases/runtime/TC-RUNTIME-004.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Request the endpoint | `HTTP_CODE:200` (not 4xx/5xx) |
| 2 | Check the top-level schema (gpus, models, errors, totals) | `schema OK` |
| 3 | Check GPU shape (≥1 GPU with id, name, vram_total) | `gpu shape OK` |
| 4 | Check error counters present | `error counters OK` |
| 5 | Check totals shape (gpu_count ≥1, loaded_models ≥0) | `totals OK` |
| 6 | Check `gpus[]` length matches `totals.gpu_count` | `gpus length consistent OK` |
| 7 | Check `gpu_count` matches the `nvidia-smi` count | `gpu_count match OK` |
| 8 | Check `gpus[].name` is the marketing name, not the runtime label | `gpu name OK` |
| 9 | Dump the full response for inspection | full JSON printed |
