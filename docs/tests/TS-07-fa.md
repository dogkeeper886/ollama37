---
id: TS-07
title: Flash-attention — K80 FA regression + benchmark
namespace: ollama37
story: STORY-005
story_hash: 8dc577f7876df4962321b6b7aff6e5ccd37e0f12d1c2590b08062eae9342b523
status: green
---

## Why this scenario exists

Flash-attention on K80 can silently corrupt output — non-empty but garbled — while the call
still "succeeds". This scenario benchmarks FA configurations and judges their coherence, and
verifies that enabling FA doesn't change the output vs an FA-off baseline — guarding the K80
build's correctness from [STORY-005](../stories/STORY-005.md). Bound to the `cli.ts test-fa`
subcommand (no YAML).

### TC-01: the FA benchmark sweep stays coherent

- **Objective:** every flash-attention configuration produces coherent output.
- **Script:** `cli.ts test-fa --benchmark`
- **Preconditions:** exclusive use of port 11434 (the workflow stops production ollama around it).

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Boot a fresh container per config (off-f16, on-f16, on-q8_0) | each container comes up |
| 2 | Run a deterministic inference per config | tok/s, VRAM, and KV-cache size captured |
| 3 | Judge each config's output (simple + agent judge) | coherent — no FA-induced garble; no `CUBLAS_STATUS` / `CUDA error` |
| 4 | Report | benchmark + judge tables + JSON; PASS only if every config is coherent |

### TC-02: enabling FA does not change the output

- **Objective:** FA-on output matches the FA-off baseline (FA is numerically safe on K80).
- **Script:** `cli.ts test-fa --baseline-compare`

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Run an FA-off baseline, then the same deterministic prompt with FA on | both produce a response |
| 2 | Compare the two outputs | exact match (FA identical to the baseline), or the difference is surfaced |
