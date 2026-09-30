---
id: TS-06
title: Throughput — per-model generation speed + output coherence on K80
namespace: ollama37
story: STORY-005
story_hash: 8dc577f7876df4962321b6b7aff6e5ccd37e0f12d1c2590b08062eae9342b523
status: green
---

## Why this scenario exists

A model is only usable on K80 if it generates at a workable speed **and** the output is coherent
— fast garbage is worthless. This scenario measures per-model throughput (tok/s) and gates it on
a content check, so a "pass" means *fast and meaningful*. It is the usable-performance facet of
[STORY-005](../stories/STORY-005.md). Bound to the `cli.ts bench-throughput` subcommand (no YAML).

### TC-01: per-model throughput is measured and the output is coherent

- **Objective:** each model generates at a measured tok/s and produces coherent output.
- **Script:** `cli.ts bench-throughput <models...> [--judge]`
- **Preconditions:** Ollama up on K80.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Ensure each model is available | pulled if missing |
| 2 | Warmup, then a deterministic benchmark generate (temp 0, seed 42) | tokens generated; tok/s + durations captured |
| 3 | Capture GPU offload % and VRAM | per-GPU VRAM + offload % recorded |
| 4 | Check the output (simple non-empty; agent judge with `--judge`) | response is coherent — not empty or garbled |
| 5 | Report | per-model table (tok/s, GPU%, VRAM) + JSON artifact; PASS only if coherent |
