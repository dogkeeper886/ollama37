# STORY-005: ollama37 builds and runs correctly on Tesla K80

## User Story

As a maintainer of the K80 Ollama fork,
I want confidence that the sm_37 build produces a working runtime that loads models and infers on Tesla K80,
So that every release is known to actually run on the target hardware — not just compile.

## The Need

ollama37 exists to run on Tesla K80 (CUDA compute 3.7), hardware mainstream Ollama no longer
supports. Compiling is not the goal — *running* is. A build can succeed yet fail silently on
K80: a CUBLAS fallback that errors, a model that loads to CPU instead of the GPU, garbled
output that still "passes" an exit-code check. The fork already has a regression suite that
guards against exactly this (build → runtime → inference → per-model), but that suite traces
back to **no story** — there's no single statement of the need it serves. This story is that
backfill: the foundational goal the existing K80 validation tests exist to verify.

## Success Looks Like

- The **build** is confirmed to produce the K80 toolchain image and runtime binary.
- The **runtime** comes up healthy: container starts, the K80 GPU(s) are detected, health and
  the metrics endpoint report correctly.
- **Inference** works on the GPU: a model pulls and generates a real response using the K80,
  not a silent CPU fallback.
- Each supported **model** passes its per-model regression on K80 — coherent output, no
  `CUBLAS_STATUS_*` / `CUDA error` / `cudaMalloc failed` and no `library=cpu` fallback.
- The existing regression tests trace back to this need, so anyone can see *why* each test
  exists.

## Open Questions

- Whether this stays one foundational story or later splits (e.g. the build/runtime/inference
  pipeline vs the per-model regression) — settled when the test docs are derived.
- How freshness between this story and its tests is enforced over time (the qa-workflow's drift
  gate) — that automation is a separate follow-up, not part of capturing the need.

## Status

- **Completed: 2026-06-16** — all tasks merged (#261, #262, #263); the four `docs/tests/TS-*.md` cover all 23 regression tests.
- Created: 2026-06-15
- Plan: #257
- Issues: #258, #259, #260
