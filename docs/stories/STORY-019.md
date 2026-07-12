# STORY-019: Support lfm2:24b and lfm2.5-thinking:1.2b, with CI coverage on K80 + RTX 2060

## User Story

As a maintainer running models on the Tesla K80 and the RTX 2060 testbed,
I want lfm2:24b and lfm2.5-thinking:1.2b to run coherently on both cards and to be
exercised by the model-suite CI on each self-hosted runner,
So that the fork gains two more Liquid LFM2-family models — a larger MoE and a small
thinking model — regression-guarded like every other supported model.

## The Need

Two more models from Liquid AI's LFM2 family: **lfm2:24b** (a larger LFM2, MoE or dense
to be confirmed) and **lfm2.5-thinking:1.2b** (a small reasoning variant with a thinking
mode). Both are text models with tool-calling and thinking.

The hard architecture work is already done. STORY-016 landed the LFM2/LFM2MOE support the
fork was missing: the vendored llama.cpp carries the arch, and the Go-side glue — the
`LFM2Renderer`/`LFM2Parser` (both `lfm2` and `lfm2-thinking`) and the converter — was ported
(#379–#381). So these two models are not a port; they build on support that already exists.

What's missing is proof and coverage. Neither model runs in CI. Every supported model earns
a model-suite regression, and — new for this story — the user wants that regression to run on
**both** self-hosted runners: the K80 (sm37, the reference target) and the RTX 2060 (sm75).
LFM2 is not in the flash-attention allowlist, so FA stays off on both cards; there is no
Turing FA hazard here (unlike the qwen35-family models). The remaining risk is size and
fit, not correctness of the engine.

Size is the open edge. lfm2.5-thinking:1.2b is tiny and fits anywhere. lfm2:24b is large:
it may need both K80 dies or spill to host, and on the 6 GB RTX 2060 it will largely offload
to CPU — runnable but slow. Whether sm75 coverage of the 24B model is worth its wall-clock
is a question for the plan, not an assumption here.

Scope is the two text models above. Vision and audio LFM2 variants are out of scope.

## Success Looks Like

- `ollama run lfm2.5-thinking:1.2b` and `ollama run lfm2:24b` produce coherent output on the
  K80 — no CUDA/CUBLAS errors, no garbage — each fitting within the K80's available VRAM
  (single die where it fits, both dies where it must, without exceeding the GPU count it needs).
- Both models are covered by the **model suite** on **both** self-hosted runners (sm37 and
  sm75), with results captured. Where a model is too large to sit in the RTX 2060's 6 GB, that
  is reported as an observed offload/throughput fact, not treated as a failure of correctness.
- A **reviewed QA test doc** drives the new model-suite regressions, reviewed before the
  implementation is accepted (consistent with how STORY-016 was handled).
- Honesty note: throughput is whatever each card delivers — reported, not held to a target.
  The thinking model's coherence is proven the way STORY-016 proved lfm2.5's (via `/api/chat`
  / `think:false`, since a thinking model's leaked chain-of-thought trips the throughput
  coherence judge).

## Open Questions

- **lfm2:24b — MoE or dense?** Which GGUF `general.architecture` does it carry (`lfm2` vs
  `lfm2moe`)? The converter auto-detects, but confirm from the real config/GGUF, since it
  drives the VRAM footprint.
- **lfm2:24b fit.** Does it fit one K80 die (~11.4 GB), need both dies, or spill to host? On
  the 6 GB RTX 2060 it will mostly offload to CPU — is sm75 coverage of the 24B model still
  wanted given the wall-clock, or is the 1.2b model the meaningful sm75 target?
- **Thinking-mode coherence judge.** Does lfm2.5-thinking:1.2b trip the throughput
  `/api/generate` coherence judge the way lfm2.5:8b did (leaked CoT), so the regression must
  use `/api/chat` with `think:false`?
- **Image currency.** Is the STORY-016 LFM2 renderer/parser work merged to `main` and present
  in the image the runners will test, or is a fresh `ollama37-ci-build` needed first?
- **Availability.** Are `lfm2:24b` and `lfm2.5-thinking:1.2b` pullable tags (do the weights
  exist in the registry the runners pull from)?

## Status

- Created: 2026-07-12
- Plan: #415
- Issues: #417 (arch/availability/image), #418 (CI coverage), #419 (K80 verify), #420 (sm75 verify)
