# STORY-016: Support lfm2.5 (Liquid LFM2 MoE) on the K80, with CI coverage

## User Story

As a maintainer running models on the Tesla K80,
I want lfm2.5:8b-a1b to run coherently and make real tool calls on the K80, and to be
exercised by the model, throughput, and MCP CI suites on the self-hosted runner,
So that the fork gains a fast on-device MoE with verified tool-calling, regression-guarded
like every other supported model.

## The Need

`ollama.com/library/lfm2.5` is Liquid AI's hybrid LFM2 MoE (8B total / ~1B active,
~5.2 GB, 125K context) — a text model with tool-calling and thinking. At ~5.2 GB it fits
on a single K80, and a 1B-active MoE is the kind of workload the K80 handles well.

The fork is already most of the way there: the vendored llama.cpp carries the
LFM2/LFM2MOE architecture (arch enums, compute graph, tensor schema), so the GGUF can load
and run. What's missing is the Go-side glue that upstream already has — the tool-call
parser and the prompt renderer (and the import converter). Without the parser and renderer,
lfm2.5 can emit text but cannot make grounded tool calls, so it can't be trusted for the
agentic / MCP workloads this fork cares about.

The model also isn't in CI yet. Every supported model earns a model-suite regression, and
the tool-calling and throughput behaviour should be measured on the real K80 runner rather
than assumed. A reviewed test doc should drive that coverage before the implementation is
accepted.

Scope is the text lfm2.5:8b-a1b (text + tools + thinking). The vision variant is a separate
model (lfm2.5-vl) with its own converter, and there is no audio variant in upstream code —
both are out of scope here.

## Success Looks Like

- `ollama run lfm2.5:8b-a1b` produces coherent output on the K80 — no CUDA/CUBLAS errors,
  no garbage — and the model fits within a single K80 GPU.
- lfm2.5 makes **real, grounded tool calls**: the MCP tool-call test passes on the
  self-hosted runner — the model calls a real tool and its final answer is grounded in the
  tool result (not hallucinated). This is the hard bar.
- A **reviewed QA test doc** (qa-workflow) exists and drives the model-suite regression
  (TC-MODELS-017), reviewed before the implementation is accepted.
- lfm2.5 is covered by all three suites on the self-hosted K80 runner, with results
  captured: the model suite (regression YAML), the throughput benchmark, and the MCP
  tool-call probe.
- Honesty note: throughput is whatever the K80 delivers for a 1B-active MoE — reported,
  not held to a target. If K80 tool-calling needs iteration to land, that is surfaced in
  the results, not hidden.

## Open Questions

- How much of upstream's lfm2 surface actually needs porting: are the parser and renderer
  enough (they are required for tool-calling), or is `convert/convert_lfm2.go` also needed
  — i.e. does pulling the prebuilt library GGUF avoid the import/convert path entirely?
- Does the "thinking" variant matter for lfm2.5:8b-a1b, or is the non-thinking renderer/parser
  enough? (Upstream registers both `lfm2` and `lfm2-thinking`.)
- Does lfm2's short-convolution (LIV) path hit any K80 op that is missing or slow on
  compute 3.7 (as happened with the gemma4 / DeltaNet ports), or does the already-vendored
  compute graph just work?
- For throughput and MCP, models are dispatch-time inputs, not YAML — should their default
  model lists be updated to include lfm2.5, or is a documented manual dispatch enough?
- VL and Audio: file the vision variant (lfm2.5-vl) as its own story if wanted; there is no
  audio implementation upstream to port.

## Status

- Created: 2026-07-09
- Docs PR: #377 (story + TS-04 TC-16 + TC-MODELS-017.yml)
- Issues: none — port issue to follow via `dw-plan`
