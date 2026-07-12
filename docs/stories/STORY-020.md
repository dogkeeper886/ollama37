# STORY-020: Support ornith:9b and ornith:35b, with CI coverage on K80 + RTX 2060

## User Story

As a maintainer running models on the Tesla K80 and the RTX 2060 testbed,
I want ornith:9b and ornith:35b to render and parse correctly and run coherently on both
cards, exercised by the model-suite CI on each self-hosted runner,
So that the fork gains the ornith models — a thinking-mode Qwen3.5 family — with their
chat template and tool-calling behaving as upstream intends, regression-guarded like every
other supported model.

## The Need

`ornith` (9b and 35b) is a thinking-oriented model family. Tracing upstream ollama shows it
is **not a new architecture**: it is a Qwen3.5-family model with its own chat template — the
underlying weights load on the `qwen35` (dense) / `qwen35moe` (MoE) architecture the fork
already runs natively. What ornith adds is a distinct **renderer and parser** (thinking
forced on, empty think-block handling) that upstream carries but this fork does not yet — it
landed upstream after the fork's last sync.

Without that renderer/parser, an ornith model would fall back to a generic template: it can
emit text, but its thinking blocks and tool calls won't be shaped the way the model was
trained to expect, so it can't be trusted for the agentic / thinking workloads it's meant
for. Porting the ornith renderer/parser — adapted to the fork's own qwen3.5 rendering, which
has deliberately diverged from upstream — closes that gap.

The models also aren't in CI. Every supported model earns a model-suite regression, and — as
with the LFM2 story — the user wants that regression to run on **both** self-hosted runners:
the K80 (sm37, reference) and the RTX 2060 (sm75).

There is one hardware hazard specific to this family. Because the ornith models run on the
qwen35 architecture, they **enable flash attention**. On the K80 that is harmless (FA is
hard-gated off below Volta). On the RTX 2060 (Turing) it is not: FA on Turing under this
fork's CUDA toolchain produces NaN and crashes the runner unless the tested image carries the
Volta-only FA gate (the fix from the earlier qwen3.5 crash work). So the sm75 validation is
only meaningful on an image that includes that gate.

Scope is the two text ornith models. Any vision or audio variant is out of scope.

## Success Looks Like

- `ollama run ornith:9b` and `ollama run ornith:35b` produce coherent output on the K80 — no
  CUDA/CUBLAS errors, no garbage — each within the VRAM / GPU-count it needs.
- ornith's chat template and thinking behaviour match upstream's intent: the ported
  renderer/parser reproduce upstream's expected rendered output (its golden-output tests pass
  on CPU), and thinking blocks render as the model expects.
- Both models run coherently on the RTX 2060 (sm75) **without the flash-attention NaN crash** —
  i.e. on an image that carries the Volta-only FA gate.
- Both models are covered by the **model suite** on **both** runners (sm37 and sm75), driven
  by a **reviewed** QA test doc, with results captured. Where a model is too large for the
  2060's 6 GB, the offload/throughput is reported as an observed fact, not a correctness fail.
- Honesty note: throughput is whatever each card delivers — reported, not held to a target.

## Open Questions

- **ornith:9b / ornith:35b arch confirmation.** Is 9b dense `qwen35` and 35b `qwen35moe`, as
  the naming and the fork's existing qwen3.6:35b-MoE testcase suggest? Confirm from the real
  GGUF `general.architecture`.
- **Renderer/parser adaptation.** Upstream's `OrnithRenderer` embeds a `Qwen35Renderer` type
  the fork doesn't have (the fork routes qwen3.5 through `Qwen3VLRenderer`). How much of
  ornith's think-block behaviour (`alwaysRenderAssistantThinkBlock`, `emitEmptyThinkOnNoThink`)
  reconstructs cleanly on the fork's renderer, and do upstream's golden-output tests port as-is
  as the acceptance bar?
- **Renderer/parser selection.** Does an ornith model get its renderer/parser via explicit
  Modelfile config, or does the server auto-resolve it — and does that resolution path need an
  `ornith` entry in the fork?
- **sm75 FA-gate precondition.** Which built image will the sm75 runs use, and does it carry
  the #385/#395 Volta-only FA gate? (Ornith on a pre-gate image crashes on Turing.)
- **Image rebuild.** The renderer/parser port is new code, so a fresh build is needed before
  the models run correctly — via CI build, not a local `make`.
- **ornith:35b fit.** Does the MoE fit one K80 die, need both, or spill? On the 6 GB 2060 it
  will offload heavily to CPU — is sm75 coverage of the 35b worth the wall-clock, or is 9b the
  meaningful sm75 target?
- **Availability.** Are `ornith:9b` and `ornith:35b` pullable tags in the registry the runners
  pull from?

## Status

- Created: 2026-07-12
- Plan: #416
- Issues: #421 (prereq confirm), #422 (renderer/parser port), #423 (CI coverage), #424 (K80 build+verify), #425 (sm75 verify)
