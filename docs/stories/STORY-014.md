# STORY-014: The prebuilt image runs on the whole 470-era datacenter GPU fleet, not just the K80

## User Story

As a user running ollama37 on NVIDIA datacenter GPUs other than — or alongside — the Tesla K80,
I want the prebuilt image to run on the full range of cards the 470 driver and CUDA 11.4 toolkit support (Pascal through Ampere, not only Kepler sm_37),
So that a P100, P40, V100, T4, A100, or a mixed box like K80 + P100 just works instead of crashing the moment a model loads.

## The Need

The shipped image compiles GPU code for the K80 (sm_37) **only**. Any other card the
470 driver can drive — P100, V100, T4, A100, and the rest of the Pascal-through-Ampere
range — has no machine code to run, so the first kernel launch faults with XID 43. This
is exactly what closed issue [#223](https://github.com/dogkeeper886/ollama37/issues/223)
reported: a K80 + P100 box where the P100 crashes after a model loads.

The project pins itself to the 470 driver / CUDA 11.4 precisely *because* that's the
last combination that still runs the K80. The quiet upside is that the same combination
can physically run everything up to Ampere — the project is leaving that whole fleet on
the table for no technical reason, only because the build targets a single architecture.
A user who has a P100 (or a mix) and is already on the 470 driver should be able to pull
the same image K80 users pull and have it run on their hardware.

The trade the user has accepted: this is one image for the whole fleet (not a separate
variant), so the K80 build gets longer and the image larger — but a K80's actual
inference stays identical.

## Success Looks Like

- A non-K80 datacenter GPU in the 470 range (P100 first, as the #223 case) **runs a
  model without XID 43** — it loads and produces sane output, instead of crashing.
- A **mixed box** (K80 + a newer card) uses all its GPUs, with the newer card no longer
  taking down the run.
- An existing **K80-only user sees no change in behavior** — same model output as before;
  the only difference is a longer build and a bigger image.
- Putting a non-K80 card in the box **does not silently degrade the K80's own
  performance** — the newer card is additive, not a regression for the K80 already there.

*Honesty note: there is no representative hardware to prove these on. The maintainer's
only non-K80 card is an RTX 2060 on a separate host with a different driver — it can't be
paired with the K80 (so no mixed-arch test) and isn't the 470 target, making it at most a
weak best-effort smoke check that the sm_75 build runs. The #223 P100 case and the
mixed-arch scenario are built-for but stay validation-pending until a real Pascal card on
the 470 stack is available; until then, confidence rests on code-trace and the fact that
the higher-arch paths are intact, upstream-maintained code.*

## Open Questions

- **Validation hardware (largely open).** The only non-K80 card available is an RTX 2060
  on a separate host with a different driver — not the 470 target, and it can't be paired
  with the K80, so no mixed-arch test. At best it's a weak best-effort smoke check of the
  sm_75 build. Real validation — the #223 mixed-arch-on-470 scenario and Pascal's specific
  gate crossings (sm_60 fast-FP16, sm_61 dp4a) — needs hardware we don't have (the #223
  reporter, a cloud instance, or a CI runner); until then we rely on code-trace + intact
  upstream code.
- The exact architecture list and how it's expressed in the build (keeping native code,
  not PTX, so there's no first-run JIT), worked out on the issue.
- ~~How to extend the flash-attention gate (`ml/device.go`) to the newer cards without
  regressing the K80's existing special-case.~~ Resolved: follow upstream — STORY-014
  leaves the FA gate untouched; the K80 special-case was removed in #350, leaving the
  pure-upstream `cc >= 7.0` gate that's already correct for the whole sweep.
- Whether the three K80-specific workarounds that assume an all-K80 box — the CUBLAS
  two-tier fallback, the VMM granularity alignment, and the ghost-GPU layout fix —
  behave correctly on a non-K80 or mixed box, or only need an audit to confirm.
- Whether the longer build time and larger image are acceptable as the cost of a single
  fleet-wide image, or warrant revisiting the one-image decision.

## Status

- Created: 2026-06-21
- Plan: #340
- Issues: #341 ✅ merged (build widened) · #343 ✅ merged (workaround audit) · #344 docs, PR open · #342 open (hardware validation gate)
- Follow-up: #355 (capability-gated vision default)
- Related track: #345 (K80 FA/rotation reversion — owned by STORY-015; the hardware FA
  gate landed in #350, the rotation + #124 model-arch reversion is still pending)
