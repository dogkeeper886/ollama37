# STORY-021: K80 model-sizing reference by memory tier

## User Story

As a K80 operator choosing which model to run,
I want a reference doc that tells me the right model tag for the hardware I have,
So that I pick a size that fits without trial-and-error OOMs or wasted dies.

## The Need

The K80 testbed exposes three usable memory tiers — one GK210 die (~11.4 GiB), two
dies (~22.8 GiB), and three dies (~34.2 GiB; the host has four, but nothing in the set
needs the fourth) — but nothing records which model, at which size, belongs on which
tier. Picking is guesswork:
too big and it OOMs or needlessly spreads across dies; too small and the hardware is
wasted. A recent session traced every model's architecture, engine, and real footprint
and produced that mapping ad hoc; without a durable home it will be lost and re-derived.

There is also a correctness trap worth capturing: within one arch, a smaller tag is not
always a drop-in for a larger one. Some `gemma4` tags do vision **and** audio, others
only vision, so "just run a smaller gemma4 to save memory" can silently drop a
capability. The doc has to reflect that, not collapse each arch into one size ladder.

## Success Looks Like

- A file under `docs/` a K80 operator can open and, knowing only their available VRAM,
  read off a model tag that fits.
- It contains **three tables**, one per tier: **single die** (~11.4 GiB), **two dies**
  (~22.8 GiB), and **three dies** (~34.2 GiB). No model in the current set needs a 4th die.
- Each table lists the recommended tag(s) per arch family that fit that tier, with size,
  laid out so generations (Gemma 3↔4, Qwen 3.5↔3.6) and like sizes compare side by side.
- Families with no tag at a tier are flagged (not silently omitted), and gemma4's three
  paths are marked as non-substitutable rather than collapsed into one size ladder.
- The **on-host roster and the model tests match the reference**: redundant/superseded
  tags are removed, representative tags are the ones tested (per-family `TC-MODELS-*`),
  and the measurement sweep's defaults name only models that exist.

## Open Questions

- Exact fit thresholds — where the weight-size cutoff sits per tier once KV cache and a
  realistic context are accounted for (short-prompt fit vs. spreads-at-context).
- Whether to list only tags currently on the host, or also known-pullable tags (some
  tier-fillers were deleted this session and would need re-pull).
- How much of the arch → engine → renderer/parser trace to carry into this doc vs. link
  out to the existing path/trace docs.
- Where exactly the file lives (`docs/` root vs. `docs/reports/` vs. `docs/research/`)
  and whether it should be a generated snapshot or hand-maintained.

## Status

- Created: 2026-07-12
- Plan: #435
- PR: #436
