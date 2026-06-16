# STORY-008: The MCP tool-call test verifies answers against live ground truth

## User Story

As a maintainer of the K80 fork,
I want the MCP tool-call test to verify a model's answer against an independent check of the live tool — not just its own internal consistency,
So that a confidently-wrong answer cannot quietly pass.

## The Need

The MCP tool-call test asks "can this model drive a real tool and faithfully use what it got
back?" — but today its semantic grader only reads the trajectory the **test harness captured and
handed it**: the prompt, the tool result, and the model's answer. The grader never
independently checks the live system. So two real failure modes slip through as PASS: a model
whose answer is internally consistent with a *wrong-but-plausible* result it received, and a
case where the captured result is stale or was mis-captured. For a test whose entire purpose is
deciding whether a model can be trusted to use real tools, the verification should confirm the
answer against the **live tool itself**, not take the captured result on faith.

This is a verification *upgrade* (not new documentation) — it changes how a result is judged, so
it is deliberately separate from the documentation/CI backfill (STORY-007).

## Success Looks Like

- The MCP test's verdict reflects an **independent** check of the live tool, not only the
  captured trajectory.
- A model whose answer is internally consistent but **contradicts the live tool** is caught and
  fails — where today it would pass.
- The independent check is **read-only**: it never creates, updates, or deletes anything in the
  real system.
- The upgrade applies to the MCP tool-call test specifically and does **not** weaken or change
  the generic semantic judge the other tests rely on.

## Open Questions

- How the grader is given **safe, read-only** access to the live tool, and how writes are
  prevented — the design and its security are worked out when this is planned.
- Whether the independent check runs always or is opt-in, given it costs a second real tool call
  per judged case — settled in the plan.
- Relationship to STORY-007: independent of the doc/CI backfill (this touches the runner/judge),
  but both concern the same MCP test — sequencing decided when planned.

## Status

- Created: 2026-06-16
- Plan: #272
- Issues: #279, #280, #281
