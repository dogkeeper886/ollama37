# STORY-011: A verifier PASS means the claim was checked against ground truth, not just believed

## User Story

As a maintainer relying on the live verifier's verdict,
I want a PASS to be backed by a fact the harness checked itself,
So that the verdict reflects the real data — not the verifying model's say-so or its tidied-up retelling.

## The Need

The live verifier retrieves real data and an independent model judges the answer against it. But
the trust currently rests on that model's prose: it decides PASS/FAIL and reports the evidence in
its own words. We saw it return evidence that looked clean and correct but was a paraphrase the
model reconstructed — plausible, partly inferred, not what the tool actually returned. For an exact
match that's harmless; for a subtle case (a near-miss id, a renamed item, partial data) a model
that will tidy up evidence may also be too quick to call PASS. The harness already captures the
real data, so the verdict should be anchored to that captured fact, with the model's opinion as a
second check rather than the sole authority.

## Success Looks Like

- A PASS is gated on a check the harness performs itself against the data it captured — the facts
  the answer claims must actually be present in the real result before a PASS stands.
- The verdict judges the model's **whole attempt against a clear guideline**, not just whether the
  final text matched. The guideline grades the three stages where tool use fails:
  - **Tool selection** — did the model make the right call(s) for the question (the necessary ones,
    no wrong or wasteful extras)? — not assuming exactly one call.
  - **Query** — did it invoke the tool with correct arguments (right filters, no invented params)?
  - **Interpretation** — is the final content both **correct/complete** *and* **grounded**: every
    fact it states actually came from the tool result, nothing plausibly invented.
- The verifying model's role is a second opinion over a verified fact, not the only judge.
- The verdict's evidence is the real captured data; the harness no longer relies on the model to
  report (and possibly invent) what it saw.
- The model's answer and the ground truth are kept clearly distinct, so the verifier isn't anchored
  to the very claim it's meant to test.
- A contradicted, incomplete, or invented answer is caught even when the verifying model would have
  waved it through.

## Open Questions

- The verifier currently sees only the model's *final answer* — to judge tool selection and query
  it needs the model's actual **tool calls** (the trajectory). What to pass, and how much.
- How strict the harness-side check should be (exact presence vs normalized/fuzzy matching) and how
  it behaves for tools whose answer isn't a simple set of facts.
- Whether the three-stage guideline yields one PASS/FAIL or a per-stage breakdown, and whether all
  three must pass or some are advisory.
- Whether the model opinion stays a required second gate or becomes advisory.
- How to keep this general across different servers/tools rather than tailored to one.

## Status

- Created: 2026-06-18
- Plan: #298
- Issues: #308, #309, #310
