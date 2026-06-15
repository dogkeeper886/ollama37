# Plan What to Test

```
Derive the scenarios that verify a story (or an on-request target) — the "what
to test", before any test doc is written.

Target: a STORY-XXX, or an ad-hoc request ("write a test for X").

## PURPOSE

The front of the qa-workflow — the test analogue of reading a story before
implementing. It produces a short, reviewable list of **scenarios** (each a TS-to-be)
that together cover the need, so `qw-cases` has a scoped plan to write against.

Fits in the qa-workflow:

    qw-plan → qw-review-plan → qw-cases → qw-review-cases → qw-bind → qw-run
    (qw-run = `make up` + the cicd runner — a phase, not a slash command)

---

## WORKFLOW

    /qw-plan STORY-003
        │
        ├─► Step 1: Read the need
        │   - If a STORY-XXX: read docs/stories/STORY-XXX.md (the need + "Success Looks Like").
        │   - If an on-request target: restate what behaviour is to be verified.
        │
        ├─► Step 2: Check what already exists (dogfood the store)
        │   - Search the step-store for steps/cases already covering this:
        │       make query Q="<the behaviour>"
        │     so the plan reuses vetted coverage instead of duplicating it.
        │   - List the docs/tests/ scenarios already linked to this story:
        │       grep -l 'story: STORY-XXX' docs/tests/
        │
        ├─► Step 3: Propose scenarios
        │   - Break the need into scenarios (TS-to-be), each:
        │     • one coherent slice of behaviour, • independently runnable,
        │     • mappable to one or more cicd executables.
        │   - For each, name the cases (TC-to-be) it will hold, at a sentence each.
        │
        └─► Step 4: Hand off
            - Present the scenario list for `/qw-review-plan`, then `/qw-cases`.

---

## API Notes

- A scenario here is a *plan item*, not yet a file — `qw-cases` writes the doc.
- The story is the goal; keep the plan to coverage, not step detail.
- Producer paired with `/qw-review-plan`.
```
