# Review the Test Plan

```
Check the proposed scenarios cover the story — and stay coverage, not a frozen
step-by-step spec.

Target: the scenario list from /qw-plan for a STORY-XXX — as proposed in the
conversation, or as already realized in docs/tests/ when reviewing after the fact.

## PURPOSE

The paired review for `/qw-plan`. Gates the plan before `qw-cases` writes any docs,
so coverage gaps are caught cheaply.

Fits in the qa-workflow:

    qw-plan → qw-review-plan → qw-cases → qw-review-cases → qw-bind → qw-run
    (qw-run = `make up` + the cicd runner — a phase, not a slash command)

---

## WORKFLOW

    /qw-review-plan STORY-003
        │
        ├─► Step 1: Coverage vs the story
        │   - [ ] Every item in the story's "Success Looks Like" maps to a scenario.
        │   - [ ] Nothing essential to verifying the story is missing.
        │   - [ ] No scenario goes beyond the story's need.
        │
        ├─► Step 2: Each scenario
        │   - [ ] One coherent slice; independently runnable.
        │   - [ ] Maps to at least one cicd executable (or names the gap to author).
        │   - [ ] No duplication of a scenario already in docs/tests/ (grep the story link).
        │
        └─► Step 3: Decision
            - PASS: covers the story → proceed to `/qw-cases`.
            - REVISE: name the missing or excess scenario; back to `/qw-plan`.

---

## API Notes

- Coverage gate only — step detail is `qw-cases`/`qw-review-cases`'s job.
- Review paired with the producer `/qw-plan`.
```
