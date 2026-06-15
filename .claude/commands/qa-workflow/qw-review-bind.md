# Review a Test Doc ↔ Script Binding

```
Audit that each test doc and its bound executable still agree — flag any case
whose doc and script have diverged as `unbound`.

Target: the docs/tests/ scenarios (all, or one named file).

## PURPOSE

The paired review for `/qw-bind`. Binding is audit-not-codegen, so something has to
*check* that the markdown and the YAML haven't drifted apart. This runs the structural
audit (by hand today) and adds a human/agent pass for meaning.

Fits in the qa-workflow:

    qw-plan → qw-cases → qw-bind → qw-review-bind → qw-run → dw-merge
    (qw-run = `npm --prefix cicd/tests test` — the cicd runner; a phase, not a slash command)

---

## WORKFLOW

    /qw-review-bind
        │
        ├─► Step 1: Run the structural audit (by hand)
        │   For each case, check:
        │     - the `Script:` path resolves to a file, and
        │     - the doc's Steps row count matches the bound YAML's `steps:` count.
        │   Either mismatch = `UNBOUND`.
        │   (An automated `audit-bind` that does this and exits non-zero for CI to
        │   gate on is a deferred port — STORY-006/Phase 2.)
        │
        ├─► Step 2: Read the meaning the audit can't
        │   For each `bound` case, skim that the doc's Actions/Expected Results
        │   still describe what the YAML actually does — structure can match while
        │   meaning has drifted. Flag any semantic mismatch.
        │
        └─► Step 3: Decision
            - PASS: every case `bound` (audit exits 0) and meaning holds.
            - REVISE: for each `UNBOUND` (or semantic mismatch), fix the doc's
              Steps/`Script:` or the binding — smallest change first — then re-run.

---

## API Notes

- The structural audit (the `Script:` path resolves + the row counts match) is done
  by hand today; an automated `audit-bind` that CI and `/qw-drift` reuse is a deferred
  port (STORY-006/Phase 2). Semantic agreement is the reviewer's job.
- `unbound` is one of the test doc's `status` values (docs/tests/README.md).
- Review paired with the producer `/qw-bind`.
```
