# Review a Test Doc ↔ Script Binding

```
Audit that each test doc and its bound executable still agree — flag any case
whose doc and script have diverged as `unbound`.

Target: the docs/tests/ scenarios (all, or one named file).

## PURPOSE

The paired review for `/qw-bind`. Binding is audit-not-codegen, so something has to
*check* that the markdown and the YAML haven't drifted apart. This runs the
deterministic audit and adds a human/agent pass for meaning.

Fits in the qa-workflow:

    qw-plan → qw-cases → qw-bind → qw-review-bind → qw-run → dw-merge
    (qw-run = `npm test` — the cicd runner; a phase, not a slash command)

---

## WORKFLOW

    /qw-review-bind
        │
        ├─► Step 1: Run the deterministic audit
        │     npm --prefix cicd/tests run audit-bind
        │   For each case it checks:
        │     - the `Script:` path resolves to a file, and
        │     - the doc's step count matches the YAML's step count.
        │   A failure prints `UNBOUND` with the reason; the command exits non-zero
        │   (so CI and the drift gate, #27, can gate on it).
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

- The audit (`audit-bind`) is structural + deterministic; it is the runnable check
  CI and `/qw-drift` reuse. Semantic agreement is the reviewer's job.
- `unbound` is one of the test doc's `status` values (docs/tests/README.md).
- Review paired with the producer `/qw-bind`.
```
