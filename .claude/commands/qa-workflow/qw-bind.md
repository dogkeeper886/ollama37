# Bind a Test Doc to its Executable

```
Link each case in a test doc to the cicd executable that runs it — or port an
existing executable into a new test-doc scaffold (the revert direction).

Target: a docs/tests/TS-*.md scenario, or a cicd YAML to port.

## PURPOSE

Binding is **audit, not codegen** (per the qa-workflow design): the markdown owns
*intent* (why / what), the cicd YAML owns *execution* (how it runs). This command
establishes the link between them; its paired review `/qw-review-bind` checks the
link still holds.

Fits in the qa-workflow:

    qw-plan → qw-cases → qw-bind → qw-review-bind → qw-run → dw-merge
    (qw-run = `npm --prefix cicd/tests test` — the cicd runner; a phase, not a slash command)

---

## WORKFLOW

### A. Forward — bind an existing test doc

    /qw-bind docs/tests/TS-02-runtime.md
        │
        ├─► For each `### TC-NN:` case, set a `Script:` line to the cicd YAML that
        │   runs it (e.g. cicd/tests/testcases/runtime/TC-RUNTIME-001.yml).
        ├─► Keep the case's Steps table aligned 1:1 with the YAML's steps — the
        │   audit treats a step-count mismatch as `unbound`.
        └─► Run `/qw-review-bind` to confirm the binding.

### B. Revert — port an executable into a doc scaffold

    /qw-bind cicd/tests/testcases/build/TC-BUILD-001.yml
        │
        ├─► Scaffold a TS doc from the YAML by hand: one `### TC-NN` per case, a
        │   `Script:` to the YAML, and a Steps row per YAML `step` (row count == the
        │   YAML's `steps:`). The STORY-005 TS docs are worked examples; the format
        │   contract is docs/tests/README.md.
        ├─► Fill the rest: `namespace`, `story` (+ `story_hash`), each TC's objective
        │   and Expected Result column.
        └─► Run `/qw-review-bind` to confirm the binding.
        (An automated `port-yaml` scaffolder is a deferred port — STORY-006/Phase 2.)

---

## API Notes

- Scaffolding is by hand today; an automated `port-yaml` is a deferred port
  (STORY-006/Phase 2). Either way it carries structure, not meaning — a human/agent fills that.
- `story_hash` = `sha256sum docs/stories/STORY-XXX.md` (the drift anchor).
- Producer paired with `/qw-review-bind` (the audit).
```
