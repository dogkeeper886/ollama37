# Write the Test Docs

```
Turn a reviewed test plan into readable test docs in docs/tests/ — reusing existing
steps instead of re-inventing them.

Target: the scenarios approved by /qw-review-plan for a STORY-XXX.

## PURPOSE

The authoring producer of the qa-workflow — the test analogue of `dw-implement`.
Writes each planned scenario as a `docs/tests/TS-*.md` doc in the format contract
(docs/tests/README.md): front-matter + cases, each case a Steps table of
Action / Expected Result rows.

Fits in the qa-workflow:

    qw-plan → qw-review-plan → qw-cases → qw-review-cases → qw-bind → qw-run
    (qw-run = `npm --prefix cicd/tests test` — the cicd runner; a phase, not a slash command)

---

## WORKFLOW

    /qw-cases STORY-003
        │
        ├─► Step 1: One file per scenario
        │   - Create docs/tests/TS-NN-<slug>.md with front-matter:
        │       id, title, namespace, story (+ the drift anchor the profile declares),
        │       issue, status: green
        │   - (Format and field meanings: docs/tests/README.md.)
        │
        ├─► Step 2: Write each case (TC) — reuse before re-inventing
        │   - Before writing a step, skim the existing docs/tests/ for one that
        │     already means the same thing; phrase yours to match it — same meaning,
        │     same good expected result — instead of coining a new one.
        │   - Fill the Steps table: each row one Action + its Expected Result.
        │
        ├─► Step 3: Bind
        │   - Bind each case to its executable:  /qw-bind  (then /qw-review-bind)
        │
        └─► Step 4: Hand off
            - Run `/qw-review-cases` to gate the docs.

---

## API Notes

- Reuse keeps coverage converging instead of duplicating: skim the existing
  docs/tests/ for a matching step before coining a new one. (A searchable step
  store is a deferred enhancement — STORY-006/Phase 2.)
- Drift anchor: record whatever the profile declares, so a later gate can tell the story
  has moved (default here: `story_hash`, the `sha256sum` of the story file — what `qw-drift`
  reads). A project that detects drift another way declares that instead.
- Producer paired with `/qw-review-cases`.
```
