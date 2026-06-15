# Write the Test Docs

```
Turn a reviewed test plan into readable test docs in docs/tests/ — reusing vetted
steps from the store instead of re-inventing them.

Target: the scenarios approved by /qw-review-plan for a STORY-XXX.

## PURPOSE

The authoring producer of the qa-workflow — the test analogue of `dw-implement`.
Writes each planned scenario as a `docs/tests/TS-*.md` doc in the format contract
(docs/tests/README.md): front-matter + cases, each case a Steps table of
Action / Expected Result rows.

Fits in the qa-workflow:

    qw-plan → qw-review-plan → qw-cases → qw-review-cases → qw-bind → qw-run
    (qw-run = `make up` + the cicd runner — a phase, not a slash command)

---

## WORKFLOW

    /qw-cases STORY-003
        │
        ├─► Step 1: One file per scenario
        │   - Create docs/tests/TS-NN-<slug>.md with front-matter:
        │       id, title, namespace, story (+ story_hash = sha256 of the story file),
        │       issue, status: green
        │   - (Format and field meanings: docs/tests/README.md.)
        │
        ├─► Step 2: Write each case (TC) — reuse before re-inventing (dogfood the store)
        │   - For each step you are about to write, ask the store first:
        │       make query Q="<the action you mean>"
        │     If a vetted step comes back close, phrase yours to match it — same
        │     meaning, same good expected result — instead of coining a new one.
        │   - Fill the Steps table: each row one Action + its Expected Result.
        │
        ├─► Step 3: Index + bind
        │   - Load the new docs into the store:  npm --prefix step-store run load-tests
        │   - Bind each case to its executable:  /qw-bind  (then /qw-review-bind)
        │
        └─► Step 4: Hand off
            - Run `/qw-review-cases` to gate the docs.

---

## API Notes

- Reuse is the point of the store: `search_step` (via `make query`) makes a vetted
  step findable so coverage converges instead of duplicating.
- `story_hash`: `sha256sum docs/stories/STORY-XXX.md`.
- Producer paired with `/qw-review-cases`.
```
