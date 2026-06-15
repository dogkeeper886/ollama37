---
paths:
  - ".claude/commands/qa-workflow/**/*.md"
---

# qa-workflow

A sibling to `dev-workflow`. Where dev-workflow turns a need into shipped code, qa-workflow
turns a story into **trustworthy test docs** — readable markdown in `docs/tests/`, authored
from a reviewed test plan. This repo owns the **authoring** half (markdown + GitHub); binding
those docs to a runner and running them is the project's own layer.

## The flow

```
   docs/stories/STORY-XXX.md   ──or──  "write a test for X"   (on request)
            │
            ▼
   qw-plan ───────► qw-review-plan      what to test — scenarios persisted as the
            │                            [STORY-XXX] Test Plan issue
            ▼
   qw-cases ──────► qw-review-cases     write docs/tests/TS-*.md (the format contract)
            │
            ▼
   → hand off to the project's binding + run layer
```

## The test-plan issue

`qw-plan`'s scenarios persist as a **GitHub issue**, titled `[STORY-XXX] Test Plan`, labelled
`test-plan` (distinct from dev's `[STORY-XXX] Plan`). `qw-review-plan` reviews it; `qw-cases`
reads it and records the issue number in each `TS-*.md` `plan:` field.

## Producer → review pairing

| Producer | Review | Covers |
|----------|--------|--------|
| `qw-plan`  | `qw-review-plan`  | does the plan cover the story? |
| `qw-cases` | `qw-review-cases` | each doc: one job, observable, traces back |

No producer ships without a review covering its output.

## What this owns — and what it hands off

- **Owns:** the authoring flow + the `docs/tests/` test-doc format (the contract). Self-contained
  — markdown + GitHub only.
- **Hands off:** binding each case to an executable and running it is the **project's binding +
  run layer**. Reusing vetted steps (a search index) is an **optional** project enhancement.

The format a test doc must follow is `docs/tests/README.md`.
