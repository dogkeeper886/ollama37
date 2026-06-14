---
paths:
  - ".claude/commands/dev-workflow/**/*.md"
---

# dev-workflow

Turns a need into shipped code. A story states the **need** (a goal, not a spec); a
**plan issue** states the agreed **approach**; task issues carry the *how* as it's worked
out. Each producer is paired with a review — no producer ships without one.

## The flow

```
   docs/stories/STORY-XXX.md
            │
   dw-story ───────► dw-review-story    the need — complete, and a goal not a spec
            │
            ▼
   dw-plan ────────► [human reviews the plan issue]   research → ONE plan issue: the approach
            │                                           (skip for trivial / single-task work)
            ▼
   dw-tasks ───────► dw-review-tasks    plan issue → task issues, each "Part of #<plan>"
            │
            ▼
   dw-implement ───► dw-review-implement   build the issue; the plan issue is the
            │                               checkpoint a fresh session resumes from
            ▼
   dw-create-pr ───► [human review + /review]   open the PR
            │
            ▼
   dw-merge          green CI + human review → merge
```

## The plan issue

The plan is a **GitHub issue**, one per story, labelled `plan`, titled `[STORY-XXX] Plan`.
Its body holds the researched approach, acceptance criteria, and the commands/files it
expects to touch. It is the **parent** of the task issues (`dw-tasks` links each task back
with "Part of #<plan>"), and the durable checkpoint that survives a lost session.

Its review is a **human gate**: a person reads, comments, and approves the issue on GitHub
before `dw-tasks` decomposes it — no `dw-*` command produces or gates it (mirrors the
`[human review + /review]` gate before a merge).

## Producer → review pairing

| Producer | Review | Covers |
|----------|--------|--------|
| `dw-story`     | `dw-review-story`     | the need: complete, and a goal not a spec |
| `dw-plan`      | **human review** (the plan issue) | the approach covers the story, before decomposition |
| `dw-tasks`     | `dw-review-tasks`     | the issues cover the plan; each lean; trace back to the plan |
| `dw-implement` | `dw-review-implement` | the change delivers the issue, surgical, fits the project |
| `dw-create-pr` | *(human review + `/review`)* | the PR overview before merge |
| `dw-merge`     | *(is the terminal gate)* | green CI + human review |

No producer ships without a review covering its output.

## Right-size it

The plan stage is for non-trivial stories. A one-line doc change or a single-task story
**skips `dw-plan`** — `dw-story` hands off straight to `dw-tasks` (or a lone issue). Three
review passes plus a plan issue on a typo is ritual, not rigor.

## What is reused, not rebuilt

- **The story + issues** are the unit of work — a story in `docs/stories/`, its task
  issues on GitHub.
- **GitHub issues** already hold the work; the plan is one too (the parent), so the
  approach, its review, and its history live where the tasks do — no separate plan store.
- **CI** is the project's existing checks + human review — the merge gate, not a new pipeline.
