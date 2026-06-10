# Plan Development Work into a Plan Issue

```
Research a story and write the plan — the agreed approach — as ONE GitHub plan
issue, then stop for human review.

Target: a STORY-XXX (or a raw request).

## PURPOSE

The front of the dev-workflow's build half: turn a story's *need* into an agreed
*approach*, captured as a durable, reviewable **plan issue** before any task issue is
opened. It researches against the repo (conventions, prior issues) so the plan is
grounded, writes one plan issue, and **stops** — a human reviews the plan on GitHub,
then `/dw-tasks` decomposes it. The plan issue is the parent of the task issues and the
checkpoint a fresh session resumes from. See `.claude/rules/dev-workflow.md`.

Fits in the dev-workflow:

    dw-story → dw-review-story → dw-plan → [human reviews the plan issue]
             → dw-tasks → dw-review-tasks → dw-implement → …

This command produces the plan only. It does NOT create task issues — that is `/dw-tasks`.

---

## WORKFLOW

    /dw-plan STORY-003
        │
        ├─► Step 1: Read the need (and right-size)
        │   - If a STORY-XXX: read docs/stories/STORY-XXX.md (the need + "Success Looks Like").
        │   - If a raw request: restate the need in a sentence.
        │   - RIGHT-SIZE: if the work is trivial — a one-line change or a single,
        │     obvious task — skip planning. Say so and hand off straight to
        │     /dw-tasks (or a lone issue). The plan stage is for non-trivial stories.
        │
        ├─► Step 2: Research (ground the approach)
        │   - Investigate the repo so the plan reflects it, not generic advice:
        │     • conventions — CLAUDE.md, .claude/rules/, neighbouring code patterns
        │     • prior art — gh issue list --state all --limit 50 ; related stories
        │   - Note the approach, the commands/files it will touch, and open questions.
        │
        ├─► Step 3: Check for an existing plan
        │   - Run: gh issue list --search "[STORY-XXX] Plan" --state all
        │   - If a plan issue already exists, report and ask how to proceed (extend it,
        │     not duplicate).
        │
        ├─► Step 4: Ensure labels (idempotent)
        │   - plan           (color: #5319e7) — The approach for a story, before tasks
        │   - priority:high / priority:medium / priority:low (as in /dw-tasks)
        │   - Use: gh label create "<name>" --color "<hex>" --description "<desc>" --force
        │
        ├─► Step 5: Write ONE plan issue
        │   - Title: [STORY-XXX] Plan   (raw request: "Plan: <short need>")
        │   - Body: the template below — approach, acceptance criteria, the
        │     commands/files it expects to touch, open questions.
        │   - Labels: plan + priority. Link the story: "Part of STORY-XXX".
        │
        ├─► Step 6: Record on the story (if a STORY-XXX)
        │   - Update docs/stories/STORY-XXX.md Status with the plan issue number:
        │     - Plan: #<plan>
        │
        └─► Step 7: Hand off — stop for human review
            - Show the plan issue URL.
            - STOP. The plan now waits for a HUMAN to read, comment, and approve it on
              GitHub. Do NOT create task issues and do NOT auto-advance.
            - Once a human is satisfied: /dw-tasks STORY-XXX breaks the plan into tasks.

---

## PLAN ISSUE BODY TEMPLATE

    ## Context
    Part of [STORY-XXX](../docs/stories/STORY-XXX.md) — <the need, one line>.

    ## Approach
    <How we'll meet the need — the agreed shape of the work, grounded in the repo.>

    ## Acceptance Criteria
    - [ ] <observable outcome that means the story is delivered>

    ## Touches
    - <commands / files / areas the work is expected to change>

    ## Open Questions
    - <what's still unresolved — settled as the tasks are worked>

---

## EXAMPLE

    /dw-plan STORY-003

**Agent reads the story, researches, writes one plan issue:**

    $ cat docs/stories/STORY-003.md
    $ gh issue list --state all --limit 50
    $ gh issue list --search "[STORY-003] Plan" --state all
    $ gh label create "plan" --color "5319e7" --description "The approach for a story, before tasks" --force
    $ gh issue create --label "plan" --label "priority:high" \
        --title "[STORY-003] Plan" \
        --body "## Context
    Part of STORY-003 — validate inbound payloads.
    ## Approach
    ...
    ## Acceptance Criteria
    - [ ] ...
    ## Touches
    - ...
    ## Open Questions
    - ..."

**Story file updated:**

    ## Status
    - Created: 2026-04-01
    - Plan: #28

**Output:**

    Plan issue #28 created: https://github.com/owner/repo/issues/28
    A human reviews + approves it; then /dw-tasks STORY-003 breaks it into tasks.

---

## API Notes

- Uses `gh` CLI — must be authenticated (`gh auth status`)
- Labels use `--force` so existing labels are updated, not duplicated
- One plan issue per story — check for duplicates before creating
- The plan issue is the parent of the task issues; the story records the need, the plan
  records the approach, the task issues carry the how (see .claude/rules/dev-workflow.md)
- Trivial work skips this command — go straight to /dw-tasks
```
