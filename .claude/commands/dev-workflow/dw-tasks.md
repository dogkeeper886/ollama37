# Break a Plan into GitHub Task Issues

```
Break a reviewed plan (or a story) into GitHub task issues, each linked back to it.

Story ID: {{input}}

## PURPOSE

Reads the reviewed **plan issue** for a story (`[STORY-XXX] Plan`, written by `/dw-plan`)
and breaks its approach into implementable GitHub task issues, each linking back to the
plan. When no plan issue exists — trivial work that skipped the plan stage — it falls
back to breaking the story directly. Updates the story file with the created issue
numbers. See `.claude/rules/dev-workflow.md`.

Fits in the dev-workflow:

    dw-story → dw-review-story → dw-plan → [human reviews the plan issue]
             → dw-tasks → dw-review-tasks → dw-implement → …

---

## WORKFLOW

    /dw-tasks STORY-003
        │
        ├─► Step 1: Read the plan (or the story)
        │   - If no story ID provided, list files in docs/stories/ and ask user to pick
        │   - Read docs/stories/STORY-XXX.md for the need + "Success Looks Like"
        │   - Find the reviewed plan issue:
        │     gh issue list --search "[STORY-XXX] Plan" --label plan --state open
        │     • If found: read it — its Approach + Acceptance Criteria are what you
        │       break into tasks (the plan is the agreed how). Note its number <plan>.
        │     • If none: fall back to breaking the story directly (trivial work that
        │       skipped the plan stage). <plan> is empty.
        │   - If the story file doesn't exist, report and stop
        │
        ├─► Step 2: Break into Tasks
        │   - Break the plan's approach (or the story, when no plan) into tasks
        │   - Each task should be:
        │     • Small — completable in one session
        │     • Independent — minimal dependencies between tasks
        │     • Testable — has a clear done condition
        │   - Detect project type to suggest task breakdown:
        │     • Check project structure, CLAUDE.md, and existing code patterns
        │     • Use these to inform sensible task boundaries
        │   - Present the proposed task list to the user for approval
        │
        ├─► Step 3: Check for Duplicates
        │   - Run: gh issue list --search "[STORY-XXX]" --state all
        │   - If matching issues exist, report and ask user how to proceed
        │
        ├─► Step 4: Create Labels (idempotent)
        │   - Ensure labels exist (same as /dw-plan):
        │     • type labels: feature, enhancement, bug, docs
        │     • priority labels: priority:high, priority:medium, priority:low
        │     • status labels: status:in-progress, status:needs-review, status:blocked
        │   - Use: gh label create "<name>" --color "<hex>" --force
        │
        ├─► Step 5: Create GitHub Issues
        │   - One issue per task
        │   - Title: [STORY-XXX] Task description
        │   - Body — start lean; the issue grows as the *how* is worked out
        │     (research, PoC, clarification, fixes) and recorded in comments:
        │       ## Context
        │       Part of [STORY-XXX](../docs/stories/STORY-XXX.md) · plan #<plan>
        │
        │       ## Goal
        │       [what this task achieves, from the plan's approach / the story's need]
        │
        │       ## Done When
        │       - [ ] [observable done condition for this task]
        │   - Labels: type + priority (infer from the plan / story content)
        │   - Trace back: "Part of #<plan>" in the body links each task to the plan
        │     issue (GitHub backlinks it). Omit the plan reference when there is none.
        │
        ├─► Step 6: Update Story File
        │   - Update the Status section in docs/stories/STORY-XXX.md:
        │     ## Status
        │     - Created: [original date]
        │     - Issues: #1, #2, #3
        │
        └─► Step 7: Report
            - Show table of created issues:
              | Issue | Title | Type | Priority |
            - Suggest implementation order based on dependencies
            - Suggest: /dw-review-tasks STORY-XXX to gate the breakdown, then
              /dw-implement <N> to start on the first task

---

## EXAMPLE

    /dw-tasks STORY-003

**Agent reads the plan, breaks it into tasks, creates issues:**

    $ cat docs/stories/STORY-003.md
    $ gh issue list --search "[STORY-003] Plan" --label plan --state open   # → plan #28
    $ gh issue list --search "[STORY-003]" --state all                      # dup check
    $ gh issue create --title "[STORY-003] Add input validation" \
        --label "enhancement" --label "priority:high" \
        --body "## Context\nPart of STORY-003 · plan #28\n\n## Goal\n...\n\n## Done When\n- [ ] ..."
    $ gh issue create --title "[STORY-003] Add error response formatting" \
        --label "enhancement" --label "priority:medium" \
        --body "..."

**Story file updated:**

    ## Status
    - Created: 2026-04-01
    - Issues: #15, #16

**Output:**

    | Issue | Title                              | Type        | Priority |
    |-------|------------------------------------|-------------|----------|
    | #15   | [STORY-003] Add input validation   | enhancement | high     |
    | #16   | [STORY-003] Add error formatting   | enhancement | medium   |

    Suggested order: #15 → #16
    Start: /dw-implement 15

---

## API Notes

- Uses `gh` CLI for issue operations
- Story files live in `docs/stories/STORY-XXX.md` (created by /dw-story)
- Issue titles use `[STORY-XXX]` prefix for traceability
- The reviewed plan issue (`[STORY-XXX] Plan`, from /dw-plan) is the source for the
  breakdown when present, and each task links back to it; no plan → break the story
- The story file records the need; the issue is the single source of truth for *how*,
  growing with research, decisions, and fixes as the work proceeds
```
