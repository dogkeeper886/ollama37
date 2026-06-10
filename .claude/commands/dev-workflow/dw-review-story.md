# Review a User Story

```
Review a story for completeness and ensure it stays a goal document, not a spec.

Story ID: {{input}}

## PURPOSE

Quality-gates a story written by `/dw-story` before it becomes work. Checks two
things: the story is **complete** (every goal section is present and substantive),
and it is a **goal document, not a spec** (it states the need and the outcome, and
leaves the *how* to the GitHub issue). Reports findings and revises the story, or
hands it back to `/dw-story` if the need itself is unclear.

---

## WORKFLOW

    /dw-review-story STORY-001
        │
        ├─► Step 1: Read the Story
        │   - If no story ID provided, list files in docs/stories/ and ask user to pick
        │   - Read docs/stories/STORY-XXX.md
        │   - If the story file doesn't exist, report and stop
        │
        ├─► Step 2: Completeness Checklist
        │   - [ ] Title is specific (not a placeholder like "[Title]")
        │   - [ ] User Story has a real role, action, AND benefit — the "so that"
        │         is a genuine benefit, not a restatement of the action
        │   - [ ] The Need explains the problem behind the request (the why), in the
        │         user's terms — not how it will be built
        │   - [ ] Success Looks Like lists outcomes someone could actually observe
        │   - [ ] Open Questions captures what's genuinely unresolved (or says
        │         "none known" — empty-because-skipped is a finding)
        │   - [ ] Status has a Created date
        │
        ├─► Step 3: Goal-not-Spec Checklist
        │   - [ ] No implementation detail in the body — no specific files, APIs,
        │         frameworks, data shapes, or step-by-step "how"
        │   - [ ] Success Looks Like describes observable user-facing results, not
        │         implementation steps ("the project shows the stories", not "parse
        │         markdown with X and render with Y")
        │   - [ ] The technical / uncertain "how" lives in Open Questions, deferred
        │         to the issue — it is not asserted as decided in the body
        │   - [ ] No acceptance-criteria checklist acting as a build spec
        │   - [ ] The whole story is decidable by someone who won't implement it
        │
        ├─► Step 4: Decision
        │   - PASS: both checklists clear → report ✓ and suggest /dw-tasks STORY-XXX
        │   - REVISE: list each finding with the exact line, then fix in place —
        │     remove leaked spec, fill missing sections, reword success into outcomes
        │   - HAND BACK: if the *need* itself is unclear or wrong (not just the
        │     wording), stop and route to /dw-story to re-capture it with the user
        │
        └─► Step 5: Report
            - Print the verdict (PASS / REVISED / HAND BACK) and the findings
            - Print the path to the story and the suggested next step

---

## EXAMPLE

    /dw-review-story STORY-001

**Agent reads the story, runs both checklists:**

    Completeness:
    ✓ Title specific — "See the project's stories"
    ✓ User Story — role + action + benefit all present
    ✓ The Need — explains why (the project is the face over the markdown source of truth)
    ✓ Success Looks Like — observable (the project shows stories, reflects changes)
    ✓ Open Questions — presentation, status, read strategy all deferred
    ✓ Status — Created date present

    Goal-not-Spec:
    ✓ No files/APIs/frameworks asserted in the body
    ✓ Success is outcome-based, not implementation steps
    ✓ The "how" sits in Open Questions
    ✓ No build-spec checklist

**Output:**

    PASS — STORY-001 is complete and stays a goal.
    docs/stories/STORY-001.md
    Next: /dw-tasks STORY-001 to open the issue where the "how" gets decided.

---

## API Notes

- Reads only the story file — no GitHub calls
- The story is a goal; the issue is where the *how* and its history live (see
  docs/stories/README.md)
- When revising, change only what a finding points to — don't rewrite a sound story
```
