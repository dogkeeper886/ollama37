# Review the Tasks for a Story

```
Review the GitHub issues a story was broken into — do they cover the story, and does
each stay a lean goal rather than a frozen spec?

Story ID: {{input}}

## PURPOSE

Quality-gates the issues created by `/dw-tasks` before implementation starts. Checks
that **together they cover the story**, and that **each is one small, testable job
that leaves the *how* to develop on the issue** (not front-loaded as a spec). Fixes
issue bodies / labels / links in place, or sends the breakdown back to `/dw-tasks` if
it's wrong.

Fits between `/dw-tasks` (creates issues) and `/dw-implement` (works one):

    dw-story → dw-review-story → dw-plan → [human reviews the plan issue]
             → dw-tasks → dw-review-tasks → dw-implement → …

---

## WORKFLOW

    /dw-review-tasks STORY-001
        │
        ├─► Step 1: Gather
        │   - If no story ID, list docs/stories/ and ask which to review
        │   - Read docs/stories/STORY-XXX.md (the goal + "Success Looks Like")
        │   - Find the plan issue (if any): note its number <plan> — none means
        │     trivial work that skipped the plan stage:
        │     gh issue list --search "[STORY-XXX] Plan" --label plan --state all
        │   - List its task issues (the plan issue is the parent, not a task):
        │     gh issue list --search "[STORY-XXX]" --state all --json number,title,labels,body
        │   - If no issues exist, report and stop (run /dw-tasks first)
        │
        ├─► Step 2: Coverage — the issues vs the story (and plan)
        │   - [ ] Every item in the story's "Success Looks Like" is covered by an issue
        │   - [ ] Nothing essential to delivering the story is missing
        │   - [ ] No issue goes beyond the story's goal — each traces back to the need
        │   - [ ] When a plan exists: each task links back to it ("Part of #<plan>") —
        │         flag any task missing the link. No plan → skip (trivial path).
        │
        ├─► Step 3: Each issue
        │   - [ ] One clear job (title + Goal say one thing)
        │   - [ ] Small enough to finish in a session; independent where it can be
        │   - [ ] "Done When" is observable — checkable without ambiguity
        │   - [ ] Body is lean (Context / Goal / Done When); the *how* is NOT frozen in
        │         — it's left to develop on the issue (research / PoC / comments)
        │   - [ ] Type + priority labels make sense
        │
        ├─► Step 4: Across the set
        │   - [ ] No two issues substantially duplicate each other
        │   - [ ] Dependencies/links are sane (no cycles) and the order is buildable
        │
        ├─► Step 5: Decision
        │   - PASS: covers the story + each issue is clean → suggest /dw-implement <first>
        │   - REVISE: specific fixes — edit the issue body / labels / links in place
        │     (gh issue edit), or comment what's missing
        │   - RE-SPLIT: the breakdown itself is wrong (gaps, wrong boundaries, overlap)
        │     → back to /dw-tasks to re-decompose
        │
        └─► Step 6: Report
            - Per issue: verdict + findings
            - A coverage verdict against the story
            - The story path + issue links + suggested next step

---

## EXAMPLE

    /dw-review-tasks STORY-007

**Agent reads the story + its issues, runs the checks:**

    Coverage (vs the story's "Success Looks Like"):
    ✓ <outcome A>  → #21
    ✓ <outcome B>  → #22
    ~ nothing essential missing; no issue exceeds the goal

    Each issue:
    ✓ #21 one job · Done When observable · lean body (Context / Goal / Done When)
    ✓ #22 one job · Done When observable · lean body

    Across the set: no overlap; deps are sane and the order is buildable

**Output:**

    PASS — the breakdown covers STORY-007 and each issue stays a goal.
    Next: /dw-implement 21

(Illustrative — a hypothetical clean breakdown. Real reviews often return REVISE.)

---

## API Notes

- Uses `gh` to read (and, on REVISE, edit) the issues — read-mostly
- The story is the goal; these issues are the agreed work — keep them lean, the how
  lives in their comments/history (see docs/stories/README.md)
- When fixing, change only what a finding points to — don't rewrite a sound breakdown
```
