# Review an Implementation

```
Review the changes an issue was implemented with — do they deliver the issue, and
do they stay surgical rather than sprawling beyond it?

Issue number: {{input}}

## PURPOSE

Quality-gates the work done by `/dw-implement` before it becomes a PR. Checks that
the changes on the branch **deliver the issue** (every "Done When" is actually
satisfied) and stay **surgical** (every changed line traces to the issue — no scope
creep, dead code, or debug leftovers) while **fitting the project**. Fixes small
findings on the branch in place, or hands back to `/dw-implement` if the approach is
wrong.

Fits between `/dw-implement` (does the work) and `/dw-create-pr` (opens the PR):

    … → dw-implement → dw-review-implement → dw-create-pr → …

---

## WORKFLOW

    /dw-review-implement 27
        │
        ├─► Step 1: Gather
        │   - Run: gh issue view <N> (the Goal + "Done When")
        │   - If the title contains [STORY-XXX], read docs/stories/STORY-XXX.md
        │     for the need behind the issue
        │   - See what changed on the branch, against the repo's default branch:
        │     git diff <default>...HEAD   (derive <default>, don't hardcode `main`)
        │   - If there are no changes, report and stop (run /dw-implement first)
        │
        ├─► Step 2: Delivers the issue
        │   - [ ] Every "Done When" box is actually satisfied by the changes —
        │         observable, not asserted. Confirm by running the project's
        │         standard tooling (build / test / render — whatever /dw-implement
        │         used), not by reading the diff alone
        │   - [ ] Nothing the issue asked for is missing or stubbed out
        │   - [ ] The changes match the issue's Goal, not a different problem
        │
        ├─► Step 3: Surgical
        │   - [ ] Every changed line traces to this issue — no unrelated refactors,
        │         "improvements", or reformatting of adjacent code
        │   - [ ] No scope creep beyond the issue's Goal (extra features, speculative
        │         abstractions, config nobody asked for)
        │   - [ ] No dead code, commented-out blocks, debug prints, or leftover TODOs
        │   - [ ] Now-unused imports/vars/files the change created are removed
        │
        ├─► Step 4: Fits the project
        │   - [ ] Matches the project's conventions and existing patterns/style
        │   - [ ] Core stays target-agnostic — no vendor names/coupling leaked in
        │         (vendor specifics belong in a profile)
        │   - [ ] Markdown stays the source of truth; cross-references resolve to
        │         files that exist
        │   - [ ] The issue records the work (start / fixes / result) per /dw-implement
        │
        ├─► Step 5: Decision
        │   - PASS: delivers the issue, surgical, fits → ready for a HUMAN to
        │     review + test, then open a PR (/dw-create-pr <N>) — don't auto-advance
        │   - REVISE: specific findings — fix on the branch, smallest blast radius
        │     first (remove leaked scope, fill a gap, tighten); commit
        │   - HAND BACK: the approach is wrong (wrong design, misread the issue) →
        │     comment the issue and route back to /dw-implement
        │
        └─► Step 6: Report
            - Per finding: the exact file:line + the smallest fix
            - A delivery verdict against the issue's "Done When"
            - The issue link + suggested next step

---

## API Notes

- Uses `gh` to read the issue; reads the branch diff with `git` — read-mostly
- Reviews the local implementation result before it becomes a PR — pairs with
  /dw-implement the way /dw-review-tasks pairs /dw-tasks
- This is the substance gate (delivers + surgical), run on the local diff before
  the PR — there is no separate PR-review command. PR mechanics (linkage, labels)
  are handled by /dw-create-pr and /dw-merge, and a human reviews + tests the PR
  before merge.
- Surgical-change bar mirrors the project's CLAUDE.md (every changed line traces to
  the request; clean up only your own orphans)
- When fixing, change only what a finding points to — don't rewrite sound work
```
