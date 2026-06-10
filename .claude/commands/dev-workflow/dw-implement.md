# Implement a GitHub Issue

```
Start work on a GitHub Issue — create branch, implement, and track progress.

Issue number: {{input}}

## PURPOSE

Picks up a GitHub Issue and drives it through implementation. Creates a feature
branch, tracks status via labels and comments, handles failures transparently,
and prepares for PR creation when done.

---

## WORKFLOW

    /dw-implement 27
        │
        ├─► Step 1: Understand the Issue
        │   - Run: gh issue view <N>
        │   - Read acceptance criteria, technical notes, dependencies
        │   - Check labels — type and priority should already be set
        │   - Check for linked/blocking issues
        │   - If the issue body links a plan ("Part of #<plan>"), read that plan
        │     issue — its Approach is the agreed checkpoint to build from, so a fresh
        │     session resumes without re-deriving it: gh issue view <plan>
        │   - If issue title contains [STORY-XXX], read docs/stories/STORY-XXX.md
        │     for full context (the user story and the need it serves)
        │   - If anything is unclear, ask the user before starting
        │
        ├─► Step 2: Create Branch
        │   - Branch name: issue-<N>-<short-slug>
        │     Example: issue-27-release-notes
        │   - Branch from the repo's default branch — derive it, don't hardcode
        │     `main` (e.g. gh repo view --json defaultBranchRef -q
        │     .defaultBranchRef.name); check it out and pull, then:
        │     git checkout -b issue-<N>-<slug>
        │   - Comment on issue:
        │     gh issue comment <N> --body "Starting work on branch \`issue-<N>-<slug>\`"
        │   - Add status label:
        │     gh issue edit <N> --add-label "status:in-progress"
        │
        ├─► Step 3: Implement
        │   - Make the changes based on acceptance criteria
        │   - Build and test using the project's standard tooling
        │     (Makefile, test framework, CI pipeline — whatever the project uses)
        │   - Commit incrementally with clear messages
        │
        ├─► Step 4a: On Success
        │   - Comment on issue:
        │     gh issue comment <N> --body "Implementation complete, tests passing. Ready for PR."
        │   - Progress is recorded on the issue, not the story — the story stays the
        │     stable statement of the need
        │   - Proceed to /dw-review-implement to gate the changes before the PR
        │
        ├─► Step 4b: On Failure
        │   - Do NOT silently retry — update the issue:
        │     gh issue comment <N> --body "Build/test failure: <what failed, error, root cause>"
        │   - If blocked, add label:
        │     gh issue edit <N> --add-label "status:blocked"
        │   - Investigate, fix, update issue:
        │     gh issue comment <N> --body "Applied fix: <what changed>. Retesting."
        │   - Remove blocked label after unblocking:
        │     gh issue edit <N> --remove-label "status:blocked"
        │   - If stuck after 2-3 attempts, comment blockers and ask the user
        │
        └─► Step 4c: On Partial Fix
            - Comment: what was fixed, what remains, blockers
            - If the partial fix is independently useful: run /dw-review-implement,
              then a human reviews + tests before a PR is opened (/dw-create-pr)
            - Create follow-up issues for remaining work

---

## ISSUE CROSS-REFERENCES

Use these patterns in issue comments and PR bodies:
- **Parent/child**: "Part of #N" or "Parent: #N"
- **Dependencies**: "Depends on #N", "Blocked by #N"
- **Related**: "Related to #N"

GitHub auto-creates backlinks when issues reference each other.

---

## EXAMPLE

    /dw-implement 27

**Agent reads issue #27, creates branch, implements:**

    $ gh issue view 27
    $ git checkout -b issue-27-release-notes   # from the repo's default branch
    $ gh issue comment 27 --body "Starting work on branch `issue-27-release-notes`"
    $ gh issue edit 27 --add-label "status:in-progress"

    ... (implementation work) ...

    $ gh issue comment 27 --body "Implementation complete, tests passing. Ready for PR."

**Next step:** /dw-review-implement 27 (local gate). Then a human reviews + tests
before a PR is opened (/dw-create-pr 27) — the workflow doesn't auto-advance to a PR.

---

## KEY PRINCIPLE

The issue is the single source of truth. Anyone reading it should see the full
history — start, failures, fixes, and resolution.

---

## API Notes

- Uses `gh` CLI for issue operations
- Branch naming convention: `issue-<number>-<short-slug>`
- Always comment on the issue before and after implementation
- Label management: `status:in-progress` while working, `status:blocked` if stuck
- For a planned task, the linked plan issue (`Part of #<plan>`) holds the agreed
  approach — the checkpoint a fresh session resumes from (see
  `.claude/rules/dev-workflow.md`)
```
