# Merge a Pull Request

```
Merge an approved pull request and clean up.

PR number: {{input}}

## PURPOSE

Merges an approved PR, deletes the remote branch, cleans up local branches
and issue labels. The linked issue auto-closes via "Fixes #N" in the PR body;
a plan issue never does — no PR targets one — so this command closes the
story's plans when its last task lands.

---

## WORKFLOW

    /dw-merge 30
        │
        ├─► Step 1: Verify Ready to Merge
        │   - Run: gh pr view <PR> --json mergeStateStatus,headRefName,reviewDecision
        │   - Must be mergeable (no conflicts)
        │   - Run: gh pr checks <PR> — any CI checks must pass (if applicable)
        │   - The merge gate is HUMAN review + test, not a GitHub approval. On a
        │     solo repo GitHub blocks self-approval, so do NOT require
        │     reviewDecision=APPROVED. Before merging, confirm a human has reviewed
        │     and tested the change (its substance was gated by /dw-review-implement
        │     before the PR). If not yet reviewed/tested, stop and say so.
        │
        ├─► Step 2: Identify Linked Issue and Story
        │   - Read PR body for "Fixes #N" or "Closes #N"
        │   - Note the issue number for label cleanup
        │   - Check if PR body or issue title contains [STORY-XXX] or "Part of STORY-XXX"
        │
        ├─► Step 3: Merge
        │   - Run: gh pr merge <PR> --merge --delete-branch
        │   - Uses --merge (not squash/rebase) to preserve commit history
        │   - --delete-branch cleans up the remote branch
        │
        ├─► Step 4: Clean Up Issue Labels
        │   - Run: gh issue edit <N> --remove-label "status:needs-review"
        │   - Issue auto-closes via "Fixes #N" — no manual close needed
        │
        ├─► Step 5: Close Out the Story (if linked)
        │   - If linked to STORY-XXX and docs/stories/STORY-XXX.md exists:
        │     • Check off completed acceptance criteria for this task
        │     • If all story tasks are closed, mark story status as Completed
        │   - On that same condition — all story tasks closed — also close its plan
        │     issues. No PR ever targets a plan, so nothing else will:
        │       gh issue list --search "[STORY-XXX]" --label plan --state open
        │       gh issue list --search "[STORY-XXX]" --label test-plan --state open
        │     Close each, naming the PR that finished the story. Both kinds orphan the
        │     same way; qa has no terminal gate of its own, so this closes both.
        │   - Skip silently if no story link, no docs/stories/, or no plan issue is open
        │
        ├─► Step 6: Clean Up Local Branch
        │   - Switch to the repo's default branch and pull — derive it, don't
        │     hardcode `main` (gh repo view --json defaultBranchRef -q
        │     .defaultBranchRef.name): git checkout <default> && git pull
        │   - Run: git branch -d <branch-name>
        │
        └─► Step 7: Report
            - Report per .claude/rules/agent-report.md; Trace carries the merged PR URL
              and any plan issues closed
            - Next: /dw-implement <next-issue> if the story has open tasks left,
              otherwise say the story is complete
            - A follow-up the merge deliberately leaves behind is Not done (a choice);
              Unresolved is for what the merge left genuinely uncertain

---

## EXAMPLE

    /dw-merge 30

**Agent verifies, merges, cleans up:**

    $ gh pr view 30 --json mergeStateStatus,headRefName,reviewDecision  # mergeable? (don't gate on self-approval)
    $ gh pr checks 30
    $ gh pr merge 30 --merge --delete-branch
    $ gh issue edit 27 --remove-label "status:needs-review"
    $ gh issue list --search "[STORY-003]" --label plan --state open   # last task → close it
    $ gh issue close 28 --comment "Delivered — last task merged in PR #30."
    $ git checkout <default-branch> && git pull
    $ git branch -d issue-27-release-notes

**Output:**

    PR #30 merged: https://github.com/owner/repo/pull/30
    Issue #27 auto-closed. Plan #28 closed — that was STORY-003's last task.
    Branch issue-27-release-notes deleted (local + remote).

---

## API Notes

- Uses `gh` CLI for PR and issue operations
- `--merge` preserves full commit history (use `--squash` only if the project convention requires it)
- `--delete-branch` removes the remote branch; `git branch -d` removes the local one
- If PR is not approved or CI fails, report the blocker instead of merging
- A plan closes only on the story's LAST open task — it stays open while any remain
- The close is hooked to this command, so a story finished another way (tasks closed
  by hand, a PR merged in the UI) still leaves its plan open
```
