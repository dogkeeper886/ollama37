# Create a Pull Request

```
Push branch and open a pull request with issue linkage.

Issue number: {{input}}

## PURPOSE

Creates a pull request for the current branch, linking it to the GitHub Issue
via "Fixes #N" for auto-closure. Updates issue labels to reflect PR status.

---

## WORKFLOW

    /dw-create-pr 27
        │
        ├─► Step 1: Verify Readiness
        │   - Confirm you're on the correct branch (issue-<N>-<slug>)
        │   - Run: git status — check for uncommitted changes
        │   - Review the branch's commits against the repo's default branch —
        │     derive it, don't hardcode `main` (gh repo view --json
        │     defaultBranchRef -q .defaultBranchRef.name):
        │     git log --oneline <default>..HEAD
        │   - If no argument given, infer issue number from branch name
        │   - Run: gh issue view <N> — check if title contains [STORY-XXX]
        │
        ├─► Step 2: Push Branch
        │   - Run: git push -u origin $(git branch --show-current)
        │
        ├─► Step 3: Create PR
        │   - Title: short, imperative, under 70 characters
        │   - Body must include "Fixes #N" or "Closes #N"
        │   - Use this template:
        │
        │       gh pr create --title "<title>" --body "$(cat <<'EOF'
        │       ## Summary
        │       <1-3 bullet points>
        │
        │       Fixes #<issue-number>
        │       (if linked to story: "Part of STORY-XXX")
        │
        │       ## Test plan
        │       - [ ] ...
        │
        │       Generated with [Claude Code](https://claude.com/claude-code)
        │       EOF
        │       )"
        │
        ├─► Step 4: Update Issue Labels
        │   - Run: gh issue edit <N> --remove-label "status:in-progress" \
        │          --add-label "status:needs-review"
        │   - Comment on issue:
        │     gh issue comment <N> --body "PR #<PR> created. Summary: <what changed>"
        │
        ├─► Step 5: Update Story File (if linked)
        │   - If the issue title contains [STORY-XXX]:
        │     update docs/stories/STORY-XXX.md status to reflect PR is open
        │   - Skip silently if no story link or docs/stories/ doesn't exist
        │
        └─► Step 6: Report
            - Show the PR URL to the user
            - Stop here — don't auto-advance. The PR now waits for a HUMAN to
              review and test it. Merge with /dw-merge <PR> only once a human is
              satisfied. (The change's substance was already gated locally by
              /dw-review-implement before the PR.)

---

## EXAMPLE

    /dw-create-pr 27

**Agent verifies, pushes, creates PR:**

    $ git status
    $ git log --oneline <default-branch>..HEAD
    $ git push -u origin issue-27-release-notes
    $ gh pr create --title "Add release notes generator command" --body "..."
    $ gh issue edit 27 --remove-label "status:in-progress" --add-label "status:needs-review"
    $ gh issue comment 27 --body "PR #30 created."

**Output:**

    PR #30 created: https://github.com/owner/repo/pull/30
    A human reviews + tests it; merge with /dw-merge 30 when satisfied.

---

## API Notes

- Uses `gh` CLI for PR and issue operations
- `Fixes #N` in PR body auto-closes the issue when PR is merged
- Copy relevant labels from the issue to the PR if needed
- If branch is already pushed, the push step is a no-op
```
