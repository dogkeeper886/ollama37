# Review a Pull Request

```
Review a pull request using a structured checklist.

PR number: {{input}}

## PURPOSE

Reviews a pull request against a structured checklist covering issue linkage,
code quality, test coverage, and documentation. Approves or requests changes.

---

## WORKFLOW

    /dw-review-pr 30
        │
        ├─► Step 1: Read the PR
        │   - Run: gh pr view <PR>
        │   - Run: gh pr diff <PR>
        │   - Run: gh pr checks <PR>
        │
        ├─► Step 2: Review Checklist
        │
        │   Issue Linkage:
        │   - [ ] PR body contains "Fixes #N" or "Closes #N"
        │   - [ ] PR title is clear and under 70 characters
        │   - [ ] Labels match the linked issue
        │
        │   Code Quality:
        │   - [ ] Changes match the issue's acceptance criteria
        │   - [ ] No unnecessary changes beyond what was requested
        │   - [ ] No security vulnerabilities (injection, hardcoded secrets)
        │   - [ ] No debug/temporary code left in
        │
        │   Project-Type Checks (detect and apply relevant items):
        │   - Detect project type by scanning for markers:
        │     • src/mcpServer.ts or @modelcontextprotocol/sdk → MCP server
        │     • package.json with react/next → Frontend
        │     • Dockerfile or docker-compose.yml → Containerized
        │     • pyproject.toml or setup.py → Python
        │     • go.mod → Go
        │
        │   IF MCP server:
        │   - [ ] Service function follows existing patterns in services/
        │   - [ ] Tool registered with clear description (REQUIRED/PREREQUISITE refs)
        │   - [ ] Async operations use standard retry/polling pattern
        │   IF Frontend:
        │   - [ ] No console.log left in production code
        │   - [ ] Components follow existing patterns
        │   IF Containerized:
        │   - [ ] Dockerfile follows best practices (multi-stage, non-root)
        │   - [ ] No secrets in image layers
        │   IF Python:
        │   - [ ] Type hints on public functions
        │   - [ ] No bare except clauses
        │   IF Go:
        │   - [ ] Errors are checked, not discarded
        │   - [ ] No goroutine leaks (context cancellation handled)
        │   (Skip this section silently if no project type detected)
        │
        │   Test Coverage (adaptive — detect what the project uses):
        │   - Detect CI: .github/workflows/, .gitlab-ci.yml, Jenkinsfile, .circleci/
        │   - Detect test infra: cicd/tests/, tests/, __tests__/, *_test.go, *.robot
        │
        │   IF CI exists:
        │   - [ ] CI pipeline passed (run: gh pr checks <PR>)
        │   - [ ] New or updated test files match the scope of changes
        │   - [ ] No test regressions in CI output
        │   IF no CI:
        │   - [ ] Manual test plan present in PR body or docs/test-plans/
        │   - [ ] Test steps are specific and verifiable
        │   - (Informational) Note: project has no CI — see /dw-test-design
        │
        │   Documentation:
        │   - [ ] Comments added where logic isn't self-evident
        │   - [ ] No unnecessary comments or docstrings
        │
        ├─► Step 3: Decision
        │   - Approve: all checks pass → proceed to /dw-merge
        │   - Request changes: specific feedback → back to /dw-implement
        │
        └─► Step 4: Submit Review
            - Check if you are the PR author (GitHub blocks self-approval)
            - If NOT the author:
              gh pr review <PR> --approve --body "LGTM"
            - If you ARE the author (self-review):
              gh pr comment <PR> --body "Self-review complete. Checklist passes. Ready to merge."
            - To request changes:
              gh pr review <PR> --request-changes --body "<specific feedback>"

---

## EXAMPLE

    /dw-review-pr 30

**Agent reads PR, runs checklist:**

    $ gh pr view 30
    $ gh pr diff 30
    $ gh pr checks 30

**Checklist result:**

    ✓ Issue linkage — Fixes #27 present
    ✓ Title — "Add release notes generator command" (42 chars)
    ✓ Code quality — Changes match acceptance criteria
    ✓ No security issues
    ✓ Tests — N/A (markdown-only change)

    $ gh pr review 30 --approve --body "LGTM"

**Output:**

    PR #30 approved. Next: /dw-merge 30

---

## API Notes

- Uses `gh` CLI for PR operations
- GitHub blocks self-approval — use comment instead if you authored the PR
- If CI checks are failing, investigate before approving
- Always read the full diff, not just the PR description
```
