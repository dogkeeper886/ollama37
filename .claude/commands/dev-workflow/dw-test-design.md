# Design Tests for a GitHub Issue

```
Write tests after implementation, adapted to the project's existing test infrastructure.

Issue number: {{input}}

## PURPOSE

After implementation is complete, this command designs and writes tests that verify
the changes. It detects what test infrastructure the project uses and generates
tests in the native format. If no test infrastructure exists, it writes a manual
test checklist and recommends adopting a test framework.

Fits into the dev-workflow chain after /dw-implement and before /dw-create-pr.

---

## WORKFLOW

    /dw-test-design 47
        │
        ├─► Step 1: Understand What Was Implemented
        │   - Run: gh issue view <N>
        │   - Read the acceptance criteria and technical notes
        │   - See what changed vs the repo's default branch — derive it, don't
        │     hardcode `main` (gh repo view --json defaultBranchRef -q
        │     .defaultBranchRef.name): git log --oneline <default>..HEAD and
        │     git diff <default> --stat
        │   - Identify what needs test coverage
        │
        ├─► Step 2: Detect Test Infrastructure
        │   - Scan the project for existing test tooling:
        │
        │     Test Frameworks:
        │     - cicd/tests/testcases/**/*.yml     → test-framework-template (YAML + dual-judge)
        │     - tests/ + pytest.ini or conftest.py → pytest
        │     - __tests__/ or *.test.ts/js         → Jest
        │     - *.spec.ts/js                       → Jest / Vitest / Mocha
        │     - *.robot                            → Robot Framework
        │     - *_test.go                          → Go test
        │     - **/*Test.java or **/*Spec.java     → JUnit / TestNG
        │     - tests/ + Cargo.toml                → Rust (cargo test)
        │
        │     Test Runners (check even if no test files found):
        │     - package.json → scripts.test        → npm test
        │     - Makefile → test target             → make test
        │     - pyproject.toml → [tool.pytest]     → pytest
        │     - tox.ini                            → tox
        │
        │     CI Pipelines:
        │     - .github/workflows/                 → GitHub Actions
        │     - .gitlab-ci.yml                     → GitLab CI
        │     - Jenkinsfile                        → Jenkins
        │     - .circleci/                         → CircleCI
        │     - .travis.yml                        → Travis CI
        │
        │   - Record findings: which framework, which runner, which CI
        │
        ├─► Step 3: Detect Issue Type (for test scope)
        │   - Read labels from the GitHub issue:
        │     - feature, epic, story       → Type A: broad coverage
        │     - bug, defect                → Type B: regression test targeting the fix
        │     - enhancement, improvement   → Type C: tests for changed behavior
        │   - If no label matches, infer from issue title and description
        │
        ├─► Step 4: Generate Tests
        │
        │   ┌─ IF test infrastructure found:
        │   │  - Write tests in the project's native format
        │   │  - Place files where the project's existing tests live
        │   │  - Follow existing naming conventions and patterns
        │   │  - Run the project's test command to verify tests pass
        │   │  - Commit test files to the issue branch
        │   │
        │   │  Examples by framework:
        │   │  - test-framework-template → YAML in cicd/tests/testcases/{suite}/
        │   │  - pytest   → Python test files in tests/
        │   │  - Jest     → *.test.ts in __tests__/ or co-located
        │   │  - Go test  → *_test.go alongside source files
        │   │  - Robot    → *.robot in tests/ or robot/
        │   │
        │   └─ IF no test infrastructure found:
        │      - Write a manual test checklist in the PR body (Step 5 will use it)
        │      - Format as markdown checklist:
        │        ## Test Plan
        │        - [ ] Step 1: description → expected result
        │        - [ ] Step 2: description → expected result
        │      - Add a recommendation note:
        │        > This project has no automated test infrastructure.
        │        > Consider adopting [test-framework-template](https://github.com/dogkeeper886/test-framework-template)
        │        > for YAML-driven CI testing with dual-judge verification.
        │      - Ask the user: "Want me to create a follow-up issue for CI setup?"
        │        If yes → create issue labeled "enhancement" + "testing"
        │
        ├─► Step 5: Report
        │   - Summarize what was created:
        │     - Test framework detected (or "none")
        │     - CI detected (or "none")
        │     - Files created/modified
        │     - Test run result (if applicable)
        │   - If tests were written, show the test run output
        │   - Suggest: proceed to /dw-create-pr
        │
        └─► Step 5b: On Test Failure
            - If generated tests fail, investigate:
              - Is the implementation incomplete?
              - Is the test expectation wrong?
            - Fix tests or flag implementation gap to the user
            - Do not proceed to /dw-create-pr with failing tests

---

## EXAMPLE 1: Project with test-framework-template

    /dw-test-design 47

**Agent detects cicd/tests/testcases/ with YAML files:**

    Detected: test-framework-template (YAML + dual-judge)
    CI: GitHub Actions (.github/workflows/test-pipeline.yml)
    Issue #47: enhancement → Type C (changed behavior)

**Generates YAML test case:**

    Created: cicd/tests/testcases/integration/TC-INTEGRATION-002.yml
    Running: npm test -- --suite integration --no-llm
    Result: 2 tests passed (0 failed)

    Next: /dw-create-pr 47

---

## EXAMPLE 2: Project with pytest

    /dw-test-design 12

**Agent detects tests/ + conftest.py:**

    Detected: pytest (tests/conftest.py found)
    CI: GitHub Actions (.github/workflows/ci.yml)
    Issue #12: bug → Type B (regression test)

**Generates pytest file:**

    Created: tests/test_issue_12_login_fix.py
    Running: pytest tests/test_issue_12_login_fix.py -v
    Result: 3 tests passed

    Next: /dw-create-pr 12

---

## EXAMPLE 3: Project with no test infrastructure

    /dw-test-design 5

**Agent finds no test framework or CI:**

    Detected: no test framework
    CI: none

    Wrote manual test checklist (will be included in PR body):
    ## Test Plan
    - [ ] Navigate to settings page → form loads without errors
    - [ ] Change display name → saves and shows success message
    - [ ] Enter invalid email → shows validation error

    > This project has no automated test infrastructure.
    > Consider adopting [test-framework-template](https://github.com/dogkeeper886/test-framework-template)
    > for YAML-driven CI testing with dual-judge verification.

    Create follow-up issue for CI setup? [y/n]

    Next: /dw-create-pr 5

---

## API Notes

- Uses `gh` CLI for issue operations
- Test detection is file-based — scans project root for known patterns
- Does not install or configure test frameworks — only generates test files
  for what already exists
- When multiple test frameworks coexist, prefer the one closest to the
  changed files
- Always run tests locally before committing to catch generation errors
```
