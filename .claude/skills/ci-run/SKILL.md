---
name: ci-run
description: Execute test cases with the simple judge by default, or opt in the agent judge
user-invocable: true
---

# Run Test Cases

Execute test cases and evaluate results. The simple (deterministic) judge is the
default verdict; the agent judge is an opt-in second opinion (`JUDGE_MODE=dual`),
keyless on a Claude Code subscription.

```
$ARGUMENTS

## PURPOSE

Run YAML test cases by executing commands and evaluating results against patterns and criteria.

---

## AGENT WORKFLOW

### Step 1: Load Test Cases

Input can be:
- A test ID (e.g., `TC-INT-001`) — run that specific test
- A suite name (e.g., `integration`) — run all tests in that suite
- A tag name (e.g., `auth`) — run all tests with that tag
- Empty — run all tests

Read YAML test case files from `cicd/tests/testcases/`.

### Step 2: Resolve Dependencies

Sort tests by:
1. Dependencies (tests that depend on others run after)
2. Priority (lower number = runs first)

If running a specific test that has dependencies, auto-include them.

### Step 3: Execute Each Test

For each test case, for each step:

1. **Substitute variables** — replace `{{varName}}` with captured values from previous steps (falls back to `process.env`)
2. **Execute the command** specified in `command`
3. **Capture the output** (stdout and stderr)
4. **Check expectPatterns** — each regex must match somewhere in the output
5. **Check rejectPatterns** — none of these regexes should match
6. **Capture variables** — if `capture` is defined, extract values from JSON output
7. **Record result**: PASS or FAIL with evidence

If a step fails, continue with remaining steps but note the failure.

### Step 4: Judge Results

For each test case, determine overall verdict:
- **PASS** — all steps passed, all patterns matched, no errors detected
- **FAIL** — any step failed, with details on which and why

If a test has `criteria` and `goal` fields, evaluate whether the output semantically satisfies them.

### Step 5: Report

Output a summary table:

```
Test Results
============
TC-INT-001  Query resources            PASS  (1.2s)
TC-INT-002  Create and verify          FAIL  (step 2: expected pattern "active" not found)
TC-E2E-001  Full lifecycle             PASS  (5.8s)

Summary: 2/3 passed
Duration: 8.2s
```

If any test failed, show:
- Which step failed
- Expected vs actual output (truncated)
- Suggested fix or investigation

### Alternative: Use the CLI

Instead of agent-based execution, use the built-in CLI:

```bash
cd cicd/tests
npm test                        # All tests (simple judge — fast, no model)
npm test -- --suite integration # Specific suite
npm test -- --id TC-INT-001     # Specific test
npm test -- --tag auth          # Tests tagged 'auth'
npm test -- --dry-run           # Preview only
JUDGE_MODE=dual npm test        # Opt in the agent judge (env, not a flag)
```

**Environment variables for CI:**
- `JUDGE_MODE` — `simple` (default) or `dual` (opt in the agent judge)
- `JUDGE_AGENT` — Command for the ACP agent the judge drives; unset uses the bundled Claude ACP agent (keyless). Set it to swap model/vendor
- `CLAUDE_CODE_OAUTH_TOKEN` — Authenticates the bundled Claude agent on a GitHub-hosted runner; unneeded on a self-hosted runner logged into Claude Code (`~/.claude`)

---

## OUTPUT

Test results summary with pass/fail status for each test case.
