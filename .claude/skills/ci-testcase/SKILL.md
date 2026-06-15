---
name: ci-testcase
description: Generate YAML test cases from requirements or acceptance criteria
user-invocable: true
---

# Create Test Case

Generate a YAML test case file from requirements or acceptance criteria.

```
$ARGUMENTS

## PURPOSE

Read requirements (story, ticket, or description) and generate YAML test case(s) that verify the acceptance criteria.

---

## AGENT WORKFLOW

### Step 1: Understand Requirements

Input can be:
- A story/requirement file path — read and analyze it
- A description of what to test — use it directly
- A ticket ID — look for related docs or ask the user

Identify:
- What functionality to test
- What commands or tools to call
- What output to expect
- What would indicate failure

### Step 2: Review Existing Tests

Read existing test cases in `cicd/tests/testcases/` to:
- Find the next available ID number per suite
- Follow existing naming and pattern conventions
- Avoid duplicating existing coverage

### Step 3: Generate Test Case YAML

Create test case file(s) in `cicd/tests/testcases/<suite>/` using this format:

```yaml
id: TC-[SUITE]-[NUMBER]
name: Descriptive test name
suite: build|integration|e2e
goal: One-line test objective for agent judge
priority: 1-10 (lower = runs first)
timeout: 30000
dependencies: []
tags: [feature-name]

steps:
  - name: Step description
    command: <shell command or mcp-client.ts call>
    timeout: 5000              # Optional
    expectPatterns:
      - "regex pattern"
    rejectPatterns:
      - "error"
    capture:                   # Optional: extract values for later steps
      varName: "json.path"

criteria: |
  Human-readable criteria for agent judge evaluation.
```

**Suite guidelines:**
- `build` — compilation, dependency checks, environment validation
- `integration` — single tool/API calls, verify output format and correctness
- `e2e` — multi-step workflows, verify end-to-end state

**ID format:** `TC-BUILD-XXX`, `TC-INT-XXX`, `TC-E2E-XXX`

**Tags:** Use feature names or capability areas (e.g., `auth`, `api`, `build`). Tags enable per-feature CI workflows via `--tag`.

**For MCP tool testing:**
```yaml
command: npx tsx cicd/tests/src/mcp-client.ts <tool_name> '{"arg":"value"}'
```

**For shell command testing:**
```yaml
command: curl -s http://localhost:3000/api/endpoint
```

**For multi-step with variable capture:**
```yaml
steps:
  - name: Create resource
    command: <command that returns JSON>
    capture:
      resourceId: "id"
  - name: Use captured value
    command: <command using {{resourceId}}>
```

Variables resolve from captured step output first, then fall back to `process.env`.

### Step 4: Report

Show the user:
- Test case files created
- What each test validates
- Suggest: `/ci-run` to execute tests

---

## OUTPUT

Paths to created test case files.
