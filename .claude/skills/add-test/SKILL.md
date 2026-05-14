---
name: add-test
description: Add a test case to the ollama37 test framework. Use when a new feature or fix needs test coverage.
argument-hint: <suite> <test-name>
---

# Add Test

Create a new test case following the **Test Management Flow**: User Story → TestLink → YAML.

## When to use
- After implementing a feature that needs test coverage
- When adding regression tests for a bug fix
- When expanding test coverage for an existing suite

## Flow

### 1. Identify the User Story
- Every test must trace to a GitHub Issue
- If no issue exists, create one first (or ask the user)

### 2. Create TestLink Test Case
- Use MCP tools to create the test case in TestLink
- TestLink is the **design authority** — define steps and expected results here first
- Include the GitHub issue reference in the summary field

**TestLink reference:**
| Suite | TestLink Suite ID |
|-------|------------------|
| Build | 2 |
| Inference | 3 |
| Runtime | 39 |
| Models | 122 |

**MCP tool:** `mcp__testlink__create_test_case`
- `project_id`: "1"
- `suite_id`: suite ID from table above
- `name`: "TC-<SUITE>-<NNN>: <Descriptive Name>"
- `summary`: include "Related to #N" for GitHub issue link
- `steps`: array of `{ actions, expected_results }`
- `importance`: 1 (low), 2 (medium), 3 (high)
- `execution_type`: 2 (automated)

### 3. Create YAML Test Script
YAML is the **execution authority** — what actually runs in CI.

**Test suites:**
| Suite | Directory | ID prefix |
|-------|-----------|-----------|
| build | `cicd/tests/testcases/build/` | TC-BUILD |
| runtime | `cicd/tests/testcases/runtime/` | TC-RUNTIME |
| inference | `cicd/tests/testcases/inference/` | TC-INFERENCE |
| models | `cicd/tests/testcases/models/` | TC-MODELS |

**Test case format (YAML):**

Required fields:
- `id` — Unique ID (e.g. `TC-BUILD-004`)
- `name` — Descriptive name
- `suite` — Suite name
- `priority` — Integer (1 = highest)
- `timeout` — Milliseconds
- `dependencies` — List of prerequisite test IDs
- `testlink_id` — TestLink external ID (e.g. `ollama37-21`)
- `issue` — GitHub issue number (e.g. `28`)
- `steps` — List of step objects
- `criteria` — LLM judge criteria string

**Step fields:**
- `name` — Step description
- `command` — Bash command to run
- `expectPatterns` — List of regex patterns that must match output
- `rejectPatterns` — List of regex patterns that must NOT match output
- `timeout` — Optional per-step timeout in milliseconds

## When to use the LLM judge vs deterministic checks

Per CLAUDE.md → "LLM Judge Scope", the LLM judge is ONLY for validating that an LLM-generated response is meaningful for its prompt. Everything else is deterministic.

| Validation | Mechanism |
|-----------|-----------|
| Response contains specific text | `expectPatterns: ["regex"]` |
| Response avoids error string | `rejectPatterns: ["CUBLAS_STATUS"]` |
| Step crashes the runner | Exit code (default; no extra config) |
| Container log signals | `expectPatterns` / `rejectPatterns` matched against captured logs |
| Generated prose is coherent and on-topic | LLM judge (run suite with `--llm`); set `criteria:` field describing what "meaningful" means |

Do not write tests that rely on the LLM judge to confirm static facts (e.g. "response contains '4'"). That's what `expectPatterns` is for.

## Related files
- Test cases: `cicd/tests/testcases/<suite>/`
- Framework docs: `cicd/README.md`
- TestLink MCP tools: `mcp__testlink__*`
