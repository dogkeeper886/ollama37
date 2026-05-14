---
name: add-test
description: Add a test case to the ollama37 test framework. Use when a new feature or fix needs test coverage.
argument-hint: <suite> <test-name>
---

# Add Test

Create a new test case following the **Test Management Flow**: User Story (GitHub Issue) → YAML test case (with `intent:` block).

## When to use
- After implementing a feature that needs test coverage
- When adding regression tests for a bug fix
- When expanding test coverage for an existing suite

## Flow

### 1. Identify the User Story
- Every test must trace to a GitHub Issue
- If no issue exists, create one first (or ask the user)
- Record the issue number — it goes into the YAML's `issue:` field

### 2. Create the YAML test case

YAML is the single source of truth for the test — both **design intent** (the `intent:` block) and **execution** (the `steps:` block) live in the same file.

**Test suites:**
| Suite | Directory | ID prefix |
|-------|-----------|-----------|
| build | `cicd/tests/testcases/build/` | TC-BUILD |
| runtime | `cicd/tests/testcases/runtime/` | TC-RUNTIME |
| inference | `cicd/tests/testcases/inference/` | TC-INFERENCE |
| models | `cicd/tests/testcases/models/` | TC-MODELS |

Auto-increment the test ID by checking existing files: `ls cicd/tests/testcases/<suite>/`.

**Required YAML fields:**
- `id` — Unique ID (e.g. `TC-BUILD-004`)
- `name` — Descriptive name
- `suite` — Suite name
- `priority` — Integer (1 = highest)
- `timeout` — Milliseconds
- `dependencies` — List of prerequisite test IDs
- `issue` — GitHub issue number (e.g. `28`)
- `intent` — Design intent block (see below)
- `steps` — List of step objects
- `criteria` — LLM judge criteria string (only used when running with `--llm`)

**`intent:` block (canonical record of why this test exists):**
- `user_story` — What value this test delivers, in plain prose
- `acceptance` — Optional list of "what must be true" criteria (human-readable)
- `notes` — Optional free-form notes: prerequisites, gotchas, acceptable warnings

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
