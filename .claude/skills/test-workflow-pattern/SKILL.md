---
name: test-workflow-pattern
description: Defines the unified design pattern every test workflow + test script follows. Use when authoring or refactoring a workflow under .github/workflows/ that runs something on the K80 runner — perf benchmarks, experiments, profiling. The TC framework (build/runtime/inference/models suites) keeps its own pattern documented in cicd/docs/CICD.md.
---

# Test Workflow Pattern

A single design pattern every test workflow + test script in this project follows. Replaces the historical situation where every perf workflow (throughput, fa-k80, kv-rotate, profile) invented its own shape.

## When to use this pattern

- Authoring a new `.github/workflows/test-<name>.yml` that runs against the K80 runner
- Refactoring an existing standalone-script workflow that doesn't yet conform
- Anything that isn't a correctness suite covered by the TC framework (build / runtime / inference / models)

The TC framework remains the right tool for correctness validation — it has YAML test cases, the TypeScript runner, and the dual-judge architecture. See `cicd/docs/CICD.md`. This pattern is for everything else.

## The shape

```
.github/workflows/test-<name>.yml
├─ checkout + deps
├─ pre:    optional, standardized — stop production ollama37 container
├─ exec:   bash cicd/scripts/test-<name>.sh    ← ONE entry point. No inline run: blocks for test logic.
├─ post:   always-on, standardized — restore production container
└─ upload: artifact at known path (/tmp/test-<name>-results.json)
```

```
cicd/scripts/test-<name>.sh
├─ sources shared helpers from cicd/scripts/lib/
├─ exits non-zero on validation failure
├─ emits structured JSON to /tmp/test-<name>-results.json
├─ defaults to bash + curl + jq for HTTP
└─ Python permitted only when bash is genuinely insufficient (heavy parsing, stats); document the choice in the script header
```

## Rules

1. **No inline test logic in workflow YAML.** Workflow `run:` blocks orchestrate (checkout, env setup, calling the script, uploading the artifact). All actual test logic lives in `cicd/scripts/test-<name>.sh`.
2. **One script per workflow.** Same name stem (`test-throughput.yml` → `test-throughput.sh`).
3. **Structured JSON output.** Every script writes its results to `/tmp/test-<name>-results.json`. This is the artifact that gets uploaded.
4. **Shared helpers, not duplicated logic.** When a behavior is needed by two or more scripts, it goes into `cicd/scripts/lib/` (see [Helpers contract](#helpers-contract) below).
5. **Standardized pre/post container handling.** Workflows that need to stop / restart the production `ollama37` container use the standardized snippet — not bespoke bash per workflow.
6. **Exit codes are load-bearing.** Validation failure → exit non-zero. The workflow's red / green status must reflect actual test pass / fail, not "the script crashed."
7. **LLM judge usage follows the scope rule.** Per `CLAUDE.md → "LLM Judge Scope"`, the judge is ONLY for "is this LLM response meaningful?". Static / log / exit-code checks stay deterministic.

## Helpers contract

`cicd/scripts/lib/` provides these shared bash functions, sourceable by any test script:

| Helper | Purpose | Signature (rough) |
|--------|---------|-------------------|
| `response_capture` | Call `/api/generate` with a prompt and capture `.response` + `.thinking` + perf metrics into JSON | `capture_response MODEL PROMPT [num_predict] [num_ctx]` → JSON to stdout |
| `simple_check` | Pass if response or thinking is non-empty (whitespace-stripped) | `simple_check RESPONSE THINKING` → `{pass, reason, source}` JSON; exit 0/1 |
| `judge_response` | Send `(prompt, output)` to the LLM judge endpoint, parse verdict | `judge_response MODEL PROMPT OUTPUT` → `{pass, reason}` JSON |
| `container_log_snip` | Capture a slice of a container's logs between markers, for log-based assertions | `container_log_snip CONTAINER START_MARKER END_MARKER` → log text |

These are the canonical implementations. Scripts must source them rather than re-implementing equivalents inline. Concrete implementations land in #159; this skill defines the contract.

## Reference implementation (today)

`cicd/scripts/benchmark-throughput.sh` is the most-evolved current example — it captures responses, runs simple + optional LLM-judge checks, exits non-zero on validation failure, and writes structured JSON. Today it has the validation logic inlined; after #159 it'll source from `cicd/scripts/lib/` as the canonical reference.

For the workflow side, `.github/workflows/test-throughput.yml` already follows the pattern: validate inputs → invoke the script → summary → unload → upload artifact. The three other perf workflows (#160) will be refactored to match.

## Why this matters

- Surfaced during the v2.1.0 release (#147 / #149 / #151): throughput shipped with no response capture and no coherence check for ~a year because the workflow was a one-off with bespoke logic. Same gap exists in `test-fa-k80`, `test-kv-rotate`, `test-profile` today.
- A unified pattern means a new workflow can be written by copying the template, not by inventing yet another shape.
- A reader (human or AI agent) needs only one mental model to understand any `test-*.yml`.

## Related

- `CLAUDE.md → "Test Management Flow"` — covers test cases inside the TC framework
- `cicd/docs/CICD.md` — TC-framework design philosophy
- `.claude/skills/add-test/SKILL.md` — authoring a new test case in the TC framework
- `/test-workflow` command — operational guide for applying this pattern
