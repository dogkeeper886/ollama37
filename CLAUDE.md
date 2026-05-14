# Claude Code Development Notes

This is an Ollama fork adding CUDA compute capability 3.7 support for Tesla K80 GPUs.

## Project Goals

### 1. CUDA Compute Capability 3.7 Support (Tesla K80)
- **Objective**: Add support for CUDA compute capability 3.7 to enable running on Tesla K80 GPUs
- **Environment**: GCC 10.5, CUDA 11.4.4, NVIDIA driver 470
- **Status**: Complete

### 2. Code Understanding and Documentation
- **Issue**: Upstream Ollama/llama.cpp lacks comments, making debugging and optimization difficult
- **Policy**: Use the `trace`, `instrument`, `profile`, and `annotate` skills to systematically understand and document code
- **Principle**: Measure first, annotate only what's verified, never guess

## Skill Loading Rule

**Skills are NOT auto-loaded.** Only the skill name and one-line description appear in context. The full skill content (workflows, rules, formats) is only available after invoking it with the Skill tool.

**You MUST invoke the Skill tool to load a skill before following its workflow.** When a task matches a skill's trigger (listed below), load it first, then proceed. Do not guess at skill contents from memory — always load.

## GitHub Workflow

### Labels
Every issue must have **one type label** and **one priority label**. Add component labels as applicable.

| Category | Labels |
|----------|--------|
| Type | `feature`, `enhancement`, `bug`, `removal` |
| Priority | `priority:critical`, `priority:high`, `priority:medium`, `priority:low` |
| Component | `component:ggml`, `component:cuda`, `component:model`, `component:go`, `component:ci` |
| Status | `status:in-progress`, `status:blocked`, `status:needs-review` |

### Project Board
- Project: "ollama37 Development" (project number 2, owner: dogkeeper886)
- Add issues to project: `gh project item-add 2 --owner dogkeeper886 --url <issue-url>`

### Issue Cross-References
GitHub auto-creates backlinks when issues reference each other. Use these patterns:
- `Depends on #N` / `Blocked by #N` — for dependencies
- `Part of #N` — for parent/child relationships
- `Related to #N` — for related issues
- `Fixes #N` / `Closes #N` — in PR body to auto-close issues on merge

## Development Lifecycle

The AI agent follows this state machine for all work. **Every state transition updates the GitHub issue** — the issue is the single source of truth.

```
          ┌─────────┐
          │ REQUEST  │  User describes work
          └────┬─────┘
               │
          ┌────▼─────┐
          │   PLAN   │  /plan — create user story + GitHub issues
          └────┬─────┘
               │
          ┌────▼─────┐
          │ APPROVAL │  User reviews and approves plan
          └────┬─────┘
               │
       ┌───────▼────────┐
       │  IMPLEMENTING   │  /implement — branch, code, build
       └───┬────────┬────┘
           │        │
     ┌─────▼──┐  ┌──▼──────┐
     │ FAILED │  │ TESTING  │  /test — run integration tests
     └───┬────┘  └──┬───┬──┘
         │          │   │
         │    ┌─────▼┐  │
         └────► BACK │  │  Fix and retry (update issue)
              └──────┘  │
                   ┌────▼──────┐
                   │ PR CREATED│  /create-pr — push branch, open PR
                   └────┬──────┘
                        │
                   ┌────▼──────┐
                   │ REVIEWING │  /review-pr — review checklist
                   └────┬──────┘
                        │
                   ┌────▼──────┐
                   │  MERGED   │  /merge — merge PR, cleanup
                   └───────────┘
```

### States and Transitions

| State | Skill/Command | Entry Action | GitHub Update |
|-------|--------------|--------------|---------------|
| **REQUEST** | — | User describes work | — |
| **PLAN** | `/plan` | Create user story as GitHub issue(s), break into tasks | Issues created, added to project board |
| **APPROVAL** | — | Present plan to user, wait for approval | — |
| **IMPLEMENTING** | `/implement` | Create branch, start coding | Comment: "Starting work on branch `issue-N-slug`", add `status:in-progress` |
| **FAILED** | — | Build/test failed | Comment: what failed + root cause, add `status:blocked` |
| **TESTING** | `/test` | Run integration tests | Comment: test results |
| **PR CREATED** | `/create-pr` | Push branch, open PR with `Fixes #N` | Comment: "PR #X created", replace `status:in-progress` → `status:needs-review` |
| **REVIEWING** | `/review-pr` | Review checklist, request changes or approve | Comment on PR |
| **MERGED** | `/merge` | Merge PR, delete branch | Remove status labels, issue auto-closes |

### Transition Conditions

| From | To | Condition |
|------|----|-----------|
| REQUEST → PLAN | User describes work | Always |
| PLAN → APPROVAL | Issues created | Always — never skip user approval |
| APPROVAL → IMPLEMENTING | User says "yes" / "go" / "start" | User must explicitly approve |
| IMPLEMENTING → TESTING | Code compiles, no obvious errors | Load `build` skill, verify |
| IMPLEMENTING → FAILED | Build or runtime error | Do NOT silently retry |
| FAILED → IMPLEMENTING | Fix applied | Comment fix on issue, remove `status:blocked` |
| TESTING → PR CREATED | Tests pass | All relevant test suites green |
| TESTING → FAILED | Tests fail | Comment failure on issue |
| PR CREATED → REVIEWING | PR opened | Always |
| REVIEWING → MERGED | Approved | User or reviewer approves |
| REVIEWING → IMPLEMENTING | Changes requested | Address feedback, re-test |

### Failure Protocol

When **any step fails** (build, test, runtime):
1. **Do NOT silently retry.** Comment on the issue: what failed, error message, root cause hypothesis
2. Add `status:blocked` label if stuck: `gh issue edit <N> --add-label "status:blocked"`
3. Investigate and apply fix, comment again: what changed and why
4. Remove `status:blocked` after unblocking: `gh issue edit <N> --remove-label "status:blocked"`
5. If stuck after 2-3 attempts, comment with blockers and ask the user

### Partial Fix Protocol

When only part of the work is complete:
- Comment: what was fixed, what remains, and blockers
- Create PR for the partial fix if it's independently useful
- Create follow-up issues for remaining work

**Key principle**: The issue is the single source of truth. Anyone reading it should see the full history — start, failures, fixes, and resolution.

## Session Summaries

- `/session-summary` — Record friction points and patterns at end of each session
- Session data saved to `docs/session_summaries/`

### Related Repos
- `ai-qa-workflow` — QA automation toolkit; origin of shared commands and skills
- `test-framework-template` — Origin of the dual-judge test framework (`cicd/tests/src/`)

## Test Management Flow

Test cases follow a **requirements-driven** flow: every test traces back to a user story.

```
GitHub Issue (User Story)  →  YAML Test Case (intent + steps)
(what to validate)            (how to validate + automated execution)
```

### 1. User Story (GitHub Issue)
- Created via `/plan` or manually
- Describes WHAT needs testing and WHY
- Has type + priority + component labels

### 2. Test Case (YAML)
- Create executable YAML via `/add-test` skill at `cicd/tests/testcases/<suite>/TC-<SUITE>-NNN.yml`
- Must include an `intent:` block (`user_story`, optional `acceptance`, optional `notes`) — the canonical record of why the test exists
- Must include the `issue:` field linking back to the GitHub issue
- PR includes the new YAML file

### Authority
- **YAML `intent:` block** = design authority (the test's purpose, in writing)
- **YAML `steps:` + `expectPatterns:` / `rejectPatterns:`** = execution authority (what actually runs in CI)
- Both live in the same file by design (one source of truth, no drift surface)

### Workflow-level pattern (non-TC workflows)

The flow above governs the TC framework (build / runtime / inference / models). Perf benchmarks, experiments, and profiling workflows (`test-throughput.yml`, `test-fa-k80.yml`, `test-kv-rotate.yml`, `test-profile.yml`) follow a separate unified pattern documented in the [`test-workflow-pattern`](./.claude/skills/test-workflow-pattern/SKILL.md) skill: one extracted bash script per workflow, shared helpers from `cicd/scripts/lib/`, structured JSON output, standardized container handling.

### LLM Judge Scope

The LLM judge has exactly one purpose: validating that an LLM-generated response is **meaningful for its prompt**. Every other check is deterministic.

| Check type | How to validate | Tool |
|------------|-----------------|------|
| LLM response is meaningful for the prompt | Send `(prompt, output)` to the judge endpoint, parse `{pass, reason}` verdict | LLM judge (`--llm` flag) |
| Stdout / API response shape | Regex match, exact match, non-empty assertion in the test step itself | Bash + `grep` / `jq` / shell tests |
| Container log signal (CUDA errors, fallback warnings, OOM) | Pattern match against captured container logs | SimpleJudge `expectPatterns` / `rejectPatterns` |
| Crash / non-crash | Exit code of the test step | Bash `set -e` and step exit status |

**Why this matters**

- The judge is slow (one LLM call per assertion) and costs GPU time.
- Using it where deterministic checks suffice hides regressions when the judge itself drifts (cf. #149 / #151 — judge prompt bias).
- Confining the judge to response-meaningfulness makes failures debuggable: a judge "FAIL" always points at the same class of problem.

**Test step design rule**

When adding or modifying a test:
- If you're validating a string contains some text → regex in `expectPatterns`, not the judge.
- If you're validating no CUDA error occurred → pattern in `rejectPatterns`, not the judge.
- If you're validating a model produced coherent prose → judge with `--llm`.

This is the principle that #142 (scope LLM judge to response checks) and #147 / #149 / #151 (throughput judge for response-meaningfulness) already moved us toward. Don't re-introduce LLM-judging for static checks.

## Skill Quick Reference

Extracted triggers and key rules from each skill. Use these to recognize when to load a skill, and as a fallback if the Skill tool is unavailable.

### git-flow
- **Trigger**: Before committing any change
- **Rule**: Code changes → branch flow (branch → PR → merge). Docs-only → commit on main. When in doubt, branch flow.

### build
- **Trigger**: Compiling from source, building Docker images, verifying compiled changes

### debug
- **Trigger**: Server startup failures, GPU detection issues, CUBLAS errors, runtime problems

### test
- **Trigger**: Validating builds, running test suites, debugging test failures

### ci
- **Trigger**: Remote builds/tests, GitHub Actions, pre-merge verification

### plan
- **Trigger**: User describes a feature, enhancement, removal, or bug to plan

### implement
- **Trigger**: Picking up a GitHub Issue to start work (after user approval)

### add-test
- **Trigger**: New feature or fix needs test coverage

### trace
- **Trigger**: Investigating unfamiliar code, understanding execution flow, before modifying unknown code
- **Rules**: One path at a time. Start from log message. Note branch conditions. Verify with runtime.
- **Save split**: `docs/traces/` = full technical knowledge. GitHub issue = short status + link to trace doc. No duplication.

### instrument
- **Trigger**: Investigating slow loading, GPU transfer, unexplained latency
- **Rules**: Measure first, optimize second. Use `// INSTRUMENT:` prefix. Use existing logging (`LLAMA_LOG_INFO`, `slog`). Remove after confirming.

### profile
- **Trigger**: Performance issues, before reading code for bottlenecks
- **Rules**: Use `nvprof` (not `nsys` — CUDA 11.4/driver 470). K80 is PCIe Gen3 x16 (~8-10 GB/s).
- **Tools**: `perf` (CPU), `strace` (syscalls/IO), `nvprof` (GPU), page faults (memory)

### annotate
- **Trigger**: After confirming behavior through tracing, profiling, or debugging
- **Rules**: Only annotate verified behavior. Comment "why" not "what". Tags: `// VERIFIED:`, `// ASSUMPTION:`, `// NOTE:`. Architecture docs in `docs/arch/`.

### script-review
- **Trigger**: After writing a video or demo script, before recording
- **Rules**: Review as spoken narration, not as a document. Check: hook (first 10s), numbers overload, transitions, spoken language, screen directions, pacing. Output uses a scannable table format.

### lint-ci
- **Trigger**: Before pushing changes to `.github/workflows/*.yml`
- **Rules**: Validate `jq -n '{...}'` object templates with `jq -n` against placeholder values (catches reserved-word collisions like `label`, `break`); shellcheck `run:` blocks (catches redirect-order bugs like `2>&1 > file`); actionlint for workflow structure. Most CI bugs in this project's history (jq reserved word, shell redirect, header staleness) were locally testable in isolation.

### test-workflow-pattern
- **Trigger**: Authoring or refactoring a `.github/workflows/test-*.yml` workflow that runs on the K80 runner (perf benchmarks, experiments, profiling) — not a TC-framework correctness suite
- **Rules**: One entry point per workflow (`bash cicd/scripts/test-<name>.sh`); no inline test logic in YAML; source helpers from `cicd/scripts/lib/`; emit structured JSON to `/tmp/test-<name>-results.json`; exit non-zero on validation failure; standardized pre/post container handling.

## Logging Guidelines

All debug logging must be **level-gated and permanent** — never add temporary log lines that need manual removal.

### Levels

| Level | Use for | Default |
|-------|---------|---------|
| ERROR | User-visible failures | On |
| WARN | Unexpected but recovered | On |
| INFO | Major state changes (model loaded, GPU count) | On |
| DEBUG | Internal decisions (layer routing, cache hits, tensor lookups) | Off |

### Go code (`slog`)

```go
// Good — silent unless OLLAMA_DEBUG=1
slog.Debug("layer uses recurrent path", "layer", i, "kvHeads", 0)

// Good — important state, always visible
slog.Info("model loaded", "gpus", 2, "vram", "21.5 GiB")

// Bad — temporary, requires manual removal
slog.Info("INSTRUMENT: tensor", "name", name)
```

Gate expensive debug work:
```go
if slog.Default().Enabled(nil, slog.LevelDebug) {
    slog.Debug("tensor details", "names", getAllTensorNames())
}
```

### C code (GGML/CUDA)

```c
// Graph building, model loading — use GGML log macros (level-gated at runtime)
GGML_LOG_DEBUG("rope: mode=%d dims=%d\n", mode, n_dims);

// CUDA hot paths — use #ifdef (compiled out in release)
#ifdef GGML_CUDA_DEBUG
    fprintf(stderr, "CUDA kernel: grid=%d block=%d\n", grid, block);
#endif
```

### Activation

```bash
OLLAMA_DEBUG=1 ollama serve                        # Go debug logs
GGML_CUDA_DEBUG=1 ollama serve                     # CUDA kernel logs
OLLAMA_DEBUG=1 GGML_CUDA_DEBUG=1 ollama serve      # Both
```

### Rules

1. **No marker prefixes** — never use `INSTRUMENT:`, `DEBUG:`, `TEMP:` in messages. Use the correct log level instead.
2. **Log decisions, not data dumps** — "layer 3 routed to DeltaNet" is useful; dumping all tensor names is not.
3. **Never remove debug logs** — if it was useful during development, keep it as `slog.Debug`. If it's not useful to anyone, don't add it.
4. **C code: two tiers** — `GGML_LOG_DEBUG` for always-compiled level-gated logs; `#ifdef GGML_CUDA_DEBUG` for per-kernel-launch tracing that would kill performance.

## Skills and Commands

When creating or updating skills and commands, follow the format guides in `.claude/references/`:
- `skill-format.md` — Skill file structure, frontmatter fields, best practices
- `command-format.md` — Slash command format, arguments, dynamic context

**Design principle**: Skills define *when* and *what*. Commands define *how* (invoked via `/slash`). Keep executable content in commands, not skills.

### Skill files (`.claude/skills/<name>/SKILL.md`)
`build`, `debug`, `ci`, `test`, `git-flow`, `plan`, `implement`, `add-test`, `test-workflow-pattern`, `trace`, `instrument`, `profile`, `annotate`, `script-review`, `lint-ci`

### Slash commands (`.claude/commands/<category>/`)
- **dev-workflow/**: `/plan`, `/implement`
- **build-test/**: `/build`, `/debug`, `/test`, `/ci`, `/add-test`, `/test-workflow`
- **code-analysis/**: `/trace`, `/instrument`, `/profile`, `/annotate`
- **project/**: `/script-review`
- **utility/**: `/session-summary`
