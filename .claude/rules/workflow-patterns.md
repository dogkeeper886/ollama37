---
paths:
  - ".github/workflows/**/*.yml"
---

# CI Workflow Patterns

## Per-Feature Workflow Pattern

Split CI into composable, independently triggerable workflows:

```
.github/workflows/
├── build.yml                # Standalone build step
├── test-run.yml             # Reusable test runner (called by feature workflows)
├── test-<feature>.yml       # One per feature (~25 lines, thin delegator)
└── ci.yml                   # Full pipeline: build -> all features in parallel
```

**Adding a new feature test:**
1. Tag test cases: `tags: [my-feature]`
2. Copy `test-feature-example.yml` -> `test-my-feature.yml`
3. Change the `tag` input value to `my-feature`
4. Add the new workflow as a job in `ci.yml`

### Suite-Based Alternative

For projects organized by test suite rather than feature:

```
.github/workflows/
├── build.yml
├── test-run.yml             # Same reusable runner
├── test-build.yml           # Uses --suite build
├── test-integration.yml     # Uses --suite integration
└── ci.yml                   # Orchestrates all suites
```

Both patterns use `test-run.yml` as the single reusable job.

## Key Design Decisions

**Dual triggers:** Each workflow supports both `workflow_dispatch` (manual, with dropdowns) and `workflow_call` (callable from pipeline). This lets you run features independently or as part of the full CI.

**Judge mode dropdown:** Each workflow offers `simple` (default — fast, deterministic, no model) or `dual` (simple + the opt-in agent judge). The workflow sets `JUDGE_MODE` from this input; the runner is simple-only unless it reads `dual`.

## Environment Variables

The agent judge is an Agent Client Protocol client: it drives an ACP agent that
authenticates itself, so the judge runs keyless — no Console API key. Swapping the
model or vendor means pointing `JUDGE_AGENT` at a different ACP agent, not changing
code. Set these via GitHub repository variables/secrets (`Settings > Variables/Secrets > Actions`):

| Variable | Purpose | Example |
|----------|---------|---------|
| `JUDGE_MODE` | `simple` (default) or `dual` (opt in the agent judge) | `dual` |
| `JUDGE_AGENT` | Command for the ACP agent the judge drives; unset uses the bundled Claude ACP agent (keyless) | (unset) |
| `CLAUDE_CODE_OAUTH_TOKEN` | Secret that authenticates the bundled Claude agent on a GitHub-hosted runner; unneeded on a self-hosted runner logged into Claude Code | `sk-ant-oat...` |

**Two auth paths for `dual`:** on a GitHub-hosted runner, add the `CLAUDE_CODE_OAUTH_TOKEN` secret. On a self-hosted runner already logged into Claude Code, the agent uses `~/.claude` and no secret is needed — point the workflow's `runs-on` at that runner. Either way, an unauthenticated agent degrades cleanly to the simple judge.

## Legacy Workflows

`test-pipeline.yml` and `test-suite.yml` are the original suite-based workflows. They still work but the per-feature pattern (`ci.yml` + `test-run.yml`) is recommended for projects with many test cases.
