---
paths:
  - ".github/workflows/**/*.yml"
---

# CI Workflow Patterns

ollama37's CI is organized by **test suite**, with a pipeline workflow that chains the suites,
plus separate perf/experiment workflows. All run on the self-hosted Tesla K80 runner.

## Suite workflows + the pipeline

```
.github/workflows/
├── test-build.yml        # build suite     (toolchain image, runtime image, image sizes)
├── test-runtime.yml      # runtime suite   (container, GPU detection, health, /api/metrics)
├── test-inference.yml    # inference suite (model pull + GPU inference smoke)
├── test-models.yml       # models suite    (per-model regression on K80)
└── test-pipeline.yml     # full pipeline: build → runtime → inference → models (via needs:)
```

Each suite workflow runs the TypeScript runner (`cli.ts run`) filtered by `--suite`. The
pipeline chains the suites with job `needs:` dependencies; the inference/models suites are the
ones that exercise the agent judge.

**Adding a suite test:** add a `TC-<SUITE>-NNN.yml` under `cicd/tests/testcases/<suite>/` (via
`ci-testcase`) — no workflow change needed; the suite workflow picks it up by `--suite`.

## Perf / experiment workflows

Performance and capability experiments run via the runner's perf subcommands, separate from the
suite workflows:

```
└── test-throughput.yml   # cli.ts bench-throughput — per-model tok/s + output check
```

(`cli.ts test-mcp` — MCP tool-call capability — is a subcommand with no workflow yet.)
`release-docker.yml` builds and publishes the image on a release.

## Key Design Decisions

**Dual triggers:** workflows support `workflow_dispatch` (manual) and, where they compose,
`workflow_call` (the pipeline calls the suites). Run a suite alone or as part of the pipeline.

**Judge mode:** the inference/models workflows offer `simple` (default — fast, deterministic, no
model) or `dual` (simple + the opt-in agent judge), set via `JUDGE_MODE`. The runner is
simple-only unless it reads `dual`.

## Environment Variables

The agent judge is an Agent Client Protocol client: it drives an ACP agent that authenticates
itself, so the judge runs keyless — no Console API key. Swapping the model or vendor means
pointing `JUDGE_AGENT` at a different ACP agent, not changing code. Set these via GitHub
repository variables/secrets (`Settings > Variables/Secrets > Actions`):

| Variable | Purpose | Example |
|----------|---------|---------|
| `JUDGE_MODE` | `simple` (default) or `dual` (opt in the agent judge) | `dual` |
| `JUDGE_AGENT` | Command for the ACP agent the judge drives; unset uses the bundled Claude ACP agent (keyless) | (unset) |
| `CLAUDE_CODE_OAUTH_TOKEN` | Secret that authenticates the bundled Claude agent on a GitHub-hosted runner; unneeded on a self-hosted runner logged into Claude Code | `sk-ant-oat...` |

**Two auth paths for `dual`:** on a GitHub-hosted runner, add the `CLAUDE_CODE_OAUTH_TOKEN` secret. On a self-hosted runner already logged into Claude Code, the agent uses `~/.claude` and no secret is needed — point the workflow's `runs-on` at that runner. Either way, an unauthenticated agent degrades cleanly to the simple judge.
