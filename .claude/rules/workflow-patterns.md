---
paths:
  - ".github/workflows/**/*.yml"
  - ".github/actions/**/*.yml"
---

# CI Workflow Patterns

ollama37's CI is organized by **test suite**, with a pipeline workflow that chains the suites,
plus separate perf/experiment workflows. All run on a labelled self-hosted GPU runner — see
[Host selection](#host-selection).

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
├── test-throughput.yml   # cli.ts bench-throughput — per-model tok/s + output check (short prompt)
└── test-context.yml      # cli.ts bench-context — long-context prefill/decode tok/s + needle + judge
```

`test-context.yml` is the flash-attention path comparison: it primes a realistic long prompt (so
prefill and decode are in the regime where FA's cost/benefit shows) and is self-contained — it
applies the experiment env (`flash_attention`/`kv_cache_type`) by recreating the container, then
`always()` reverts to the stable baseline (FA off). `test-throughput.yml` stays the short-prompt K80
model-regression tool.

(`cli.ts test-mcp` — MCP tool-call capability — is a subcommand with no workflow yet.)
`release-docker.yml` builds and publishes the image on a release.

## Host selection

There is more than one possible GPU testbed, so **never use a bare `runs-on: self-hosted`.**
It matches any runner, and a K80 sweep landing on a small consumer card produces numbers that
look valid and are not.

Select the host by the compute capability it provides:

```yaml
runs-on: [self-hosted, sm37]      # Tesla K80 — the only hardware-validated target
```

| Label | Host | Notes |
|---|---|---|
| `sm37` | Tesla K80 box (`rocky9-k80-cicd-1`) | 4 dies, ~11.4 GiB each. The reference testbed. |
| `sm75` | RTX 2060 box (`rocky9-2060-cicd-1`) | 1 card, **5.1 GiB usable** (display attached). |

A workflow that can run on more than one testbed takes a `runner_label` input defaulting to
`sm37`, and uses `runs-on: [self-hosted, "${{ inputs.runner_label || 'sm37' }}"]`.

**Not every suite fits every host.** `sm75` holds no model the `models` suite uses
(`deepseek-r1:32b`, `gemma3:27b`, …), and the `build` suite is a multi-hour no-cache compile that
belongs on `sm37` where the builder image is cached. The `runtime` suite needs no model and runs
anywhere.

**Order of operations when adding a runner:** add the label to the runner *first*, then pin
workflows to it. Pinning to a label no runner carries makes every job queue forever.

**Identify the host** as the first step after checkout, so a result is attributable to a
testbed rather than to "self-hosted":

```yaml
- name: Identify host
  uses: ./.github/actions/identify-host
```

It writes the runner name and every GPU's name / compute capability / VRAM / driver to the step
summary **and** the log — a run that dies before the summary renders is still attributable. If
`nvidia-smi` is installed but unqueryable, the step **fails** rather than reporting "none
detected": a broken host is not a CPU host, and an unattributable result is worse than a red job.

It deliberately exposes no outputs. Add them when a job actually needs to branch on the hardware,
and when you do, decide what a multi-GPU or mixed-architecture host should report — the first
device's compute capability is not the host's.

**Per-host knobs don't travel.** `cicd/scripts/gpu-temp-guard.sh` defaults to an 80 °C abort,
which is tuned for the K80. Another card with a different thermal envelope needs its own
`GPU_TEMP_LIMIT`, set where that host's jobs are defined — not by changing the default.

## Key Design Decisions

**No workflow triggers on `push` or `pull_request`.** The test workflows are `workflow_dispatch`
and, where they compose, `workflow_call`; `release-docker.yml` adds `release: [published]`. That
is deliberate: the runners are self-hosted on a public repository, so a PR-triggered workflow
would let a fork execute arbitrary code on them. Publishing a release requires write access, so
that trigger is safe. Adding a `push` or `pull_request` trigger is a security decision, not a
convenience.

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
