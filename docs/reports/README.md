# `docs/reports/` — the MCP capability matrix

A structured record of MCP capability-test results: each model run against each test on the
multi-server menu, captured as a **model × test matrix** with per-cell verdict, three-stage
rubric, context headroom, and perf. One run = one report (a snapshot).

## Files

| File | What it is |
|---|---|
| [`TEMPLATE.md`](./TEMPLATE.md) | The blank form. Copy it to start a new report. |
| [`mcp-capability-matrix-2026-06-20.md`](./mcp-capability-matrix-2026-06-20.md) | A worked example, filled with real data. |
| `README.md` | This file — the format and how to produce it. |

New reports are dated snapshots: `mcp-capability-matrix-YYYY-MM-DD.md`.

## Why a matrix

The single-run report (`cli.ts test-mcp` stdout) covers one prompt across models. The matrix adds
the **test** dimension — the capability ladder rungs — so you can see, per model, both:

- **T1 — list projects:** the model picks the right tool from the merged menu (tool selection).
- **T2 — list suites (derived arg):** the model must *derive* `project_id` from a name, so the
  **query** stage does real work — the failure T1 can't catch.

The results table is one row per (model × test): reading a model's rows together answers "how
capable is this model"; filtering to a test (e.g. all T2 rows) answers "which models clear this
rung".

## How to produce a report

Run each test on the distractor menu with `--output`, then fill `TEMPLATE.md` from the JSON. From
`cicd/tests/`:

```bash
# T1 — list projects
npx tsx src/cli.ts test-mcp <models> \
  --verify-live --verify-allow list_projects --verify-server-name testlink \
  --distractor-command npx --distractor-args "@playwright/mcp@latest" \
  --prompt "List the projects." --output t1.json

# T2 — list suites (derived arg)
npx tsx src/cli.ts test-mcp <models> \
  --verify-live --verify-allow list_projects,list_test_suites --verify-server-name testlink \
  --distractor-command npx --distractor-args "@playwright/mcp@latest" \
  --prompt "List the test suites in the ollama37 project." --output t2.json
```

`num_ctx` (8192) and `timeout` (1800s) default to the multi-server values; raise `--num-ctx` to
12288–16384 for the chained T2 case on models that already sit near 8k on T1.

## Field map (JSON → cell)

Each `results[]` entry in the `--output` JSON (`McpModelResult` / `Judgment` in
`cicd/tests/src/mcp/test-mcp.ts`) fills one cell:

| Column | JSON field |
|---|---|
| verdict | `check.overall_pass` → ✅/❌; `supported === false` → ⚪ |
| stages (tool·query·content) | `check.agent.stages.{tool,query,content}` (verify-live/dual only; else `—`) |
| cross-check | `check.agent.crossCheckUnsupported` (empty → grounded) |
| peak/ctx | `max_prompt_tokens` / the run's `num_ctx` (⚠️ when >90%, ✂️ when ==) |
| time | `total_duration_s` |
| tok/s | `eval_tps` |
| tools | `tool_calls[].name` |
| detail: reason / evidence | `check.agent.reason` / `check.agent.evidence` (+ `evidenceStatus`) |

## Markers

| Mark | Meaning |
|---|---|
| ✅ | pass |
| ❌ | fail |
| ⚪ | no tool support (model template can't do tools — not a harness failure) |
| ⚠️ | peak prompt > 90% of `num_ctx` (near truncation) |
| ✂️ | truncated (peak == `num_ctx` — the menu didn't fit) |
| ⏳ | not run |
| — | not graded (e.g. simple mode has no rubric stages) |

## Future

A small `merge-mcp-tests t1.json:T1 t2.json:T2` aggregator would read the per-test JSONs and emit
the matrix automatically (markdown + a machine-readable twin). Not built yet — reports are filled
from the template by hand for now.
