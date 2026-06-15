# `docs/tests/` — the test-doc format

Each test is a **readable markdown document** that lives here, close to the story it verifies.
The markdown owns **why / what** (intent); the bound `cicd/tests/` YAML owns **how it runs**
(execution). `qw-cases` authors these docs, `qw-review-cases` gates them, and (when the binding
tooling is wired — see "Binding & drift" below) `qw-bind`/`qw-drift` keep the doc and its
executable in sync.

## One file = one scenario (TS), many cases (TC)

A **scenario** groups related **cases**, each case a sequence of **steps**.

```
docs/tests/
  TS-01-<slug>.md     # a scenario: TC-01, TC-02, … each with a Steps table
  TS-02-….md
```

- **TS** (scenario) — the file. Holds the front-matter and a `## Why this scenario exists`.
- **TC** (case) — a `### TC-NN:` section. Has an objective, a **`Script:`** line (the bound
  `cicd/tests/` YAML), and a **Steps** table.
- **Step** — one row of a case's Steps table: an **Action** and its **Expected Result**. The
  binding check compares the doc's step count against the bound YAML's `steps:` count.

## Front-matter (scenario level)

```yaml
---
id: TS-02                       # scenario id, unique within the namespace
title: Runtime comes up healthy on K80
namespace: ollama37             # which repo this test belongs to
story: STORY-005                # the need this scenario verifies (→ docs/stories/STORY-005.md)
story_hash: 7474d8b6…           # sha256 of the linked story file at last sync (drift anchor)
plan: 257                       # the [STORY-XXX] Test Plan issue it was authored from (optional)
status: green                   # green | stale | unbound
---
```

- `story` + `story_hash` are the **drift anchor**: when the story changes, its hash no longer
  matches and the scenario is `stale`. Set it with `sha256sum docs/stories/STORY-XXX.md`.
- The **`Script:` binding is per-TC, not in front-matter** — a scenario's cases can map to
  different executables.

## Case (TC) structure

```markdown
### TC-01: Container starts with GPU passthrough and reports healthy

- **Objective:** the runtime container comes up and the K80 GPU is detected.
- **Script:** cicd/tests/testcases/runtime/TC-RUNTIME-001.yml
- **Preconditions:** the ollama37 image is built; the host has a Tesla K80 + driver.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Start the container (`docker compose up -d`) | comes up with no `error` in the logs |
| 2 | Wait for container health | reports `healthy` within the timeout |
```

The Steps table is **machine-extractable** on purpose: one row = one `Action → Expected Result`,
and the row count is what the binding check compares to the YAML's `steps:`.

## Suites (the `cicd/tests/testcases/` layout)

Scenarios map to ollama37's four executable suites (see [`cicd/README.md`](../../cicd/README.md)):

| Suite | Verifies | YAML |
|-------|----------|------|
| `build` | toolchain image + runtime binary | `cicd/tests/testcases/build/TC-BUILD-*.yml` |
| `runtime` | container, GPU detection, health, metrics | `cicd/tests/testcases/runtime/TC-RUNTIME-*.yml` |
| `inference` | model pull + GPU inference smoke | `cicd/tests/testcases/inference/TC-INFERENCE-*.yml` |
| `models` | per-model regression on K80 | `cicd/tests/testcases/models/TC-MODELS-*.yml` |

## Binding & drift

- **Bind:** each TC's `Script:` points at the `cicd/tests/testcases/**/*.yml` that runs it.
- **Run:** `npm --prefix cicd/tests test` (the runner — `cli.ts run`; filter with `-- --suite <suite>`).
- **Audit / drift:** the automated `audit-bind` / `drift` scripts are **not yet ported** to
  ollama37 (a planned follow-up). Until they are, the binding invariant — each TC's Steps-table
  **row count equals its bound YAML's `steps:` count** — and `qw-review-cases` are checked by
  hand. When the scripts land, `status` (green | stale | unbound) becomes their verdict.

## Traceability

- **story → tests:** `grep -l 'story: STORY-XXX' docs/tests/`
- **test → story / script:** the front-matter `story:` and each case's `Script:` line.
- **script → test:** the `Script:` path points at the YAML.

No hand-maintained index — the links live in the files and resolve by `grep`/path.
