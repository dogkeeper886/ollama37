# cicd

Vitest suites check every ollama37 build on real GPUs, run the same way locally or in GitHub Actions. A second host now joins the K80 box: it shares its cards with the agent judge's server and runs just the models that fit one card. A pipeline that hard-codes one container, one address and one model list breaks that host when it runs there. Each self-hosted runner's `.env` names its own server under test, and the workflows borrow a card and return it.

## 1. One runner, three layers

A workflow in `.github/workflows/test-*.yml` picks a self-hosted runner by `runner_label`. The correctness suites are TypeScript files in `cicd/tests/suites/`, run by `vitest run`; they call the Ollama client directly, so a reply is typed fields rather than text (#535). The benchmarks and the MCP test are still `cicd/tests/src/cli.ts` subcommands until they move onto Vitest (#536).

| Workflow | Command |
|---|---|
| `test-suites-v2.yml` | `vitest run [suites/<suite>.test.ts]` |
| `test-pipeline.yml` | `test-suites-v2.yml` per suite: build → runtime → inference → models |
| `test-throughput.yml` | `cli.ts bench-throughput`, then `cli.ts judge-throughput` |
| `test-context.yml` | `cli.ts bench-context` |
| `test-mcp.yml` | `cli.ts test-mcp` |
| `test-report-sweep.yml` | `cli.ts bench-throughput` · `test-mcp` · `model-bounds` |

## 2. Folder → command

Run these from `cicd/tests/`:

| Folder | Holds | Command |
|---|---|---|
| `tests/suites/` | one Vitest file per suite | `npx vitest run suites/<suite>.test.ts [-t <name>]` |
| `tests/src/` | Ollama client, perf and MCP commands, judges | `npm run typecheck` |
| `tests/.env.example` | config for a run outside CI | `cp .env.example .env` |
| `tests/scripts/` | runner helpers | `npx tsx scripts/validate-agent-judge.ts` |
| `scripts/` | tools the workflows call | `bash ../scripts/gpu-temp-guard.sh -- <command>` |
| `tests/results/` | `junit.xml`, written under GitHub Actions, gitignored | — |
| `specs/` · `infrastructure/` | notes | — |

## 3. The host decides the server under test

A self-hosted runner reads its own `.env` (in the runner's folder), so each host names its server there and the workflows stay the same. An unset knob takes the default in the suite or the workflow step.

| Knob | Default | Read by |
|---|---|---|
| `OLLAMA37_CONTAINER` | `ollama37` | runtime suite; VRAM attribution in `suites/gpu.ts` |
| `OLLAMA_HOST` | `http://localhost:11434` | every suite, perf commands, workflow steps |
| `OLLAMA37_MODELS` | the K80 list in `suites/models.test.ts` | models suite |
| `OLLAMA37_SERVICE` | unset: no service switch | Yield GPU / Restore GPU actions |
| `OLLAMA37_YIELD_SERVICE` | unset | Yield GPU / Restore GPU actions |

Two gates guard destructive or hour-long work, off unless set: `OLLAMA37_RESTART_CONTAINER` (runtime suite runs `docker compose down` and `up`) and `OLLAMA37_BUILD_IMAGE` (build suite compiles the image from source and retags it).

For a local run, set them on the command line. The suites do not read `cicd/tests/.env`; the `cli.ts` commands do.

```bash
OLLAMA37_MODELS=gemma3:4b npx vitest run suites/models.test.ts
```

## 4. A run on a shared host

The Yield GPU action stops `OLLAMA37_YIELD_SERVICE` and starts `OLLAMA37_SERVICE` as user systemd units, on a host whose runner sets `OLLAMA37_SERVICE`. The suites run, each model unloading in its own `afterAll`, and Restore GPU restarts the yielded service if it is enabled. `test-throughput.yml` then runs the agent judge on the saved report, after the card is back. Both actions skip on a host that leaves `OLLAMA37_SERVICE` unset.

**Run one job at a time on a host with several runners: its test services share port 11434.**

## Quick start

```bash
cd cicd/tests
npm ci
npx vitest list                                   # every test
npx vitest run suites/models.test.ts -t gemma3    # every model whose name matches
```

In CI:

```bash
gh workflow run test-suites-v2.yml -f runner_label=sm86 -f suite=suites/models.test.ts
```

## Diagrams

`docs/diagrams/render.sh` renders the PNGs from `docs/diagrams/*.svg`:

```bash
docs/diagrams/render.sh
```
