# cicd

Before an image ships, CI answers one question: does this build still produce language on a K80? Three tests ask it, and every model reply they get goes through one call path that checks it and has the agent judge read it. A reply nobody read cannot pass, and "0 failed" means every reply was checked and judged (#542).

## Layers

| Layer | Tool | Holds |
|---|---|---|
| Shell | bash, `cicd/scripts/` | build the image, deploy the container and health-check it, hand the GPU over and back, the canary, the structure check |
| Library | TypeScript, `cicd/tests/lib/` | the only call to the model, the script check, the judge, the prompt loader, the run wrapper |
| Tests | three scripts, `cicd/tests/` | a prompt name × a model list, nothing else |
| Workflow | GitHub Actions, `.github/workflows/` | wiring only |

Setup is shell, not tests: it leaves a usable image and a healthy server, or fails the job, and adds nothing to a test count.

## Every reply is checked and judged

```
runTest('models', body)
  ├─ yield GPU          cicd/scripts/gpu-yield.sh
  ├─ body: generate(model, name) → reply → check → recorded
  ├─ restore GPU        cicd/scripts/gpu-restore.sh
  ├─ judge every recorded reply
  └─ report + exit      "N models, N replies, N judged, K failed"
```

- `lib/ollama.ts` is the only code that talks to Ollama. `generate` (and `converse`, the same with a tool menu) is the only way to get model text, and it refuses outside `runTest`.
- `lib/check.ts`: the reply has text, is not a loop, holds what the prompt planted, and is not a refusal.
- `lib/judge.ts`: Claude Code on Ollama (`JUDGE_BASE_URL`, `JUDGE_MODEL`), in an empty working directory, one fresh session per question. The answer must open with yes or no; anything else is an abstain, and an abstain fails.
- The judge runs after the GPU is back, because on a shared host its server is the one the test borrowed the card from.

## One file for every prompt and judge question

`cicd/tests/prompts.yaml` holds what the tests send and what the judge asks. A test names a prompt; the prompt names its judge question and the answer that passes. `cicd/scripts/check-structure.sh` fails a change that calls the model outside `lib/ollama.ts` or writes prompt text anywhere else. `OLLAMA37_PROMPTS=<file>` runs an experiment without a code change.

| Prompt | Test | Judge |
|---|---|---|
| `short-answer` | models | `broken`: "Is this text garbage, crash output, or words in random order?" (passes on no) |
| `image`, `audio` (gemma4:12b), `time-tool` (qwen3.8:27b) | models | `broken`, `grounded` |
| `speech-rewrite`, `farewell-summary` | throughput | `broken` |
| `tool-call` | mcp | `grounded`: "Does this answer use the result?" (passes on yes) |

## Run it

From `cicd/tests/` after `npm ci`:

```bash
npx tsx models.ts --models "gemma3:4b"                  # one model = an inference test
npx tsx throughput.ts --models "gemma3:4b" [--prompt farewell-summary]
npx tsx mcp.ts --models "qwen3.8:27b"
../scripts/canary.sh                                    # the judge still catches garbage
../scripts/check-structure.sh                           # the structure still holds
```

Each prints a markdown report and takes `--output <file>` for JSON.

In CI:

| Workflow | What it runs |
|---|---|
| `pipeline.yml` (Build + Test Pipeline) | build → deploy → canary → models, each only after the one before passed |
| `build.yml` (Build) | toolchain check; with `build_image`, compile and retag |
| `deploy.yml` (Deploy) | GPU hand-over, container up, health checks, GPU back |
| `test-canary.yml` (Canary) | the judge still catches garbage |
| `test-models.yml` (Models) | every model; one model is an inference test |
| `test-throughput.yml`, `test-mcp.yml` | run on their own, outside the pipeline |
| `pr-check.yml` | typecheck and the structure check, on every pull request |

## The host decides the server under test

Each runner's `.env` (in the runner's folder, e.g. `~/actions-runner/.env`; template: [`runner.env.example`](runner.env.example)) names its own. The runner reads it only at start, so restart its service after a change:

| Knob | Read by |
|---|---|
| `OLLAMA37_SERVICE`, `OLLAMA37_YIELD_SERVICE` | the GPU hand-over; unset = test whatever serves `OLLAMA_HOST` |
| `OLLAMA37_CONTAINER` (default `ollama37`) | `deploy.sh`, VRAM attribution |
| `OLLAMA_HOST` (default `http://localhost:11434`) | everything |
| `OLLAMA37_MODELS`, `OLLAMA37_THROUGHPUT_MODELS`, `OLLAMA37_MCP_MODELS` | the model list when `--models` is not given |
| `MCP_COMMAND`, `MCP_ARGS`, `MCP_ENV` | the MCP server under test (default testlink-mcp) |

The previous harness, and the report sweep that regenerated `docs/reports`, are in [`archive/v2-test-framework/`](../archive/v2-test-framework/).
