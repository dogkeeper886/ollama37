# CI/CD Infrastructure

This folder contains CI/CD infrastructure and the test framework for validating ollama37 builds on Tesla K80 GPUs.

## Quick Start

```bash
# Navigate to test framework
cd cicd/tests

# Install dependencies
npm install

# Run all tests
npm run test

# Run specific suite
npm run test -- --suite build
npm run test -- --suite runtime
npm run test -- --suite inference

# Also run the agent judge (default is simple judge only)
JUDGE_MODE=dual npm run test

# List available tests
npm run list
```

## Components

### Test Framework (`tests/`)

TypeScript-based test framework with dual-judge architecture.

**Features:**
- YAML-based test case definitions
- Docker log collection with precise boundaries
- Dual judge system (simple + keyless agent judge)
- JSON and console output formats
- Pattern matching for test validation

**Test Suites:**
| Suite | Tests | Purpose |
|-------|-------|---------|
| Build | 3 | Verify Docker images, toolchain, image sizes |
| Runtime | 4 | Container startup, GPU detection, health check, /api/metrics schema |
| Inference | 2 | Model pull + API inference smoke |
| Models | 13 | Per-model regression — gpt-oss, ministral-3, gemma3 (4b/27b), gemma4 (e4b/26b), qwen3.5 (9b/27b), qwen3-vl (8b/30b), deepseek-r1 (14b/32b) |

Each test case lives in `cicd/tests/testcases/<suite>/TC-<SUITE>-NNN.yml`. The `intent:` block in every YAML — `user_story`, optional `acceptance`, optional `notes` — is the single design authority for that test.

### Agent Judge

The semantic judge is the **keyless ACP agent judge** — it spawns a Claude agent over the
Agent Client Protocol; no container and no `ANTHROPIC_API_KEY`. Auth is the runner's
`~/.claude` (local) or the `CLAUDE_CODE_OAUTH_TOKEN` secret (CI). Enable it with
`JUDGE_MODE=dual`; a test passes only if both the simple judge and the agent judge pass.
See [`infrastructure/README.md`](infrastructure/README.md).

## GitHub Actions Workflows

Located in `.github/workflows/`. Two families:

### TC-framework workflows (correctness validation)

These workflows execute the YAML test suites via the TypeScript runner with the dual-judge architecture described above.

| Workflow | Description |
|----------|-------------|
| `test-pipeline.yml` | Full pipeline: build → runtime → inference → models |
| `test-build.yml` | Build suite (image verification) |
| `test-runtime.yml` | Runtime suite (container, GPU, health, metrics) |
| `test-inference.yml` | Inference suite (pull + smoke); supports `JUDGE_MODE=dual` opt-in agent judge |
| `test-models.yml` | Models suite (all 13 per-model regressions); supports `JUDGE_MODE=dual` opt-in agent judge |

### Perf / experiment workflows (the unified test-workflow pattern)

These run via the TypeScript runner's perf subcommands (`cli.ts bench-throughput`, `cli.ts test-fa`, `cli.ts test-mcp`), which capture metrics or a capability verdict and judge output through the same agent judge, emitting a JSON report plus a markdown summary. Full bullet list in [`docs/CICD.md`](docs/CICD.md) → "Perf / experiment workflows".

| Workflow | Script | Description |
|----------|--------|-------------|
| `test-throughput.yml` | `cli.ts bench-throughput` | Per-model tok/s benchmark with simple + optional agent-judge output check |
| `test-fa-k80.yml` | `cli.ts test-fa` | K80 flash-attention regression + benchmark (FA off/on × KV cache type) |
| _(subcommand only)_ | `cli.ts test-mcp` | Probe whether a model can drive a real MCP server's tools — server-agnostic (default `testlink-mcp`, override via `--mcp-command`/`--mcp-args`); structural check + optional agent judge; per-model verdict (PASS / FAIL / NO TOOL SUPPORT) |

### Release workflow

| Workflow | Description |
|----------|-------------|
| `release-docker.yml` | Builds and publishes Docker image on release publication |

**Usage:**
- Trigger manually via GitHub Actions "Run workflow" or `gh workflow run`
- The TC pipeline runs all suites in sequence
- Individual TC workflows assume the production container is already running
- Perf workflows manage their own container lifecycle (stop production → boot test container → restart production)

## Folder Structure

```
cicd/
├── docs/
│   ├── CICD.md              # Design philosophy
│   └── PLAN.md              # Infrastructure planning
├── infrastructure/
│   └── README.md            # Agent judge auth notes (no container)
├── scripts/                 # Perf / experiment helper scripts
│   ├── format-results.sh
│   └── test-mlx-smoke.sh
├── specs/
│   ├── build.md             # Build test specifications
│   ├── runtime.md           # Runtime test specifications
│   ├── inference.md         # Inference test specifications
│   └── models.md            # Models test specifications
├── tests/
│   ├── src/                 # Framework source code
│   ├── testcases/           # YAML test definitions (intent + steps)
│   ├── package.json
│   └── tsconfig.json
├── results/                 # Test output (gitignored)
└── README.md                # This file
```

## Related Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Test subject | `docker/docker-compose.yml` | Ollama build being tested |
| Builder image | `docker/builder/Dockerfile` | Build toolchain container |
| Runtime image | `docker/runtime/Dockerfile` | Compiled binary container |
| Test framework | `cicd/tests/` | Test execution and judging |
| Test specs | `cicd/specs/` | Test case specifications |

## Documentation

- [CICD.md](docs/CICD.md) - Design philosophy and architecture
- [PLAN.md](docs/PLAN.md) - Infrastructure planning and checklist
