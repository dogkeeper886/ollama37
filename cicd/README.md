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

# Run with LLM judge enabled (default is simple judge only)
npm run test -- --llm

# List available tests
npm run list
```

## Components

### Test Framework (`tests/`)

TypeScript-based test framework with dual-judge architecture.

**Features:**
- YAML-based test case definitions
- Docker log collection with precise boundaries
- Dual judge system (simple + LLM)
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

### LLM Judge (`infrastructure/docker-compose.judge.yml`)

A stable reference Ollama instance for semantic test evaluation.

**Architecture:**
```
Port 11434 → ollama37 (test subject, local build)
Port 11435 → ollama37-judge (stable reference, DockerHub)
```

**Usage:**
```bash
# Start judge container
cd cicd/infrastructure
docker compose -f docker-compose.judge.yml up -d

# Pull model for judging (first time)
curl -X POST http://localhost:11435/api/pull -d '{"name": "gemma3:4b"}'

# Stop judge
docker compose -f docker-compose.judge.yml down
```

## GitHub Actions Workflows

Located in `.github/workflows/`. Two families:

### TC-framework workflows (correctness validation)

These workflows execute the YAML test suites via the TypeScript runner with the dual-judge architecture described above.

| Workflow | Description |
|----------|-------------|
| `test-pipeline.yml` | Full pipeline: build → runtime → inference → models |
| `test-build.yml` | Build suite (image verification) |
| `test-runtime.yml` | Runtime suite (container, GPU, health, metrics) |
| `test-inference.yml` | Inference suite (pull + smoke); supports `--llm` opt-in judge |
| `test-models.yml` | Models suite (all 13 per-model regressions); supports `--llm` opt-in judge |

### Perf / experiment workflows (the unified test-workflow pattern)

These follow a shared contract: one extracted bash script per workflow at `cicd/scripts/test-<name>.sh`, sourcing shared helpers from `cicd/scripts/lib/`, emitting structured JSON to `/tmp/test-<name>-results.json`. Full bullet list in [`docs/CICD.md`](docs/CICD.md) → "Perf / experiment workflows".

| Workflow | Script | Description |
|----------|--------|-------------|
| `test-throughput.yml` | `cli.ts bench-throughput` | Per-model tok/s benchmark with simple + optional agent-judge output check |
| `test-fa-k80.yml` | `cicd/scripts/test-fa-k80.sh` | K80 flash-attention regression + benchmark (FA off/on × KV cache type) |

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
│   ├── docker-compose.judge.yml
│   └── README.md
├── scripts/                 # Perf / experiment workflow scripts (unified pattern)
│   ├── test-fa-k80.sh
│   ├── format-results.sh
│   └── lib/                 # Shared helpers (sourceable from any script)
│       ├── response_capture.sh
│       ├── simple_check.sh
│       ├── judge_response.sh
│       └── container_log_snip.sh
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
