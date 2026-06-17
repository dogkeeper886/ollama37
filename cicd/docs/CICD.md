# CI/CD Pipeline for Ollama37

## Project Goal

Enable Ollama to run on Tesla K80 GPUs (CUDA compute capability 3.7), which are no longer supported by mainstream Ollama builds. This requires custom compilation with CUDA 11.4 and legacy CUBLAS function fallbacks.

## Infrastructure

The CI/CD pipeline runs on a self-hosted GitHub Actions runner with Tesla K80 hardware. This is necessary because:

1. K80 requires specific NVIDIA driver (470.x) and CUDA version (11.4)
2. Cloud runners don't provide legacy GPU hardware
3. Real GPU testing is essential - emulation cannot catch CUBLAS compatibility issues

## Build Strategy

The build uses a two-stage Docker approach:

**Stage 1 (Builder)**: A cached base image containing the complete toolchain - Rocky Linux 8, CUDA 11.4, GCC 10 (compiled from source due to RHEL limitations), CMake, and Go. This image takes ~90 minutes to build but is reused across all builds.

**Stage 2 (Runtime)**: Built on each commit, this stage clones the source, compiles with K80-specific CMake presets, and packages the binary. Build time is ~10 minutes.

This separation means routine builds are fast while toolchain changes are rare and cached.

## Test Framework Design

### Philosophy

The test framework is designed around two key insights:

1. **Exit codes lie**: A CUDA operation can fail silently, returning success while producing garbage output. Traditional pass/fail based on exit codes misses these failures.

2. **Logs tell truth**: The real story is in the execution logs - CUBLAS errors, memory allocation failures, and GPU fallback warnings appear there even when commands "succeed".

### Judge System

The framework implements a dual-judge architecture:

**Simple Judge**: Fast, deterministic verification based on:
- Exit codes (all steps must return 0)
- Pattern matching (expected patterns found, rejected patterns absent in output)
- CUDA error detection (no CUBLAS/CUDA errors in logs)

**Agent Judge**: Semantic analysis via the keyless ACP agent judge — a Claude agent over the Agent Client Protocol (no API key, no container). It receives test criteria + logs and evaluates whether the actual behavior matches the expected behavior. This catches:
- CUDA errors that don't cause exit failures
- Subtle GPU fallback to CPU mode
- Memory allocation warnings
- Incorrect but non-crashing output

**Dual Mode** (default): Both judges must pass. This combines the speed of simple checking with the depth of semantic analysis.

### Log Collection

A critical problem with container log analysis is temporal precision. Using `docker compose logs --since=5m` creates issues:

- Logs from previous tests contaminate current test analysis
- Long-running tests may exceed the time window
- Fast tests include irrelevant historical logs

The LogCollector solves this using a file-based architecture with embedded text markers.

#### Architecture

```
docker compose logs --follow --timestamps
         │
         ▼
/tmp/ollama37-session-{timestamp}.log  (persistent file)
         │
===TEST:TC-001:START:2024-01-15T10:30:05Z===
[container logs during test]
===TEST:TC-001:END:2024-01-15T10:30:15Z===
         │
         ▼ sed extraction
cicd/results/{run-timestamp}/TC-001.log  (test-specific logs)
```

#### Session File Format

The session file contains all docker logs with embedded markers:

```
===SESSION:START:2024-01-15T10:30:00.000Z===
ollama37  | 2024-01-15T10:30:01Z msg="starting server"
===TEST:TC-RUNTIME-001:START:2024-01-15T10:30:05.123Z===
ollama37  | 2024-01-15T10:30:06Z msg="inference compute" library=CUDA
ollama37  | 2024-01-15T10:30:07Z msg="loaded model" layers=28
===TEST:TC-RUNTIME-001:END:2024-01-15T10:30:15.789Z===
===TEST:TC-RUNTIME-002:START:2024-01-15T10:30:16.000Z===
[more logs...]
===TEST:TC-RUNTIME-002:END:2024-01-15T10:30:45.000Z===
===SESSION:END:2024-01-15T10:31:00.000Z===
```

**Marker Format**: `===TEST:{TEST_ID}:{TYPE}:{ISO_TIMESTAMP}===`
- TEST_ID: Test case identifier (e.g., `TC-RUNTIME-001`)
- TYPE: `START` or `END`
- ISO_TIMESTAMP: When the marker was written

#### Log Extraction

Test-specific logs are extracted using sed:

```bash
# Extract logs for TC-RUNTIME-001 (excluding marker lines)
sed -n '/===TEST:TC-RUNTIME-001:START:/,/===TEST:TC-RUNTIME-001:END:/{/===TEST:/d;p}' \
  /tmp/ollama37-session-*.log
```

For tests still running (no END marker yet), extraction continues to EOF:

```bash
sed -n '/===TEST:TC-RUNTIME-001:START:/,${/===TEST:/d;p}' /tmp/ollama37-session-*.log
```

#### Design Benefits

1. **Crash Resilience**: The session file persists at `/tmp/ollama37-session-{timestamp}.log` even if the test process crashes. This enables post-mortem log analysis.

2. **Bounded Memory**: No in-memory array growth. All logs are written directly to disk.

3. **Precise Boundaries**: Text markers provide exact test boundaries regardless of docker log buffering delays.

4. **Race Condition Prevention**: All writes (both docker log data and markers) go through a serialized write queue with line buffering, ensuring markers never interleave with log lines.

#### Cleanup

- Old session files (> 24 hours) are automatically cleaned up at LogCollector startup
- Stale test log files are removed when a new test with the same ID starts

### Test Execution Flow

1. **Load Phase**: YAML test definitions are parsed and sorted by dependency order
2. **Collection Start**: LogCollector begins streaming container logs
3. **Execution Phase**: Tests run sequentially, each step receiving current test ID via environment
4. **Log Capture**: Before each step, accumulated logs are written to a test-specific file
5. **Judgment Phase**: Both judges evaluate results - simple checks exit codes, LLM analyzes logs
6. **Cleanup**: Models are unloaded from VRAM, log collector stops

### Test Architecture

Tests are organized into four suites that must run in order:

**Build Suite** (3 tests): Verifies the builder Docker image contents, runtime image build, and image-size invariants. No GPU required.

**Runtime Suite** (4 tests): Starts the container and verifies GPU detection, container health, and `/api/metrics` endpoint schema. Critical validation that the driver/toolkit/container integration works.

**Inference Suite** (2 tests): Tests model pull + API inference with gemma3:4b. Validates that models load correctly and generate responses using GPU.

**Models Suite** (13 tests): Per-model regression coverage on K80 hardware — gpt-oss:20b, ministral-3:3b, gemma3 (4b / 27b), gemma4 (e4b / 26b), qwen3.5 (9b / 27b), qwen3-vl (8b / 30b), deepseek-r1 (14b / 32b). Each model size unloads after testing to free VRAM for the next.

### Model Unload Strategy

K80 has limited VRAM (12GB per GPU). The framework explicitly unloads each model after its tests complete, rather than relying on automatic eviction. This ensures:

- Predictable VRAM state between tests
- No interference from cached models
- Clean baseline for each model size test

Workflow-level cleanup provides a safety net if individual test unloads fail.

## Error Detection

The framework specifically watches for K80-related failure patterns:

- `CUBLAS_STATUS_*` errors indicate the legacy CUBLAS fallback isn't working
- `CUDA error` messages suggest driver/toolkit mismatch
- `cudaMalloc failed` indicates VRAM exhaustion
- `id=cpu library=cpu` means GPU detection failed entirely

These patterns are checked by both the simple judge (via grep in test steps) and the agent judge (via semantic log analysis).

## Design Decisions

**Why YAML test cases?** Declarative test definitions separate test logic from execution machinery. Adding a new test requires no code changes.

**Why LLM judging?** Traditional test assertions require anticipating every failure mode. LLM evaluation can recognize novel failures and evaluate fuzzy criteria like "response should mention Paris".

**Why sequential execution?** Log collection with precise boundaries requires knowing which test is running. Parallel execution would interleave logs unpredictably.

**Why Docker-based builds?** Reproducibility. The exact toolchain that works is captured in the builder image. No "works on my machine" issues.

**Why self-hosted runners?** K80 hardware. No cloud provider offers compute capability 3.7 GPUs for CI/CD.

## Limitations

- Tests must run sequentially for accurate log collection
- The agent judge requires Claude auth on the runner (`~/.claude` or `CLAUDE_CODE_OAUTH_TOKEN`)
- K80 VRAM limits restrict maximum model size to ~27B parameters
- Build times are significant due to CUDA compilation

## Framework Structure

The test framework is located at `cicd/tests/`:

```
cicd/
├── docs/
│   ├── CICD.md              # This document
│   └── PLAN.md              # Infrastructure planning
├── specs/
│   ├── build.md             # Build test specifications
│   ├── runtime.md           # Runtime test specifications
│   └── inference.md         # Inference test specifications
├── tests/
│   ├── src/
│   │   ├── cli.ts           # CLI entry point
│   │   ├── types.ts         # TypeScript interfaces
│   │   ├── loader.ts        # YAML test case loader
│   │   ├── executor.ts      # Test execution engine
│   │   ├── log-collector.ts # Docker log capture
│   │   ├── judge/
│   │   │   ├── simple-judge.ts
│   │   │   └── agent-judge.ts
│   │   └── reporter/
│   │       ├── json.ts
│   │       └── console.ts
│   ├── testcases/
│   │   ├── build/           # TC-BUILD-001..003
│   │   ├── runtime/         # TC-RUNTIME-001..004
│   │   ├── inference/       # TC-INFERENCE-001..002
│   │   └── models/          # TC-MODELS-001..013
│   ├── package.json
│   └── tsconfig.json
├── results/                 # Test output (gitignored)
└── README.md                # Quick start guide
```

## GitHub Actions Integration

The test framework integrates with GitHub Actions via reusable workflows in `.github/workflows/`:

**Pipeline Workflow** (`test-pipeline.yml`):
Runs all test suites in sequence: build → runtime → inference → models. This is the primary workflow for full validation.

**Individual TC-framework Workflows**:
- `test-build.yml` — Build suite (3 tests)
- `test-runtime.yml` — Runtime suite (4 tests)
- `test-inference.yml` — Inference suite (2 tests); supports `JUDGE_MODE=dual` opt-in agent judge
- `test-models.yml` — Models suite (13 tests); supports `JUDGE_MODE=dual` opt-in agent judge

**Design Notes**:
- Individual TC workflows do not manage container lifecycle
- Container must be running before runtime / inference / models tests
- Pipeline workflow orchestrates the sequence via `needs:` dependencies
- All workflows run on self-hosted runners with K80 hardware

### Perf / experiment workflows (separate pattern)

The TC framework above handles **correctness validation** — "did this model load and produce coherent output?". Performance benchmarks and experiments live alongside under a separate, simpler contract:

- A TypeScript subcommand per workflow or experiment (`cli.ts bench-throughput`, `cli.ts test-fa`, `cli.ts test-mcp`)
- Metrics + coherence judging reuse the framework's perf helpers and the one agent judge
- A JSON report (`--output`) plus a markdown summary on stdout
- Workflow manages container lifecycle (stop production → run → restart production → upload artifacts)

Current perf workflows: `test-throughput.yml` (tok/s benchmark) and `test-fa-k80.yml` (FA regression + benchmark). The agent judge is invoked here too but only ever to answer "is this response meaningful for its prompt?" — never for static / log / exit-code checks.

`cli.ts test-mcp` is a **capability** probe rather than a perf metric: it drives a **real** stdio MCP server (server-agnostic — default `testlink-mcp`, override via `--mcp-command`/`--mcp-args`), lists and translates its tools to Ollama's `/api/chat` `tools[]`, runs the model's tool loop against the live server, and reuses the one agent judge to grade whether the final answer is grounded in the tool result. It reports a per-model verdict (PASS / FAIL / NO TOOL SUPPORT) and runs via the manual `test-mcp.yml` workflow (`workflow_dispatch`; testlink-mcp creds supplied as `cicd-1` secrets and forwarded with `--mcp-env`). See [`docs/traces/mcp-tool-call-validation.md`](../../docs/traces/mcp-tool-call-validation.md) for an end-to-end validation run.

Add `--verify-live --verify-allow <read-only-tool>` for a stronger check: instead of trusting the result the model captured, the verifier makes its **own** read-only call to the live MCP server and the sandboxed agent judge grades the model's answer against that **fresh** ground truth — so a confidently-wrong (or stale-captured) answer FAILs where consistency-only would pass. Read-only is by exact allow-list (empty ⇒ fail-closed), and it stays keyless (the grading agent is never exposed to the tool). Background on why the verifier calls the tool itself rather than the agent: keyless ACP adapters either don't surface a client-injected stdio server or flood the agent with account connectors disableable only via `ANTHROPIC_API_KEY`.

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
