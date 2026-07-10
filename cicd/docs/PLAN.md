# Infrastructure Plan for ollama37 CI/CD

Based on analysis of test cases and existing documentation.

---

## Test Case Summary

| Suite | Test Cases | Requirements |
|-------|------------|--------------|
| Build | 3 | Docker images, build toolchain |
| Runtime | 4 | GPU passthrough, container health, metrics schema |
| Inference | 2 | Model download, GPU inference, API |
| Models | 16 | Large model testing on K80 |

---

## Infrastructure Components

### 1. Hardware Requirements

| Component | Specification | Purpose |
|-----------|---------------|---------|
| GPU | Tesla K80 (compute 3.7) | Target hardware for testing |
| NVIDIA Driver | 470.x | Last driver supporting K80 |
| RAM | 32GB+ recommended | Docker builds, model loading |
| Storage | 100GB+ | Docker images (~40GB), models (~20GB) |

### 2. Host Software Stack

```
┌─────────────────────────────────────────┐
│            Host Machine                  │
├─────────────────────────────────────────┤
│  OS: Linux (Rocky/RHEL/Ubuntu)          │
│  Docker: 20.10+ with nvidia runtime     │
│  NVIDIA Driver: 470.256.02              │
│  nvidia-container-toolkit               │
└─────────────────────────────────────────┘
```

**Required packages:**
- `docker-ce` / `docker-compose-plugin`
- `nvidia-driver-470`
- `nvidia-container-toolkit`
- `curl` (for API tests)

### 3. Docker Images

| Image | Size | Purpose | Build Time |
|-------|------|---------|------------|
| `ollama37-builder:latest` | ~15GB | Toolchain (CUDA 11.4, GCC 10, Go) | ~90 min (first build; CMake from source) |
| `ollama37:latest` | ~1.9GB | Runtime with compiled binary | full no-cache 9-arch nvcc compile |

`TC-BUILD-003` asserts the runtime image is between 1GB and 3GB — it is a slim
multi-stage artifact, not a copy of the builder.
### 4. Container Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Host Network                             │
│                                                             │
│  ┌─────────────────────┐    Semantic judge: the keyless    │
│  │   ollama37          │    ACP agent judge (a Claude       │
│  │   (Test Subject)    │    agent over ACP) — no container, │
│  │                     │    no port, no API key.            │
│  │  Port: 11434        │                                    │
│  │  GPU: Tesla K80     │                                    │
│  │  Volume:            │                                    │
│  │  ollama-data        │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 5. Network Configuration

| Service | Container Port | Host Port | Purpose |
|---------|---------------|-----------|---------|
| ollama37 | 11434 | 11434 | Test subject API |

The agent judge needs no port — it runs as a stdio subprocess, keyless.

### 6. Volume Configuration

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `ollama-data` | `/root/.ollama` | Test subject models |

---

## Test Framework Components

### 1. LogCollector

**Purpose:** Capture container logs with precise test boundaries

**Architecture:**
```
docker compose logs --follow --timestamps
         │
         ▼
/tmp/ollama37-session-{timestamp}.log
         │
    ┌────┴────┐
    │ Markers │
    └────┬────┘
         │
===MARKER:START:TC-XXX:timestamp===
[test logs]
===MARKER:END:TC-XXX:timestamp===
         │
         ▼
/tmp/test-TC-XXX-logs.txt (extracted)
```

**Implementation:** File-based with text markers for crash resilience

### 2. Judge System

```
┌─────────────────────────────────────────┐
│              Test Result                 │
│                  │                       │
│         ┌───────┴───────┐               │
│         ▼               ▼               │
│   ┌──────────┐    ┌──────────┐          │
│   │  Simple  │    │   LLM    │          │
│   │  Judge   │    │  Judge   │          │
│   └────┬─────┘    └────┬─────┘          │
│        │               │                │
│   Exit codes      Log analysis          │
│   Grep patterns   Semantic eval         │
│        │               │                │
│        └───────┬───────┘                │
│                ▼                        │
│         PASS (both pass)                │
│         FAIL (either fails)             │
└─────────────────────────────────────────┘
```

### 3. Test Executor

**Responsibilities:**
- Parse YAML test definitions
- Manage test execution order
- Coordinate LogCollector
- Invoke judges
- Report results

---

## File Structure

```
cicd/
├── docs/
│   ├── CICD.md                    # Design philosophy
│   └── PLAN.md                    # This document
├── infrastructure/
│   └── README.md                  # Agent judge auth notes (no container)
├── specs/
│   ├── build.md                   # Build test specifications
│   ├── runtime.md                 # Runtime test specifications
│   ├── inference.md               # Inference test specifications
│   └── models.md                  # Models test specifications
├── tests/                         # Test framework (v2)
│   ├── src/
│   │   ├── cli.ts                 # CLI entry point
│   │   ├── types.ts               # TypeScript interfaces
│   │   ├── loader.ts              # YAML loader
│   │   ├── executor.ts            # Test runner
│   │   ├── log-collector.ts       # Docker log capture
│   │   ├── judge/
│   │   │   ├── simple-judge.ts    # Exit code + patterns
│   │   │   └── agent-judge.ts     # Semantic analysis (keyless ACP agent)
│   │   └── reporter/
│   │       ├── json.ts            # JSON output
│   │       └── console.ts         # Terminal output
│   ├── testcases/
│   │   ├── build/                 # TC-BUILD-001, 002
│   │   ├── runtime/               # TC-RUNTIME-001, 002, 003
│   │   ├── inference/             # TC-INFERENCE-001, 002
│   │   └── models/                # TC-MODELS-001, 002, 003
│   ├── package.json
│   └── tsconfig.json
├── results/                       # Test output (gitignored)
└── README.md                      # Quick start guide
```

---

## Implementation Checklist

### Phase 1: Infrastructure Setup
- [x] Verify host meets hardware requirements
- [x] Install NVIDIA driver 470.x
- [x] Install nvidia-container-toolkit
- [x] Configure Docker with nvidia runtime
- [x] Test subject uses existing `docker/docker-compose.yml`

### Phase 2: Docker Images
- [x] Build or verify `ollama37-builder:latest`
- [x] Build or verify `ollama37:latest`
- [x] Pull `dogkeeper886/ollama37:latest` for judge

### Phase 3: Test Framework (v2) - COMPLETE
- [x] Create YAML test case definitions (9 tests)
- [x] Implement LogCollector (file-based markers)
- [x] Implement Simple Judge (exit codes + patterns)
- [x] Implement the agent judge (keyless ACP semantic analysis)
- [x] Implement Test Executor
- [x] Implement JSON/Console reporters
- [x] Implement CLI

### Phase 4: Integration
- [x] Test suite execution order (Build → Runtime → Inference → Models)
- [x] Model unload after inference tests
- [x] Result reporting to `cicd/results/`
- [x] CI/CD pipeline integration (GitHub Actions)

---

## Two-testbed CI: `sm37` + `sm75`

The K80 is the only hardware-validated target, but it can never exercise a tensor-core
code path — `turing_mma_available(370)` is false. An RTX 2060 (cc 7.5) does. Adding it
as a second testbed is what surfaced #385.

**The two hosts are independent and cannot reach each other.** Both have internet.

### What constrains the design

1. **`ollama37:latest` is a tag in one machine's Docker daemon.** It is not a name that
   means anything on the other host. The test framework has no image parameter: the
   string appears only in `docker/docker-compose.yml` and `TC-BUILD-002/003`.
2. **`TC-BUILD-002` *is* the build** — `make build-runtime-local-no-cache`, a full
   no-cache nine-architecture `nvcc` compile. `test-build.yml` does not wrap a build.
3. **`TC-RUNTIME-001` owns the container** (`docker compose down` → `up -d`).
   Inference and models inherit it; the perf workflows manage their own lifecycle.
4. **`gpu-temp-guard.sh` defaults to 80 °C**, tuned for the K80, and is invoked by six
   workflows. It reads `GPU_TEMP_LIMIT` from the environment.

### The decision: build once, test the artifact

Building on `sm75` would take hours (i3-14100, no builder cache) and would produce a
*second* artifact — so a disagreement between testbeds could not be attributed. Build on
`sm37`, publish, pull.

```
   ┌──────────── sm37 (Tesla K80) ────────────┐
   │  TC-BUILD-002: make build-…-no-cache      │
   │    → ollama37:latest (local)              │
   │  docker tag  → dogkeeper886/ollama37-ci:ci-<sha>
   │  docker push ─────────────────────────────┼──┐
   │  → emit digest sha256:abc…  as output     │  │
   └───────────────────────────────────────────┘  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │  Docker Hub             │
                                    │  ollama37-ci:ci-<sha>   │
                                    │  @sha256:abc…           │  immutable
                                    └───────┬────────┬────────┘
                          pull by digest    │        │   pull by digest
              ┌───────────────────────────────┘        └──────────────────────┐
              ▼                                                               ▼
   ┌──────── sm37 ────────┐                                     ┌──────── sm75 ────────┐
   │ runtime, inference,  │                                     │ runtime              │
   │ models, perf sweeps  │                                     │ (+ FA regression once │
   │ → K80 no-harm gate   │                                     │    #385 is fixed)     │
   └──────────────────────┘                                     └───────────────────────┘
              └────────────────────┬──────────────────────────────────┘
                                   ▼
                    both ran the SAME digest — comparisons are sound
                                   │
                                   ▼
              (manual) release-docker: rebuild at the tag → Docker Hub :latest
```

**Pull by digest, never by tag.** A tag can be overwritten; a digest cannot. That is the
entire reason for the registry hop — so "sm37 says no-harm" and "sm75 says fixed" provably
describe the same bits.

### Why Docker Hub, and why no new credentials

`release-docker.yml` pushes to Docker Hub with **no `docker login` step and no Docker Hub
secret** — the `cicd-1` environment holds only `TESTLINK_*` and `OLLAMA_VERSION`. The
`sm37` daemon is already authenticated from a one-time login on that box.

So: `sm37` pushes with the credential it already has, and `sm75` pulls from a **public**
repo with no auth at all. Zero new accounts, zero new secrets, zero `docker login` steps.

CI images go to a **separate repo, `dogkeeper886/ollama37-ci`**, so half-tested `ci-<sha>`
tags never clutter the tag list users pull `:latest` from. Nothing leaks — the source is
public already.

GHCR was considered and rejected: it would require adding `docker login ghcr.io` to both
runners for no benefit here.

### Which suite runs where

| Suite | `sm37` (K80, 4×11.4 GiB) | `sm75` (RTX 2060, 5.1 GiB usable) |
|---|---|---|
| `build` | **yes** — builder image cached here | no — multi-hour compile, no cache |
| `runtime` | yes | **yes** — no model; the first suite to enable |
| `inference` | yes | **blocked by #385** — `TC-INFERENCE-001` pulls `gemma3:4b`, an FA-whitelisted arch; the stock compose auto-enables FA and the runner panics on cc>=7.5 |
| `models` | yes | **never** — 16 cases incl. `deepseek-r1:32b`, `gemma3:27b` |
| perf (`throughput`, `mcp`) | yes | only for models that fit |

`test-inference` on `sm75` is the natural regression test for #385, once #385 lands.

### Per-host knobs live on the host

| Knob | `sm37` | `sm75` |
|---|---|---|
| `GPU_TEMP_LIMIT` | 80 (the built-in default) | **85** — the driver reports an 83 °C *target*, so 80 would abort healthy runs |

Set it in the runner's `~/actions-runner/.env`, not by changing the script default and not
per workflow.

### Phases

- **A — select and identify the host.** `identify-host` composite action; `sm37` label;
  `runner_label` input on `test-runtime.yml` defaulting to `sm37`. *(#390, #394)*
- **B — stand up `sm75`.** Register `rocky9-2060-cicd-1` with label `sm75`;
  `GPU_TEMP_LIMIT=85` in the runner `.env`; a local `ollama37:latest` (pull the published
  image and retag). Dispatch `test-runtime` with `runner_label: sm75`.
- **C — publish the CI image.** `test-build.yml` tags and pushes
  `dogkeeper886/ollama37-ci:ci-<sha>` and emits the digest as a workflow output.
- **D — consume it.** `docker-compose.yml` takes `image: ${OLLAMA37_IMAGE:-ollama37:latest}`;
  the suites take an `image_ref` input. A job pulls by digest and retags locally.
- **E — verify a code change end to end.** The #385 flow: build on `sm37` → `sm37` runs the
  no-harm gate → `sm75` runs the FA regression → benchmark tile-FA vs FA-off at ≥6.8k
  context (`docs/porting-k80.md` §3) → only then promote.

### Open questions

- Does the `sm37` daemon's existing Docker Hub credential permit `push` to a **new** repo?
  A token scoped to `dogkeeper886/ollama37` alone would not. Verify before Phase C.
- Tag retention on `ollama37-ci`. Deleting tags needs a Hub API token, which does not exist
  today. Accumulate, or add the secret.

### Non-goals

- Building on `sm75`.
- Running the `models` suite on `sm75`.
- GHCR.
- Changing `release-docker.yml`. It rebuilds from a fresh GitHub clone at the release
  tag — same source as tested, different mechanism. Promoting a tested digest instead
  would be an improvement, but it is not required for two testbeds.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `0.0.0.0:11434` | Ollama bind address |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU visibility |
| `OLLAMA_DEBUG` | `0` | Verbose logging |
| `GGML_CUDA_DEBUG` | `0` | CUDA debug output |
| `TEST_MODEL` | `gemma3:4b` | Default test model |

---

## Port Conflict Prevention

The test subject and judge must use different ports:

```yaml
# Test Subject (docker-compose.test.yml)
ports:
  - "11434:11434"
```

The semantic judge is the keyless ACP agent judge — a stdio subprocess, no port.

---

## GPU Sharing Considerations

Only the test subject occupies the K80 GPUs; the agent judge runs no model on the GPU.

| Container | Model | VRAM Usage |
|-----------|-------|------------|
| Test Subject | gemma3:4b | ~3GB |
| **Total** | | ~3GB |

K80 has 2x12GB = 24GB total, so both can run concurrently.

**Note:** Larger test models (12B, 27B) may require unloading judge model temporarily.
