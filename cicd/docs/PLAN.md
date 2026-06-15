# Infrastructure Plan for ollama37 CI/CD

Based on analysis of test cases and existing documentation.

---

## Test Case Summary

| Suite | Test Cases | Requirements |
|-------|------------|--------------|
| Build | 2 | Docker images, build toolchain |
| Runtime | 3 | GPU passthrough, container health |
| Inference | 2 | Model download, GPU inference, API |
| Models | 3 | Large model testing on K80 |

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
| `ollama37-builder:latest` | 10-20GB | Toolchain (CUDA 11.4, GCC 10, Go) | ~90 min (cached) |
| `ollama37:latest` | 15-25GB | Runtime with compiled binary | ~10 min |
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
