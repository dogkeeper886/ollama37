# Phase 0 delta analysis — sync to upstream v0.40.13

Generated 2026-11-01 on branch `sync/upstream-v0.40.13`. Feeds Phases 1–3 of the
upstream sync plan. All findings verified against local git objects and a
read-only checkout of llama.cpp `b11963`.

## Verified SHAs

| Item | Value |
|---|---|
| Fork base (merge-base with upstream) | `114c3f22657750cfb57f70c4a0d6e7389fb7a9fe` |
| Target tag `v0.40.13` | `04088e7c9d41cba5d33bafbf788d87a465c79d5b` |
| Upstream `LLAMA_CPP_VERSION` | `b11963` |
| Old fork vendor pin (`Makefile.sync`) | `7049736b2dd9011bf819e298b844ebbc4b5afdc9` |
| Rollback assets | `ollama37:v2.3.5` manifest pullable; judge pin `v2.3.4@sha256:2804a259…` (`cicd/scripts/judge-up.sh:15`) |
| Upstream `go.mod` | `go 1.26.10` (fork: `go 1.24.0`) |

## 1. Model architectures — ABSORBED upstream (drop from fork)

llama.cpp `b11963` natively ships: `qwen3next.cpp`, `qwen35.cpp`, `qwen35moe.cpp`,
`delta-net-base.cpp` (`GGML_OP_GATED_DELTA_NET`), `gemma4.cpp`,
`gemma4-assistant.cpp`, `lfm2.cpp`, `lfm2moe.cpp`, `lfm3.cpp` — arch enums
`LLM_ARCH_QWEN3NEXT/QWEN35/QWEN35MOE/GEMMA4/GEMMA4_ASSISTANT/LFM2/LFM2MOE/LFM3`
all present in `src/llama-arch.cpp`.

Upstream ollama `v0.40.13` natively ships: `create/qwen35.go`,
`create/qwen35_decision.go`, `create/gemma4.go`, `create/lfm3.go`,
`model/renderers/{qwen35,gemma4,gemma4_audio,lfm2,lfm3,…}.go`,
`model/parsers/{qwen35,gemma4,lfm2,lfm3,…}.go`.

**Verdict:** the fork's qwen3.5/DeltaNet graph+tensor work (~20 commits),
gemma4 (`ea675471` `283d50ec`) and lfm2 (`2c3caf29` `b14bc8b5` `14e9fc12`
`9ea19974`) architecture work is absorbed. Do NOT re-implement in
compat-k80. Exceptions (FORK-ONLY, keep as patch series):

- K80-specific CUDA kernels in `ml/backend/ggml/ggml`: `dequantize_block_q4_K_k80`
  (convert.cu), `mmid-k80.cu` grouped-expert prefill, `clock-k80.cu` clock
  pinning, K80 mat-vec variants + `ggml_cuda_should_use_mmvq_k80` guards,
  K80 dense-prefill path in fattn.cu, Kepler CUDA-graph decode work.
- Fork's `gated_delta_step_f32` (GGML_OP_GATED_DELTA_STEP): b11963's
  `gated_delta_net` MMA path requires SM_70+; K80 needs the simpler step
  kernel or a Kepler fallback → compat-k80 item.
- Fork `SetThinkValue`/`honorThink` deltas in qwen35 renderer/parser and
  lfm2 thinking support (`ThinkingClose`, `PreservedTokens`) — re-check
  against upstream's versions during Phase 2 (upstream gained its own
  thinking plumbing; port only genuine gaps).

Fork's native-engine `model/models/{qwen35,gemma4}` (Go engine) become dead
code under upstream's llama-server-only engine and are dropped with `runner/`.

## 2. `harmony/` + `thinking/` — no fork deltas to port

Fork commits touching these dirs: only `ef14fb5b` (initial sync) and
`27dd062b` (upstream merge) — no fork-authored logic. All diff hunks vs
`v0.40.13` are upstream additions (`ThinkingClose`, `PreservedTokens`,
`contentPrefill`, `TemplateSupportsThinking`, …) or typo fixes.
**Policy: take upstream wholesale. Port nothing.**

## 3. Fork Go semantics → upstream homes

| Fork semantic | Fork location | Upstream v0.40.13 target | Notes |
|---|---|---|---|
| Graph safety multiplier 2.5 + `OLLAMA_GRAPH_SAFETY_MULTIPLIER` | `llm/memory.go` (`graphSafetyMultiplier`) | No Go-side estimator remains; llama-server estimates internally. Re-express via `-ngl` capping / `--cache-ram` from Go, or compat-k80 patch to llama.cpp estimate | Upstream parses llama-server log via `memoryParsingWriter` in `llm/llama_server.go` |
| FA gate: CUDA < 11.8 → Volta-only (#385) | `ml/device.go` `FlashAttentionSupported()` | `ml/device.go` `FlashAttentionSupported()`/`cudaFlashAttentionSupported()` + `llm/llama_server.go` `appendFlashAttentionArgs()` (`--flash-attn`) | Same gate logic, new call path |
| K80 metrics baselines + labeled VRAM components (#98) | `server/metrics.go` (fork-only file) | Keep fork file; rewire to upstream `MemorySize()`, `VRAMByGPU()`, `GetDeviceInfos()` in `llm/llama_server.go` | Per-component breakdown (weights/KV/graph) needs new llama-server log parsing or compat-k80 metrics output |
| Ghost-GPU pruning / device reset (#98/#509) | `discover/runner.go`, `discover/gpu.go`, metrics `attributeGPUUsage()` | `discover/runner.go` `GPUDevices()`, `discover/llama_server.go` bootstrap, `discover/native_probe*.go` | Upstream discovery is richer (native probe, vulkan, cuda_compat, amd); graft fork pruning onto it |
| qwen3.5/gemma4/lfm2 registration/templates | `convert/convert_qwen3*.go`, `kvcache/recurrent*.go`, `model/models/*` | Upstream `create/` + `model/renderers` + `model/parsers` (native) | Absorbed; only thinking-value deltas re-applied |

## 4. `llama/patches/` classification (32 patches)

- **17 upstream-owned at merge-base** (renumbered): fork 0001–0015, 0018, 0020
  → **DROP** (upstream deleted the patch system; `llama/compat/` +
  `llama-ollama-compat.*` replace what ollama still needs).
- **15 fork-added**:

| Patch | Phase 3 disposition |
|---|---|
| 0016 mtmd C API | Drop — was for cgo bindings; llama-server uses mtmd internally |
| 0017 win32 GNUC power-throttle guard | Drop — Windows-only, out of K80 scope |
| 0019 batch-size hint | Drop — embedded-runner infra, gone with new engine |
| 0021 mtmd windows build fix | Drop — Windows-only |
| 0022 no-alloc sched mode | Drop — embedded-runner infra |
| 0023 decode disable output_all | Candidate — decode perf patch; evaluate on llama-server |
| 0024 backend device reset (#509) | **Port** → compat-k80 (ghost-GPU reset) |
| 0025 uncaught-exception hardening | Port → compat-k80 |
| 0026 GPU discovery enhancements (#98) | **Port** → compat-k80 (metrics/discovery) |
| 0027 NVML unified-memory fallback | Port → compat-k80 (metrics) |
| 0028 CUDA spin scheduling | **ABSORBED** in b11963 (`ggml-cuda.cu` SM121 spin) — drop |
| 0029 report LoadLibrary failures | Drop — Windows-only |
| 0032 interleave multi-rope | Verify — b11963 IMROPE differs; validate qwen3.5 output in matrix before porting |
| 0033 lfm2 token_embd_norm optional | Likely obsolete — b11963 native lfm2; validate lfm2.5:24b load in matrix |
| 0034 lfm2 feed_forward shape | Likely obsolete — same as 0033 |

## 5. Upstream structure notes (for Phase 1)

- Upstream `ml/` is slim (`device.go`, `path.go`); fork's `ml/backend/ggml/**`
  K80 kernels leave the tree and re-enter as compat-k80 patches.
- Upstream `discover/` gains `llama_server.go`, `native_probe*`, `vulkan*`,
  `cuda_compat.go`, `amd.go`, `types.go`.
- Upstream `llama/compat/` = 2 patches + compat shim (`llama-ollama-compat.*`,
  `compat.cmake`); applied by `llama/server` FetchContent configure.
- Upstream adds `mlx/`, `mlxrunner/`, `app/`, `anthropic/`, `internal/`,
  `decision/`, `manifest/`, `transfer/`, `compatmigrate/`, `parser/` etc.
- `fs/ggml` → `fs/gguf`; layer offload decided inside llama-server (`-ngl auto`).
