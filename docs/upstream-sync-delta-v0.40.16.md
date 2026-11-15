# Phase 0 delta analysis — sync to upstream v0.40.16

Generated 2026-11-15 on branch `sync/upstream-v0.40.13` (in-place merge of
`v0.40.16`, per maintainer request). Feeds Phases 1–3 of the upstream sync
plan. All findings verified against local git objects; the llama.cpp
`b12177` scratch re-audit is Phase 3.

## Verified SHAs

| Item | Value |
|---|---|
| Branch tip before merge | `c293a5a6a` (13 fork commits over the v0.40.13 merge) |
| Target tag `v0.40.16` | `d63b702609c4300a2ca07be2bdb0adb9834725e5` |
| Merge-base (branch, v0.40.16) | `04088e7c9` — the v0.40.13 sync point on upstream main |
| `LLAMA_CPP_VERSION` | `b11963` → `b12177` |
| `MLX_VERSION` | bumped (`065ec46…` → `54a7051…`); fork does not build MLX — inert |
| Upstream `go.mod` / `go.sum` | **byte-identical** v0.40.13 → v0.40.16 (no Go toolchain move) |
| Rollback assets | `ollama37:sync-v0.40.13-c293a5a6` (`8449819978f7`, tagged `:latest`) and `ollama37:pre-sync-rollback` (`d4fb57d3262c`) pullable; judge pin `v2.3.4@sha256:2804a259…` (`cicd/scripts/judge-up.sh:15`) unchanged |

Delta size: 28 upstream commits, 151 files, +15582/−1627. Concentrated in
`mlxrunner/` (47), `cmd/launch/` (26), `mlx/` (14), `model/` (11), `create/`
(9), `openai/` (7), `server/` (9), docs restructuring (4a89b76cc). No `ml/`,
`template/`, `envconfig/`, `tools/` or `llama/server/CMakeLists.txt` changes
upstream.

## 1. What did upstream absorb?

- **llama.cpp b11963 → b12177.** The Metal unary-cast workaround landed in the
  engine; upstream **deleted** `llama/compat/002-metal-unary-cast-constexpr.patch`
  (commit 04088e7c9). The fork's series applies after `llama/compat/*.patch` —
  the glob just has one fewer file, no fork patch references it. **No compat-k80
  patch is absorbed.**
- **Go num_parallel guard removed.** Upstream deleted the
  `slices.Contains([]string{"mllama","qwen3vl",…,"nemotron_h_omni"}, ModelFamily)`
  → `numParallel = 1` block in `server/sched.go` (load()), because
  llama-server b12177 now allows parallel requests for recurrent and vision
  models (84dc81db1). Fork policy: **take the upstream deletion** — no fork delta
  lives there. Validation-gate item: the K80 smoke matrix must include a
  qwen3.5 / qwen3next (DeltaNet recurrent) family, and the CI canary stays as
  the arbiter before any release.
- **New upstream features, taken as-is** (no fork interaction): AutoTrust
  decision model (`decision/`, `create/autotrust_decision_test.go`,
  `mlxrunner/model/autotrustdecision/`), Kimi Code CLI + Pi catalog/launch
  refinements (`cmd/launch/`), deferred tool loading for Claude Code,
  ChatGPT CLI restore/graceful restart, MLX concurrent generation + YaRN
  (Qwen3/Qwen3.5) + ripple preview, openai `/v1/models` empty-array and
  truncation forwarding, Vulkan iGPU mmap, Windows ARM64 CUDA 13.5/SM 103.
- **Docs restructuring** (4a89b76cc) moved/renamed upstream doc pages. Fork
  pages (`upstream-sync.md`, `porting-k80.md`, `server-log-metrics.md`,
  `upstream-sync-delta*.md`) are fork-owned and intentionally divergent; per
  path policy `docs/` (fork pages) → keep fork. Upstream-only doc moves are
  taken, fork pages are not re-merged onto the new structure in this sync.

## 2. Where does the fork's work live now?

| Fork behaviour | Home (unchanged by this delta) |
|---|---|
| Graph safety multiplier + `OLLAMA_GRAPH_SAFETY_MULTIPLIER` | `llm/llama_server.go` (`PredictServerVRAM`), `envconfig/config.go` |
| Flash-attention gate (CUDA 11.4, Volta-only, #337/#385) | `llm/llama_server.go` (`LlamaServerFlashAttention`), `ml/device.go` (`FlashAttentionSupported`) |
| Per-device VRAM components (weights/KV/graph) | `llm/llama_server.go` (`VRAMComponents`, `componentsByDevice`) |
| `/api/metrics` with K80 baselines (#98) | `server/metrics.go`, `server/routes.go` (route), `server/sched.go` (counters) |
| functiongemma template fallback + escape-args parser (#38) | `template/functiongemma.*`, `template/template.go` |
| Sorted tool definitions / tool-call args (#555) | `tools/tools.go` |
| compat-k80 patch series 1000–1006 | `llama/compat-k80/` (re-audited vs b12177 in Phase 3) |
| K80 build entry (`llama_cuda_v11_linux` preset) | `llama/server/CMakePresets.json`, `llama/server/CMakeLists.txt`, `build-local.sh`, `docker/` |

## 3. Which fork patches matter under b12177?

The 7-patch `llama/compat-k80/` series is **re-audited, not assumed**: clone
b12177, apply `llama/compat/001` + `1000..1006`, `git apply --check` each,
compile the touched targets (nvcc 11.4 / sm_37 for 1000/1001/1002/1003/1005/1006,
GCC 10 `cpu` preset for 1004), and re-run the pre-Volta constraint
checklist in `upstream-sync.md` §5 (cublasEx cc≥5.0, BF16 F32 pin, FA
MMA_F16 CUDA≥11.8, graphs on Kepler, `__CUDA_ARCH_LIST__`, CPU variant
toolchain guards). Patches that no longer apply are re-based onto b12177
context; the README manifest is updated row by row.

## 4. Conflict forecast (dry-run `git merge-tree`, 2026-11-15)

7 files changed on both sides; **2 real conflicts**, 5 clean auto-merges:

| File | Outcome |
|---|---|
| `.github/workflows/release.yaml` | CONFLICT (delete/modify). Fork deleted all upstream workflows (29cb9d3a2, 84210db18); upstream change is the Windows ARM64 CUDA 13.5 cross installer — irrelevant here. **Resolve: keep deletion.** |
| `README.md` | CONFLICT (content). **Resolve: keep fork** (two-build payload description); port no upstream content (its delta is Windows/MLX/docs). |
| `llama/server/CMakePresets.json` | Auto-merged. Upstream hunk: `121` → `103;121` in the CUDA 13 ARM64 preset. Fork's `llama_cuda_v11_linux` preset is in a separate hunk — verify intact post-merge. |
| `llm/llama_server.go` | Auto-merged. Upstream hunk: `bufferSizeRegex` gains `layer input`/`rotation` kinds (b12111). Fork hunks (FA gate, graph multiplier, VRAMComponents) verified present post-merge. |
| `llm/llama_server_test.go` | Auto-merged. Upstream added a b12111 buffer-kinds case to `TestMemoryParsingWriter`; fork modified `TestAppendFlashAttentionArgs`. Both verified present post-merge. |
| `server/routes.go` | Auto-merged. Upstream hunk: `autotrust-decision` renderer in `SystemOneHandler`; fork hunk: `/api/metrics` route — verified present. |
| `server/sched.go` | Auto-merged. Upstream hunk: num_parallel guard removal; fork hunks: `serverMetrics` field + 4 counter sites — verified present. |

## 5. Findings post-merge (2026-11-15)

- **b12177 NCCL allreduce path.** `ggml/CMakeLists.txt` adds
  `GGML_CUDA_NCCL` (**ON by default**); with it set, `ggml-cuda` calls
  `ncclCommInitRankConfig` / `ncclConfig_t` (NCCL ≥ 2.17). Hosts with an
  older distro NCCL (the `kilo-compilecheck:cuda11.4` helper ships one)
  fail the build. The `ollama37-builder` image has **no NCCL**, so
  `find_package(NCCL)` degrades to a warning there and the option is inert;
  keep any rebuilt builder NCCL-free or pass `-DGGML_CUDA_NCCL=OFF`.
  No patch filed: the K80 build never wants NCCL.
- **`go test ./...` gate, MLX link.** The `mlx` cgo dlopen layer
  (`mlx/dynamic.c`, present since before v0.40.13) does not link on the
  builder's glibc 2.28 (`dlopen` needs `-ldl`; upstream CI's glibc ≥ 2.34
  has it in libc). Reproduced on pre-merge `c293a5a6a` → **pre-existing,
  not a merge regression**; the gate runs
  `go test $(go list ./... | grep -vE 'mlx|mlxrunner')`.
- **compat-k80 vs b12177.** Full chain (001 + 1000–1006) `git apply --check`
  green on a fresh b12177 clone after re-basing 1001 (include list gained
  `batch-get.cuh`); 1000/1002–1006 apply byte-identical. b12177 adds no new
  x86 CPU variants, so 1004's guard set stays complete (no gcc10 rebuild).
