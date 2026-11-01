# Syncing the fork with a new upstream Ollama release

The fork carries two kinds of work over upstream Ollama: Go-level behaviour the K80 needs
(memory estimate, flash-attention gate, metrics, templates) and a numbered patch series on
the pinned llama.cpp. A sync lands both on top of one immutable upstream tag, on a
dedicated branch, without ever touching `main` until the gates pass.

This document is the procedure. The K80 *porting* rules — never bend the builder, benchmark
every ported optimization — live in [porting-k80.md](./porting-k80.md).

## 0. Ground rules

- **Target an immutable tag**, never upstream `main`. Record its SHA.
- **One branch per sync**, cut from `main`: `sync/upstream-<tag>`.
- **`main` stays deployable** on the previous release throughout; production keeps serving it.
- **Never re-pin the agent judge** (`cicd/scripts/judge-up.sh`) to a build under test.

## 1. Prepare

```sh
git remote add upstream https://github.com/ollama/ollama.git   # once
git fetch upstream --tags
git merge-base main upstream/v<tag>        # record it
git switch -c sync/upstream-v<tag> main
```

Record the rollback assets before building anything: the previous release image and the
judge image digest must be pullable.

Since v0.40.13, llama.cpp is **not vendored**: `LLAMA_CPP_VERSION` names the pinned ref,
`llama/server/CMakeLists.txt` fetches it with FetchContent, upstream's `llama/compat/`
patches are applied at configure time, and the fork's `llama/compat-k80/` series is
applied right after (see [section 5](#5-the-llamacompat-k80-patch-series)).

## 2. Delta analysis

Before merging, answer three questions in a report committed to the sync branch:

1. **What did upstream absorb?** Anything natively supported in the pinned llama.cpp
   (architectures, kernels, gates) is dropped from the fork's patch series, not duplicated.
2. **What did the fork change and where does it live now?** The engine rewrite
   (cgo → llama-server) moved Go semantics: map each fork behaviour onto its new home
   (`llm/llama_server.go`, `server/`, `discover/`, `model/`, `create/`).
3. **Which old `llama/patches/*.patch` still matter?** Classify each as upstream-owned,
   fork-added, absorbed, or obsolete under the new engine.

## 3. Structural merge

```sh
git merge --no-commit --no-ff upstream/v<tag>
```

Resolve per path, by intent:

| Path | Policy |
|---|---|
| `cicd/`, `docker/`, `docs/` (fork pages), `.github/`, `build-local.sh` | keep fork |
| `llama/`, `ml/`, `cmake/`, root `CMakeLists.txt`, root `Dockerfile`, `LLAMA_CPP_VERSION` | take upstream; fork additions come back in Phase 3/4 |
| `convert/`, `runner/`, `kvcache/`, `sample/` | accept upstream deletions |
| `harmony/`, `thinking/` | take upstream; port fork deltas only |
| `llm/`, `server/`, `discover/`, `model/`, `template/`, `api/`, `cmd/` | take upstream structure, re-apply fork semantics in Phase 4 |
| `go.mod`, `go.sum` | take upstream |

Never resolve a conflict by keeping a stale textual copy of fork code on a rewritten
engine — re-express the behaviour instead. Commit the merge (no green build expected).

## 4. Go layer

Re-apply fork semantics one commit per area, then gate:

```sh
go build ./... && go vet ./... && go test ./...      # CPU-only, as pr-check.yml runs it
```

Current areas: graph safety multiplier (+ `OLLAMA_GRAPH_SAFETY_MULTIPLIER`), flash-attention
gate on CUDA < 11.8, `/api/metrics` K80 baselines, template/renderer fallbacks.

## 5. The `llama/compat-k80` patch series

Ordered `*.patch` files applied to the fetched llama.cpp **after** `llama/compat/`,
against the pinned `LLAMA_CPP_VERSION`. `llama/compat-k80/README.md` is the manifest:
one row per patch with the issue it answers.

Rules:

- Numbering starts at **1000** to stay disjoint from upstream's compat numbers.
- One concern per patch; it must apply after the whole chain, not just after upstream.
- A kernel that cannot compile under CUDA 11.4 is CMake-gated off and filed — the K80
  build stays green.

Adding a patch:

```sh
# scratch tree at the pinned ref
git clone --depth 1 --branch "$(cat LLAMA_CPP_VERSION)" https://github.com/ggml-org/llama.cpp wc
cd wc
for p in ../llama/compat/*.patch ../llama/compat-k80/1*.patch; do git apply "$p"; done
git commit -am base            # freeze the chain as the diff base
# ... edit the source ...
git diff > ../llama/compat-k80/1NNN-<slug>.patch
```

Then verify, in this order:

1. `git apply --check` the new patch on a fresh chain (`compat` + all `1*.patch`).
2. Compile the touched target under nvcc 11.4 / sm_37 (scratch image, or the builder).
3. For a CPU-side patch, compile the `cpu` preset under GCC 10.
4. Ship it: rebuild the runtime image, then test on the K80.

Fast runtime check without a full image rebuild: build `libggml-cuda.so` in a scratch tree
with the same architecture list and bind-mount it over the shipped one:

```sh
docker run -d --runtime nvidia -v /path/to/build/bin/libggml-cuda.so.0.28.0:/usr/lib/ollama/cuda_v11/libggml-cuda.so:ro ...
```

### Pre-Volta constraints — the checklist for the next sync

Every new upstream llama.cpp release must be re-audited against these; the fork's patches
exist because of them.

| Constraint | Where it bites | Fork answer |
|---|---|---|
| `cublasGemmEx` / `*Ex` batched need cc ≥ 5.0 | every cuBLAS matmul | 1000: legacy `cublasSgemm*` + `CUBLAS_GEMM_DEFAULT` |
| BF16 GEMM (`CUDA_R_16BF`) needs cc ≥ 8.0 | small-matrix matmul keeps BF16 | 1005: pin compute type F32 pre-Volta |
| `CUBLAS_GEMM_DEFAULT_TENSOR_OP` needs Volta | matmul algorithm choice | 1000 |
| MMA_F16 flash attention needs CUDA 11.8+ on cc ≥ 7.5 | `fattn.cu` NaN | 1003: Volta-only else VEC |
| CUDA graphs gated to Ampere upstream | decode launch cost | 1002: keep them on Kepler |
| No autoboost clock filler between dies | K80 decode clock drop | 1001 |
| `__CUDA_ARCH_LIST__` only from nvcc ≥ 11.5 | ollama cannot filter unmatched GPUs | 1006: emit `ARCHS` from CMake |
| AMX / AVX512-BF16 / AVX-VNNI CPU variants need GCC 11 + binutils 2.36 | `cpu` preset build | 1004: compiler feature guards |
| cuBLAS paths gated away on pre-Volta (`use_blas` needs fast FP16; `ssm-scan` SSD needs Turing) | conv2d/conv3d, Mamba | no patch — verify the gate still exists |

## 6. Build entry

- Presets live in `llama/server/CMakePresets.json`; the K80 preset is
  `llama_cuda_v11_linux` (`OLLAMA_RUNNER_DIR=cuda_v11`, `sm_37`–`sm_86` `-real` CUBIN).
  **Not `cuda_v12`**: the runner dir names the ABI, and `cuda_v12` selects
  `filterOldCUDADriver`, which hides the K80 on 470 drivers.
- The payload is **two builds**, as upstream's root Dockerfile does: the `cpu` preset
  installs `llama-server` and the core libraries to `dist/lib/ollama/`, the CUDA preset
  installs the backend and bundled CUDA runtime to `dist/lib/ollama/cuda_v11/`.
- `cmake --build <dir> --preset <name>` resolves build presets from the *repository root*
  `CMakePresets.json`; pass targets instead (`--target llama-server --target ggml-cuda`).
- Builder pins stay: CUDA 11.4, GCC 10, CMake 4. Only Go tracks upstream's `go.mod`.

## 7. Validation gates

1. **Local image + smoke matrix** (per iteration, minutes): build
   `ollama37:sync-<tag>-<sha>`, run it beside production on another port, check CUDA
   detection (`compute=3.7` via `cuda_v11`), `/api/metrics`, and one short generation per
   model family present in the store. Each family proves a different path: DeltaNet
   (`qwen3.5`), tool/thinking parsers (`deepseek-r1`), BF16/vision models (`gemma4`,
   `qwen3-vl`), embeddings.
2. **CI canary** (`test-canary.yml` on sm37): the pinned v2.3.4 judge must still call word
   salad garbage and pass a sound reply. The judge model only exists on the runner.
3. **Full matrix** (`pipeline.yml` on sm37): build → deploy → canary → models → throughput →
   mcp, with throughput compared against the previous release.
4. Spot-check `sm61`/`sm86` for the multi-arch CUBIN claim.

## 8. Release

PR the sync branch into `main` (merge commit, keep history), tag `v2.4.0`, update
`.env.example`/docs, and rehearse the rollback: redeploy the previous release image,
judge pin unchanged.
