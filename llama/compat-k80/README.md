# llama/compat-k80 — fork-owned K80 patch series

Ordered `*.patch` set restoring Tesla K80 (compute 3.7) support on top of the
pinned llama.cpp (`LLAMA_CPP_VERSION`, currently `b12177`). Applied by
`llama/compat-k80/compat-k80.cmake` **after** upstream's `llama/compat/`
patches via the FetchContent `PATCH_COMMAND` chain in `llama/server`.

Numbering starts at 1000 to stay disjoint from upstream compat numbers.
Discipline (docs/porting-k80.md): port the source; never bend the builder's
CUDA 11.4 / GCC 10 pin. A kernel that cannot compile under CUDA 11.4 gets
CMake-gated off and filed as an issue, keeping the K80 build green.

## Patches

| # | File | Issue | What |
|---|---|---|---|
| 1000 | `1000-cublas-pre-volta-fallback.patch` | #367-era | `cublasGemmEx`/`*Ex` batched need CC >= 5.0 and TENSOR_OP needs Volta+; pre-Volta uses `CUBLAS_GEMM_DEFAULT` + legacy `cublasSgemm{StridedBatched,Batched}` for F32 |
| 1001 | `1001-k80-clock-pinning.patch` | #587 | Autoboost clock filler for multi-die K80 decode (`clock-k80.cu`). Re-based on b12177 (2026-11-15): the `ggml-cuda.cu` include list gained `batch-get.cuh`, shifting the include hunk |
| 1002 | `1002-kepler-cuda-graphs.patch` | #573 | Keep CUDA graphs on Kepler (upstream gates them to Ampere+) |
| 1003 | `1003-fa-guard-cuda11.patch` | #385 | Disable MMA_F16 FA under CUDA 11.4 for cc >= 7.5 (NaN); Volta keeps a working path |
| 1004 | `1004-cpu-variant-toolchain-guards.patch` | sync | Guard cooperlake/zen4/alderlake/sapphirerapids CPU backends behind compiler/assembler feature checks (need GCC 11 + binutils 2.36); the CUDA 11.4 pin's GCC 10 skips them like upstream's POWER11/z17 pattern |
| 1005 | `1005-cublas-bf16-pre-volta.patch` | sync | Pin the cuBLAS compute type to F32 on pre-Volta NVIDIA: the BF16 branch keeps `GGML_TYPE_BF16` for small matrices, and `cublasGemmEx(CUDA_R_16BF)` needs cc >= 8.0 → `CUBLAS_STATUS_ARCH_MISMATCH` (repro: `gemma4:e4b` on K80 aborted llama-server) |
| 1006 | `1006-cuda-arch-list-from-cmake.patch` | sync | nvcc < 11.5 never defines `__CUDA_ARCH_LIST__`, so `system_info` carried no `CUDA : ARCHS` and ollama disabled GPU/arch filtering (startup WARN, unmatched GPUs not filtered). Emit the list from `CMAKE_CUDA_ARCHITECTURES` in the same 3-digit form |

## Deferred (follow-up commits, same numbering)

- Perf kernels (correctness-neutral, throughput): `mmvq-k80`, `mmvf-k80`,
  `mmid-k80` grouped-expert prefill, `dequantize_block_q4_K_k80`, K80 dense
  prefill in `fattn.cu` (#563 #565 #567 #571 #572 #580 #583 #584 #585).
  Baseline: b12177 compiles and runs on sm_37 without them — just slower.
- Volta WMMA FA kernel port (`fattn-wmma-f16.cu`): 1003 ships the guard
  (falls back to VEC); the WMMA kernel restores Volta FA speed.

## Absorbed upstream / obsolete under the llama-server engine (do NOT port)

- Upstream `llama/compat/002` (Metal unary cast) → landed in b12177 (04088e7c9)
  and deleted from the upstream compat chain. Not in the fork series; the
  `llama/compat/*.patch` glob in the apply loop simply has one fewer file.
- Go num_parallel guard for recurrent/vision arches → removed upstream in
  v0.40.16 (llama-server b12177 allows parallel requests for recurrent and
  vision models, 84dc81db1); the fork took the deletion in the merge.
- BF16 cc >= 8.0 gate for the **kernels** → `fast_bf16_hardware_available()` upstream.
  This does not cover the cuBLAS compute type, where BF16 GEMM also needs cc >= 8.0 —
  that layer is patch 1005.
- VMM granularity alignment → upstream aligns `reserve_size` to
  `vmm_granularity`.
- Patch 0028 CUDA spin scheduling → upstream (SM121 spin).
- Patches 0033/0034 lfm2 → native lfm2/lfm2moe (b11963, still in b12177).
- Patch 0024 device reset + #509 + 04028f51 ghost pruning → the ghost came
  from the Go ggml engine touching unused devices; that engine is gone.
  The new engine still opens a CUDA context (~112 MB on a K80) per device for
  discovery, so `/api/metrics` only flags `ghost` while a model is loaded
  (server/metrics.go); reintroduce the reset only if a real ghost reappears.
- Patches 0016/0019/0021/0022/0029 → embedded-runner/Windows infra, gone
  with the old engine.
- Model architectures (qwen3.5/DeltaNet, gemma4, lfm2) → native in b11963,
  still native in b12177.
  DeltaNet sm_37 verified: `select_gdn_mma_path` declines pre-Ampere
  (`ampere_mma_available` false on cc 370) and falls through to the pure-FP32
  `gated_delta_net_cuda` step kernel (no wmma/ldmatrix/cp.async/bf16) — no
  patch needed. Re-confirm at Phase 5/6 on sm37.

## Known limitations (accepted)

- **1001 host flag across two devices.** `k80_fill_prev` is a plain static
  pointer read/written from `graph_compute`; two K80 dies computing at the
  same time race on it. Benign in practice (one store per launch, the value
  is a context that outlives the call) and inherited from the fork-era patch;
  an atomic would remove the formal UB but is not worth re-generating a
  runtime-verified patch for. Revisit if a TSan-style audit lands here.
- **1005 ignores `GGML_CUDA_CUBLAS_COMPUTE_TYPE` on pre-Volta.** The pin is a
  hardware-safety guard (BF16 GEMM needs cc >= 8.0), so it deliberately wins
  over the debug override. Set `GGML_CUDA_CUBLAS_COMPUTE_TYPE` on Volta+ only.

## Scratch verification (no CUDA 11.4 on the workstation)

```sh
# one-time helper image (nvcc 11.4.3 + gcc 8.5 + cmake 3.31; the real builder
# uses GCC 10 — final gate is the ollama37-builder image, Phase 4):
docker build -t kilo-compilecheck:cuda11.4 -f - . <<EOF
FROM nvidia/cuda:11.4.3-devel-rockylinux8
RUN dnf -y install git make && dnf clean all
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz -o /tmp/c.tgz \
    && tar xzf /tmp/c.tgz -C /opt && ln -s /opt/cmake-3.31.6-linux-x86_64/bin/cmake /usr/local/bin/cmake
EOF

# reproduce the patched tree (what ollama builds):
git clone --depth 1 --branch b12177 https://github.com/ggml-org/llama.cpp work && cd work
git apply ../llama/compat/*.patch            # in numeric order
cmake -DPATCH_DIR=$PWD/../llama/compat-k80 -DPATCH_LABEL=compat-k80 -P ../cmake/apply-git-patches.cmake

# compile-check CUDA 11.4 + sm_37:
docker run --rm -v $PWD:/work -w /work kilo-compilecheck:cuda11.4 sh -c \
  'cmake -B b -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=37 -DGGML_CCACHE=OFF . && cmake --build b --target ggml-cuda -j 8'

# compile-check the CPU preset under GCC 10 (patch 1004). llama/compat side
# sources are wired by ollama's llama/server/CMakeLists.txt, not by the
# patches, so replicate that here (copy llama-ollama-compat*.{cpp,h} into
# src/, target_sources(llama ...) after add_library(llama), and add src/ to
# mtmd's includes) before building target llama-server:
docker run --rm -v $PWD:/work -w /work kilo-compilecheck:gcc10 sh -c \
  'cmake -B bc -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=ON -DGGML_CCACHE=OFF . && cmake --build bc --target llama-server -j 8'
```

Regenerating a patch after editing the scratch tree:
`git add -N <new files> && git diff <base-commit> > ../llama/compat-k80/NNNN-*.patch`

Status: 1000–1003 compile-verified together (b11963 + llama/compat + all four)
under nvcc 11.4.3 / sm_37 — `ggml-cuda` target green. 1004 compile-verified
with the CPU preset (`GGML_CPU_ALL_VARIANTS`, llama-server + llama-quantize)
under GCC 10.3 — the four guarded variants are skipped, all others build.
1005 runtime-verified on a Tesla K80 (driver 470.256.02): `gemma4:e4b` failed
with `CUBLAS_STATUS_ARCH_MISMATCH` before the patch and answers correctly
after, using the rebuilt `libggml-cuda.so` over the shipped image.

Re-audit 2026-11-15 against **b12177** (v0.40.16 sync): the full chain
`llama/compat/001 + 1000..1006` applies cleanly on a fresh b12177 clone —
1001 re-based (the `ggml-cuda.cu` include list gained `batch-get.cuh`),
1000/1002–1006 apply with byte-identical context. `ggml-cuda` compiles green
(nvcc 11.4.3, sm_37, `GGML_CUDA_NCCL=OFF` — the builder ships no NCCL;
b12177's new NCCL allreduce path, ON by default, needs NCCL ≥ 2.17 and breaks
on hosts with an older distro NCCL). 1004 re-audited without rebuild: b12177
adds no new x86 CPU variants (the set still ends at sapphirerapids, guards
intact).

Audit of every cuBLAS call site against pre-Volta (cc 3.7): `mul_mat_cublas`
covered by 1000 + 1005; `conv2d`/`conv3d` never take the cuBLAS path on K80
(`use_blas` requires fast FP16); `ssm-scan`'s `GemmStridedBatchedEx` calls sit
behind upstream's `cc >= TURING` SSD gate, so the pre-Volta build uses the
cuBLAS-free scan path; `out-prod` already uses legacy `cublasSgemm*`.
