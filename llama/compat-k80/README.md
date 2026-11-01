# llama/compat-k80 — fork-owned K80 patch series

Ordered `*.patch` set restoring Tesla K80 (compute 3.7) support on top of the
pinned llama.cpp (`LLAMA_CPP_VERSION`, currently `b11963`). Applied by
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
| 1001 | `1001-k80-clock-pinning.patch` | #587 | Autoboost clock filler for multi-die K80 decode (`clock-k80.cu`) |
| 1002 | `1002-kepler-cuda-graphs.patch` | #573 | Keep CUDA graphs on Kepler (upstream gates them to Ampere+) |
| 1003 | `1003-fa-guard-cuda11.patch` | #385 | Disable MMA_F16 FA under CUDA 11.4 for cc >= 7.5 (NaN); Volta keeps a working path |

## Deferred (follow-up commits, same numbering)

- Perf kernels (correctness-neutral, throughput): `mmvq-k80`, `mmvf-k80`,
  `mmid-k80` grouped-expert prefill, `dequantize_block_q4_K_k80`, K80 dense
  prefill in `fattn.cu` (#563 #565 #567 #571 #572 #580 #583 #584 #585).
  Baseline: b11963 compiles and runs on sm_37 without them — just slower.
- Volta WMMA FA kernel port (`fattn-wmma-f16.cu`): 1003 ships the guard
  (falls back to VEC); the WMMA kernel restores Volta FA speed.

## Absorbed by b11963 / obsolete under the llama-server engine (do NOT port)

- BF16 cc >= 8.0 gate → `fast_bf16_hardware_available()` upstream.
- VMM granularity alignment → upstream aligns `reserve_size` to
  `vmm_granularity`.
- Patch 0028 CUDA spin scheduling → upstream (SM121 spin).
- Patches 0033/0034 lfm2 → b11963 native lfm2/lfm2moe.
- Patch 0024 device reset + #509 + 04028f51 ghost pruning → the ghost came
  from the Go ggml engine touching unused devices; that engine is gone.
  Re-verify via `/api/metrics` `ghost` flag on sm37 (Phase 5); reintroduce
  here only if ghosts reappear.
- Patches 0016/0019/0021/0022/0029 → embedded-runner/Windows infra, gone
  with the old engine.
- Model architectures (qwen3.5/DeltaNet, gemma4, lfm2) → native in b11963.
  Verify `gated_delta_net` non-MMA path covers sm_37 at Phase 5/6
  (`ggml_cuda_gdn_mma_launch` must decline pre-Volta); port the fork's
  `gated-delta.cu` step kernel only if it does not.

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
git clone --depth 1 --branch b11963 https://github.com/ggml-org/llama.cpp work && cd work
git apply ../llama/compat/*.patch            # in numeric order
cmake -DPATCH_DIR=$PWD/../llama/compat-k80 -DPATCH_LABEL=compat-k80 -P ../cmake/apply-git-patches.cmake

# compile-check CUDA 11.4 + sm_37:
docker run --rm -v $PWD:/work -w /work kilo-compilecheck:cuda11.4 sh -c \
  'cmake -B b -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=37 -DGGML_CCACHE=OFF . && cmake --build b --target ggml-cuda -j 8'
```

Regenerating a patch after editing the scratch tree:
`git add -N <new files> && git diff <base-commit> > ../llama/compat-k80/NNNN-*.patch`
