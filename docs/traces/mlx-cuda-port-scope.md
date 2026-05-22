# MLX-CUDA port to K80 (sm_37) — build-feasibility scope (#188)

Investigation result for #188. Conclusion: the version gaps are a **rewrite surface, not a
wall** (decision: pursue the port, not the MLX→GGUF conversion pivot). This doc scopes the
rewrite so the port is concrete. Sources: `ml-explore/mlx` @ `e8ebdebe…` (pinned by upstream
Ollama's `MLX_VERSION`), `mlx-c` @ `fba4470…` (`MLX_C_VERSION`).

## The two gaps, and which is hard

### 1. Toolchain version (mechanical rewrite)
- MLX core requires **C++20**: `mlx/CMakeLists.txt:25` `set(CMAKE_CXX_STANDARD 20)`. Its
  headers are shared between host and device `.cu` code.
- `nvcc` accepts `-std=c++20` only from **CUDA 12.0**; our toolchain is **CUDA 11.4.4** (max
  for K80 / driver 470, C++17 ceiling).
- **CUDA 12.0 dropped Kepler (sm_37)** — 11.8 was the last toolkit to support it. So no single
  toolkit gives both C++20 *and* sm_37.
- MLX already ships CUDA-version-gated GEMM files (`gemms/cublas_gemm_batched_12_0.cpp`,
  `…_12_9.cu`) — version-specific variants are an established pattern; we add an 11.4 path.
- **Work:** lower MLX device code C++20 → C++17 where nvcc 11.4 chokes; add an 11.4 GEMM
  variant. Broad but mechanical.

### 2. Hardware features (real kernels to write)
- `steel/mma.cuh` = **tensor-core MMA**; Kepler has **no tensor cores** (Volta sm_70+).
- `cutlass_utils.cuh` uses `cutlass::bfloat16_t` — **bf16 is Ampere sm_80+**.
- CUTLASS 3.x kernels target sm_70+/sm_80+; no Kepler path.
- **But a Kepler-capable path exists:** `gemms/cublas_gemm.cpp` routes through **cuBLAS**,
  which runs on K80. Elementwise/`reduce`/`copy`/`unary` are classic SIMT.

## Strategy: fallback-first (bounds the rewrite)
K80 lacks tensor cores / bf16, so those paths should **fall back fast** to Kepler-capable code
rather than be reimplemented:
1. Route all GEMM through the **cuBLAS** path on Kepler; disable/skip `steel/mma` + CUTLASS.
2. Force fp16/fp32 compute; avoid bf16.
3. Only **write new kernels** where there is *no* fallback. Prime suspect: **quantized matmul**
   (`quantized/qmm/`) — MLX's own quant kernels, likely no cuBLAS fallback → must port to SIMT.
4. `conv/` needs a cuDNN 8.x compatible with CUDA 11.4 (cuDNN-frontend v1.16.0 is fetched).

The open empirical question (the optimism to verify): how many ops have **no** Kepler fallback?
That count *is* the kernel-writing workload. Best measured by building and seeing what fails.

## Size signal
`mlx/backend/cuda`: **197 files, 129 `.cu/.cuh`**, subsystems: `gemms/ steel/ quantized/qmm/
conv/ reduce/ copy/ unary/ binary/ device/`.

## Rewrite areas → sub-issues
1. **#188** — build MLX+CUDA for sm_37 on CUDA 11.4: C++20→17 sweep, 11.4 GEMM variant,
   cuBLAS-only GEMM routing, force fp16/fp32, cuDNN 8.x. First move: a CI build to get the
   *actual* first error and turn this scope into an ordered worklist.
2. **#189** — port `x/mlxrunner` + `x/models` MLX impls (cgo) once the lib links.
3. **#190** — `IsMLX()` safetensors routing into our `sched.go`.
4. **#191** — build/CI integration (MLX_ENGINE wiring, Docker, test-build).
5. **#192** — end-to-end smoke test on K80.

## Note vs llama.cpp
GGML's CUDA is C++17, hand-written SIMT kernels, no CUTLASS — sm_37 there was a flag + small
patches. MLX is C++20 + CUTLASS/tensor-core — bigger, but the fallback-first strategy keeps the
must-write surface to the no-fallback ops (quantized matmul foremost).
