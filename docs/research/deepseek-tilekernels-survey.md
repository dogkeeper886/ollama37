# Survey: DeepSeek TileKernels / TileLang for K80 (sm_37)

**Issue**: [#103](https://github.com/dogkeeper886/issues/103)
**Date**: 2026-04-26
**Status**: Survey complete (revised after code-level investigation) — verdict in §6

> **Revision note.** The first pass of this survey concluded "definitively infeasible" based on upstream version pins (CUDA 13.1+, PyTorch 2.10+, SM90/SM100). That conclusion was wrong as stated: version pins describe what the upstream package supports as-installed, not what TileLang's compiler can emit or what would run on sm_37. After cloning both repos, tracing the codegen, and empirically lowering two TileKernels kernels for `arch=sm_37`, the real picture is: the toolchain can technically emit sm_37-compatible CUDA for the non-tensor-core kernels, but the kernels that emit successfully don't address ollama37's actual bottlenecks. **Verdict moves from "infeasible" to "feasible-but-unjustified."**

## 1. Executive Summary

DeepSeek released [TileKernels](https://github.com/deepseek-ai/TileKernels) on 2026-04-24, a kernel library written in [TileLang](https://github.com/tile-ai/tilelang), a Python-embedded DSL that lowers to CUDA via TVM. TileKernels' README requires SM90/SM100 hardware, CUDA 13.1+, and PyTorch 2.10+; this is what the *prebuilt package* needs to install and run as-is.

The *underlying compiler* is more capable than those pins suggest. TileLang's C++ codegen (`src/target/utils.cc`, `src/transform/lower_ptx_async_copy.cc`) has explicit graceful-fallback paths for older architectures: when `TargetHasAsyncCopy(target)` returns false, the `cp.async` lowering pass returns the function unchanged; when `TargetHasBulkCopy(target)` returns false, TMA injection is skipped; the `T.gemm` template (`src/tl_templates/cuda/gemm.h`) only has implementations for sm_70+ but is only included when `T.gemm` is actually called. Empirically, calling `tilelang.lower(func, target=Target("cuda -arch=sm_37"), enable_device_compile=False)` on TileKernels' `batched_transpose_kernel` and `topk_gate_kernel` produces clean vanilla CUDA using only `__shared__`, `__syncthreads`, vectorized `uint2`/`float2` loads, `#pragma unroll`, and `__shfl_xor_sync` — all supported on sm_37 with CUDA 11.4. Samples in §5.

So *technically* a non-trivial subset of TileKernels could be code-generated for sm_37 ahead-of-time on a non-K80 host and the resulting `.cu` files compiled into ollama37 with `nvcc 11.4 -arch=sm_37`. The honest blockers are:

1. **The kernels that compile aren't the ones we need.** TileKernels targets MoE routing for 256-expert Hopper-class deployments, FP8/FP4 quantization, and DeepSeek-internal model architectures (Engram, Manifold HyperConnection). ollama37's K80 bottlenecks are KV cache size, scalar attention throughput on the `fattn-vec` path, and PCIe Gen3 x16 decode bandwidth — none of which TileKernels addresses.
2. **The kernels that would address ollama37's needs (FP8 quant, fused SwiGLU+cast)** require FP8/FP4 hardware that K80 does not have. Software emulation defeats the purpose.
3. **Integration cost is real.** A separate Python toolchain (TileLang + a modern Python + a host with CUDA) would have to live in CI, generate `.cu` files on each kernel change, and slot them into ollama37's CMake-based GGML build. Significant ongoing maintenance for kernels we don't need.

**Recommendation: do not adopt at this time.** Keep TileLang on the radar as a kernel-authoring DSL for any *future* ollama37 work that targets sm_70+ hardware. Do not import any TileKernels kernel into the K80 path.

## 2. What TileKernels Is

- **Authors**: DeepSeek (Wang, Xu, Cao, Tian, Zhao, Yu, Zhao). License MIT.
- **Stated requirements** (`pyproject.toml` + README): Python 3.10+, PyTorch 2.10+, TileLang 0.1.9+, CUDA 13.1+, SM90/SM100 GPU.
- **Surface model**: Python `@tilelang.jit` decorators wrap PrimFunc-style kernels. PyTorch `torch.autograd.Function` wrappers in `tile_kernels/modeling/` provide trainable layers.

### Kernel categories

Confirmed by listing `tile_kernels/<dir>/` and grepping for primitives used:

| Dir | Files | Uses `T.gemm` | Uses `T.Pipelined` | Uses FP8/FP4 |
|---|---|---|---|---|
| `quant/` | 13 cast + swiglu+cast files | no | no | **yes (all)** |
| `moe/` | 13 routing/gating/scoring files | no | no | no |
| `transpose/` | 1 batched transpose | no | no | no |
| `engram/` | engram gate + grad reduce | no | grad-reduce only | mixed |
| `mhc/` | 5 manifold-HC layer kernels | **norm_fn only** | most | mixed |
| `modeling/` | autograd wrappers | n/a | n/a | n/a |

`T.gemm` only appears in `mhc/norm_fn_kernel.py` (3 sites). `T.Pipelined` is used in mhc/engram fused kernels but is gated by TileLang's async-copy capability check — falls back to synchronous loops when async copy is unavailable.

## 3. TileLang's Code-Gen Behaviour for Older Architectures

This is the section v1 of the survey skipped. The actual capability detection is in `src/target/utils.cc`:

```cpp
bool TargetIsVolta(Target target) { return arch >= 70 && arch < 75; }
bool TargetHasAsyncCopy(Target target) { return TargetIsCuda(target) && arch >= 80; }
bool TargetHasLdmatrix(Target target) { return TargetIsCuda(target) && arch >= 75; }
bool TargetHasBulkCopy(Target target) { return TargetIsCuda(target) && arch >= 90; }
```

For `arch=37` (Kepler), every `TargetIs*` check returns false (lowest checked is Volta) and every `TargetHas*` capability returns false. There is no `TargetIsKepler` because TileLang has no Kepler-specific code path — but this is fine, because the codegen treats sm_37 as a generic CUDA target without modern features.

Critically, the lowering passes **honour these capability flags rather than asserting**:

- `src/transform/lower_ptx_async_copy.cc:721` — `if (!TargetHasAsyncCopy(target)) { return f; }`. Async-copy lowering is skipped; the original synchronous loads remain.
- `src/transform/inject_fence_proxy.cc:541` — TMA proxy fences are only injected when `TargetHasBulkCopy(target)` is true. Otherwise no-op.
- `src/transform/pipeline_planning.cc:833,906,1022` — async pipeline annotations only emitted when async-copy is supported. Otherwise the pipeline lowers to a plain unrolled/sync version.

The `T.gemm` template (`src/tl_templates/cuda/gemm.h`) cascades on `__CUDA_ARCH_LIST__`: sm_120 → sm_100 → sm_90 → sm_89 → sm_80 → sm_70 → empty `#else`. So any kernel that calls `T.gemm` *will* fail to link on sm_37. But this only affects the one `mhc/norm_fn_kernel.py`; no other TileKernels kernel uses `T.gemm`.

## 4. Empirical Verification

To check this isn't theoretical, I installed `tilelang==0.1.9` in a clean venv on a host without CUDA, constructed `Target({"kind": "cuda", "arch": "sm_37", ...})` explicitly, and called `tilelang.lower(..., enable_device_compile=False)` on two TileKernels kernels.

Both lowered successfully. The generated CUDA uses only intrinsics supported on sm_37 with CUDA 11.4.

### 4.1 batched_transpose, sm_37, fp16

```cpp
extern "C" __global__ void __launch_bounds__(256, 1) batched_transpose_kernel_kernel(
    half_t* __restrict__ out, const half_t* __restrict__ x,
    int num_batches, int shape_x, int shape_y, int stride_x) {
  extern __shared__ __align__(1024) half_t out_shared[];
  half_t tmp[16];
  half_t tmp_row[4];
  #pragma unroll
  for (int i_ = 0; i_ < 4; ++i_) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      *(uint2*)(tmp_row + 0) = *(uint2*)(x + ...);
      // ... vectorized transpose into shared memory ...
    }
  }
  __syncthreads();
  // ... store tile back to global memory via uint2 ...
}
```

No `mma.sync`, no `cp.async`, no `ldmatrix`, no TMA, no FP8. Just `extern __shared__`, `__syncthreads()`, vectorized `uint2` loads, and `#pragma unroll`. `half_t` is the project's typedef for `__half`, which Kepler has had since CUDA 7.5. Full output saved as `transpose_sm37.cu` in the survey scratch dir.

### 4.2 topk_gate (num_experts=64, num_topk=8), sm_37, fp32

```cpp
extern "C" __global__ void __launch_bounds__(32, 1) topk_gate_kernel_kernel(
    const float* __restrict__ scores, int64_t* __restrict__ topk_idx, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float scores_fragment[2];
  // ... per-thread top-k via repeated max + warp reduction ...
  amax_fragment[0] = tl::AllReduce<tl::MaxOp, 32, 1, 0>::run(amax_fragment[0]);
  // ...
  idx_reducer[0] = tl::AllReduce<tl::MinOp, 32, 1, 0>::run(idx_reducer[0],
      (&(((int*)buf_dyn_shmem)[0])));
  // ...
}
```

`tl::AllReduce` is a thin template (`src/tl_templates/cuda/reduce.h:146`) that uses `tl::shfl_xor_sync`, which is a wrapper around `__shfl_xor_sync`. `__shfl_xor_sync` requires CUDA 9+ (we have 11.4) and works on sm_30+. K80 (sm_37) supports it.

### 4.3 What we did **not** verify

- We did not actually invoke `nvcc 11.4 -arch=sm_37` on the generated `.cu` files. The host running this survey has no CUDA toolchain. Risk that some emitted typedef or template requires a sm-specific intrinsic we missed. Probability low (the generated code uses only well-known portable intrinsics) but non-zero.
- We did not verify that the kernels produce *correct* results on a K80 — only that the source compiles in principle. A real adoption would need to run them on the CI runner against a reference implementation.

## 5. Per-Category Feasibility (Code-Grounded)

| Category | Compiler can emit for sm_37? | Useful for ollama37 K80? | Net |
|---|---|---|---|
| `quant/` (FP8/FP4/E5M6 cast) | Maybe — depends on whether TileLang's FP8 path emits hardware intrinsics. Likely no useful output without FP8 silicon. | No — K80 has no FP8/FP4. Software emulation defeats the purpose. | Skip |
| `moe/` (gating, routing, scoring, top-k) | **Yes** — verified for `topk_gate`. Other MoE kernels use the same primitive set. | No — llama.cpp upstream MoE works on K80; ollama37 doesn't currently target MoE; useful MoE models won't fit in 24 GB anyway. | Skip |
| `transpose/batched_transpose` | **Yes** — verified empirically. | No — GGML has transpose; this is a bandwidth-bound op where TileLang offers no algorithmic advantage. | Skip |
| `engram/` | Likely yes for non-T.gemm kernels (uses `T.Pipelined` which falls back gracefully). | No — DeepSeek-research-specific layers, no model in ollama37's lineup uses them. | Skip |
| `mhc/norm_fn` | **No** — uses `T.gemm`, which has no sm_37 path in `gemm.h`. | No — mHC is research-specific. | Skip |
| `mhc/` (other) | Likely yes. | No — same as Engram. | Skip |
| `modeling/` (autograd wrappers) | n/a — pure PyTorch. | No — ollama37 has no PyTorch dependency. | Skip |

## 6. Verdict

**Do not adopt TileKernels into ollama37 at this time.**

This is not because it's technically impossible — it isn't. The toolchain can emit clean sm_37 CUDA for the non-tensor-core, non-FP8 kernels. It's because the intersection of {kernels TileLang can emit for sm_37} and {kernels useful to ollama37 K80} is empty:

- The kernels we *can* generate (MoE routing, batched transpose, parts of engram/mHC) target Hopper-class workloads that don't bottleneck K80 inference.
- The kernels that *would* address K80 bottlenecks (FP8 quantization for KV/weights) need hardware K80 doesn't have.
- The dense-LLM operations that *do* bottleneck K80 (FlashAttention-style scalar attention, KV cache reads, GGML matmul kernels) aren't in TileKernels at all — TileKernels assumes those live elsewhere (FlashAttention, CUTLASS, vendor kernels), and on Hopper those use tensor cores, which K80 lacks.

### What would change this verdict

1. ollama37 expands to support sm_70+ as a build target (V100-and-newer dev cards). At that point TileLang as a kernel-authoring DSL becomes worth a separate evaluation, possibly without needing TileKernels at all.
2. A future TileKernels release adds genuinely portable algorithmic kernels (KV quant, attention) that don't depend on FP8/FP4 or tensor cores.
3. ollama37 takes on serving MoE models on a multi-K80 cluster where the routing kernels would actually move the needle — currently not on the roadmap.

Until one of those happens, the integration cost (separate Python codegen toolchain in CI, generated-source check-in or build-time generation, ongoing version compatibility with TileLang) exceeds the benefit (none of the generated kernels solve a current bottleneck).

## 7. References

- TileKernels repo — https://github.com/deepseek-ai/TileKernels
- TileLang repo — https://github.com/tile-ai/tilelang
- TileKernels `pyproject.toml` (PyTorch 2.10, TileLang 0.1.9) — https://github.com/deepseek-ai/TileKernels/blob/main/pyproject.toml
- TileKernels README (CUDA 13.1, SM90/SM100) — https://github.com/deepseek-ai/TileKernels#requirements
- TileLang `src/target/utils.cc` (capability flags, lowest arch sm_70) — https://github.com/tile-ai/tilelang/blob/main/src/target/utils.cc
- TileLang `src/transform/lower_ptx_async_copy.cc` (graceful fallback) — https://github.com/tile-ai/tilelang/blob/main/src/transform/lower_ptx_async_copy.cc
- TileLang `src/tl_templates/cuda/gemm.h` (sm_70+ only) — https://github.com/tile-ai/tilelang/blob/main/src/tl_templates/cuda/gemm.h
- TileLang README "Tested Devices" — https://github.com/tile-ai/tilelang#tested-devices
- Original article (zh): "别卷CUDA了！DeepSeek开源TileKernels，用Python写出榨干H800的算力内核？"
