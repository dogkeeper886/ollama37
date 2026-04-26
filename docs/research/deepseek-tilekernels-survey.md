# Survey: DeepSeek TileKernels / TileLang for K80 (sm_37)

**Issue**: [#103](https://github.com/dogkeeper886/ollama37/issues/103)
**Date**: 2026-04-26 (revised twice)
**Status**: Survey complete — verdict in §6

> **Framing note (v3).** The first two passes of this survey leaned too hard on upstream version pins ("requires CUDA 13.1+", "PyTorch 2.10+") as if those were verdicts. They are not. ollama37 *exists* because we routinely port code that upstream has stopped supporting on Kepler — version pins describe upstream's installer, not our porting capacity. The questions this survey actually has to answer are:
>
> 1. **Hardware ceiling** — what does K80 silicon physically lack that an algorithm depends on? (Real, immovable.)
> 2. **Logic portability** — for each kernel, can the *idea* be lifted into ollama37's GGML/CUDA code, regardless of upstream's packaging?
> 3. **K80 benefit** — would the ported logic actually buy us anything?
>
> Software/version pins are noted but never used as the verdict.

## 1. Executive Summary

DeepSeek released [TileKernels](https://github.com/deepseek-ai/TileKernels) on 2026-04-24, a kernel library written in [TileLang](https://github.com/tile-ai/tilelang), a Python-embedded DSL that lowers to CUDA via TVM.

After cloning both repos, tracing the TileLang codegen, empirically lowering two TileKernels kernels for `arch=sm_37` (both produced clean vanilla CUDA), and auditing the algorithmic content of every kernel category against K80's hardware ceiling and ollama37's existing implementations, the picture is:

- **TileLang's compiler can emit sm_37 CUDA** for kernels that don't use tensor cores or modern memory accelerators. Verified empirically. The version pins on the prebuilt package are upstream policy, not a real blocker for our use case (we'd be running TileLang on a non-K80 dev host as an ahead-of-time codegen tool, then compiling the generated `.cu` files with our existing nvcc 11.4 → sm_37 path).
- **The hardware ceiling is the real story.** K80 lacks tensor cores (Volta+), FP8/FP4 silicon (Hopper/Blackwell), `cp.async` (Ampere+), TMA (Hopper+), `ldmatrix` (Turing+), and native bfloat16 (Ampere+). Any algorithm whose benefit is rooted in those features cannot be preserved on K80 — software emulation defeats the purpose.
- **Logic portability is high for ~half of the kernels** (MoE routing, batched transpose, parts of engram/mHC). The algorithms are pure indexing + memory ops + simple fp32 arithmetic.
- **K80 benefit is low to zero for almost all of them.** Most TileKernels kernels target Hopper-class workloads (MoE routing for 256-expert models, FP8 cast, DeepSeek-internal architectures) that ollama37's K80 path does not run. The *one* possible exception is the transpose kernel's bank-conflict-avoiding shared-mem layout, which is a generic technique GGML's current `cpy.cu`-based transpose doesn't appear to apply.

**Recommendation: do not import any TileKernels kernel into the K80 path now. There is one optional follow-up worth considering: study TileLang's transpose swizzle pattern as a reference if/when GGML's transpose shows up as a profiling hotspot.** Sections §3-§5 give the per-kernel analysis behind this.

## 2. K80 Hardware Ceiling vs. Software Pins

Every "blocker" found in the upstream packaging falls into one of two buckets. Only the red ones are real ceilings.

| Constraint | Type | Notes |
|---|---|---|
| No tensor cores (no `mma.sync`, no `wmma`, no `wgmma`) | 🔴 Hardware | Volta+ feature. Kepler has no equivalent silicon. Any MMA-based GEMM falls back to scalar. |
| No FP8 / FP4 / E5M6 datapath | 🔴 Hardware | Hopper+ (FP8) / Blackwell (FP4). Software emulation defeats the bandwidth point of the operation. |
| No native bfloat16 arithmetic | 🔴 Hardware | Ampere+. K80 has fp16 *storage* (since CUDA 7.5) but not bf16. |
| No TMA / bulk copy | 🔴 Hardware | Hopper-only async memory accelerator. Not implementable in software at the same speed. |
| No `cp.async` (async copy) | 🟡 Hardware-rooted | Ampere+. Falls back gracefully to sync copies in TileLang — not a blocker, just slower. |
| No `ldmatrix` / `stmatrix` | 🔴 Hardware | Turing+ / Hopper+ instructions. No equivalent on Kepler. |
| TileKernels package wants CUDA 13.1+ | 🟢 Software pin | We don't run the package; we'd run TileLang as an offline codegen tool on a different host. Irrelevant to porting. |
| TileKernels package wants PyTorch 2.10+ | 🟢 Software pin | Same — only matters if running their autograd wrappers, which we wouldn't. |
| TileKernels' `modeling/` autograd layer | 🟢 Software | Architectural choice, not a hardware constraint. We'd ignore it and use the raw kernels. |
| TileLang's `T.gemm` template has no sm_37 path | 🟢 Software (rooted in 🔴) | Authors didn't write a non-MMA implementation. They could, but the result would be a triple-nested-loop scalar GEMM on K80 — no faster than what GGML already has, because the underlying ceiling is "no tensor cores." |

So the *real* question is: what algorithms in TileKernels avoid the red items, and are any of them worth porting?

## 3. TileLang Codegen for sm_37 (Verified Empirically)

To make sure the porting picture isn't theoretical, I installed `tilelang==0.1.9` on a host without CUDA, constructed `Target('cuda -arch=sm_37 ...')`, and called `tilelang.lower(..., enable_device_compile=False)` on two TileKernels kernels.

Both lowered successfully. The generated CUDA uses only intrinsics that work on sm_37 + CUDA 11.4 (`__shared__`, `__syncthreads`, vectorized `uint2/float2` loads, `#pragma unroll`, `__shfl_xor_sync`).

Two key code paths in TileLang's C++ codegen support this:

- `src/transform/lower_ptx_async_copy.cc:721` — `if (!TargetHasAsyncCopy(target)) { return f; }`. Async-copy injection is skipped on pre-Ampere; sync copies remain.
- `src/transform/inject_fence_proxy.cc:541` and `pipeline_planning.cc` — TMA fence/pipeline annotations are gated behind `TargetHasBulkCopy`. Skipped on pre-Hopper.

The capability checks (`src/target/utils.cc`) are runtime predicates, not assertions — sm_37 simply gets a "no async, no TMA, no ldmatrix" downgrade rather than rejection. Sample emitted source:

```cpp
// batched_transpose, lowered for arch=sm_37, fp16
extern "C" __global__ void __launch_bounds__(256, 1) batched_transpose_kernel_kernel(
    half_t* __restrict__ out, const half_t* __restrict__ x, int num_batches, ...) {
  extern __shared__ __align__(1024) half_t out_shared[];
  half_t tmp[16]; half_t tmp_row[4];
  #pragma unroll
  for (int i_ = 0; i_ < 4; ++i_) { ...
    *(uint2*)(tmp_row + 0) = *(uint2*)(x + ...);  // vectorized load
    ...
  }
  __syncthreads();
  // ... store back via uint2 ...
}
```

Caveat: I did not run `nvcc 11.4 -arch=sm_37` on the generated source. The host running this survey has no CUDA. Risk that some emitted typedef or template needs a tweak is low (the intrinsics used are all Kepler-supported in CUDA 11) but non-zero. A real adoption would need to compile on the K80 CI runner.

## 4. Per-Kernel Logic Portability

This is the table the v1 and v2 reports should have led with. For each kernel category, what's the algorithmic idea, can we lift it into our code regardless of upstream packaging, and would it actually help K80?

| Kernel | Algorithmic idea | Logic portable to K80? | K80 benefit if ported? |
|---|---|---|---|
| `transpose/batched_transpose` | Tiled transpose with `+ block_k` shared-mem padding to avoid bank conflicts; 4-element vectorized loads/stores | ✅ Yes — pure CUDA, no special instructions | ⚠️ Maybe — GGML transposes via `cpy.cu` stride manipulation, doesn't appear to use tiled+padded shared memory. Worth measuring *if* transpose ever shows up as a profiling hotspot (it usually doesn't). |
| `moe/topk_gate`, `top2_sum_gate` | Top-k expert selection via repeated max + warp reduction | ✅ Yes — integer indexing + fp32 + `__shfl_xor_sync` (works on sm_37) | ❌ No — ollama37 doesn't currently target MoE on K80. K80's 2×12 GB can't hold most useful MoE models anyway. |
| `moe/expand_to_fused`, `reduce_fused`, `normalize_weight`, etc. | Token-to-expert dispatch + weighted reduction | ✅ Yes — same primitive set | ❌ No — same reason. |
| `quant/*` (FP8/FP4/E5M6 cast and SwiGLU+cast fusions) | Per-token / per-block / per-channel quantization with fused activation | ❌ No — the *benefit* (FP8 bandwidth) is hardware-rooted. The *idea* of per-token scaling is portable to int8/int4, but GGML's Q8_0 / Q4_0 already implement per-block scaling and the per-token variant is a research direction, not a clear win. | ❌ No — even a portable per-token int8 scheme would need a full re-implementation independent of TileKernels; nothing to *lift* directly. |
| `engram/engram_grad_w_reduce` and gating | DeepSeek's "Engram" architecture (research-specific layers + gradient reductions) | ⚠️ Partial — depends on whether `T.Pipelined` falls back cleanly | ❌ No — no model in ollama37's lineup uses Engram. Forward-only inference doesn't need the gradient-reduce path at all. |
| `mhc/post`, `pre_apply_mix`, `pre_big_fuse` | Manifold HyperConnection layer ops (DeepSeek research architecture) | ⚠️ Partial — uses `T.Pipelined` but no `T.gemm`; should fall back | ❌ No — no model uses mHC. |
| `mhc/norm_fn` | Fused RMSNorm + GEMM + grad reduce | ❌ No (uses `T.gemm` which has no sm_37 emit path) | ❌ No — also research-architecture-specific. The idea of RMSNorm+GEMM fusion is interesting in general but would need a custom CUDA kernel; not a "port the logic" job. |
| `modeling/*` (autograd wrappers) | PyTorch-facing trainable layer composition | ❌ N/A (not a kernel) | ❌ N/A — ollama37 has no PyTorch and is inference-only. |

## 5. What ollama37's K80 Path Actually Needs

For comparison, the things that are slow on K80 today, and whether TileKernels has a relevant kernel:

| K80 bottleneck | TileKernels has a relevant kernel? |
|---|---|
| KV cache size (VRAM-bound long context) | ❌ No — closest is FP8 quant which K80 can't run; the algorithmic idea (per-token scaling) is generic, not TileKernels-specific |
| Scalar attention throughput (`fattn-vec`, no tensor cores) | ❌ No — TileKernels has no attention kernel; assumes FlashAttention/CUTLASS lives elsewhere |
| PCIe Gen3 x16 decode bandwidth | ❌ No — kernel-level work doesn't help |
| Matmul on K80 (no tensor cores) | ❌ No — `T.gemm` requires tensor cores; GGML's existing scalar/Q-quantized matmul is what we have |
| MoE routing speed | ✅ Yes — but ollama37 doesn't currently target MoE on K80 |

The intersection of {portable from TileKernels} and {addresses K80 bottleneck} contains at most one item, and it's a "maybe" (the transpose swizzle, if transpose ever becomes a hotspot).

## 6. Verdict

**Don't import any TileKernels kernel into the K80 path now.**

The reasoning is grounded in §2 (hardware ceiling) and §4 (per-kernel logic portability + K80 benefit), not in upstream version pins.

- **What's blocked at the hardware level (immovable):** anything that depends on tensor cores, FP8/FP4, native bf16, TMA, ldmatrix, async copy. That covers the entire `quant/` directory, the `mhc/norm_fn` GEMM fusion, and any future TileKernels kernel that adds wgmma / TMA paths.
- **What's portable in principle (algorithm lifts cleanly):** MoE routing, batched transpose, most engram/mHC layer kernels.
- **What's worth porting now:** essentially nothing. The portable kernels target workloads ollama37's K80 path doesn't run (MoE, DeepSeek-internal architectures), and the transpose tiling trick is a known generic technique whose value to ollama37 depends on transpose being a hotspot — which it isn't, in the profiles we have.

### Optional follow-up (not blocking)

If transpose ever appears in a top-N profile:
- Read TileLang's emitted `batched_transpose` source (~30 lines of plain CUDA) as a reference for tiled+padded shared-memory layout.
- Don't import TileKernels as a dependency. Just lift the swizzle idea into a new GGML CUDA kernel.

### Conditions to fully revisit

1. ollama37 expands to support sm_70+ as a target — TileLang as a kernel-authoring DSL becomes worth a fresh evaluation (without TileKernels).
2. ollama37 takes on multi-K80 MoE serving — the routing kernels become relevant.
3. TileKernels adds a kernel that addresses a current K80 bottleneck (KV quant, attention) without depending on tensor cores or FP8 silicon.

## 7. References

- TileKernels repo — https://github.com/deepseek-ai/TileKernels
- TileLang repo — https://github.com/tile-ai/tilelang
- TileLang `src/target/utils.cc` (capability flags) — https://github.com/tile-ai/tilelang/blob/main/src/target/utils.cc
- TileLang `src/transform/lower_ptx_async_copy.cc` (graceful fallback at line 721) — https://github.com/tile-ai/tilelang/blob/main/src/transform/lower_ptx_async_copy.cc
- TileLang `src/tl_templates/cuda/gemm.h` (sm_70+ only) — https://github.com/tile-ai/tilelang/blob/main/src/tl_templates/cuda/gemm.h
- Original article (zh): "别卷CUDA了！DeepSeek开源TileKernels，用Python写出榨干H800的算力内核？"
