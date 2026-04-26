# Survey: DeepSeek TileKernels / TileLang for K80 (sm_37)

**Issue**: [#103](https://github.com/dogkeeper886/ollama37/issues/103)
**Date**: 2026-04-26
**Status**: Survey complete — verdict in §5

## 1. Executive Summary

DeepSeek released [TileKernels](https://github.com/deepseek-ai/TileKernels) on 2026-04-24, a Python-authored GPU kernel library built on the [TileLang](https://github.com/tile-ai/tilelang) DSL. The project is real and is in DeepSeek's production training/inference path. The pitch — "write H800-class kernels in Python instead of CUDA C++" — is technically accurate for SM90/SM100 hardware.

For Tesla K80 (sm_37) it is a hard no on every axis. Three independent stack-level blockers each rule it out by themselves: (1) TileKernels requires CUDA 13.1+, but CUDA 12 already dropped Kepler; (2) it requires PyTorch 2.10+, which dropped Kepler in the 2.x line; (3) the kernels themselves require SM90 or SM100 hardware features (FP8/FP4 native types, Hopper TMA, transformer-engine MMA) that don't exist on a 2014-era Kepler card. The TileLang DSL itself is more interesting in principle, but its lowest-tested NVIDIA architecture is sm_70 (V100), its `T.gemm` primitive dispatches to CUTLASS CuTe (which assumes tensor cores), and validating it on sm_37 would mean maintaining a Kepler-only TileLang fork — disproportionate effort for a single downstream project that already has working hand-written CUDA kernels in GGML.

**Recommendation: do not port. Do not open a prototype issue.** The remaining sections document the constraints in detail so this question doesn't get re-asked.

## 2. What TileKernels Is

- **Authors**: DeepSeek (Xiangwen Wang, Chenhao Xu, Huanqi Cao, Rui Tian, Weilin Zhao, Kuai Yu, Chenggang Zhao)
- **License**: MIT
- **Stated purpose**: "Optimized GPU kernels for LLM operations, built with TileLang… most kernels approach the limit of hardware performance regarding compute intensity and memory bandwidth."
- **Package**: `pip install tile-kernels`. PyTorch front-end via `torch.autograd.Function` wrappers in `tile_kernels/modeling/`.
- **Hard requirements** (`pyproject.toml` + README):
  - Python ≥ 3.10
  - PyTorch ≥ 2.10
  - TileLang ≥ 0.1.9
  - CUDA Toolkit ≥ 13.1
  - GPU arch: SM90 (Hopper) or SM100 (Blackwell)

### Kernel categories present

Confirmed by listing `tile_kernels/<dir>/`:

| Dir | Files (representative) | Purpose |
|---|---|---|
| `quant/` | `per_token_cast_kernel.py`, `per_block_cast_kernel.py`, `per_channel_cast_kernel.py`, `cast_back_e5m6_kernel.py`, `swiglu_forward_and_per_token_cast_kernel.py` | FP8/FP4/E5M6 cast and SwiGLU+cast fusions |
| `moe/` | `topk_gate_kernel.py`, `top2_sum_gate_kernel.py`, `expand_to_fused_kernel.py`, `reduce_fused_kernel.py`, `normalize_weight_kernel.py`, `scoring.py` | MoE gating, expert routing, weight normalization |
| `transpose/` | `batched_transpose_kernel.py` | Batched transpose |
| `engram/` | (engram-specific layers) | DeepSeek "Engram" architecture pieces |
| `mhc/` | (Manifold HyperConnection layers) | DeepSeek "mHC" architecture pieces |
| `modeling/` | autograd Function wrappers | High-level PyTorch-facing layers (engram, mHC) |

## 3. Hard Blockers for sm_37 / K80

These are *each, individually*, sufficient to rule out adoption. They are not stylistic mismatches — they are stack-level "won't compile, won't load, won't run" constraints.

### 3.1 CUDA 13.1+ requirement

TileKernels' README requires CUDA Toolkit ≥ 13.1. NVIDIA dropped Kepler (sm_3x) support in CUDA 12; CUDA 11.x was the last toolkit line that could compile for sm_37. ollama37 is pinned to CUDA 11.4.4 for exactly this reason. There is no path to a CUDA 13 toolchain that emits sm_37 device code.

### 3.2 PyTorch 2.10+ requirement

TileKernels' `pyproject.toml` declares `torch>=2.10`. PyTorch's official wheels in the 2.x line dropped Kepler — recent prebuilt PyTorch only ships kernels for sm_50+, and many ops require sm_70+. Even building PyTorch 2.10 from source against CUDA 11 + sm_37 would require multiple patches and is not a tested configuration. The TileKernels Python wrappers (`tile_kernels/modeling/`) wouldn't load.

### 3.3 SM90/SM100 hardware features

The actual kernels use Hopper-class hardware:
- **FP8 / FP4 / E5M6 cast** kernels assume native FP8 hardware types (introduced Hopper). On K80, "FP8" only exists as software emulation, which defeats the bandwidth-saving point of the operation.
- **TMA (Tensor Memory Accelerator)** — Hopper-only async bulk copy. The one MoE kernel inspected (`topk_gate_kernel.py`) explicitly opts out with `disable_tma=True`, which suggests TileLang has a non-TMA path, but the broader TileKernels suite is designed around having TMA available.
- **Transformer-engine MMA / wgmma** — Hopper warp-group MMA. K80 has no tensor cores at all (Kepler MMA ops don't exist).

### 3.4 TileLang's lowest validated arch is sm_70

TileLang's README lists tested NVIDIA devices: H100, A100, V100, RTX 4090, RTX 3090, RTX A6000. Lowest is sm_70 (V100). No Pascal (sm_60), Maxwell (sm_50), or Kepler (sm_37). TileLang's headline `T.gemm` primitive states "we dispatch to the cute/hip on Nvidia/AMD GPUs" — CUTLASS CuTe paths assume tensor cores. On a tensor-core-free architecture the entire performance model breaks; you'd be using a Python DSL to emit slow scalar code.

### 3.5 ollama37 has no PyTorch dependency

ollama37's runtime is Go + C++/CUDA via GGML. It has no Python or PyTorch in the inference path. TileKernels' `torch.autograd.Function` wrappers are unusable as-is; you'd be using the raw TileLang-generated CUDA only, which forfeits the project's framing as a Python-friendly kernel library.

## 4. Per-Category Portability

For completeness, the question "could we re-implement the *algorithms* in CUDA for sm_37" is addressed per category:

| Category | Algorithm portable? | Useful for ollama37? | Verdict |
|---|---|---|---|
| **Quant (FP8/FP4/E5M6)** | No — hardware data types are the entire point | No — K80 has no FP8/FP4 silicon | Skip |
| **MoE gating / routing** | Yes — top-k / scoring / index work is plain integer + fp32 | No — llama.cpp upstream already has working MoE (Mixtral, etc.) and ollama37 inherits it | Skip |
| **SwiGLU fusion** | Partial — base SwiGLU is portable, but the TileKernels variants are SwiGLU+FP8-cast fusions | No — GGML already has fp16 SwiGLU; the FP8-fused value doesn't apply | Skip |
| **Batched transpose** | Yes — generic memory-bound op | No — GGML has transpose | Skip |
| **Engram, Manifold HyperConnection** | Partial — these are model-architecture pieces specific to DeepSeek's research models | No — none of ollama37's target models use them | Skip |
| **Modeling (autograd wrappers)** | N/A — PyTorch-only | No — ollama37 has no PyTorch | Skip |

There is no category where "port the idea by hand to CUDA 11 / sm_37" produces a result that is both achievable and not already covered by GGML.

## 5. Verdict

**Do not port. Do not open a prototype issue.**

TileKernels is a sound, well-engineered library — for the hardware it targets. Every hard blocker in §3 is independent and stack-level, not stylistic; fixing any one of them would still leave the other two. Even setting all three aside and treating this purely as an algorithmic study, §4 finds no kernel where the algorithm is novel-to-ollama37 *and* the K80-feasible re-implementation would beat what GGML already has.

The TileLang DSL itself is interesting as a kernel-authoring approach for new hardware, and worth keeping on the radar for any future ollama37 work that runs on sm_70+ GPUs (V100 and newer). It is not appropriate for the sm_37 K80 target.

### What would change this verdict

For completeness, the conditions under which this should be re-opened:

1. NVIDIA back-ports Kepler to a future CUDA toolkit (will not happen).
2. The TileLang project adds a tested sm_37 backend path with non-tensor-core code generation (not on their roadmap; unlikely without external contribution).
3. ollama37 expands to support sm_70+ GPUs as a build target — at which point TileLang becomes worth a separate evaluation, but not via TileKernels' Hopper-specific kernels.

## 6. References

- TileKernels repo — https://github.com/deepseek-ai/TileKernels
- TileLang repo — https://github.com/tile-ai/tilelang
- TileKernels `pyproject.toml` (PyTorch 2.10, TileLang 0.1.9 requirements) — https://github.com/deepseek-ai/TileKernels/blob/main/pyproject.toml
- TileKernels README (CUDA 13.1, SM90/SM100 requirements) — https://github.com/deepseek-ai/TileKernels#requirements
- TileLang README "Tested Devices" section (lowest NVIDIA arch sm_70 / V100) — https://github.com/tile-ai/tilelang#tested-devices
- TileLang CuTeDSL backend PR (CUTLASS dispatch for `T.gemm`) — https://github.com/tile-ai/tilelang/pull/1421
- Original article (zh): "别卷CUDA了！DeepSeek开源TileKernels，用Python写出榨干H800的算力内核？"
