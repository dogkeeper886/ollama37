# Trace: qwen3-vl CUDA "invalid configuration argument" at batched cublas

**Issue**: #138 — TC-MODELS-011 (qwen3-vl:8b) and TC-MODELS-012 (qwen3-vl:30b) crash on load
**Date**: 2026-05-12
**Run**: [25308846241](https://github.com/dogkeeper886/ollama37/actions/runs/25308846241)

## Symptom

Both qwen3-vl tests die during worst-case graph reservation:

```
CUDA error: invalid configuration argument
  current device: 0, in function ggml_cuda_mul_mat_batched_cublas_impl
    at ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:2182
  cudaGetLastError()
llama runner terminated, error="exit status 2"
```

`do load request: Post "http://127.0.0.1:XXXX/load": EOF` surfaces in the API client.

## Crash site (verified from stack)

`ggml-cuda.cu:2182` is the `CUDA_CHECK(cudaGetLastError())` immediately after a
`k_compute_batched_ptrs` kernel launch on the broadcasting / non-contiguous
branch of `ggml_cuda_mul_mat_batched_cublas_impl`:

```c
// ggml-cuda.cu:2161-2182
} else {
    // use cublasGemmBatchedEx
    const int64_t ne23 = ne12*ne13;

    ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
    ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

    size_t src1_stride_size = sizeof(cuda_t);

    dim3 block_dims(ne13, ne12);
    k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(...);
    CUDA_CHECK(cudaGetLastError());   // ← line 2182
```

The launch packs **all batch indices into a single block**: `gridDim = (1,1,1)`,
`blockDim = (ne13, ne12, 1)`. K80 (CC 3.7) caps **threads per block at 1024**.
When `ne12 * ne13 > 1024`, `cudaLaunchKernel` returns `cudaErrorInvalidConfiguration`
and the next `cudaGetLastError()` surfaces it.

The kernel itself supports a 2D grid+block (it indexes with
`blockIdx.x * blockDim.x + threadIdx.x` and bounds-checks `i12 < ne12`,
`i13 < ne13` — see `k_compute_batched_ptrs` at ggml-cuda.cu:1933-1938). Only
the launch config under-uses that capability — fixable, but out of scope for
this trace.

## Dispatch flow

```
runner.allocModel()                                     # runner.go:1208
  └── reserveWorstCaseGraph(multimodal=true)            # runner.go:1208
        ├── maxPixels = 2048 (s.flashAttention=true)    # runner.go:1055-1062  ← the regression
        ├── synth image: 2048×2048 grayscale            # runner.go:1063-1065
        ├── multimodalProcessor.EncodeMultimodal(img)   # qwen3vl/imageprocessor.go:85
        │     └── SmartResize → grid {H=W=102, T=1}     # ~10,404 raw patches at 2048×2048
        ├── batch.Multimodal = ...                      # runner.go:1108-1110
        ├── model.Forward(ctx, batch)                   # runner.go:1126
        │     └── VisionModel.Forward(pixelValues,grid) # qwen3vl/model_vision.go:221
        │           ├── PatchEmbedding (Conv3D)
        │           ├── PositionEmbedding
        │           ├── 27 × VisionEncoderLayer
        │           │     └── VisionAttention.Forward
        │           │           └── nn.Attention(Q,K,V, cache=nil)   # ml/nn/attention.go:24
        │           │                 └── (cache==nil → manual path) # attention.go:84
        │           │                       ├── K.MulmatFullPrec(Q)
        │           │                       └── V.Mulmat(KQ)
        │           ├── PatchMerger
        │           └── DeepstackMerger (×N indexes)
        └── ctx.Reserve()                               # ggml.go:855
              └── ggml_backend_sched_reserve()
                    └── while executing the graph, ggml_cuda_mul_mat dispatches
                        to ggml_cuda_mul_mat_batched_cublas for any matmul
                        where src1->ne[2]*src1->ne[3] > 1                 # ggml-cuda.cu:2303-2306
                          → ggml_cuda_mul_mat_batched_cublas_impl
                            → broadcast/non-contig branch                  # ggml-cuda.cu:2161
                              → k_compute_batched_ptrs launch              # ggml-cuda.cu:2171  ← crashes
```

## What changed (the real regression chain)

| Change | Effect |
|---|---|
| `c379fd96` (Apr 27) — Productize K80 FA (#117) | FA is now actually enabled on K80 instead of being warned-off |
| `5d09f609` (May 3)  — K80 perf defaults (#137)  | docker-compose sets `OLLAMA_FLASH_ATTENTION=1` by default |
| Net effect on `s.flashAttention` | was `false` pre-Apr-27 → now `true` |

Effect at `runner.go:1055-1062`:

```go
maxPixels := int(envconfig.VisionMaxPixels())
if maxPixels == 0 {
    if s.flashAttention {
        maxPixels = 2048      // ← active path after #117 + #137
    } else {
        maxPixels = 512       // ← previous path; #50cba688 lowered this for K80
    }
}
img := image.NewGray(image.Rect(0, 0, maxPixels, maxPixels))
```

The 2048-vs-512 branch chooses image resolution for the **worst-case graph
reservation only** — it has nothing to do with actual inference image sizes.
The vision-tower forward at 2048×2048 produces enough patches that some
batched-cublas matmul in the graph hits `ne12 * ne13 > 1024` and the kernel
launch fails.

### Why not caught on main

No Models-suite full run has been recorded on `main` since #117 and #137
landed (last full main run was `24251137369` on 2026-04-10, before either
change). The breakage was first surfaced on the `issue-138-...` branch
because that's the first branch to run a full Models suite on top of the
new defaults.

The branch's three commits only touch `llm/server.go` (layout planner) —
none of them are upstream of the vision-tower or CUDA dispatch path.
Initial bisection-by-CI-run misattributed the crash to `e20ce9c9`; that was
a false positive caused by the prior runs being scoped to TC-MODELS-010
only. Not a regression from this branch.

## Pre-regression evidence

Run `24251137369` on `main` (2026-04-10, before #117) was the last full
Models suite — qwen3-vl:8b passed:

```
"flash attention enabled but not supported by gpu"
FlashAttention:false                  ← FA warned-off ⇒ maxPixels=512
Per-GPU memory: 7608 MiB              ← model on 1 GPU, no crash
```

In the failing 2026-05-04 run, same model, same K80 hardware:

```
"enabling flash attention"
FlashAttention:true                   ← FA actually on ⇒ maxPixels=2048
CUDA error: invalid configuration argument
```

## Which specific tensor overflows

Not yet pinpointed from logs alone — the crash is `cudaGetLastError()` after
the launch, with no GGML node name in the back-trace. Hypothesis space:

1. **Vision attention K.MulmatFullPrec(Q) or V.Mulmat(KQ).** Direct dim
   analysis suggests `ne12 = numHeads = 16` and `ne13 = 1` for these — that's
   only 16 threads, under the cap. So pure attention shouldn't overflow at
   numHeads ≤ 1024.
2. **A reshape/permute in PositionEmbedding or PatchMerger** that produces a
   tensor with one of `ne[2]` or `ne[3]` equal to the per-side patch count
   (~102 for the 2048×2048 case) and the other multiplied up by mergeSize /
   spatial-merge structure. `102 × ≥10` is enough to exceed 1024.
3. **The mRoPE positional path** in qwen3-vl (`mropeSections: [24, 20, 20]`)
   may introduce a section-dimension that contributes to `ne[2]` or `ne[3]`.
4. **DeepstackMerger** runs after specific vision blocks (per
   `deepstackVisualIndexes`); its mul_mat dispatch could differ from the
   normal layer matmul.

To pin down (1)–(4) we need the actual ne dims at the crashing dispatch.
The existing `GGML_CUDA_DEBUG` log block at `ggml-cuda.cu:2117-2123` only
captures the GEMM compute-type config, not the kernel launch dims. The
trace adds a level-gated launch-dim log at the kernel call site (see
"Debug log additions" below) so the next CI run with `GGML_CUDA_DEBUG=1`
can finish the trace.

## Debug log additions

`ggml-cuda.cu` — at the kernel launch site of
`ggml_cuda_mul_mat_batched_cublas_impl` (gated by `GGML_CUDA_DEBUG=1`,
matching the existing static-bool style at line 2118):

```c
if (debug_enabled) {
    fprintf(stderr,
            "DEBUG batched_cublas k_compute_batched_ptrs: "
            "ne02=%lld ne03=%lld ne12=%lld ne13=%lld threads_per_block=%lld "
            "src0_name=\"%s\" src1_name=\"%s\" dst_name=\"%s\"\n",
            (long long)ne02, (long long)ne03, (long long)ne12, (long long)ne13,
            (long long)(ne12 * ne13),
            src0->name, src1->name, dst->name);
}
```

This is permanent (level-gated, not removed), surfaces the launch
configuration before the failing `<<<1, block_dims>>>` call, and lets
future debugging of similar K80-block-overflow crashes find the offending
tensor immediately by name. The kernel launch is per-graph-node so the log
volume with `GGML_CUDA_DEBUG=1` is bounded (≈ once per dispatched matmul,
not per element).

## Fix options

In priority order:

1. **Drop the `s.flashAttention` gate on the vision reservation.** The
   gate is unsound: every `model/models/*/model_vision.go` passes
   `cache=nil` to `nn.Attention`, and the SDPA fast-path in
   `ml/nn/attention.go:84` requires `cache!=nil`. So vision always runs
   the manual matmul path regardless of the server-level FA flag — the
   `if s.flashAttention { maxPixels = 2048 }` branch was reserving at the
   FA-friendly size while the actual graph still materializes the full
   non-FA attention matrix. Smallest surface; restores the working
   behavior from before #117 flipped `s.flashAttention` to `true` on K80.

2. **Fix the kernel launch to use grid+block split.** The kernel already
   supports `<<<grid, block>>>` (see indexing at ggml-cuda.cu:1933-1934).
   Change `<<<1, dim3(ne13, ne12)>>>` to
   `<<<dim3(ceil_div(ne13, bx), ceil_div(ne12, by)), dim3(bx, by)>>>` with
   e.g. `bx = by = 32`. Upstream-shaped fix; affects all GPUs (no
   regression risk on CC ≥ 7.0 because they hit the strided branch first).

3. **Route around the broadcast/non-contig branch for K80.** Force the
   strided path when possible — but the dispatch into the broadcast branch
   is driven by tensor stride / broadcast factors, not GPU CC, so this
   would require model-side reshape changes.

## Resolution

Applied **option 1** (`runner/ollamarunner/runner.go:1055-1067`): drop the
`s.flashAttention` branch, cap `maxPixels` at 512 unconditionally when the
user has not set `OLLAMA_VISION_MAX_PIXELS`. The comment in-source links
back to this trace for the reasoning. Option 2 remains a worthwhile
upstream-shaped fix; not done here to keep the change minimal and aimed at
the immediate CI red.

CI verification is pending — the K80 runner is the only build/run
environment for this code path, so the proper validation is the next full
Models suite run on this branch.

## Related

- #138 (this issue) — first encountered the failure
- #117 — productized K80 FA (changed `s.flashAttention` from false to true)
- #137 — productized FA-on default (made #117's effect happen by default)
- #63 / #50cba688 — added the `maxPixels = 512` fallback for non-FA path;
  the FA-on path retains the original 2048 upstream value
