# MLX CUDA kernels actually on the qwen3.5/3.6 path (#188 scoping)

Don't port all 181 CUDA files — only those the qwen3.5/3.6 inference path executes.
Traced from the MLX-engine model (`ollama` upstream `x/models/qwen3_5/qwen3_5.go`
+ `x/models/nn/{nn,recurrent,rope,sdpa}.go`): the ops it calls map to a bounded
kernel set, and the hardest subsystems are **off-path or avoidable**.

## Ops the model uses → kernel category

| Op (mlx.* / nn.*) | Kernel file(s) | Status |
|---|---|---|
| Reshape/Slice/Transpose/Concatenate/Squeeze/ExpandDims/Flatten/Stack | copy.cu, copy/, slicing.cpp | needed (mechanical) |
| Add/Mul/Div/Neg/*Scalar | binary/, binary_two.cu | needed (mechanical) |
| SiLU/Sigmoid/Exp/SwiGLU | unary/ | needed (mechanical) |
| Where | ternary.cu | needed (mechanical) |
| RMSNorm/LayerNormFn | rms_norm.cu, layer_norm.cu | needed (mechanical) |
| RoPEWithBase | rope.cu | needed (mechanical) |
| FastScaledDotProductAttention | scaled_dot_product_attention.* | needed |
| Take/TakeAlongAxis (embedding/gather) | indexing.cpp | needed |
| Sum | reduce.cu, reduce/ | needed |
| SoftmaxAxis | softmax.cu | needed |
| Argsort/Argpartition (MoE top-k) | sort.cu | needed |
| Dequantize / QuantizedMatmul / GatherQMM / GatherMM | see "quant" below | needed (re-route) |
| Conv1d / CausalConv1D (DeltaNet) | see "conv" below | needed (rewrite) |
| GatedDelta / FastGatedDelta | — (composed of the above) | no kernel |

## The hard parts are avoidable

- **Quant matmul without CUTLASS.** All `quantized/qmm/*` kernels pull CUTLASS, and
  the dispatch (`qmm.cu`) is arch-gated (sm90/sm80/naive). Instead of porting CUTLASS,
  re-route `QuantizedMatmul`/`GatherQMM::eval_gpu` to **`affine_quantize` dequant
  (CUTLASS-free) → cuBLAS GEMM**. cuBLAS runs on K80. Exclude `qmm_sm80/sm90.cu`,
  `qmv.cu`, `qmm_naive.cu`, `cublas_qqmm` (CUDA-12 cuBLASLt), and `cute_dequant.cuh`.
- **conv1d without cuDNN.** `conv.cpp` is cuDNN-routed, but the only conv used is
  DeltaNet's depthwise **causal conv1d** — a small custom SIMT kernel. Exclude cuDNN.
- **GatedDelta** is composed of matmul/elementwise/scan in `recurrent.go` — no kernel.

## Droppable entirely (never called on this path)

`fft.cu`, `hadamard.cu`, `distributed.cu`, all FP8/mxfp8/nvfp4 quant, the CUTLASS
`qmm_sm80/sm90`, `cublas_qqmm`, cuDNN (`conv.cpp`, `cudnn_utils.*`), and complex-only
paths. These can be excluded from the build (the primitives are never instantiated).

## Net effect

The port shrinks to: the mechanical elementwise/norm/rope/sdpa/copy/reduce/softmax/
sort/indexing kernels (the C++17/array/fp16/coop-groups patterns already established),
plus two focused pieces — **dequant+cuBLAS quant matmul** and a **causal conv1d kernel**
— and **no CUTLASS, no cuDNN**. That removes the multi-week "hard core" the earlier
scope assumed.
