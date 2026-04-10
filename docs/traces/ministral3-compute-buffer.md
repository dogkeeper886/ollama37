# Trace: ministral-3:3b compute buffer allocation

**Issue**: #63 — ministral-3 uses 13.3 GiB for a 3GB model on K80
**Date**: 2026-04-10

## The Anomaly

CUDA3 (3 layers + output) has 9.1 GiB compute graph buffer, while CUDA0 (12 layers) has only 411.8 MiB.

## Reservation Flow

```
allocModel()                                        # runner.go:1125
  └── reserveWorstCaseGraph(prompt=true)            # runner.go:1193
        ├── batchSize = s.batchSize = 512           # runner.go:1024
        ├── inputs = [512 tokens]                   # runner.go:1027
        ├── model.Forward(ctx, batch)               # runner.go:1111
        │     └── Full forward pass through all 27 layers + output
        ├── ctx.SetBatchSize(512)                   # runner.go:1116
        └── ctx.Reserve()                           # runner.go:1117
              └── ggml_backend_sched_reserve()       # ggml.go:855
                    ├── ggml_backend_sched_split_graph()  # ggml-backend.cpp:948
                    │     Pass 1: assign backends by weight location
                    │     Pass 2: expand GPU assignments to adjacent nodes
                    │     Pass 3: upgrade to higher prio backends
                    │     Pass 4: assign remaining from dst/view_src
                    │     Pass 5: split graph, insert cross-GPU copies
                    └── ggml_gallocr_reserve_n()     # ggml-alloc.c:799
                          └── ggml_gallocr_alloc_graph_impl()
                                For each node in order:
                                  - allocate on assigned buffer_id
                                  - free parents when no longer needed
                                  - peak memory per buffer = reported size
```

## Attention Path (non-flash, K80)

```go
// ml/nn/attention.go:61-74 (manual path, flash not available)
query = query.Permute(0, 2, 1, 3)
key = key.Permute(0, 2, 1, 3)
value = value.Permute(1, 2, 0, 3).Contiguous()

kq = key.MulmatFullPrec(query)     // [kv_size, batch, heads] F32
kq = kq.Scale(scale)
kq = kq.Add(mask)
kq = kq.Softmax()                  // [kv_size, batch, heads] F32
kqv = value.Mulmat(kq)             // [d_v, batch, heads]
kqv = kqv.Permute(0, 2, 1, 3).Contiguous()
```

### Per-layer attention intermediate sizes (batch=512, heads=24, kv_size=4096)

| Tensor | Shape | Size |
|---|---|---|
| kq (QK^T) | [4096, 512, 24] F32 | 192 MiB |
| kq after softmax | [4096, 512, 24] F32 | 192 MiB |
| kqv | [128, 512, 24] F32 | 6 MiB |
| Contiguous copies | various | ~200 MiB |

Peak per layer: ~400-500 MiB. The allocator reuses memory between layers within the same GPU split, so 12 layers on CUDA0 still peak at ~400 MiB (one layer's worth).

### Expected vs actual

| GPU | Layers | Expected peak | Actual |
|---|---|---|---|
| CUDA0 | 12 | ~400-500 MiB | 411.8 MiB ✓ |
| CUDA1 | 12 | ~400-500 MiB | 400.0 MiB ✓ |
| CUDA3 | 3 + output | ~400-500 MiB | **9.1 GiB** ✗ |

CUDA0 and CUDA1 match expectations. CUDA3 is ~18× too high.

## Hypotheses for CUDA3 Anomaly

### H1: Output layer logit computation
The output layer projects hidden_dim → vocab_size: [3584, 32768].
With batch=512: `[32768, 512] * 4 bytes = 64 MiB`. Not enough to explain 9.1 GiB.

### H2: Cross-GPU tensor copies
When hidden state transfers from CUDA1 → CUDA3, the scheduler creates copy tensors on CUDA3.
Size: `[3584, 512] * 4 = 7 MiB`. Negligible.

### H3: Graph scheduler assigns non-layer ops to CUDA3
The scheduler's "expand GPU" passes (2-4 in split_graph) may assign embedding or other
non-layer operations to CUDA3 because it hosts the output layer. If the embedding
lookup or positional encoding happens on CUDA3, those tensors could be large.

### H4: Vision model tensors
ministral-3 has a vision model. Even for text-only inference, the reservation
graph may include vision processing paths. The 2048×2048 worst-case image
(runner.go:1048) is encoded during reservation. If vision layers run on CUDA3,
the pixel processing could use significant memory.

### H5: GGML allocator fragmentation
The dynamic allocator may not optimally reuse memory across the graph split
boundary. Tensors from early nodes may not be freed if they're referenced by
later nodes across the split.

## Root Cause: Vision Encoder Self-Attention

**Confirmed by code trace.** The 9.1 GiB is from the vision encoder's self-attention
on image patches during worst-case graph reservation.

### Reservation creates a 2048×2048 image

```go
// runner.go:1048
img := image.NewGray(image.Rect(0, 0, 2048, 2048))
```

### Vision model processes it into patches

```go
// model_vision.go:169 — imageSize=1540, patchSize=14
numPatches = (1540/14) × (1540/14) = 110 × 110 = 12,100 patches
```

### Vision self-attention WITHOUT flash attention

```go
// model_vision.go:42 — nil cache, no flash attention on K80
attention := nn.Attention(ctx, query, key, value, scale, nil)
```

Falls through to manual path (attention.go:61-74):
```go
kq = key.MulmatFullPrec(query)  // [12100, 12100, 16] F32
kq = kq.Softmax()               // [12100, 12100, 16] F32
```

### The math

```
kq tensor = [12100, 12100, 16 heads] × 4 bytes/F32
          = 12100 × 12100 × 16 × 4
          = 9,388,960,000 bytes
          ≈ 8.9 GiB
```

Plus softmax output (same size but reuses memory) and other intermediates → **~9.1 GiB**.

### Why only CUDA3?

The vision encoder weights are not part of the layer split (they're separate from
blk.0-26). The scheduler assigns vision ops to whichever GPU it expands to. Since
the output layer is on CUDA3, and vision processing feeds into the text model's
input, the scheduler puts vision on CUDA3.

## Fix Options

1. **Reduce reservation image size for K80**: 2048×2048 → 512×512 = 1,369 patches
   → kq = [1369, 1369, 16] × 4 = 115 MiB (78× reduction)
2. **Skip vision reservation on non-flash GPUs**: Only reserve text graph, since
   vision inference won't work well without flash attention anyway
3. **Tile the vision attention**: Process patches in smaller groups (requires
   changes to the vision model's attention path)
4. **Reduce patch count**: Use larger patchSize for K80 (but this changes model behavior)
