# Trace: qwen3.5:27b GPU0 uses 10 GiB

**Issue**: #72 — qwen3.5:27b GPU0 uses 10 GiB while GPU1-3 use 4 GiB each
**Date**: 2026-04-11

## nvidia-smi vs llama.cpp Reported Allocations

| GPU | Weights | KV | RS | Compute | llama.cpp total | nvidia-smi | Gap |
|---|---|---|---|---|---|---|---|
| CUDA0 | 3492 | 64 | 37 | 1493 | 5086 | **10149** | **5063** |
| CUDA1 | 3405 | 64 | 37 | 266 | 3772 | 4325 | 553 |
| CUDA2 | 3383 | 64 | 37 | 266 | 3750 | 4303 | 553 |
| CUDA3 | 3514 | 64 | 37 | 266 | 3881 | 4434 | 553 |

All values in MiB.

## GPU1-3 Gap: Normal (~553 MiB each)

CUDA runtime overhead per GPU. Expected.

## GPU0 Gap: 5063 MiB

### Accounted allocations on GPU0

```
CUDA0 model weights:       3492 MiB
CUDA0 KV cache:              64 MiB
CUDA0 RS recurrent state:    37 MiB
CUDA0 compute buffer:      1493 MiB
CUDA_Host model (pinned):   995 MiB  ← cudaMallocHost, nvidia-smi counts on GPU0
CUDA_Host compute (pinned):  22 MiB
CUDA runtime overhead:     ~500 MiB
─────────────────────────────────
Subtotal:                  6603 MiB
nvidia-smi:               10149 MiB
Unexplained:               3546 MiB
```

### Two known issues

**1. Compute buffer imbalance**: GPU0 = 1493 MiB vs GPU1-3 = 266 MiB.
This is the llama.cpp scheduler putting more compute on GPU0 (first graph split,
embedding lookup, cross-GPU transfer staging).

**2. Unexplained 3.5 GiB on GPU0**: Not reported by llama.cpp logging. Possible causes:

#### H1: Cross-GPU transfer staging buffers
qwen3.5 has 98-109 graph splits across 4 GPUs. Each split boundary needs
staging buffers for tensor copies between GPUs. These are allocated on
GPU0 (the "main" GPU) by the GGML scheduler but not reported in the
per-backend buffer size logs.

Graph stats: `nodes=16749, splits=98` (batch=512). Extremely high split count
due to the hybrid DeltaNet+attention architecture — each layer type transition
creates a new split.

#### H2: DeltaNet state buffers
DeltaNet layers use `ssm_d_state × ssm_d_inner = 128 × 6144` state per layer.
With 48 DeltaNet layers, the state storage is significant. These may be
allocated on GPU0 as the "recurrent state host" beyond what's reported in
the RS buffer logs.

#### H3: CUDA memory fragmentation
Multiple allocation/free cycles during the iterative graph reservation
may leave fragmented GPU memory that nvidia-smi counts but isn't usable.

## Model Parameters

```
Architecture: qwen35 (DeltaNet hybrid)
Layers: 64 (48 DeltaNet + 16 full attention)
n_embd: 5120, n_head: 24, n_ff: 17408
ssm_d_inner: 6144, ssm_d_state: 128
BatchSize: 512, KvSize: 4096, FlashAttention: false
File: Q4_K_M, 16.21 GiB, 1307 tensors (456 vision skipped)
```

## TODO

- [ ] Run with `GGML_CUDA_NO_PINNED=1` to see if pinned memory accounts for more than reported
- [ ] Run with `CUDA_LAUNCH_BLOCKING=1` and check nvidia-smi during each allocation phase
- [ ] Check if reducing graph splits (by changing layer assignment) reduces GPU0 memory
- [ ] Compare with a non-hybrid model of similar size (e.g., gemma3:27b) to see if the gap exists
