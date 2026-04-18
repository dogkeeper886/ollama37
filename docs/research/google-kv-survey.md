# Survey: Google's KV Cache Compression Research for K80 (sm_37)

**Issue**: [#99](https://github.com/dogkeeper886/ollama37/issues/99)
**Date**: 2026-04-18
**Status**: Survey complete — recommendation in §5

## 1. Executive Summary

Google/DeepMind's most significant recent KV cache work is **TurboQuant** (arXiv 2504.19874, ICLR 2026), a post-training KV quantization scheme that reaches 3-3.5 bits per channel with near-zero quality loss. Its companion **PolarQuant** (AISTATS 2026) targets the same problem from a polar-coordinate angle. Both are algorithmic — no hardware-specific instructions are required by the core method — which is the primary reason they are interesting for sm_37 (Tesla K80), a pre-Volta architecture without tensor cores, FP8, or native bfloat16.

The primary risk for K80 is not the algorithm but the **surrounding infrastructure**: every public TurboQuant integration so far targets sm_86+ with Flash Attention MMA kernels. ollama37's K80 path uses the scalar `fattn-vec` kernels, which do not yet have a TurboQuant instance. Landing TurboQuant on sm_37 means porting the quantize/dequantize path into the vec kernels and accepting that the reported end-to-end speed gains (driven by tensor-core-aware MMA paths) will not carry over — the value for K80 is **VRAM reduction**, not tok/s.

## 2. Techniques Surveyed

### 2.1 TurboQuant (Google Research + DeepMind, ICLR 2026)

- **Paper**: Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
- **Blog**: [research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **Method**:
  1. Apply a random rotation (Walsh-Hadamard Transform, O(d log d)) to each KV vector block. This induces a concentrated Beta distribution on coordinates and makes them approximately independent.
  2. Apply an optimal scalar quantizer (Lloyd-Max) per coordinate at the target bit-width (3 or 4 bits).
  3. Optional: 1-bit Quantized Johnson-Lindenstrauss correction on residuals to compensate for inner-product distortion.
- **Reported results**: 3.5 bits/channel = neutral quality, 2.5 bits/channel = marginal degradation. 6-8x KV memory reduction on Gemma and Mistral. Downstream benchmarks (LongBench, MMLU, etc.) unchanged at 3.5 bits.
- **Tested hardware** (paper + public forks): H100, RTX 5090 (sm_120), RTX 3090 (sm_86), Apple M-series. **No sm_37 testing reported.**

### 2.2 PolarQuant (Google, AISTATS 2026)

- Companion technique. Converts KV vectors from Cartesian to polar form (radius + angles) before quantization, exploiting the concentrated angular distribution that emerges from random rotation.
- Used as the first stage inside TurboQuant. Standalone performance is weaker than TurboQuant, so TurboQuant is the better integration target.

### 2.3 Grouped-Query Attention (Google, 2023)

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/pdf/2305.13245).
- 4-8x KV reduction by sharing K/V heads across query head groups.
- **Status for ollama37**: already in all recent models (Gemma, Llama 3, Qwen, Mistral). No integration work required — this is a **model architecture** choice, not a runtime-pluggable technique. Listed here only to note the baseline: any TurboQuant gains compose on top of GQA, not in place of it.

### 2.4 Non-Google eviction work (for completeness)

- **SAGE-KV** ([arXiv:2503.08879](https://arxiv.org/abs/2503.08879)) — self-attention-guided top-k eviction after prefill. Not Google-authored, but representative of an alternative axis (eviction vs. quantization). Out of scope for this study but flagged as a future option.

## 3. sm_37 Feasibility Table

K80 constraints: compute capability 3.7, CUDA 11.4 (driver 470), GCC 10.5, no tensor cores, no FP8, no native bf16, 2×12 GB VRAM, PCIe Gen3 x16.

| Technique | Core op | sm_37 compatible? | Blockers / notes |
|---|---|---|---|
| **TurboQuant (vec path)** | WHT rotation + Lloyd-Max quantize/dequantize | **Yes, in principle** | WHT is integer shuffle + add/sub, runs on any GPU. Lloyd-Max dequant is table lookup + FP32 scalar — compatible. Needs a new `fattn-vec` template instance for tbq3/tbq4 key and value types. |
| **TurboQuant (MMA path)** | Tensor-core MMA with quantized KV | **No** | Requires sm_80+ (Ampere MMA) or sm_90 (Hopper WGMMA). K80 has zero tensor cores. |
| **PolarQuant standalone** | Polar conversion + angle quantization | **Yes** | Pure arithmetic. Lower priority — subsumed by TurboQuant. |
| **GQA** | Architectural | **Yes (already present)** | No runtime change; depends on model definition. |
| **bf16 KV** | Any op reading/writing bf16 | **No (native)** | K80 has no bf16. Would need emulation — not worth it. Use fp16 instead. |
| **FP8 KV** | Any FP8 op | **No** | No FP8 support anywhere on K80. |
| **Flash Attention (fast path)** | Warp-level MMA | **No** | K80 uses `fattn-vec` fallback. Already the case — not a TurboQuant-specific regression. |

**Verdict**: TurboQuant's algorithm is sm_37-compatible via the scalar vec path. The tensor-core acceleration that makes it fast on H100/RTX 5090 is unavailable on K80, so the expected win is **VRAM footprint**, not throughput.

## 4. Expected Wins on K80

For a 7B model at 4K context (typical K80 ceiling today):

| KV format | Bits/value | Footprint vs. fp16 | K80 24GB fits |
|---|---|---|---|
| fp16 (current) | 16 | 1.0× | baseline |
| q8_0 (current, supported) | 8 | 0.50× | ~2× context |
| q4_0 (current, supported) | 4 | 0.25× | ~4× context |
| **tbq4 (TurboQuant 4-bit)** | 4 | 0.25× | ~4× context, **better quality than q4_0** |
| **tbq3 (TurboQuant 3-bit)** | 3 | 0.19× | ~5× context, neutral quality |

The comparison that matters is **tbq3 vs. q4_0**: TurboQuant at 3 bits claims to match fp16 quality, while q4_0 KV shows measurable degradation on long-context tasks. If the claim holds on K80, users get ~33% more context at equal or better quality compared to existing q4_0 KV quantization.

## 5. Recommendation

**Prototype TurboQuant's 3-bit KV (tbq3) for the K80 `fattn-vec` path.** Reject PolarQuant standalone (subsumed) and any MMA-dependent variant (incompatible).

Rationale:
- Algorithm is sm_37-compatible; only integration work is required.
- Existing quantized KV infrastructure (q4_0, q8_0 fattn-vec instances, type registration in `ml/backend/ggml/ggml/src/ggml-cuda/`) provides a direct template.
- The primary K80 pain point is VRAM, which is exactly what TurboQuant optimizes — throughput gains were never the K80 story.
- A reference CUDA implementation exists ([AmesianX/TurboQuant fork of llama.cpp](https://github.com/AmesianX/TurboQuant)) to crib kernels from, even though it targets sm_86+.

**Explicit non-goals for the prototype**:
- Do not port the MMA kernel. K80 has no tensor cores.
- Do not aim for tok/s parity with fp16 KV. The WHT + per-block dequant cost is real on scalar hardware; measure it, report it, accept it if VRAM wins.

## 6. Integration Sketch

Where the change would land in ollama37:

| Component | File(s) | Change |
|---|---|---|
| **KV cache allocator** | `kvcache/causal.go:617,622,624` — `c.ctxs[i].Zeros(c.DType, ...)` | Extend `c.DType` to accept new quantized types (tbq3, tbq4). Propagate from model config. |
| **GGML type registration** | `ml/backend/ggml/ggml/src/ggml-cuda/convert.cu`, `quantize.cu` | Register new type enums for tbq3/tbq4. Implement quantize (WHT + Lloyd-Max) and dequantize kernels. |
| **Flash attention kernels** | `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` + new `template-instances/fattn-vec-instance-*-tbq3-tbq3.cu` | Add tbq3/tbq4 instances matching the existing q4_0/q8_0 pattern. |
| **CMake** | `ml/backend/ggml/ggml/src/ggml-cuda/CMakeLists.txt` (around the `fattn-vec*q4_0-q4_0.cu` glob) | Add tbq3/tbq4 globs for the K80 path. |
| **New op (optional)** | `GGML_OP_TURBO_WHT` | If WHT needs to be a first-class graph op rather than baked into quantize/dequantize. Simpler to inline initially. |
| **CLI flag** | Wherever KV type is selected (grep for `cache-type-k` / `OLLAMA_KV_CACHE_TYPE`) | Add tbq3/tbq4 as user-selectable values. |

**Test strategy**:
- Unit: round-trip quantize → dequantize, compare MSE against fp16.
- Inference: Gemma 3 2B at tbq3 vs. fp16 on perplexity (wikitext) and one long-context benchmark. Target: ≤1% perplexity delta.
- Runtime: measure decode tok/s on K80 for fp16, q4_0, tbq3. Expect tbq3 to be slower than q4_0 (extra WHT + correction) but comparable.
- VRAM: confirm ~5× context headroom gain vs. fp16 on a 7B model.

## 7. Open Questions

1. **WHT block size on sm_37**: llama.cpp discussion notes block-32 is preferred on modern hardware for Flash Attention parallelism. K80's vec path may prefer a different block size — needs measurement.
2. **Asymmetric K/V quantization**: TurboQuant forks use asymmetric quant (K norms are 50-180× V norms). ollama37's current cache treats K and V symmetrically — does the cache API support per-tensor-type configuration?
3. **CUDA 11.4 compatibility of the reference kernels**: public forks were developed on 12.8/13.0. Some kernel features (cooperative groups, launch bounds semantics) may need backporting.
4. **K80 double-precision cost of Lloyd-Max codebooks**: codebooks are typically fp32 tables. Should be fine but confirm against shared memory budget (sm_37: 48 KB/SM).

## 8. References

- Zandieh et al. *TurboQuant.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Google Research blog. [TurboQuant: Redefining AI efficiency with extreme compression.](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- llama.cpp discussion #20969. [TurboQuant - Extreme KV Cache Quantization.](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Reference CUDA fork. [AmesianX/TurboQuant.](https://github.com/AmesianX/TurboQuant)
- Ainslie et al. *GQA.* [arXiv:2305.13245](https://arxiv.org/pdf/2305.13245)
- InfoQ coverage. [Google's TurboQuant Compression May Support Faster Inference on Less Capable Hardware.](https://www.infoq.com/news/2026/04/turboquant-compression-kv-cache/)
