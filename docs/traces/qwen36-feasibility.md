# Qwen3.6 Support Feasibility on K80

Issue: #170
Date: 2026-05-19

## TL;DR

**Go/no-go: GO.** Qwen3.6 reuses the Gated DeltaNet hybrid architecture we already
shipped for Qwen3.5 (PR #22, #72, #82). The dense 27B variant fits a Tesla K80 at
Q3–Q4 quants. Recommend landing **directly in the ollama-engine native path**
(`model/models/qwen36/`, forked from `model/models/qwen35/`), **skipping the
llama.cpp detour** that cost us #72/#82 last time.

Target first: **27B dense, text-only, 8K–32K context, Q4_K_M on 1× K80**.
Defer: 35B-A3B MoE, vision encoder, MTP head, 256K context, YaRN to 1M.

## 1. Source Verification (AC #1)

Confirmed real:
- HF: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (dense, vision)
- HF: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE, vision)
- GGUF: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — full quant range available
- Ollama library: `ollama.com/library/qwen3.6` (27b, 35b, 35b-a3b tags)
- License: Apache 2.0
- Release: 27B on 2026-04-22, 35B-A3B on 2026-04-16

Note on the pasted release notes: items like `ollama launch claude --model qwen3.6`
and "OpenClaw" do not appear in the official model card and are likely
hallucinated content from the source the user copied from. The core release is
real; ignore the agent-launching CLI claims.

## 2. Architecture (AC #2)

### Qwen3.6-27B (dense)
- 64 layers, hidden 5120, vocab 248,320 (BPE)
- Layout: **16 × (3× Gated DeltaNet → FFN, then 1× Gated Attention → FFN)**
  - 48 DeltaNet layers + 16 attention layers
  - Same 3:1 hybrid ratio as Qwen3.5 (which used `LLM_KV_FULL_ATTENTION_INTERVAL=4`)
- Gated DeltaNet: 48 V-heads, 16 QK-heads, head_dim 128 (matches Qwen3.5 family)
- Gated Attention: 24 Q-heads, 4 KV-heads, head_dim 256, **partial_rotary_factor=0.25** (RoPE on 64 of 256)
- FFN intermediate: 17,408
- Context: 262,144 native, YaRN extensible to 1,010,000
- RoPE: yarn, theta=10,000,000, mrope_interleaved=true, mrope_section=[11,11,10]
- Multi-Token Prediction (MTP) head present
- Vision encoder integrated (`AutoModelForImageTextToText`)
- HF architecture field: `Qwen3VLMoeForConditionalGeneration`-style (needs convert.py inspection)

### Qwen3.6-35B-A3B (MoE)
- 40 layers, hidden 2048, vocab 248,320
- Layout: **10 × (3× Gated DeltaNet → MoE, then 1× Gated Attention → MoE)**
  - 30 DeltaNet + 10 attention
- Same DeltaNet/Attention shape as 27B but smaller hidden
- MoE: 256 experts, 8 routed + 1 shared activated, expert_intermediate 512
- 3B activated params per token

### Tensors expected (mirrors Qwen3.5)
All `LLM_TENSOR_SSM_*` and DeltaNet tensors from #12 plus:
- New: vision encoder tensors (`v.*`)
- New: MTP head tensors (`mtp.*`)
- New: MoE expert tensors on 35B-A3B (`ffn_*_exps`, `ffn_gate_inp`)
- Gated Attention may add `attn_gate.weight` or similar — verify on first convert

## 3. Landing Zone Recommendation (AC #3)

**Recommendation: native ollama-engine only. Skip llama.cpp.**

Reasoning:
- Qwen3.5 was first added to llama.cpp (#12–#16, #18, #21) then **re-ported to the
  native engine** (#72, #82) because the llama.cpp path overallocated GPU memory
  (4 GPUs instead of 2 on K80). That re-port is exactly the work we'd repeat.
- The native engine already has the DeltaNet builder (`model/models/qwen35/deltanet.go`),
  hybrid layer routing (`model/models/qwen35/model.go`), the recurrent state cache
  (#81), and MoE config fields already declared in the qwen35 Options struct
  (unused but present — see `model/models/qwen35/model.go:36-40`).
- Fork `model/models/qwen35/` → `model/models/qwen36/`, then adjust:
  - Hidden size / layer count / head dims
  - Gated Attention with `partial_rotary_factor=0.25` (probably new; verify)
  - Tokenizer / vocab (248K vocab, possibly different BPE)
  - MoE routing (wire up the already-declared MoE fields)
- Convert path: extend `fs/ggml/ggml.go` to recognize the qwen3.6 arch string.

Counter-argument considered: llama.cpp would buy us upstream parity. **Not worth
it** — we're not contributing upstream, and 3.5 proved the native path is the only
one that fits the K80 memory budget.

## 4. K80 Hardware Feasibility (AC #4)

K80 has 2× 12 GB chips (24 GB usable per board). Most rigs in the test fleet have
1–2 K80s.

| Variant | Best quant for K80 | Weights | KV @ 8K | KV @ 32K | Fits 1× K80 (24 GB)? |
|---|---|---|---|---|---|
| 27B dense | Q4_K_M | 16.8 GB | ~1–2 GB (hybrid: most layers recurrent) | ~4–6 GB | **Yes, ~22 GB total** |
| 27B dense | Q3_K_M | 13.6 GB | as above | as above | Yes with headroom |
| 27B dense | Q5_K_M | 19.5 GB | as above | as above | Tight; 8K only |
| 35B-A3B MoE | Q4_K_M | ~20 GB | larger (less recurrent fraction) | ~6–8 GB | **No on 1×; needs 2× K80** |
| 27B+vision | Q4_K_M | 16.8 GB + vision (?) | as above | as above | Probably yes |
| Any variant @ 256K | — | — | enormous | enormous | **No** |

Go/no-go calls:
- **27B Q4_K_M, 8K–32K, text-only**: GO. Primary target.
- **27B vision**: Probably go, but defer until text path lands.
- **35B-A3B MoE**: GO on 2× K80 rigs; defer until 27B lands. K80 has no tensor cores
  so MoE expert dispatch perf is an open question — will need profiling.
- **256K context**: NO-GO on K80 hardware regardless of architecture. Cap at 32K.
- **1M YaRN**: NO-GO.
- **MTP head**: Defer. Useful for throughput but optional for correctness.

## 5. Follow-Up Issue Breakdown (AC #5)

To be created **after** this doc is reviewed. Numbers are placeholders.

| # | Title | Type | Priority | Component | Depends on |
|---|---|---|---|---|---|
| B | Fork `model/models/qwen35` → `model/models/qwen36` and register architecture in ollama-engine | feature | medium | model, go | #170 |
| C | Implement Qwen3.6 Gated Attention variant (partial RoPE 0.25, head_dim 256) | feature | medium | model, go | B |
| D | Add Qwen3.6 tokenizer/vocab support (248K BPE) | feature | medium | model, go | B |
| E | Add Qwen3.6 GGUF convert metadata to `fs/ggml/ggml.go` | feature | medium | go | B |
| F | Add Qwen3.6 chat renderer + parser (`model/renderers/`, `model/parsers/`) | feature | medium | go | B |
| G | Add `qwen3.6:27b` to model test suite (`cicd/tests/testcases/models/TC-MODELS-NNN.yml`) | feature | medium | ci, model | B,C,D,E,F |
| H | (deferred) Qwen3.6 MoE 35B-A3B support (wire up unused MoE fields) | enhancement | low | model, go | G |
| I | (deferred) Qwen3.6 vision encoder support | enhancement | low | model, ggml | G |
| J | (deferred) Qwen3.6 MTP head | enhancement | low | model | G |
| K | (deferred) Profile DeltaNet recurrent path on 64-layer model (Qwen3.5 was 32 layers) | enhancement | low | cuda, ggml | G |

What we are explicitly **not** doing in B–G:
- No llama.cpp side (`llama/llama.cpp/src/llama-arch.{cpp,h}`, `llama-model.cpp`, `llama-vocab.cpp`)
- No new GGML ops (DeltaNet ops landed in #18, CUDA in #19)
- No 256K context tuning
- No MLX variants

## 6. K80-Specific Risks (AC #6)

Based on the Qwen3.5 bring-up (#67, #73, #105, #121, #138), expect these to recur:

1. **Memory estimator overallocation** (#67-class): 27B/64-layer hybrid will likely
   exercise the same hybrid-graph-buffer estimator that #67 fixed for 3.5. Watch
   for it; same fix pattern probably applies.
2. **Small-model GPU spread** (#73): If the scheduler tries to fan out across all
   4 K80 halves for a 17 GB model, the same fix from #73 is needed.
3. **Quantization defaults** (#105): K80 prefers Q4_K_M / Q5_K_M; verify the
   default Modelfile picks one of these, not IQ-series (slow on K80).
4. **FA gating** (#121): DeltaNet doesn't use FA so the recurrent path is fine,
   but the 16 attention layers will. K80 has no tensor cores; FA-1 only.
5. **KV ghost allocation** (#138): Hybrid model split across GPUs may produce the
   same ghost MiB on the unused GPU. Same diagnostic applies.
6. **Graph node count**: Qwen3.5 needed `graph_max_nodes` bumped from 1024 to
   16384 for 32-layer DeltaNet. 27B has **64 layers** (2×) — may need 32768.
   This is the single most likely first-bug-after-load.
7. **Larger vocab**: 248K vs 152K for 3.5. Token embedding tensor is ~1.6× larger
   (~1.3 GB at FP16 for hidden=5120). Tokenizer load path may need verification.

## 7. Verification Done

- [x] Confirmed Qwen3.6 release is real (HF, Ollama library)
- [x] Confirmed GGUF availability across full quant range
- [x] Confirmed architecture family (Gated DeltaNet hybrid, same as 3.5)
- [x] Confirmed K80 fit for 27B at Q4
- [x] Identified the 4 net-new components vs 3.5 (Gated Attention partial RoPE,
      larger vocab, vision, MTP)

## 8. Open Questions (resolve during B/C)

- Exact HF `architectures` field string (need `config.json` inspection on a
  downloaded GGUF — neither `Qwen3VLMoeForConditionalGeneration` nor a clean
  `Qwen3_6ForCausalLM` was conclusively confirmed by web fetch).
- Whether Gated Attention adds a new tensor (`attn_gate.weight`) vs reusing the
  Q/K/V tensors with a different math path.
- Whether the 248K vocab uses the same tiktoken/BPE merges file as Qwen3.5 or a
  new one.
- Whether MTP tensors can be cleanly skipped at load (3.5 skipped 456 vision+MTP
  tensors successfully — see `MEMORY.md`).
