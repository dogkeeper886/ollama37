# Qwen3.6 Load Path Trace

Related: #170, [qwen36-feasibility.md](qwen36-feasibility.md)
Date: 2026-05-20

## Purpose

Trace the full load path for a `qwen35`-architecture GGUF file (which is what Qwen3.6's
converter emits — confirmed via HF `config.json`: `"architectures": ["Qwen3_5ForConditionalGeneration"]`,
`"model_type": "qwen3_5"`). The goal is to predict, before runtime, where Qwen3.6
could trip on a code path versus where it must succeed.

## TL;DR

The runtime path for a `qwen35`-arch GGUF is **architecturally complete**. Risk surface
reduces to three independently-verifiable items:

1. **GGUF metadata keys must match** what `qwen35/model.go:New()` reads (lines 405–498).
   If the qwen3.6 GGUF were converted with different key names than qwen3.5, `New()`
   would fail loudly with a clear error from one of its named-field validators.
2. **Tensor names must match** the `gguf:` tags on `Layer`, `GatedDeltaNet`, and
   `FullAttention` structs. If qwen3.6 added a new tensor (e.g. for MTP) or renamed
   one, `populateFields` would leave the pointer nil and `Validate()` would fail
   with `qwen35: layer N missing <X>`.
3. **Renderer/parser is selected by name string** (`"qwen3.5"`), not by arch. If the
   published `qwen3.6:27b` Modelfile names a renderer that isn't in the switch
   (`model/parsers/parsers.go:51`, `model/renderers/renderer.go:59`), chat formatting
   falls back to the Modelfile's `TEMPLATE` block — usually safe, but worth verifying.

Everything else (engine selection, FA gating, hybrid layer routing, BPE tokenizer,
recurrent cache, attention math including Gated Attention with `attn_gate.weight`)
is reused as-is.

## Call Flow

```
ollama run qwen3.6:27b
  │
  ├── (server) handles request → schedules model load
  │
  ▼
llm/server.go:NewLlamaServer()                                       # server.go:154
  │
  ├── arch = f.KV().Architecture()                                   # "qwen35" (from GGUF general.architecture)
  │
  ├── if envconfig.NewEngine() || f.KV().OllamaEngineRequired():     # server.go:159
  │     OllamaEngineRequired() returns true for "qwen35"             # ggml.go:241–255
  │     │
  │     └── textProcessor, err = model.NewTextProcessor(modelPath)   # server.go:161
  │           │
  │           ├── meta = fsggml.Decode(r, -1)                        # model.go:129
  │           ├── m = modelForArch(meta.KV())                        # model.go:134
  │           │     └── arch = "qwen35" → models["qwen35"] = qwen35.New
  │           │
  │           └── return m as TextProcessor (BytePairEncoding)       # model.go:139
  │
  ├── (memory estimation, batch sizing, etc.)
  │
  └── runner spawn → eventually loads weights and runs Forward
        │
        ▼
qwen35.New(c fs.Config)                                              # qwen35/model.go:404
  │
  ├── numLayers = c.Uint("block_count")                              # GGUF: qwen35.block_count
  ├── headCountKV = c.HeadCountKV()                                  # GGUF: qwen35.attention.head_count_kv[]
  │
  ├── isRecurrent = inferRecurrentLayers(...)                        # model.go:350
  │     ├── if any headCountKV[i] == 0: that layer = recurrent       # primary path
  │     └── else: derive from full_attention_interval (compat path)  # qwen35.full_attention_interval
  │
  ├── isMoE = c.Uint("expert_count") > 0                             # false for 27B dense
  │
  ├── per-layer Operator:
  │     ├── recurrent → &GatedDeltaNet{Layer: i}                     # deltanet.go:36
  │     └── attention → &FullAttention{}                             # attention.go:20
  │
  ├── opts := &Options{...}                                          # model.go:457
  │     reads ~20 GGUF keys: embedding_length, attention.head_count,
  │     attention.head_count_kv, attention.key_length,
  │     attention.value_length, rope.dimension_count,
  │     attention.layer_norm_rms_epsilon, rope.type, rope.freq_base,
  │     rope.scaling.factor, rope.scaling.original_context_length,
  │     attention.scale, expert_count, expert_used_count,
  │     norm_top_k_prob, ssm.inner_size, ssm.state_size,
  │     ssm.group_count, ssm.time_step_rank, ssm.conv_kernel,
  │     ssm.v_head_reordered, mrope_sections | rope.mrope_section,
  │     rope.mrope_interleaved | mrope_interleaved
  │
  ├── validation: numKVHeads != 0, headKDim == headVDim              # model.go:496, 530
  │
  ├── m := Model{
  │     BytePairEncoding: model.NewBytePairEncoding(                 # model.go:536
  │       &Vocabulary{
  │         Values:  c.Strings("tokenizer.ggml.tokens"),             # 248K entries
  │         Types:   c.Ints("tokenizer.ggml.token_type"),
  │         Merges:  c.Strings("tokenizer.ggml.merges"),
  │         BOS/EOS: tokenizer.ggml.{bos,eos}_token_id(s),
  │       },
  │       <GPT-2 style pretokenize regex hardcoded>,                 # model.go:549
  │     ),
  │     Layers:  layers,
  │     Options: opts,
  │   }
  │
  ├── m.Cache = NewHybridCache(...)                                  # model.go:555 → cache.go
  │
  └── return &m
        │
        ▼
(framework) populateFields(base, v.Elem())                           # model.go:118, 160
  │
  └── reflectively walks Model struct, binds tensors by `gguf:` tag:
        │
        ├── Model fields:
        │     TokenEmbedding   `gguf:"token_embd"`
        │     OutputNorm       `gguf:"output_norm"`
        │     Output           `gguf:"output,alt:token_embd"`        # tied embed fallback
        │     Layers           `gguf:"blk"`                          # → "blk.N.*"
        │
        ├── Layer fields (per-layer, common to all):
        │     AttnNorm, AttnPostNorm (RMSNorm)
        │     FFN: gate_proj/up_proj/down_proj OR MoE experts+router
        │
        ├── FullAttention fields (attention layers only):
        │     Query      `gguf:"attn_q"`                             # outputs Q + gate interleaved
        │     QueryNorm  `gguf:"attn_q_norm"`
        │     Key        `gguf:"attn_k"`
        │     KeyNorm    `gguf:"attn_k_norm"`
        │     Value      `gguf:"attn_v"`
        │     Output     `gguf:"attn_output"`
        │     (Q has 2× width: query + sigmoid gate, split at runtime — line 62–63)
        │
        └── GatedDeltaNet fields (recurrent layers only):
              SSMQKV       `gguf:"attn_qkv"`
              SSMQKVGate   `gguf:"attn_gate"`                        # the #57 fix
              SSMIn        `gguf:"ssm_in"`
              SSMBetaAlpha `gguf:"ssm_ba"`                           # legacy combined
              SSMBeta      `gguf:"ssm_beta"`                         # qwen35 split
              SSMAlpha     `gguf:"ssm_alpha"`
              SSMConv1D    `gguf:"ssm_conv1d"`
              SSMDT        `gguf:"ssm_dt,alt:ssm_dt.bias"`
              SSMA         `gguf:"ssm_a"`
              SSMNorm      `gguf:"ssm_norm"`
              SSMOut       `gguf:"ssm_out"`

        ▼
(framework) m.Validate()                                              # model.go:297
  │
  └── For each recurrent layer, asserts the required GatedDeltaNet
      tensors were bound. Failure → clear "qwen35: layer N missing <X>"
      error — easy to diagnose.

        ▼
(runtime, first decode) Forward()                                     # model.go:264
  │
  ├── positions = buildPositions(...)                                 # mrope-aware if mrope_sections set
  ├── hiddenStates = TokenEmbedding.Forward(...)                      # 248K × 5120 embed
  │
  └── for each layer:
        if isRecurrent[i]:  GatedDeltaNet.Forward(...)                # deltanet.go:75
                            uses recurrent state in cache.SetLayer(i)
        else:               FullAttention.Forward(...)                # attention.go:29
                            uses standard KV cache slot
                            Gated Attention math: attn × sigmoid(gate)
```

## Engine Selection Gate (the key decision point)

`llm/server.go:159` is where ollama-engine vs llama.cpp is chosen:

```go
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    textProcessor, err = model.NewTextProcessor(modelPath)
    ...
}
if textProcessor == nil {
    llamaModel, err = llama.LoadModelFromFile(...)   // falls back to llama.cpp
}
```

`OllamaEngineRequired()` for `qwen35` returns `true` (`fs/ggml/ggml.go:253`). The
llama.cpp fallback is unreachable for this arch — confirming that the native engine
path is the only one we need to validate.

## Flash Attention Path

`fs/ggml/ggml.go:893`: legacy FA gate **denies** qwen35 (because the hybrid layout
breaks the head-count equality check).

`fs/ggml/ggml.go:926`: new-engine FA allowlist **permits** qwen35 ("validated #107:
qwen3.5:9b run 25799484945, qwen3.5:27b run 25799818450"). Only the 16 attention
layers exercise FA; the 48 recurrent layers are unaffected.

A new Qwen3.6 GGUF would inherit this gate automatically — the arch string is the
key, and it remains `qwen35`.

## Renderer / Parser (the actual risk surface)

`model/renderers/renderer.go:59` and `model/parsers/parsers.go:51` both have
explicit `case "qwen3.5":` handlers that alias to `Qwen3VLRenderer{isThinking:true}`
and `Qwen3VLParser{hasThinkingSupport:true}`.

**Selection is by name string, not by GGUF architecture.** The name comes from the
Modelfile's `TEMPLATE` or `PARSER`/`RENDERER` directives. So three scenarios for
the published `qwen3.6:27b` blob:

| Modelfile names | Result |
|---|---|
| `qwen3.5` | Uses existing renderer/parser. No code change. |
| `qwen3.6` (or new name) | Falls into `default: return nil`. Behavior depends on whether `TEMPLATE` is provided as a Go template instead. |
| `qwen3-vl-thinking` | Uses existing renderer/parser (it's the same struct under the hood). No code change. |

**This is the one item that needs runtime verification before we can claim "we
support 3.6 end-to-end."** Pull the blob, inspect its Modelfile, see what name is
declared.

## What This Trace Confirms vs Leaves Open

### Confirmed (no Qwen3.6-specific code needed)
- Architecture string `qwen35` is the runtime key everywhere (engine selection, FA
  gate, model factory, tensor binding).
- All 27B-shape hyperparameters (5120 hidden, 64 layers, head_dim 256,
  partial_rotary_factor 0.25, attn_output_gate, 248K vocab, MRoPE) are read by
  generic GGUF keys — no name-specific dispatch.
- Gated Attention with `attn_gate.weight` (the #57 shape-mismatch fix) is built
  into `FullAttention` (`attention.go:62–63`) and `GatedDeltaNet`
  (`deltanet.go:38`).
- Hybrid 3:1 layer layout supported via `inferRecurrentLayers()` from either
  per-layer `head_count_kv` (preferred) or `full_attention_interval` (compat).
- BPE tokenizer with 248K vocab works the same way as for qwen3.5:27b — same
  GGUF keys, same pretokenize regex.
- FA enabled on attention layers via the new-engine allowlist.

### Leaves open (verify by pulling the GGUF)
1. **Modelfile name for renderer/parser** — see "Renderer / Parser" section above.
2. **Tensor name set parity** — does the qwen3.6 GGUF introduce any tensor not
   already tagged in `Layer` / `GatedDeltaNet` / `FullAttention`? Likely no (MTP
   tensors would just be skipped at load, as they were for qwen3.5), but
   confirmable only by inspecting the file.
3. **BPE merges identity** — vocab size matches (248K), but the actual merges
   file contents could differ from qwen3.5:27b. If they differ, tokenization
   correctness is at risk regardless of count.
4. **MRoPE section values** — config reports `mrope_section: [11, 11, 10]`. The
   loader (`model.go:441–450`) accepts this from `mrope_sections` /
   `rope.mrope_section` / `rope.dimension_sections` keys. Confirm GGUF uses one
   of these.

## Suggested Verification (no code change needed)

```bash
# Pull and inspect
ollama pull qwen3.6:27b
ollama show qwen3.6:27b --modelfile          # see TEMPLATE / RENDERER / PARSER names
ollama show qwen3.6:27b --parameters         # confirm context, defaults

# Run with debug logs — the qwen35 path is highly instrumented
OLLAMA_DEBUG=1 ollama run qwen3.6:27b "Say one short sentence about pumpkins."

# Expected debug lines if successful:
#   "engine selected" engine=ollama arch=qwen35
#   "model config" arch=qwen35 layers=64 recurrent=48 attention=16 moe=false
#   "ssm config" d_inner=6144 d_state=128 n_group=16 dt_rank=48 ...
#   "rope config" type=mrope sections=[11,11,10] base=10000000 ...
#   "first forward" batch_size=N layers=64
```

If those lines appear and output is coherent: the trace's "confirmed" section is
fully borne out and only the renderer/parser may need adjustment. If `Validate()`
errors with "missing <X>" or `populateFields` leaves a tensor nil, the open
question on tensor parity is the bug — fixable by adding a new `gguf:` tag.

## References

- Engine selection: `llm/server.go:154-177`
- OllamaEngine allowlist: `fs/ggml/ggml.go:241-255`
- Model factory dispatch: `model/model.go:122-158`
- qwen35 `New()`: `model/models/qwen35/model.go:404-557`
- Per-layer recurrence inference: `model/models/qwen35/model.go:350-402`
- Validation: `model/models/qwen35/model.go:297-335`
- GatedDeltaNet tensor tags: `model/models/qwen35/deltanet.go:36-48`
- FullAttention (Gated Attention) tensor tags: `model/models/qwen35/attention.go:20-27`
- FA gate (legacy deny): `fs/ggml/ggml.go:893`
- FA gate (new-engine allow): `fs/ggml/ggml.go:918-935`
- Renderer switch: `model/renderers/renderer.go:45-69`
- Parser switch: `model/parsers/parsers.go:37-67`
- Precedent: #57 (qwen3.5:27b shape mismatch — attn_gate fix), #107 (FA validation)

---

## Addendum: Qwen3.6-35B-A3B (MoE variant)

The 35B-A3B variant is a **different arch string** (`qwen35moe`) than the 27B
dense, and its routing is **incomplete** in the current code. The MoE math is
implemented but not wired to the dispatch tables.

### HF config (verified raw)
- `architectures: ["Qwen3_5MoeForConditionalGeneration"]`
- `model_type: "qwen3_5_moe"`
- 40 layers, hidden 2048, 16 Q-heads, 2 KV-heads, head_dim 256
- `num_experts: 256`, `num_experts_per_tok: 8`, `moe_intermediate_size: 512`
- `shared_expert_intermediate_size` present → uses shared-expert path

The converter (`convert_hf_to_gguf.py` upstream) maps `qwen3_5_moe` → GGUF arch
`qwen35moe`. That string is the routing key.

### MoE math: implemented (✅)
`model/models/qwen35/model.go:97-166` defines `sparse` with the full Qwen MoE
machinery:
- Router (`ffn_gate_inp`)
- Batched expert linears (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)
- Optional shared experts (`ffn_gate_inp_shexp`, `ffn_gate_shexp`,
  `ffn_up_shexp`, `ffn_down_shexp`) — Qwen3.6-35B-A3B uses this path
- Top-k selection, optional `norm_top_k_prob` renormalization
- SILU activation, gated output, summed expert outputs

Selection of dense-vs-sparse is **per-layer, GGUF-metadata-driven**
(`model.go:425`): `isMoE = c.Uint("expert_count") > 0` — independent of arch
string. So a `qwen35moe` GGUF would correctly produce `sparse` MLPs once the
arch routing dispatches to this package.

### Routing tables: **NOT plumbed for `qwen35moe`** (❌)

| Location | Current | Needed |
|---|---|---|
| `model/models/qwen35/model.go:560` | `model.Register("qwen35", New)` | Add `model.Register("qwen35moe", New)` |
| `fs/ggml/ggml.go:253` (`OllamaEngineRequired`) | only `"qwen35"` | Add `"qwen35moe"` — else falls to llama.cpp which doesn't know it |
| `fs/ggml/ggml.go:893` (legacy FA deny) | only `"qwen35"` | Add `"qwen35moe"` |
| `fs/ggml/ggml.go:926` (new-engine FA allow) | only `"qwen35"` | Add `"qwen35moe"` **only after empirical validation** per the methodology comment in that block |
| `fs/ggml/ggml.go:943` (`FlashAttention` enable) | only `"qwen35"` | Add `"qwen35moe"` |

The package already anticipates `qwen35moe` — `defaultVHeadReordered(arch)` at
`model.go:347` accepts both — but the dispatch additions above were never made.
This is the only reason the existing `sparse` implementation isn't reachable.

### Real risks for 35B-A3B (beyond the missing dispatch wiring)

1. **MoE path has never been exercised in this fork.** Qwen3.5's MoE variants
   were 397B-A17B and 122B-A10B — too big for K80, never tested. The `sparse`
   struct compiles and the math is upstream-derived, but no run has validated
   it end-to-end. First load is genuinely first-load.
2. **256 experts is large.** Per layer: 3× `LinearBatch` tensors with first
   dim 256. Per layer total MoE weights at FP16 ≈
   3 × 256 × 2048 × 512 × 2 bytes ≈ 1.6 GB. Across 40 layers ≈ 64 GB at FP16
   or ~16 GB at Q4. The `LinearBatch.Forward` path needs to handle this size;
   probably fine since it's just batched matmuls, but worth profiling for
   K80-specific bottlenecks (no tensor cores → expert dispatch could stall).
3. **K80 fit.** 35B-A3B Q4_K_M is ~20+ GB before KV cache. Will not fit on 1×
   K80 (24 GB); needs 2× K80 (4 GPU halves). Tensor-parallel split across MoE
   experts is its own performance question.
4. **Memory estimator + scheduler.** The estimator has had MoE-specific bugs
   (e.g. shared-expert and routing-tensor accounting). Expect to revisit #67 /
   #138 patterns when this lands.

### Practical scope for 35B-A3B support
A separate issue beyond the current set:
- Add the 5 dispatch-table entries listed above (10-line PR)
- Verify Qwen3.6-35B-A3B actually loads through the `sparse` path
- Profile expert dispatch on K80
- Empirically validate FA before adding to the new-engine allowlist (per #123 methodology)

This is dramatically smaller than "implement MoE from scratch" but larger than
the dense 27B case (which needs nothing).

