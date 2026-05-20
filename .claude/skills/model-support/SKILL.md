---
name: model-support
description: Classify a "support model X" request before planning. Use when asked to support/run/load a new model, to decide if it is already-supported, a variant, renamed-from-upstream, or genuinely new — before creating issues.
---

# Model Support

Decision framework for "add support for model X" requests. **Verify the model exists
(Rule 0), then run the identity check — both before `/plan` or creating any issues.**
Most "new model" requests are not new architectures — they reuse, rename, or vary
something already in the tree.

## When to use
- A request to support / run / load a model (e.g. "support qwen3.6", "add deepseek-v4")
- Before `/plan` would create implementation issues for model work

## Rule 0 — Verify before you doubt (the most important rule)
**Never reject, downgrade, or call a model request "fabricated" because it is missing
from your training data or because the pasted source has odd-looking lines.** Your
knowledge has a cutoff; "I've never heard of it" is not evidence it does not exist. A
new model released after the cutoff will *always* look unfamiliar — that is expected,
not suspicious. The first action for any model request is a web search to confirm it
exists. Only after confirming do you proceed to the identity check below.

## Why this exists
"Support Qwen3.6" (#170) was first met with skepticism — it was flagged as "likely
fabricated/future content" purely because it post-dated the training cutoff and a few
marketing lines looked off. That was wrong: one web search confirmed it was real in
seconds. The request was nearly dropped on the basis of training data + surface
pattern-matching instead of verification.

Then it swung the other way: once confirmed real, it was over-planned as six issues for
a from-scratch port. The HuggingFace config showed `model_type: qwen3_5` — Qwen3.6
*reuses* the architecture we already ship as `qwen35` (itself a rename of upstream
Ollama's `qwen3next`). The real work was a smoke test plus restoring ~6 dispatch lines
our fork had dropped for the MoE sibling.

Two failures, two rules: **verify before doubting** (Rule 0), then **identity-check
before planning** (below).

## Flow at a glance

```
REQUEST: "support / run model X"
        │
        ▼
[ RULE 0 — VERIFY ]  web-search the name
        │            (post-cutoff = expected, NOT fake)
   ┌────┴────┐
  NO         YES
   │          │
 STOP         ▼
 (can't    [ IDENTITY CHECK ]
 confirm)   1. HF config.json → architectures / model_type
            2. grep our model.Register
            3. gh api upstream ollama (fork may have stripped it)
                  │
                  ▼
            [ K80 REALITY GATE ]
            fits 24GB@Q4? MLX? FP8/FP4/BF16 HW? 256K ctx?
                  │
                  ▼
            [ CLASSIFY + ROUTE ]
            ├─ Already supported ──→ smoke test only (/test)
            ├─ Variant (MoE/size) ─→ one small "extend" issue
            ├─ Renamed/stripped ───→ one small "restore" issue
            └─ Genuinely new ──────→ full plan via /plan
```

## The identity check (after Rule 0 confirms it exists)

1. **Read the real architecture from HuggingFace — not the marketing name:**
   ```bash
   curl -sL "https://huggingface.co/<org>/<model>/raw/main/config.json" \
     | python3 -c "import json,sys;c=json.load(sys.stdin);print(c.get('architectures'),c.get('model_type'))"
   ```
   For multimodal configs, also inspect `text_config` and `vision_config`.

2. **Check our registry** for that architecture / model_type:
   ```bash
   grep -rn "model.Register" model/models/*/model.go
   grep -rn "<arch>" fs/ggml/ggml.go
   ```

3. **Check upstream Ollama** — our fork may have renamed or stripped it:
   ```bash
   gh api repos/ollama/ollama/contents/model/models --jq '.[].name'
   gh api repos/ollama/ollama/contents/model/models/<pkg>/model.go --jq '.content' \
     | base64 -d | grep "model.Register"
   ```

## Classify, then route

| Finding | Classification | Action |
|---|---|---|
| Same `model_type`/arch already registered and previously run | **Already supported** | Smoke test only (`/test`); no implementation issues |
| Sibling variant (MoE/size) of a registered arch | **Variant** | Restore/extend dispatch entries; one small issue |
| Arch exists upstream but our fork renamed/stripped it | **Renamed/stripped** | Diff our fork vs upstream, restore the dropped wiring; small issue |
| No matching arch in our tree or upstream | **Genuinely new** | Full implementation plan via `/plan` |

## K80 reality gate (always, before committing to a variant)
- Weights at Q4 + KV cache vs **24 GB per K80**; large MoE needs multiple boards.
- No FP8 / FP4 / BF16 hardware (compute capability 3.7) — those quants dequant to FP32.
- **MLX format is unreadable** — the converter only takes safetensors / pytorch.
- 256K context is impractical on K80.

See `docs/traces/qwen36-feasibility.md` and `docs/traces/qwen36-load-path.md` for a
worked example of this whole flow.

## Output
A one-paragraph classification (which row above, with the arch string and the evidence)
plus the routing decision. Only then proceed — to `/plan` for genuinely-new, or to a
smoke-test / restore-wiring issue for everything else.
