# Classify a Model-Support Request Before Planning

```
Identify a model's true architecture and route the request before creating issues.
Reference `.claude/skills/model-support/SKILL.md` for context.

Model: $ARGUMENTS

## PURPOSE

Two jobs. First, stop a real request from being wrongly dropped as "fabricated" just
because the model post-dates the training cutoff. Second, stop confirmed requests from
being over-planned — most are not new architectures, they reuse/rename/vary something
already in the tree. Verify, then identity-check, then route to the (usually small) action.

---

## WORKFLOW

    /model-support qwen3.6
        │
        ├─► Step 0: VERIFY BEFORE ANY DOUBT (existence AND reachability)
        │   - Exists? A model unknown to training data is expected, not suspicious.
        │     WebSearch the name; confirm HF repo / Ollama library entry.
        │   - Reachable? "MLX/can't load/wrong quant/won't fit" are CLAIMS, not facts.
        │     "Can't load directly" ≠ "can't be made to work" (conversion may exist).
        │   - Never reject as "fabricated" or "out of scope" before tracing it.
        │
        ├─► Step 1: Read the REAL architecture (not the marketing name)
        │   curl -sL "https://huggingface.co/<org>/<model>/raw/main/config.json" \
        │     | python3 -c "import json,sys;c=json.load(sys.stdin);print(c.get('architectures'),c.get('model_type'))"
        │   - For multimodal: also inspect text_config / vision_config
        │   - Note: HF model_type is the truth; the release name (e.g. "3.6") is not
        │
        ├─► Step 2: Check OUR registry
        │   grep -rn "model.Register" model/models/*/model.go
        │   grep -rn "<arch>" fs/ggml/ggml.go
        │   - Is the model_type / GGUF arch string already registered?
        │
        ├─► Step 3: Check UPSTREAM Ollama (our fork may have renamed/stripped it)
        │   gh api repos/ollama/ollama/contents/model/models --jq '.[].name'
        │   gh api repos/ollama/ollama/contents/model/models/<pkg>/model.go --jq '.content' \
        │     | base64 -d | grep "model.Register"
        │   - Compare upstream's registered arch strings + dispatch lists to ours
        │
        ├─► Step 4: K80 reality gate — QUESTIONS to trace, not verdicts to apply
        │   - Fit: weights @ Q4 + KV vs 24 GB/board (fit or split?)
        │   - Quant: in our GGML enum? if not, requantizable from bf16 source?
        │   - Format: directly loadable, or CONVERTIBLE? (MLX→safetensors→GGUF — trace
        │     before rejecting; the arch underneath may already be supported)
        │   - Context: practical, or runnable at reduced ctx?
        │   - "Out of scope" is written only AFTER a trace shows a dead-end
        │
        ├─► Step 5: Classify + route
        │   ┌────────────────────────────────────────────┬───────────────────────────┐
        │   │ Already supported (arch registered + run)   │ Smoke test only (/test)   │
        │   │ Variant (MoE/size sibling of a reg. arch)   │ One small "extend" issue  │
        │   │ Renamed/stripped from upstream              │ One small "restore" issue │
        │   │ Reachable via conversion (fmt/quant gap,    │ Trace conversion path;    │
        │   │   arch already supported)                   │ never drop on sight       │
        │   │ Genuinely new arch (no match anywhere)      │ Full plan via /plan       │
        │   └────────────────────────────────────────────┴───────────────────────────┘
        │
        └─► Step 6: Report
            One paragraph: classification + arch string + evidence + routing decision.
            Do NOT create implementation issues until the classification is stated.

---

## EXAMPLE

    /model-support qwen3.6

**Agent finds:**
- HF config: `Qwen3_5ForConditionalGeneration` / `qwen3_5` (27B), `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` (35B-A3B)
- Our registry: `model.Register("qwen35", New)` only
- Upstream: `qwen3next` package registers `qwen35`, `qwen35moe`, `qwen3next`

**Agent reports:**

    27B → "Already supported" (arch qwen35, registered + validated #107). Smoke test only.
    35B-A3B → "Renamed/stripped": upstream registers qwen35moe; our fork dropped it.
              MoE `sparse` code is present but unreachable. One small restore issue.
    Not a from-scratch port. No qwen36 package needed.

---

## API Notes

- HF `config.json` is authoritative for architecture; the release name is marketing.
- Upstream comparison via `gh api` — must be authenticated (`gh auth status`).
- Output of this command feeds `/plan` (only for genuinely-new) or a single small issue.
```
