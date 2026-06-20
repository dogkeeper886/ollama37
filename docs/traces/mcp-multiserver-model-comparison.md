# Multi-server MCP test — model comparison (50-tool menu)

Records the first end-to-end runs of the multi-server MCP capability test (issue #324,
[STORY-012](../stories/STORY-012.md)): the model is given a **merged menu of two servers**
and must pick the right tool, ignoring the other server's tools as distractors.

## Setup

- **Servers merged into one menu:** `testlink-mcp` (27 tools, via Docker) + `@playwright/mcp`
  (23 browser tools, via npx) = **50 tools**.
- **Prompt:** `List the projects.` — the correct answer uses testlink's `list_projects`; the 23
  Playwright tools are distractors the model must **not** pick.
- **Mode:** simple (structural check — did it call a real tool, succeed, and answer?). Not yet
  verify-live (the three-stage rubric).
- **Context:** `--num-ctx 8192`. **Timeout:** 1800s (30 min) ceiling. **Hardware:** Tesla K80.
- Date: 2026-06-20.

## Results

| Model | Verdict | Tool called | Peak in/ctx | Time | tok/s |
|---|---|---|--:|--:|--:|
| **gpt-oss:20b** | ✅ PASS | `list_projects` | **3783/8192** | **94.7s (1.6 min)** | **11.36** |
| qwen3.5:9b | ✅ PASS | `list_projects` | ⚠️ 7518/8192 | 186.9s (3.1 min) | 6.89 |
| gemma4:e4b | ✅ PASS | `list_projects` | 6796/8192 | 1104s (18.4 min) | 1.13 |

All three correctly picked `list_projects` and **ignored all 23 Playwright distractors** — the
multi-server tool-selection capability works across models. A correct run never calls a browser
tool, so Playwright never launches Chromium.

## Findings

1. **Prompt token size is tokenizer/template-dependent for the *same* menu** — gpt-oss renders the
   50-tool menu in **3783** tokens, gemma4 in **6796**, qwen3.5 in **7518**. A ~2× spread for
   identical content.
2. **gpt-oss:20b is the best fit** — fastest (1.6 min, 11.36 tok/s) *and* the most context-efficient
   (54% headroom). No near-limit warning.
3. **qwen3.5:9b runs near the 8k ceiling already on the simple case** (⚠️ 7518/8192, ~8% headroom).
   It works well here (GPU 100%, ~70 °C — healthy), but a **chained case** (TC-05, which appends
   `list_projects` *and* `list_test_suites` results to the history) would very likely cross 8192 and
   truncate. The `Peak in/ctx` column flagged this with ⚠️.
4. **gemma4:e4b is slow** (vision-model overhead, 1.13 tok/s → 18.4 min). It's why the per-call
   timeout has to be generous; faster models finish in 1–3 min and never approach the ceiling.

## Recommendations

- **Default model for the multi-server scenario:** `gpt-oss:20b` (fast + efficient). CI currently
  defaults to `gemma4:e4b`.
- **For the next-level chained cases** (distractors + a derived-argument chain): raise context to
  **12288 or 16384**, especially for qwen3.5-class models whose simple-case prompt already sits near
  8k. gpt-oss has the headroom to stay at 8k.
- **Timeout:** 1800s (30 min) ceiling is fine — it only bounds the slow gemma4 path; fast models
  return in seconds-to-minutes well under it.
