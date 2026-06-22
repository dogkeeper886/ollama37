# K80 MCP Capability by Model Family

Tool-call capability across families, swept over context — the **capability** axis that pairs with
`k80-vram-by-family.md` (the **speed/fit** axis). Read together: an MCP fail at high ctx with a low
GPU% in the VRAM report is a speed/timeout artifact, not a capability gap.

## Setup

| | |
|---|---|
| Hardware | 4 × Tesla K80 · 11441 MiB/die |
| Menu | `testlink + playwright` (~50 tools) |
| Judge | verify-live (LLM judge + deterministic cross-check) |
| `graphSafetyMultiplier` | 2.5 (default, #352) |
| Timeout | 3600 s |
| Context sweep | 2k / 4k / 8k / 16k |

## Tests

- **T1 — list projects:** `List the projects.` · allow `list_projects`
- **T2 — list suites (derived arg):** `List the test suites in the ollama37 project.` · allow `list_projects, list_test_suites`

## Reading the metrics

- **T1 / T2** — per-test verdict: ✅ pass · ❌ fail · ⚪ menu doesn't fit ctx · ⏳ not run.
- **Peak/ctx** — highest prompt tokens used vs the context window (truncation risk near 100%).
- **Time(s) / tok/s** — cost of the heavier test (T2); ties directly to the VRAM report's GPU%/offload.
- **Context floor** — the ~50-tool menu alone is large (qwen/gemma tokenizers render it bigger than
  gpt-oss). At **2k/4k** the menu likely **exceeds the window** for most models → ⚪, not a capability
  result. MCP is generally only meaningful at **8k/16k**; the low-ctx rows document where the floor is.

---

## gemma4

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `gemma4:e2b` | 2k | | | | | | |
|              | 4k | | | | | | |
|              | 8k | | | | | | |
|              | 16k | | | | | | |
| `gemma4:e4b` | 2k | | | | | | |
|              | 4k | | | | | | |
|              | 8k | | | | | | |
|              | 16k | | | | | | |
| `gemma4:12b` | 2k | | | | | | |
|              | 4k | | | | | | |
|              | 8k | | | | | | |
|              | 16k | | | | | | |
| `gemma4:26b` | 2k | ⚪ | ⚪ | | | | menu floor |
|              | 4k | | | | | | |
|              | 8k | | | | | | ⚠️ ~95% ctx (see 2026-06-20 note) |
|              | 16k | ✅ | ✅ | 7103/16384 | 1733.8 | 1.14 | vision overhead, ~29 min |
| `gemma4:31b` | 2k | | | | | | |
|              | 4k | | | | | | |
|              | 8k | | | | | | |
|              | 16k | | | | | | |

---

## qwen3.6

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `qwen3.6:27b` | 2k | | | | | | |
|               | 4k | | | | | | |
|               | 8k | | | | | | ⚠️ ~95% ctx (T2 may truncate) |
|               | 16k | ✅ | ❌ | 7819/16384 | 979.9 | 2.33 | T2: cross-check caught unsupported "top-level" |
| `qwen3.6:35b` | 2k | | | | | | |
|               | 4k | | | | | | |
|               | 8k | | | | | | |
|               | 16k | | | | | | |

---

## deepseek-r1

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `deepseek-r1:1.5b` | 2k | | | | | | |
|                    | 4k | | | | | | |
|                    | 8k | | | | | | |
|                    | 16k | | | | | | |
| `deepseek-r1:7b`   | 2k | | | | | | |
|                    | 4k | | | | | | |
|                    | 8k | | | | | | |
|                    | 16k | | | | | | |
| `deepseek-r1:8b`   | 2k | | | | | | |
|                    | 4k | | | | | | |
|                    | 8k | | | | | | |
|                    | 16k | | | | | | |
| `deepseek-r1:14b`  | 2k | | | | | | |
|                    | 4k | | | | | | |
|                    | 8k | | | | | | |
|                    | 16k | | | | | | |
| `deepseek-r1:32b`  | 2k | | | | | | |
|                    | 4k | | | | | | |
|                    | 8k | | | | | | |
|                    | 16k | | | | | | |

---

## gpt-oss

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `gpt-oss:20b` | 2k | | | | | | |
|               | 4k | | | | | | |
|               | 8k | | | | | | most context-efficient menu render |
|               | 16k | ✅ | ✅ | 4038/16384 | 105.1 | 11.41 | fastest; best for routine runs |

<!-- 16k rows for gpt-oss:20b / qwen3.6:27b / gemma4:26b seeded from the
     2026-06-20 worked example (CI runs 27873966435 / 27874945352). -->
