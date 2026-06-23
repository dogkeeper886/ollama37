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
| Context | 8k / 16k (2k/4k omitted — the ~50-tool menu exceeds those windows) |
| Swept | 2026-06-23 — one model per run, verify-live |

## Tests

- **T1 — list projects:** `List the projects.` · allow `list_projects`
- **T2 — list suites (derived arg):** `List the test suites in the ollama37 project.` · allow `list_projects, list_test_suites`

## Reading the metrics

- **T1 / T2** — per-test verdict: ✅ pass · ❌ fail · 🚫 no tool-call support (model template) · ⏳ not run.
- **Peak/ctx** — highest prompt tokens used vs the context window (truncation risk near 100%); shown for the heavier test (T2).
- **Time(s) / tok/s** — cost of the heavier test (T2); ties directly to the VRAM report's GPU%/offload.
- **Context floor** — the ~50-tool menu alone is large (qwen/gemma tokenizers render it bigger than
  gpt-oss), so 2k/4k are below the floor and not run. Even at 8k the qwen menu sits at ~92–95% of
  ctx, so the chained T2 can truncate — `qwen3.6:35b` passes T2 at 16k but truncates at 8k.

---

## gemma4

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `gemma4:e2b` | — | ⏳ not on host | | | | | |
| `gemma4:e4b` | 8k  | ✅ | ❌ | 6150/8192 | 134.78 | 10.64 | T2 ❌ — same guessed-arg miss |
|              | 16k | ✅ | ❌ | 6150/16384 | 135.68 | 10.65 | T2 ❌ — called `list_test_suites` without first deriving `project_id` (guessed arg) |
| `gemma4:12b` | — | ⏳ not on host | | | | | |
| `gemma4:26b` | 8k  | ✅ | ✅ | 7103/8192 | 184.51 | 8.81 |  |
|              | 16k | ✅ | ✅ | 7103/16384 | 186.42 | 8.79 | 100% GPU on 2.5 image → 8.8 tok/s (the 2026-06-20 1.2 tok/s was pre-#352 CPU offload) |
| `gemma4:31b` | — | ⏳ not on host | | | | | |

---

## qwen3.6

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `qwen3.6:27b` | 8k  | ✅ | ❌ | 7819/8192 | 983.47 | 2.2 | T2 ❌ — same `top-level` confabulation |
|               | 16k | ✅ | ❌ | 7819/16384 | 961.04 | 2.25 | T2 ❌ — cross-check caught unsupported `top-level` (LLM judge passed all 3 stages) |
| `qwen3.6:35b` | 8k  | ✅ | ❌ | 7819/8192 | 474.43 | 6.93 | T2 ❌ — peak 7819/8192 (95%) truncated the chain |
|               | 16k | ✅ | ✅ | 7819/16384 | 473.77 | 6.83 | drives the derived-arg chain correctly |

---

## deepseek-r1

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `deepseek-r1:1.5b` | all | 🚫 | 🚫 | — | — | — | no tool-calling support (deepseek-r1 distill template) |
| `deepseek-r1:7b` | all | 🚫 | 🚫 | — | — | — | no tool-calling support (deepseek-r1 distill template) |
| `deepseek-r1:8b` | all | 🚫 | 🚫 | — | — | — | no tool-calling support (deepseek-r1 distill template) |
| `deepseek-r1:14b` | all | 🚫 | 🚫 | — | — | — | no tool-calling support (deepseek-r1 distill template) |
| `deepseek-r1:32b` | all | 🚫 | 🚫 | — | — | — | no tool-calling support (deepseek-r1 distill template) |

---

## gpt-oss

| Model | ctx | T1 | T2 | Peak/ctx | Time(s) | tok/s | Notes |
|---|---|:--:|:--:|--:|--:|--:|---|
| `gpt-oss:20b` | 8k  | ✅ | ✅ | 4038/8192 | 112.85 | 9.99 |  |
|               | 16k | ✅ | ✅ | 4038/16384 | 113.89 | 9.91 | fastest · most ctx-efficient menu render (~3.8k) |
