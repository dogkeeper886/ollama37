<!-- A worked example of the MCP capability matrix (see TEMPLATE.md + README.md). Filled with
     real data from two verify-live CI runs on the K80: T1 = run 27873966435, T2 = run 27874945352. -->

## 🧪 MCP capability matrix — ❌ 5 passed · 1 failed

**Commit:** `9e04a7ed` · **Date:** 2026-06-20 · **Judge:** verify-live
**Menu:** `testlink + playwright` · **num_ctx:** 16384 · **timeout:** 3600s

### Tests
- **T1 — list projects:** `List the projects.` · allow `list_projects`
- **T2 — list suites (derived arg):** `List the test suites in the ollama37 project.` · allow `list_projects, list_test_suites`

## Results

<!-- markers: ✅ pass · ❌ fail · ⚪ no-tool-support · ⚠️ peak >90% num_ctx · ✂️ truncated · ⏳ not run · — not graded -->

| Model | Test | Verdict | Stages (tool·query·content) | Cross-check | Peak/ctx | Time (s) | tok/s | Tool calls |
|---|---|:--:|:--:|:--:|--:|--:|--:|---|
| `gpt-oss:20b` | T1 projects | ✅ | ✅·✅·✅ | ✅ grounded | 3783/16384 | 95.5 | 11.68 | list_projects |
| `gpt-oss:20b` | T2 suites | ✅ | ✅·✅·✅ | ✅ grounded | 4038/16384 | 105.1 | 11.41 | list_projects, list_test_suites |
| `qwen3.6:27b` | T1 projects | ✅ | ✅·✅·✅ | ✅ grounded | 7518/16384 | 665.0 | 2.33 | list_projects |
| `qwen3.6:27b` | T2 suites | ❌ | ✅·✅·✅ | ❌ "top-level" | 7819/16384 | 979.9 | 2.33 | list_projects, list_test_suites |
| `gemma4:26b` | T1 projects | ✅ | ✅·✅·✅ | ✅ grounded | 6796/16384 | 1485.9 | 1.18 | list_projects |
| `gemma4:26b` | T2 suites | ✅ | ✅·✅·✅ | ✅ grounded | 7103/16384 | 1733.8 | 1.14 | list_projects, list_test_suites |

## Detail

<details><summary>gpt-oss:20b</summary>

- **T1 — ✅ PASS** — called `list_projects`, ignored the 23 Playwright distractors; answer matched the live two-project list.
- **T2 — ✅ PASS** — drove the chain `list_projects` → `list_test_suites(project_id="1")`, deriving the id from the name; answer matched the real suites. Fast and most context-efficient.

</details>

<details><summary>qwen3.6:27b</summary>

- **T1 — ✅ PASS** — correct `list_projects`, grounded.
- **T2 — ❌ FAIL** — drove the chain correctly (the **query stage is ✅** — it derived `project_id` right), and the LLM judge marked all three stages ✅. But the **deterministic cross-check caught an unsupported `"top-level"` claim** in the answer that isn't in the live `list_test_suites` result, and overrode the PASS → FAIL. This is the STORY-011 trust mechanism working: a harness-side fact-check catching a confabulation the model judge waved through.

</details>

<details><summary>gemma4:26b</summary>

- **T1 — ✅ PASS** — correct `list_projects`, grounded. Slow (vision model: 1.18 tok/s → 24.8 min).
- **T2 — ✅ PASS** — drove the derived-argument chain correctly; answer matched the real suites. 28.9 min — inside the 3600s budget.

</details>

## Notes / flags

- ❌ **Failure:** `qwen3.6:27b` · T2 — **not** a query-stage miss (it derived the id correctly); the **deterministic cross-check** rejected an unsupported `"top-level"` claim the LLM judge accepted. The matrix's value is exactly this: T1 can't catch it, and the cross-check catches what the judge model misses.
- **Context:** no truncation — at 16384 the hottest cell is `qwen3.6:27b` T2 at **7819/16384** (48%). Note that at `num_ctx 8192` that prompt would sit at ⚠️ ~95% and the chained T2 could truncate — which is why this campaign used 16384 (qwen/gemma-family tokenizers render the 50-tool menu larger than gpt-oss).
- **Speed:** gpt-oss:20b is far the fastest (1.6–1.8 min/test, 11+ tok/s); qwen3.6:27b ~11–16 min; gemma4:26b 25–29 min (vision overhead). gpt-oss is the best fit for routine runs.
- **Provenance:** verify-live CI runs T1 = `27873966435`, T2 = `27874945352`; menu testlink + playwright; the temp guard was active (peak 67 °C T1, 69 °C T2 — both well under 80). Full per-model evidence is in each run's JSON artifact.
