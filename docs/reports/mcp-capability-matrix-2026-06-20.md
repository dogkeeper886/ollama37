<!-- A worked example of the MCP capability matrix (see TEMPLATE.md + README.md). Filled with real
     data from this date's runs; T2 (derived-arg) was not run on the multi-server menu, so its cells
     are honestly marked ⏳ not run rather than invented. -->

## 🧪 MCP capability matrix — ✅ 3 passed · 0 failed · ⏳ 3 not run

**Commit:** `79e1e5b6` · **Date:** 2026-06-20 · **Judge:** mixed — verify-live + simple (see provenance)
**Menu:** `testlink + playwright` · **num_ctx:** 8192 · **timeout:** 1800s

### Tests
- **T1 — list projects:** `List the projects.` · allow `list_projects`
- **T2 — list suites (derived arg):** `List the test suites in the ollama37 project.` · allow `list_projects, list_test_suites`

## Matrix

<!-- cell = verdict · stages(tool·query·content) · peak/ctx · time · tok/s · tools
     markers: ✅ pass · ❌ fail · ⚪ no-tool-support · ⚠️ peak >90% num_ctx · ✂️ truncated · ⏳ not run · — not graded -->

| Model | T1 — list projects | T2 — list suites (derived) |
|---|---|---|
| `gpt-oss:20b` | ✅ · ✅·✅·✅ · 3783/8192 · 96.4s · 11.7 · `list_projects` | ⏳ not run |
| `qwen3.5:9b` | ✅ · — (simple) · ⚠️ 7518/8192 · 186.9s · 6.9 · `list_projects` | ⏳ not run |
| `gemma4:e4b` | ✅ · — (simple) · 6796/8192 · 1104s · 1.1 · `list_projects` | ⏳ not run |

## Detail

<details><summary>gpt-oss:20b</summary>

**T1 — list projects — ✅ PASS** (verify-live, all three stages, cross-check grounded)
- **Reason:** Called list_projects myself; ground truth shows exactly two projects: id 1 'ollama37' (prefix ollama37, tc_counter 33, active, is_public) and id 162 'testlink-mcp' (prefix tm, tc_counter 32, active, is_public). The model called the correct single tool with no arguments, and every fact in its table matches the retrieved data.
- **Tool calls:** `list_projects` (ignored all 23 Playwright distractors)
- **Live evidence:**
<pre>[
  { "id": "1",   "prefix": "ollama37", "tc_counter": "33", "is_public": "1",
    "api_key": "[redacted]", "name": "ollama37" },
  { "id": "162", "prefix": "tm",       "tc_counter": "32", "is_public": "1",
    "api_key": "[redacted]", "name": "testlink-mcp" }
]</pre>

**T2 — list suites — ⏳ not run**

</details>

<details><summary>qwen3.5:9b</summary>

**T1 — list projects — ✅ PASS** (simple mode — structural check only, stages not graded)
- **Tool calls:** `list_projects` (ignored all 23 Playwright distractors)
- **Note:** peak prompt 7518/8192 — ⚠️ within 10% of the window.

**T2 — list suites — ⏳ not run**

</details>

<details><summary>gemma4:e4b</summary>

**T1 — list projects — ✅ PASS** (simple mode — structural check only, stages not graded)
- **Tool calls:** `list_projects` (ignored all 23 Playwright distractors)
- **Note:** vision model — slow on the K80 (1.1 tok/s → 1104s).

**T2 — list suites — ⏳ not run**

</details>

## Notes / flags

- ⚠️ **Near context limit:** `qwen3.5:9b` · T1 · 7518/8192 (~92% of num_ctx). It tokenizes the 50-tool menu larger than the others (gpt-oss 3783, gemma4 6796), so **T2 — which appends tool results to the prompt — would likely truncate at 8k**. Chained cases want `num_ctx` 12k–16k for qwen3.5-class models.
- ⏳ **T2 pending:** the derived-arg case hasn't been run on the multi-server menu yet (issue #321 territory — proving the query stage catches a wrong/guessed `project_id`).
- **Provenance:** `gpt-oss:20b` T1 is the **verify-live** CI run 27865608868 (full three-stage rubric + live evidence). `qwen3.5:9b` and `gemma4:e4b` T1 are local **simple-mode** runs (structural check; no stages graded — see `docs/traces/mcp-multiserver-model-comparison.md`). A fully verify-live matrix would re-run all three.
