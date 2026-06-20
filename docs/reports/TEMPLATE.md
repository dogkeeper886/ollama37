<!-- MCP capability matrix — blank form. One row per (model × test). Copy to
     docs/reports/mcp-capability-matrix-YYYY-MM-DD.md and fill the {{placeholders}}
     from the test-mcp JSON output. See README.md for the field map and the commands
     that produce the data. Snapshot form: one campaign = one report. -->

## 🧪 MCP capability matrix — {{✅|❌|⚪}} {{N}} passed · {{M}} failed

**Commit:** `{{sha}}` · **Date:** {{YYYY-MM-DD}} · **Judge:** {{verify-live | dual | simple}}
**Menu:** `{{server-a}} + {{server-b}}` · **num_ctx:** {{16384}} · **timeout:** {{3600}}s

### Tests
- **T1 — list projects:** `{{prompt}}` · allow `{{list_projects}}`
- **T2 — list suites (derived arg):** `{{prompt}}` · allow `{{list_projects, list_test_suites}}`

## Results

<!-- One row per (model × test). markers: ✅ pass · ❌ fail · ⚪ no-tool-support
     · ⚠️ peak >90% num_ctx · ✂️ truncated (peak==ctx) · ⏳ not run · — not graded -->

| Model | Test | Verdict | Stages (tool·query·content) | Cross-check | Peak/ctx | Time (s) | tok/s | Tool calls |
|---|---|:--:|:--:|:--:|--:|--:|--:|---|
| `{{model}}` | T1 projects | {{✅}} | {{✅·✅·✅}} | {{✅ grounded}} | {{peak/ctx}} | {{t}} | {{tps}} | {{list_projects}} |
| `{{model}}` | T2 suites | {{❌}} | {{✅·✅·✅}} | {{❌ <claim>}} | {{peak/ctx}} | {{t}} | {{tps}} | {{list_projects, list_test_suites}} |

## Detail

<!-- One <details> per model; T1 and T2 inside. -->

<details><summary>{{model}}</summary>

- **T1 — {{verdict}}** — {{the verifier's reason}}. Tool calls: {{list_projects}}.
- **T2 — {{verdict}}** — {{the verifier's reason}}. Tool calls: {{list_projects → list_test_suites(project_id)}}.

<pre>{{captured live evidence — secrets must be [redacted]}}</pre>

</details>

## Notes / flags

<!-- Pull the cells that need eyes up here; omit a bullet when it doesn't apply. -->

- ⚠️ **Near limit / ✂️ truncation:** {{model · test · peak/ctx}}
- ❌ **Failures:** {{model · test — why (which stage / cross-check)}}
- **Provenance / caveats:** {{CI run ids; which cells are verify-live vs simple; what's pending}}
