<!-- MCP capability matrix — blank form. Copy to docs/reports/mcp-capability-matrix-YYYY-MM-DD.md
     and fill the {{placeholders}} from the test-mcp JSON output. See README.md for the field map
     and the commands that produce the data. Snapshot form: one run = one report. -->

## 🧪 MCP capability matrix — {{✅|❌|⚪}} {{N}} passed · {{M}} failed

**Commit:** `{{sha}}` · **Date:** {{YYYY-MM-DD}} · **Judge:** {{verify-live | dual | simple}}
**Menu:** `{{server-a}} + {{server-b}}` · **num_ctx:** {{8192}} · **timeout:** {{1800}}s

### Tests
- **T1 — list projects:** `{{prompt}}` · allow `{{list_projects}}`
- **T2 — list suites (derived arg):** `{{prompt}}` · allow `{{list_projects, list_test_suites}}`

## Matrix

<!-- One row per model. cell = verdict · stages(tool·query·content) · peak/ctx · time · tok/s · tools
     markers: ✅ pass · ❌ fail · ⚪ no-tool-support · ⚠️ peak >90% num_ctx · ✂️ truncated (peak==ctx) · ⏳ not run · — not graded -->

| Model | T1 — list projects | T2 — list suites (derived) |
|---|---|---|
| `{{model}}` | {{✅}} · {{✅·✅·✅}} · {{peak/ctx}} · {{time}}s · {{tps}} | {{⏳ not run}} |

## Detail

<!-- One <details> per model; nest its T1 and T2 inside. -->

<details><summary>{{model}}</summary>

**T1 — list projects — {{verdict}}**
- **Reason:** {{the verifier's own reason}}
- **Tool calls:** {{list_projects}}
- **Live evidence:**
<pre>{{captured live data — secrets must be [redacted]}}</pre>

**T2 — list suites — {{verdict}}**
- **Reason:** {{…}}
- **Tool calls:** {{list_projects → list_test_suites(project_id="1")}}
- **Live evidence:**
<pre>{{…}}</pre>

</details>

## Notes / flags

<!-- Pull the cells that need eyes up here; omit a bullet when it doesn't apply. -->

- ⚠️ **Near context limit:** {{model · test · peak/ctx}}
- ✂️ **Truncation:** {{model · test — the menu didn't fit num_ctx}}
- ⚪ **No tool support:** {{model}}
- **Provenance / caveats:** {{which cells are verify-live vs simple; what's pending}}
