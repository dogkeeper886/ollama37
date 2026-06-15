# MCP tool-call test — end-to-end validation (STORY-004 / #251)

Validates `cli.ts test-mcp` against a **real** stdio MCP server driving a **live**
backend — no mock. Confirms the harness distinguishes tool-capable models from
those without tool support, and that a tool-capable model's grounded answer is
graded correctly by the keyless agent judge.

- **Date:** 2026-06-15
- **MCP server:** `dogkeeper886/testlink-mcp` (local build `dist/index.js`), pointed at
  a live TestLink via its `.env` (`TESTLINK_URL` / `TESTLINK_API_KEY`, loaded by dotenv).
- **Models:** `gemma4:e4b` (subject), `qwen3:4b` (tool-capable control), `gemma3:4b`
  (no-tool control), on the Tesla K80 stack at `:11434`.
- **Judge:** dual (structural check + keyless ACP agent judge).

## Command

```bash
MCP_CWD=/path/to/testlink-mcp \
npx tsx src/cli.ts test-mcp gemma4:e4b qwen3:4b gemma3:4b \
  --judge \
  --mcp-command node --mcp-args "/path/to/testlink-mcp/dist/index.js" \
  --prompt "List the projects in TestLink." \
  --num-ctx 8192 --output report.json
```

## Result

| Model | Verdict | Tool call | Final answer (grounded in real data) |
|---|---|---|---|
| `gemma4:e4b` | **PASS** | `list_projects` | "The following projects are available in TestLink: 1. **ollama37** (ID: 1) 2. **testlink-mcp** (ID: 162)" |
| `qwen3:4b` | **PASS** | `list_projects` | "Here are the projects in TestLink: 1. ollama37 (ID: 1) 2. testlink-mcp (ID: 162). Both projects are active and publicly accessible." |
| `gemma3:4b` | **NO TOOL SUPPORT** | — | (clean verdict — Ollama: "does not support tools"; not a crash) |

The model decided the call, the **real** testlink-mcp executed `list_projects` against the
live TestLink, the **real** project list (ollama37 / testlink-mcp) came back, and the model
produced a grounded answer. Agent-judge verdict for `gemma4:e4b`:

> The final answer correctly uses the list_projects tool result to answer the prompt, naming
> ollama37 (ID 1) … plus testlink-mcp (ID 162) … the answer reflects the tool data rather than
> contradicting or ignoring it.

For `qwen3:4b` the judge even cross-checked the model's "active and publicly accessible" claim
against the tool result fields (`active=1`, `is_public=1`).

## Bug found and fixed by this validation

The first run **failed** `gemma4:e4b` and `qwen3:4b` even though both produced correct grounded
answers. Root cause: `toTestResult` placed the final answer **after** a ~1560-char tool result,
and the agent judge truncates step stdout to 1000 chars — so the judge never saw the answer and
rejected it as "no final answer". Fixed by leading the judge payload with the prompt + final
answer and capping each tool result, so the answer always survives the judge's window. Re-run →
both PASS (above). This is exactly the kind of harness defect end-to-end validation exists to catch.

## Server-agnostic (no testlink coupling)

Pointing the same host at a different real stdio MCP server needs only flags — no code change:

```bash
npx tsx src/cli.ts test-mcp qwen3:4b \
  --mcp-command npx --mcp-args "-y @modelcontextprotocol/server-everything" \
  --prompt "Use the echo tool to echo exactly: ollama37 works"
# → qwen3:4b PASS, called `echo`, real result "Echo: ollama37 works"
```

## No regression

`npm run build` (strict tsc) passes; `cli.ts list`, `bench-throughput`, and `test-fa` still
load and run. The change only adds `src/mcp/*`, the `test-mcp` subcommand, a `config.ts` block,
and one `host.ts` field — the existing chat tests' logic is untouched.
