---
id: TS-05
title: MCP tool-call capability — can a model drive a real MCP server's tools
namespace: ollama37
story: STORY-004
story_hash: 4e602782b1a4a6813fc9cf3bf0ef58ec85929e78553d68eba81351171282582d
status: green
---

## Why this scenario exists

Chatting is not the same as *driving tools*. This scenario proves whether a given model can take
a real MCP server's tool menu, pick the right tool with valid arguments, run it against the live
backend, and use the result in its answer — and that the harness reports a model that *can't* do
tools cleanly rather than crashing. It is the capability [STORY-004](../stories/STORY-004.md)
delivers. Bound to the `cli.ts test-mcp` subcommand (no YAML), so the Steps below describe its
logical flow.

### TC-01: a tool-capable model drives `list_projects` on the real testlink-mcp

- **Objective:** the model calls the right tool against the live server and produces a grounded answer.
- **Script:** `cli.ts test-mcp <tool-capable-model> --judge` (default server: testlink-mcp)
- **Preconditions:** Ollama up with the model pulled; testlink-mcp reachable with `TESTLINK_URL` / `TESTLINK_API_KEY`. Locally these are read from `cicd/tests/.env` (copy `cicd/tests/.env.example`); in CI they come from the matching GitHub secrets.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Run the host against the real testlink-mcp | the server is spawned and its tools are listed + translated to Ollama tools |
| 2 | The model receives the menu + the prompt ("List the projects") | it emits a `list_projects` tool call with valid args (a real tool, required args present) |
| 3 | The host executes the call against live TestLink | a real project list comes back (e.g. ollama37, testlink-mcp), `isError=false` |
| 4 | The model produces a final answer from the result | a non-empty answer that names the real projects |
| 5 | Judge the trajectory | simple check **and** the agent judge both PASS |

### TC-02: a model with no tool support is reported cleanly

- **Objective:** a model whose template can't do tools yields a clean verdict, not a crash.
- **Script:** `cli.ts test-mcp <no-tool-model>` (e.g. gemma3:4b)

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Run the host with a no-tool model | Ollama responds "does not support tools" |
| 2 | Report the verdict | **NO TOOL SUPPORT** — a clean per-model verdict, not a crash, and it does not redden the exit code on its own |

### TC-03: server-agnostic — the same host drives a different MCP server

- **Objective:** the host works against any stdio MCP server — no testlink coupling.
- **Script:** `cli.ts test-mcp <tool-capable-model> --mcp-command <cmd> --mcp-args <args>`

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Point the host at a second real stdio MCP server (e.g. an echo server) via the flags | the host lists + drives that server's tools, no testlink involved |
| 2 | The model calls one of its tools and uses the result | tool executed against the real server; grounded answer → PASS |

### TC-04: `--verify-live` — an independent keyless verifier checks the answer against live data

- **Objective:** the verifier itself calls the read-only tool against the live server and confirms the model's answer, keyless and read-only.
- **Script:** `cli.ts test-mcp <tool-capable-model> --verify-live --verify-allow list_projects --verify-server-name testlink`
- **Preconditions:** as TC-01, plus a keyless login in `~/.claude` (no `ANTHROPIC_API_KEY`); in CI the `CLAUDE_CODE_OAUTH_TOKEN` secret.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | The verifier spawns the ACP agent in the isolated config dir | only the injected `mcp__testlink__*` tool surfaces — the account connectors are gone (flood cleared) |
| 2 | The verifier calls the allow-listed `list_projects` itself | the read-only gate ALLOWS it; live data comes back and is captured |
| 3 | A non-allow-listed / write tool is attempted | the gate DENIES it (fail-closed) — the verifier can never write |
| 4 | The verdict is reported | PASS only if the answer matches the live data; the **Live evidence** column shows the captured result |
| 5 | The verifier can't authenticate or start | the run **fails closed** (FAIL), never a silent downgrade to the structural check |

### TC-05: derived-argument tool use — the model must work out a tool argument ([STORY-012](../stories/STORY-012.md))

- **Objective:** the model must pick the right tool *and* supply a **derived** argument, so the verifier's **query** stage does real work — unlike the no-argument `list_projects`, where "queried correctly" is a no-op. The model is asked by project *name* and must resolve it to the id itself.
- **Script:** `cli.ts test-mcp <tool-capable-model> --prompt "List the test suites in the ollama37 project." --verify-live --verify-allow list_projects,list_test_suites --verify-server-name testlink`
- **Preconditions:** as TC-04, plus the testlink `ollama37` project has test suites (Build/Inference/Runtime/Models Tests). Chained tool use is slow on the K80 (~8 min end-to-end); use a generous `--timeout`.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | The model is asked by **name** ("the ollama37 project") — the id is never given | it must *derive* the id, not copy it from the prompt |
| 2 | The model drives the chain | `list_projects` (finds ollama37 = id 1) → `list_test_suites(project_id="1")` with the derived id and exact arg key |
| 3 | The verifier grades the three stages | tool selection (right tools) ✅, **query (correct derived `project_id`)** ✅, interpretation (answer matches the real suites) ✅ |
| 4 | A model guesses/uses a wrong `project_id`, or skips the lookup | FAILs on the **query** stage — no longer an automatic pass (the failure the no-arg case can't catch) |
