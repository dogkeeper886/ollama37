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
- **Preconditions:** Ollama up with the model pulled; testlink-mcp reachable with `TESTLINK_URL` / `TESTLINK_API_KEY`.

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
