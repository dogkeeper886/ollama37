# STORY-004: Validate that Ollama models can perform MCP tool calls

## User Story

As a maintainer of the K80 Ollama fork,
I want to test whether a given model can correctly drive a real MCP server's tools,
So that I know which models are usable for tool-calling / agent workloads — not just chat.

## The Need

The test framework (`cicd/tests/`) proves a model can **chat** over HTTP. It never proves a
model can **drive tools** — given a menu of functions, pick the right one with valid
arguments, run it, and use the result. That is a different, increasingly important capability,
and it **varies by model**: this repo already recorded that `gemma3` has no tool support while
`llama3.2` does. Without a dedicated test we can't answer a basic question like *"can `gemma4`
actually call `list_projects` on `testlink-mcp` and use the answer?"* — we'd be guessing.

The check must be **real and end-to-end** (no mocks): the model decides the tool call, a real
MCP server executes it against real data, and the model's final answer is graded on whether it
used that result. It must **not** disturb the existing HTTP chat tests, which work well.

## Success Looks Like

- A maintainer can run a single on-demand check and get a per-model verdict on tool-calling
  capability against a **real** MCP server.
- For a tool-capable model (e.g. `gemma4`/`llama3.2`): the report shows the model called the
  right tool with valid arguments, the real server returned real data, and the final answer
  was graded correct.
- For a model with no tool support (e.g. `gemma3`): the report says so cleanly — a clear
  "no tool support" verdict, not a crash.
- The check works against **any** real MCP server, not just the default — switching servers
  needs no code change.
- The existing chat tests are unchanged and still pass.

## Open Questions

The technical approach has been discussed and will be carried by the plan issue (`/dw-plan`),
where the *how* is decided and recorded. Genuinely unresolved details to settle there / on the
task issues:

- How the default MCP server is launched and how its credentials are supplied (locally vs in
  CI) without hardcoding secrets.
- What exactly counts as a correct tool call for the verdict, and how a "no tool support"
  outcome is classified as a clean result rather than a failure.
- Whether a manual on-demand CI entry point is worth adding now or deferred.

## Status

- Created: 2026-06-15
- Plan: #248
- Issues: #249, #250, #251, #252
