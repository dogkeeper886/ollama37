# STORY-009: The MCP capability verdict is trustworthy — checked against live data, keyless

## User Story

As a maintainer (or agent) running ollama37's MCP capability test,
I want the verdict to be backed by an independent check against the real server's live data,
So that a PASS means the model's tool answer is actually true — not just that it called a tool and produced well-formed prose.

## The Need

`test-mcp` proves a model can drive a real MCP server's tools. But "drove a tool and answered"
is not the same as "answered correctly." The structural check confirms a tool call happened; the
agent judge grades whether the prose looks grounded in the captured trajectory. Neither one goes
back to the live server to confirm the answer is *true*. A model that calls `list_projects` and
then confidently names projects that don't exist can still look like a PASS.

What's missing is an independent verifier that retrieves the ground truth itself and confirms the
answer against it. For this to be usable it has to hold three properties at once, and that
combination is what made it hard:

- **Keyless** — it must run in CI on the subscription login, never a paid `ANTHROPIC_API_KEY`.
- **Read-only and safe** — it touches a live backend, so it must never be able to write or mutate.
- **Honest** — if it can't actually verify, it must say so, not quietly fall back to a weaker check
  and report green.

## Success Looks Like

- A verify-live verdict means an independent checker retrieved the real data and confirmed the
  model's answer matches it; a contradicted or invented answer comes back FAIL.
- The check runs keyless — no Console API key — both locally and in CI.
- The verifier can only ever read: its tool access is an exact, fail-closed allow-list, so it can
  never write to the real backend even if it tried.
- The report shows the live data the verdict was based on, so a human can see the ground truth and
  not just the verdict.
- When the verifier can't run (can't authenticate, can't reach the server), the run fails closed —
  it never downgrades silently to the structural check and calls it a pass.
- It runs on demand from CI the same way the other capability tools do.

## Open Questions

- Whether the verifier should call the tool *itself* (agent-calls-tool) or have the client
  pre-fetch the data and only grade the answer (client-calls-tool) — settled in favour of
  agent-calls-tool once the blocker below was understood.
- How to keep the spawned agent's toolset clean enough that the one injected server is actually
  reachable — the operator's account connectors and the repo's own config were flooding it.
- How to stay keyless without a credential that silently expires mid-run.

## Status

- **Completed: 2026-06-18** — delivered via #292 (merged in #293). The verifier spawns its agent in
  an isolated, connector-free config dir so the injected server surfaces, calls the read-only
  allow-listed tool against live data, surfaces that data as report evidence, and fails closed when
  it can't run. Keyless via a symlink to the live login token (CI uses `CLAUDE_CODE_OAUTH_TOKEN`);
  wired into `test-mcp.yml` as `judge_mode: verify-live`.
- Plan B (client-calls-tool, #290) was explored and **superseded** by the agent-calls-tool design
  above. The earlier dead-end issues (#284–#288) were closed not-planned before the root cause —
  config pollution, not a binary limitation — was found.
