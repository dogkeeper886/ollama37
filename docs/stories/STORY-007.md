# STORY-007: Capability test tools are documented and CI-runnable

## User Story

As a maintainer of the K80 fork,
I want each capability/performance test tool to have a readable test doc and be runnable on demand from CI,
So that anyone can see what a tool verifies and trigger it without reading the source.

## The Need

The fork has three on-demand capability tools — a throughput benchmark, a flash-attention
regression, and an MCP tool-call probe. They already work and report the same way (a per-model
result table plus a JSON artifact), but they're only half-wired into how the project tracks
tests: none has a readable test doc, two of the three don't trace back to the need they verify,
and the MCP tool-call probe can't be triggered from CI at all — it only runs by hand on a
developer's machine. So a maintainer can't discover what each tool checks, or run the MCP probe
through CI, without going into the code. The structured suites (build/runtime/inference/models)
already have this — these tools should match.

This is a **backfill**: document what already exists and wire the missing CI trigger. It does
not change how any tool verifies a result.

## Success Looks Like

- Each of the three capability tools has a readable test doc that states what it verifies and
  traces to the need it serves.
- The MCP tool-call probe can be triggered on demand from CI — the same way the throughput and
  flash-attention tools already can — with its credentials supplied securely.
- The test-doc review gate passes over the new docs.
- No change to how the tools judge results — they're documented and wired, not re-designed.

## Open Questions

- How a test doc records its binding to a tool that is a **command/workflow rather than a YAML
  test case** — settled when the work is planned (a small format convention).
- Stronger MCP verification (having the judge independently re-run the tool to check the answer
  against live data) and the broader MCP capability ladder are **separate, later stories**, not
  part of this backfill.

## Status

- Created: 2026-06-16
- Plan: #271
- Issues: #273, #274, #275
