# STORY-012: The MCP test climbs the capability ladder — real tool use, not just the trivial rung

## User Story

As a maintainer of ollama37's MCP capability test,
I want test cases that exercise harder tool use — starting with a tool that takes arguments,
So that the verifier's three-stage rubric actually does work, instead of rubber-stamping the easiest possible case.

## The Need

The MCP test only ever drives one read-only, **no-argument** tool (`list_projects`). We just built
a verifier that grades three stages — tool selection, query, interpretation — but against a single
no-arg tool two of those stages are no-ops: there is only one tool to "select", and "query" is
meaningless when the tool takes no arguments. So a trustworthy judge is rubber-stamping the
trivial rung.

To actually know a model can drive tools — and to know the verifier catches a model that drives
them *badly* — the test has to climb the ladder. The first real rung is a tool that takes
arguments: the model must supply the right ones to get the right result, and the verifier's
"query" stage genuinely passes or fails on whether it did. Higher rungs (choosing the right tool
among many, create-then-verify write chains, multi-step where one result drives the next call)
come later; this story is about taking the first real step off the trivial case.

## Success Looks Like

- At least one test case drives a tool that **takes arguments** against the real server.
- The model has to supply the **correct arguments** to get the right result — a no-arg call won't do.
- The verifier's **query stage genuinely passes/fails** on whether the model queried correctly — no
  longer an automatic pass.
- A model that picks the right tool but calls it with **wrong or invented arguments** is caught: the
  verdict reflects the query failure, not just whether the final answer happened to look right.
- The harder case runs the same way as today's — locally and on the self-hosted CI runner, keyless,
  read-only.

## Open Questions

- Which testlink tool to use for the first arguments case (e.g. a get-by-id or a filtered query) —
  decided when the work is planned.
- Whether to add cases by writing more runner code, or by introducing **declarative test-case
  definitions** (the "MCP YAML" idea) so new rungs can be added without editing the runner.
- The higher rungs — multi-tool selection, CRUD write-chains (which need a write-safe strategy),
  recursive multi-step — sequenced after this first arguments rung.
- How to keep any write-path cases read-only-by-default and safe, given the verifier's exact
  allow-list.

## Status

- Created: 2026-06-19
- Plan: #318
- Issues: #320, #321, #322
