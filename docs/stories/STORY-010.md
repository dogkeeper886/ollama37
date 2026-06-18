# STORY-010: A test verdict explains itself without a debugging session

## User Story

As a maintainer (or agent) running ollama37's capability tests,
I want each verdict to carry the facts that decided it,
So that I can tell why a PASS / FAIL / no-evidence happened without re-running under a debug flag.

## The Need

When a capability test reports a result, the result hides what determined it. A run can report
PASS when the deeper check never actually ran; an empty or "—" result could mean the tool was
denied, never called, returned nothing, or the checker itself failed to start — the report can't
tell you which. The only way to find out today is to re-run under a debug flag and read a firehose
of low-level events, or to add temporary instrumentation by hand. We did exactly that several
times this week. A maintainer should be able to read a verdict and immediately know what happened
and why — not reverse-engineer it.

## Success Looks Like

- Every verdict states the decisive facts behind it: whether the capability under test actually
  exercised the thing it measures (for the MCP test: the tool surfaced, was permitted, was called,
  returned data; for a perf test: the run executed and produced a real measurement) and whether the
  deeper check truly ran or silently fell back.
- A missing or empty result says *why* it's missing, in one glance — not an ambiguous dash.
- A maintainer can diagnose a failed or surprising run from the normal output, without turning on
  a debug flag or editing code.
- There's a sensible middle ground between the terse default and the full debug firehose.
- The capability tools report this consistently, so what you learn from one carries to the others.

## Open Questions

- Where the decisive facts belong — the human-facing report, the machine-readable output, the log,
  or some of each — settled when the work is planned.
- How log verbosity should be structured (levels; what's on by default).
- How far to standardize the "explain yourself" shape across the capability tools vs doing it
  per-tool first.

## Status

- Created: 2026-06-18
- Plan: #297
- Issues: #300, #301, #302, #303
