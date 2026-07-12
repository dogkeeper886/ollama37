---
name: codebase-memory-verify
description: |
  Graph-first verify discipline: re-ground in the codebase knowledge graph before AND
  after any non-trivial code change, and especially after a failed build/test or wrong
  output. Re-index the changed repo, re-trace the symbols you touched (callers, callees,
  data flow), and — for a port/adaptation — diff against the reference project, then
  verify against the live file before editing. Confirm any relationship- or
  reachability-based claim (who-calls, what-derefs, which-case-runs) with a graph trace,
  not grep alone. Use when implementing or porting
  non-trivial code, when an attempt fails, or before iterating on a fix. Complements (does
  not replace) the codebase-memory tool reference and codebase-graph-navigation
  IDE-movement skills.
---

# Codebase Memory — Graph-First Verify

The graph is a **snapshot of the last index**. Acting on a stale graph — or on an
assumption you never traced — is how wrong-reference and broken-caller bugs slip in.
Re-ground in the graph at the moments a change is most likely to drift from reality.

This skill is the *discipline* (WHEN to re-ground). The *tools* live in `codebase-memory`
(decision matrix, Cypher, gotchas); the *IDE-movement → tool* mapping lives in
`codebase-graph-navigation`. Don't duplicate them — reach for them from here.

## The loop

**Before** a non-trivial change:
1. **Fresh index?** If anyone edited since the last index, `index_repository` (or
   `detect_changes`) first — a stale graph traces the old code.
2. **Trace what you're about to change:** `trace_path(inbound)` for the blast radius
   (every caller a new precondition or return-shape touches), `trace_path(outbound)` for
   what it depends on, `data_flow` when a value's path matters.
3. **Verify against the live file** before you Edit — confirm the symbol/line the graph
   points at still exists and reads as the graph claims.

**Porting / adapting from a reference:** index BOTH projects, find the symbol in each, and
diff them. Aim for the **minimal delta** vs the reference, not a presumed rewrite — and
confirm the reference you're copying is the one that actually applies to your case.

**After** the change, or **on failure** (build/test fails, output wrong):
1. **Re-index** — your edit made the graph stale.
2. **Re-trace** the same path: did the change land as intended, and do the callers still
   hold?
3. **Re-ground before the next attempt — don't iterate on stale assumptions.** A failed
   attempt is the signal to re-verify the *premise*, not just the code: re-trace the path
   AND re-confirm the reference. This catches the worst class of bug — comparing against a
   source that doesn't apply (e.g. a code path that never runs for your model/config).

## grep/read vs. the graph — when each

`grep`/`read` and the graph answer different questions; a real finding often needs both.

- **grep + read** — *locating and reading text.* Fast for finding a literal string or
  symbol, reading a symbol you've already pinned, checking a comment or constant. It shows
  **text, not relationships**: it can land on the wrong block (several arches share a tensor
  name; several `case`s look alike) and it can't tell you whether a line is ever reached.
- **codebase-memory** (`search_graph` / `trace_path` / `get_code_snippet`) — *confirming
  relationships and reachability.* Use it for who-calls, what-derefs, which-`case`-actually-
  runs, does-this-path-execute, and a symbol's exact boundaries.

**Condition — when grep isn't enough, confirm with a trace.** If your claim rests on a
**relationship or reachability** — a call, a deref, a dispatch, "X is required because Y
uses it", "this path runs for that model/config" — a grep hit is a *lead, not a
confirmation*. Verify it with `get_code_snippet`/`trace_path` before you record or act on
it. A grep that "confirms" a behavioral claim is the classic near-miss: right string, wrong
block (e.g. a `token_embd_norm` hit that turns out to be a *different* arch's case, while
the one that actually loads sits elsewhere and the graph builder is what proves it required).

## Right-size it

Skip this for trivial edits — a typo, a one-line tweak, a comment. The loop earns its cost
on changes where a stale graph or an unverified assumption would actually bite: new code,
ports, cross-cutting edits, and any fix you are about to attempt a second time.
