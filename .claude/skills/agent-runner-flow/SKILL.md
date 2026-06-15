---
name: agent-runner-flow
description: |
  Integration test design principles for a connected end-to-end flow. Use when
  writing, reviewing, refactoring, or adding test cases under cicd/tests/ (the YAML
  flow and the runner). Enforces one self-contained, connected flow with no
  hardcoded or standalone tests. The principles are project-agnostic; the section at
  the end is a fill-in template for your project's concrete flow.
---

# Integration Test Flow

## The goal (non-negotiable)

The integration tests are **one self-contained, connected end-to-end flow** — not a
bag of independent tests. A run provisions its own fixtures, threads them through
every stage, and tears them all down, leaving the backend as it found it. It must
pass against a **fresh** backend (the only precondition being that the backend's API
is reachable), repeatably.

Mechanics — the YAML schema and how to run — live in `.claude/rules/test-yaml-format.md`
and the runner's README. This skill is the *why* and the *rules*.

## The five rules (portable to any project)

1. **Embed every test in the flow — never an island.**
   A new test consumes fixtures produced upstream and declares `dependencies` on the
   stage that produced them. It does **not** bootstrap its own project/parent
   objects. If it creates a fixture, that fixture is removed in the shared teardown
   stage, not by the test itself.

2. **No hardcoded instance IDs.**
   Never write a real backend ID into a test. Every ID is created at runtime by a
   stage and threaded downstream. The only allowed literals are values that are
   portable across any fresh spin-up (e.g. a built-in default account).

3. **No IDs in names; no random/timestamp suffixes.**
   Fixtures are **stable named test data**. Do not embed a runtime id or a
   random/`$$`/timestamp value into a name to get uniqueness — use a stable name with
   **idempotent reuse-or-create** (look up by name, reuse if present, else create).
   This survives a fixture leaked by a failed run.

4. **Everything is connected — no parallel islands.**
   Fixtures must relate to each other, mirroring the system's real graph. A new
   entity that doesn't connect to the chain is not embedded — link it.

5. **Teardown leaves the backend clean.**
   The final stage removes everything the flow created and depends on every
   fixture-consuming test so it runs last. It verifies removal where it can (e.g. a
   read-back reports the entity gone).

## Checklist for a new/edited test

- [ ] `dependencies` point at the stage(s) producing the IDs it reads
- [ ] Reads fixture IDs from the shared hand-off — zero hardcoded instance IDs
- [ ] Any fixture it creates has a **stable name** + idempotent reuse-or-create
- [ ] Any fixture it creates is published to the hand-off and removed in teardown
- [ ] The new entity is **linked into** the connected graph, not standalone
- [ ] Steps emit a marker for `expectPatterns`; parse structured responses rather
      than dumping raw output (keeps the deterministic judge's error scan clean)
- [ ] The runner still passes against a fresh backend, twice (idempotent)

## Anti-patterns (reject these)

- ❌ A "self-contained" test that creates *and* tears down its own parent objects in
  isolation — duplicates setup and breaks the shared lifecycle. Embed it instead.
- ❌ Hardcoded instance IDs, or IDs/random values baked into fixture names.
- ❌ An entity created but never connected to the flow's chain.
- ❌ A fixture created with no corresponding teardown (leaks across runs).

---

## This project (fill this in)

> **Replace this block with your project's concrete flow.** The framework ships
> example cases under `cicd/tests/testcases/{build,integration,e2e}/` — standalone
> illustrations of the YAML schema, **not** a connected flow. Once your integration
> tests form a real end-to-end chain, describe its instantiation here so the rules
> above have a concrete referent:
>
> - **Hand-off:** how stages publish and read fixture IDs (e.g. files under
>   `/tmp/<your>-flow/`).
> - **Stable fixtures:** the named test data each stage creates idempotently
>   (reuse-or-create), and any portable literal (e.g. a built-in default account).
> - **The connected graph:** how the fixtures relate, mirroring your system's real
>   graph — what *covers*, *belongs to*, or *records* what.
> - **Teardown:** the final stage that removes everything the flow created and
>   depends on every fixture-consuming test so it runs last.
> - **Judge hygiene:** steps `echo` a marker for `expectPatterns` and parse
>   structured responses rather than dumping raw output, so the simple judge's error
>   scan stays clean.
> - **Run:** the command and the requirement that it passes twice in a row against a
>   fresh backend.
>
> For a worked example of a filled-in block — a `s1 → … → s7` TestLink flow with a
> `/tmp/tl-flow` hand-off and a teardown stage — see `testlink-mcp`, the sibling repo
> this skill was ported from.

**Backporting upstream:** keep "The goal", "The five rules", the checklist, and the
anti-patterns verbatim; replace only this section with the target project's concrete
fixtures, hand-off mechanism, and teardown stage.
