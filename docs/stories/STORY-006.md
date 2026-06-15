# STORY-006: The synced qa-workflow tooling fits ollama37 and is trustworthy

## User Story

As a maintainer (or agent) using ollama37's `.claude` tooling,
I want the synced qa-workflow and CI skills/commands to match how this repo actually works,
So that I can trust and run them without hitting dangling references, contradictions, or steps that assume infrastructure that isn't here.

## The Need

ollama37 recently synced its qa-workflow and test-framework tooling from an upstream template.
The sync brought the agent-facing commands, rules, and skills, but several of them describe a
setup ollama37 doesn't have or a testing model it doesn't use — so following them leads to dead
ends. Concretely, the tooling currently: points at files and scripts that don't exist here;
describes a CI structure and a runner flag the repo doesn't use; carries a design skill written
for a different kind of test suite than ollama37's; and contains a command set that contradicts
its own rule about how big the workflow is. A maintainer or agent who trusts these as-is wastes
time discovering, one failure at a time, that the instructions don't apply. The tooling should
either fit this repo or honestly say what isn't wired yet — nothing should silently mislead.

Phase 1 (STORY-005) already made the authoring half real (the format contract + the test docs);
this is the cleanup that makes the rest of the synced tooling coherent.

## Success Looks Like

- Every synced command, rule, and skill either applies to ollama37 as written, or is trimmed/
  removed/marked so it no longer claims something untrue.
- No dangling references: anything a synced file points to (a file, a script, a workflow, a
  command flag) exists, or the reference is dropped.
- No internal contradiction: the qa-workflow rule and its commands agree on the same scope.
- A design skill that doesn't match ollama37's testing model is adapted to it or set aside, not
  left as a misleading rule.
- Where capability genuinely isn't built yet (e.g. automated binding/freshness checks), the
  tooling says so plainly rather than implying it works.
- A maintainer/agent can read any piece of the synced tooling and act on it without hitting a
  step that can't be followed.

## Open Questions

- Whether to *port* the missing automation (the binding/freshness scripts) or just *document*
  that it isn't wired yet — settled per item when the work is planned; porting may be its own
  separate effort.
- Whether the contradictory command set should be reconciled toward the lighter or the heavier
  shape — decided against how ollama37 actually intends to use it.
- Which synced pieces are worth keeping at all versus removing as not-applicable.

## Status

- **Completed: 2026-06-16** — all tasks merged (#268, #269, #270). The synced
  qa-workflow/CI tooling now fits ollama37 or honestly states what isn't wired.
- Created: 2026-06-16
- Plan: #264
- Issues: #265, #266, #267
