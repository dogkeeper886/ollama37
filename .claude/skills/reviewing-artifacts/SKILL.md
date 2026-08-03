---
name: reviewing-artifacts
description: |
  Reviews any workflow artifact — the commands, skills, and project docs that are the
  tooling (READMEs, stories, CLAUDE.md, and the like) — against a few goal questions:
  one clear job, complete, a goal not a frozen spec, fits the project, right for its
  reader, and (for producers) paired with a review. Its pairing question flags any
  producer shipped without a paired review. It judges whether an artifact does its job;
  how a human-read doc (the README, docs/ prose) looks and reads goes to the typography +
  phrasing review. Floor, not ceiling.
---

# reviewing-artifacts

One reviewer for every workflow artifact. It does not score against a fixed checklist
or a published standard — it asks a few questions about whether the artifact does its
job and fits this project, and trusts your judgment for the rest.

**The questions are a floor, not a ceiling.** If something hurts the artifact and
isn't listed below, flag it anyway.

**Scope.** Review whatever artifact you're handed, *by kind* — commands, skills,
READMEs, stories, CLAUDE.md, and anything like them. Don't tie this skill to a fixed
inventory of the current commands and skills; new ones appear and old ones change. The
fine-grained **look and words** of a human-read doc — the README, the prose in `docs/` —
go to `reviewing-typography` (the look) + `reviewing-phrasing` (the words); this skill
judges whether the artifact does its job (Q5 still asks whether a doc serves its reader).

## The questions

Ask these of any artifact. The artifact's type shifts which ones bite hardest.

1. **One clear job.** Can you say what this file is for in a sentence? Does everything
   in it serve that one job? Flag sprawl (steps that wander off) and heavy overlap with
   another artifact (could it merge, or go away?).
2. **Complete.** Does it deliver that job end to end — no missing steps, placeholder
   text, or dead instructions that produce nothing? A reader/agent should be able to act.
   **Says what it reports?** A unit that ends at a human gate is not complete until it
   says what it hands that human. Flag one that finishes silently, one that defines a
   decision it never prints, one that restates the sections of
   `.claude/rules/agent-report.md` inline instead of resolving to it, and one that
   invents verdict words outside `project-profile.md` → Reports. A unit deliberately
   exempt (its output is not a gate) says so — silence is not an exemption.
3. **Goal, not frozen spec — and no hardcoding.** Does it state intent and leave room
   where room belongs, instead of freezing a "how" that will drift? Flag stale paths or
   filenames, magic values that should be derived, rigid step-by-step where a principle
   would do, and references to tools or layouts that have moved.
   **Bakes in a mechanism?** Hardest bite of this question. A value `project-profile.md`
   owns may appear inline as a default (see its "Two wiring styles"), but a *procedure* —
   how drift is detected, how files are laid out — cannot be overridden by any value, so a
   project that does it differently must edit the shipped unit. Flag every "compute X this
   way" or "write it at this path" that the profile cannot redirect; the fix is to state
   the goal and let the profile or the project's layer name the mechanism.
4. **Fits the project.** Does it match the project's conventions as declared in
   `project-profile.md` — the **canonical format** (its source of truth) and the **live
   integrations** listed there — rather than a stack it has moved past? Flag coupling to
   a tool the profile does not list as live (one genuinely retired or relocated); an
   integration the profile lists as live, or a deliberate adapter, is not a violation.
   Cross-references resolve to files that exist. Skills must be flat under
   `.claude/skills/<name>/` — a foldered skill is undiscoverable; a unit that needs
   folders to group is a command, not a skill.
5. **Right for its reader.** Agent-facing (commands, skills): unambiguous instructions
   the agent can follow. Human-facing (README, story): reads like a person wrote it for
   a person — clear, concrete, scannable.
6. **Paired (producers only).** If this artifact *produces or changes* a **deliverable**
   (as `project-profile.md` → Review semantics defines one), does a review cover its
   output? Every producer needs a
   paired review — a standing rule (the `## Producer → review pairing` tables in
   `.claude/rules/*.md`). A producer with no
   review is a gap to flag, not an exception; one that yields no outward deliverable
   (internal scaffolding, an authoring input) is exempt, not a gap. This skill is the
   enforcement arm. Bites only on producers.

Where each type leans:

| Artifact | Leans on |
|----------|----------|
| Command / skill | Q3 (no hardcoding), Q5 (agent can follow it), Q6 if it produces |
| README / user doc | Q5 (reads for a human), Q1 (one clear job) |
| Story | Q3 (goal, not spec) — this is what `dw-review-story` checks at the story stage |
| CLAUDE.md | Q2/Q4 (matches the repo as it actually is — no orphaned references) |

## Steps

1. **Scope.** A single file, a folder, or "the files I just changed." Find where the
   artifacts actually live in *this* repo — don't assume a fixed layout.
2. **Read** the target(s).
3. **Ask the questions** of each. Checklists are a floor — note anything else that
   weakens the artifact.
4. **Frontmatter hygiene (commands/skills).** Flag and recommend removing:
   - `disable-model-invocation: true` — a unit's user-only nature comes from living in
     `.claude/commands/`, not a flag. On a skill the flag just makes it dormant (Claude
     can't auto-invoke it); a genuinely user-only entry point belongs in `.claude/commands/`.
   - a `tools:` / `allowed-tools:` allow-list — legacy baggage that pins the unit to
     specific tools/servers. Drop it so the unit inherits the session's tools.
5. **Report** (below).
6. **Fix (if asked).** Smallest blast radius first: remove leaked hardcoding, fill gaps,
   tighten wording. Structural changes — merging, splitting, or removing an artifact —
   need explicit confirmation. Never delete an artifact without approval; flag it for
   removal instead.

## Report

Report per `.claude/rules/agent-report.md` — the verdict first, findings as a table,
and a section with nothing to report saying so. The verdict vocabulary is
`project-profile.md` → Reports. No numeric score. This review also uses
**CUT**, declared there as the artifact review's own. In this review they mean:

- **PASS** — does its job, fits the project, nothing leaked.
- **REVISE** — specific, fixable findings (gaps, hardcoding, drift, readability).
- **CUT** — duplicates another artifact or does nothing useful; propose removal (with approval).

Each finding names the question it came from (`[Q#]`), the line, and the smallest fix.
Trace carries the path(s) reviewed.
