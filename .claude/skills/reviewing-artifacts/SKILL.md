---
name: reviewing-artifacts
description: |
  Reviews any workflow artifact — the commands, skills, and project docs that are the
  tooling (READMEs, stories, CLAUDE.md, and the like) — against five goal questions:
  one clear job, complete, a goal not a frozen spec, fits the project, right for its
  reader. Also runs a producer→review pairing coverage pass that flags any producer
  shipped without a paired review. It judges whether an artifact does its job; how a
  human-read doc (the README, docs/ prose) looks and reads goes to the typography +
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

## The five questions

Ask these of any artifact. The artifact's type shifts which ones bite hardest.

1. **One clear job.** Can you say what this file is for in a sentence? Does everything
   in it serve that one job? Flag sprawl (steps that wander off) and heavy overlap with
   another artifact (could it merge, or go away?).
2. **Complete.** Does it deliver that job end to end — no missing steps, placeholder
   text, or dead instructions that produce nothing? A reader/agent should be able to act.
3. **Goal, not frozen spec — and no hardcoding.** Does it state intent and leave room
   where room belongs, instead of freezing a "how" that will drift? Flag stale paths or
   filenames, magic values that should be derived, rigid step-by-step where a principle
   would do, and references to tools or layouts that have moved.
4. **Fits the project.** Does it match the conventions this project actually uses —
   markdown as the source of truth, plus the tools and layout the repo relies on now —
   rather than a stack it has moved past? Flag coupling to a tool or layout the project
   has genuinely retired or relocated; an integration the project still uses, or a
   deliberate adapter, is not a violation. Cross-references resolve to files that exist.
5. **Right for its reader.** Agent-facing (commands, skills): unambiguous instructions
   the agent can follow. Human-facing (README, story): reads like a person wrote it for
   a person — clear, concrete, scannable.

Where each type leans:

| Artifact | Leans on |
|----------|----------|
| Command / skill | Q3 (no hardcoding), Q5 (agent can follow it) |
| README / user doc | Q5 (reads for a human), Q1 (one clear job) |
| Story | Q3 (goal, not spec) — this is what `dw-review-story` checks at the story stage |
| CLAUDE.md | Q2/Q4 (matches the repo as it actually is — no orphaned references) |

## Producer→review pairing (coverage pass)

A standing rule (CLAUDE.md → "Review pairing"): every **producer** has a **paired
review**. When the scope is the whole workflow — or any change that adds/edits a
producer — run this coverage pass on top of the five questions.

It is a **method, not a fixed list** — derive the producers and reviews from whatever
units exist now; don't hardcode an inventory that will drift.

1. **List the producers.** A producer is any unit that *creates, syncs, publishes, or
   drafts a deliverable* — by name (`create-`, `sync-`, `publish-`, `draft-`, `init-`)
   or by what it does (a producing gerund skill — a name like `planning-…`, `drafting-…`).
2. **List the reviews.** Any unit whose job is to *check a result* — `*-review`,
   `*-verify`, the `reviewing-*` skills, a typography/format audit.
3. **Match each producer to the review that covers its output.** A pairing is real only
   if some review actually inspects what that producer makes.
4. **Flag the gaps.** Name every producer with **no** review covering its output — that
   is a pairing violation. Note the missing review and where it would live.
5. **Mark the exempt.** A producer that yields no outward deliverable to review —
   internal scaffolding, a visual folded into an already-reviewed doc, an authoring
   input, tooling logs — is **exempt**, not a gap. List it as exempt and say why.

Report pairings as a small table and list the unpaired producers as findings:

```
Producer → Review
<producer>           → <review>            ✓
<producer>           → (none)              ✗  needs: <proposed review + home>
<producer>           → (exempt)            —  <why it has no outward deliverable>
```

## Steps

1. **Scope.** A single file, a folder, or "the files I just changed." Find where the
   artifacts actually live in *this* repo — don't assume a fixed layout.
2. **Read** the target(s).
3. **Ask the five questions** of each. Checklists are a floor — note anything else that
   weakens the artifact.
4. **Pairing coverage pass** (when reviewing the workflow or a producer change) — run
   the section above and report unpaired producers.
5. **Report** (below).
6. **Fix (if asked).** Smallest blast radius first: remove leaked hardcoding, fill gaps,
   tighten wording. Structural changes — merging, splitting, or removing an artifact —
   need explicit confirmation. Never delete an artifact without approval; flag it for
   removal instead.

## Report

Per artifact, a short verdict and the specific findings — no numeric score.

```
<artifact path> — PASS | REVISE | CUT

- [Q#] <finding, with line reference> → <smallest fix>
```

- **PASS** — does its job, fits the project, nothing leaked.
- **REVISE** — specific, fixable findings (gaps, hardcoding, drift, readability).
- **CUT** — duplicates another artifact or does nothing useful; propose removal (with approval).

End with the path(s) reviewed and the suggested next step.
