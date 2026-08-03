---
paths:
  - ".claude/commands/**/*.md"
  - ".claude/skills/**/*.md"
---

# agent-report

Every unit in this toolkit ends at a **human gate**. What that human reads to decide is
the **report**. This states what a report owes its reader.

The report is where the workflow's cost actually lands: producing is cheap, judging is
not. A report that has to be re-read from the top makes the human the slowest part of a
pipeline built entirely around their judgment.

This file states the **goal**. The words a report uses — the verdicts, the section
names, the marker for an empty section, what each medium allows — are **values**: see
`project-profile.md` → Reports. A unit resolves them from there; it does not name its
own.

## What a report answers

Seven questions, in this order. The reader should find each without hunting.

```
   1  What am I being asked to decide?   ──► the verdict, first, alone on a line
   2  What is wrong, and where?          ──► the findings
   3  What does this verdict cover?      ──► what was actually examined
   4  What was left out on purpose?      ──► a CHOICE
   5  What is still uncertain?           ──► a RISK
   6  Where are the artifacts?           ──► the trace
   7  What happens next?                 ──► exactly one step
```

## The rules that make it work

**Lead with the decision.** Question 1 is answerable without reading the report. A
verdict that arrives after a paragraph of narration has already cost the reader the
thing the report exists to save.

**4 and 5 are not the same question.** Something skipped on purpose is a *choice*;
something unresolved is a *risk*. Collapsing them is the failure this contract exists to
prevent — it is how a workaround and a concern end up reading alike.

**Silence is not an answer.** A section with nothing to report says so. An absent
section cannot be told apart from a question never asked, and a reader cannot tell
whether an area was clean or simply never looked at.

**Say what the verdict covers.** A pass is scoped to what was examined (3). Without it a
reader assumes the scope, and assumes wrong in the direction of comfort.

**Structure before prose.** A verdict is a line; findings are a table; a flow, a
pipeline, or a shape is a diagram; options compared are a table with a row each. Prose is
for an argument that has to be *followed* — and then it is short. Rebuilding structure
out of paragraphs in your head *is* the review cost.

**Be legible where it lands.** A report read in a chat session must work as plain text.
One that lands in a document or an issue may use whatever renders there. The formats each
medium allows are a value — see `project-profile.md` → Reports. Do not bake one medium's
markup into a unit; a project that reports elsewhere would have to edit it.

## Right-size it

Not every answer is a report. A one-line reply, a confirmation, a single fact a reader
asked for directly — these are the answer, and wrapping them in seven sections is ritual,
not rigor. The contract binds a unit's **gate output**: what a command or skill prints
when it finishes and hands a decision to a human. Below that, use judgment.

## Why this is a rule, not a template

A frozen block of headings is a *procedure* — no profile value can override it, so a
project that reports differently would be forced to edit a shipped unit and its repo
would then read as drifted when it was only working around us (see
`project-profile.md` → what belongs here vs. not). State the questions; let the profile
name the words.
