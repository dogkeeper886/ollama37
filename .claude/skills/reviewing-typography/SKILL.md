---
name: reviewing-typography
description: |
  Reviews how a human-read document looks — the README, the prose and tables in docs/, or
  any markdown written for a person — for visual hierarchy, Gestalt proximity, restraint,
  and walls of text, so a reader can find the point at a glance. The look half of the
  human-read doc review (reviewing-phrasing handles the words). Use when such a doc is
  written or restructured. Judgment over checklist; floor, not ceiling.
---

# reviewing-typography

One reviewer for how a human-read document **looks**. Its partner `reviewing-phrasing`
judges how a doc *reads*; this judges how it *scans*. Together they are the human-read doc
review.

This skill's job is **human-read** markdown — the README, the prose, tables, and lists in
`docs/`, and any doc aimed at a person. The **agent-read** workflow tooling — commands,
skills, rules, CLAUDE.md, stories — is **not** this skill's job; whether those do their job
goes to `reviewing-artifacts`.

A markdown doc has no fonts to set, but it has the same levers UI typography uses, and the
same principles decide whether it works: **heading levels** are size/weight, **blank lines
and grouping** are spacing, **bold/italic** are weight, and **paragraph and list length**
decide whether the page reads as structure or as soup.

## Why this is a judgment skill, not a checklist

The right structure depends on the content's actual shape, which no rule can predict. A
metadata line reads fine until real prose lands under it and the two fuse. One bold label
anchors the eye; ten bold labels compete and the hierarchy collapses. A 4-item list reads
cleanly; a 17-item list reads as a paragraph. **Look at the doc and decide** — there is no
rule set that survives contact with arbitrary content.

## What to weigh

These are **lenses, not steps**. Weigh them together; flag anything that weakens the look
even if it isn't named here.

- **Hierarchy.** Can the eye find the point of focus? A doc with no heading structure reads
  as one undifferentiated blob — the reader has no idea what they're looking at. The title
  and section heads should be visibly heavier than the body; each level distinct from the
  next.
- **Proximity (grouping).** Spacing groups or separates. Related lines sit together; a real
  break — a blank line, a heading, a rule — sits between things that aren't one group.
  Where ideas cross layers (metadata → prose, intro → first section) they need the loosest
  separation, or they fuse.
- **Restraint.** You need very few levels. A handful of heading depths plus weight is enough
  to build any doc; piling on nested headings, or bolding every other phrase, *destroys*
  hierarchy rather than adding it.
- **Emphasis by de-emphasis.** To make something stand out, tone the rest down. If every
  paragraph opens with a bold phrase, none of them anchor anything. Emphasis is a budget —
  spend it on the few things the reader should land on.
- **Wall of text.** A long undifferentiated paragraph, or an 800-word stretch with no
  heading break, reads as soup. Break at the natural boundary; promote a `**Label:**`
  followed by a long list into a real heading.
- **Structure for the true shape, then for focus.** Use heading levels for the doc's real
  hierarchy — but also use common sense about what the reader should focus on and let that
  stand out. Don't structure for structure's sake; structure for whether the reader can
  *use* the doc.

## The test

Squint at the rendered doc — or scan it without reading any word. The visible weight order
(title → headings → emphasis → body) should be obvious at a glance. If you can't tell the
levels apart unfocused, the hierarchy isn't doing its job. When unsure, show it to someone
and watch where their eye snags or stalls; that spot is the finding.

## Steps

1. **Scope.** Which doc(s). If unclear, ask before reviewing.
2. **Scan it as a reader would** — unfocused first (does the shape read?), then through.
3. **Weigh the lenses** above by judgment; note anything else that hurts the look.
4. **Report** (below).
5. **Fix (if asked).** The smallest change that fixes the look — a heading break, a blank
   line, removing bold from the labels that aren't anchors. Don't restructure what already
   reads well; don't invent findings on a clean doc (it trains everyone to ignore the real
   ones).

## Report

Per doc, a short verdict and specific findings — no numeric score.

```
<doc> — PASS | REVISE

- <what hurts the look, with the location> → <smallest fix>
```

- **PASS** — the eye finds the point; hierarchy and grouping hold.
- **REVISE** — specific, fixable findings (no hierarchy, fused groups, bold inflation, wall of text).

End with what was reviewed and the suggested next step.
