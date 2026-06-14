---
name: reviewing-phrasing
description: |
  Reviews the words of a human-read document — the README, the prose in docs/, or any
  text written for a person rather than an agent — for whether the phrasing fits its
  reader: leads with the point, stays brief, carries the right tone, and says the true
  complete thing and only that. The words half of the human-read doc review
  (reviewing-typography handles the look). Use when such a doc is written or updated and
  is about to reach a reader. Judgment over checklist; floor, not ceiling.
---

# reviewing-phrasing

One reviewer for the **words** of any document written for a person. Its partner
`reviewing-typography` judges how a doc *looks*; this judges how it *reads*. Together
they are the human-read doc review.

This skill's job is **human-read** text — the README, the prose inside `docs/`, and any
text aimed at a person. The **agent-read** workflow tooling — commands, skills, rules,
CLAUDE.md, stories — is **not** this skill's job; whether those do their job goes to
`reviewing-artifacts`.

## Why this is a judgment skill, not a checklist

Good phrasing can't be frozen into a rule. The sentence that's right in a quickstart is
wrong in a design rationale; a line that lands for an engineer loses a newcomer. There is
no fixed order and no score — read the doc as its **actual reader** would and decide what
helps and what hurts. Use common sense.

## What to weigh

These are **lenses, not steps** — they interact, and which bites hardest depends on the
doc in hand. Weigh them together, and flag anything that weakens the writing even if it
isn't named here.

- **Reader.** Who actually reads this, and what do they already know? The phrasing meets
  them there — no unexplained jargon for a newcomer, no over-explaining to a peer.
- **Lead with the point.** The reader should hit what matters first, not after a runway of
  setup. Context that arrives before the point reads as not knowing the priority — or as
  hiding something. Front-load the conclusion; let the detail follow for those who want it.
- **Brevity.** Less is more. Every word earns its place; cut filler, hedging, and the
  second sentence that restates the first. A reader skimming a long file rewards a doc that
  respects their time.
- **Content.** Does it say the true, complete thing the reader needs — and only that? No
  vague filler, nothing burying the one fact that matters, no gap that forces the reader to
  go ask the obvious. Brevity is not omission: precise, not contextless.
- **Tone.** Does it fit the relationship and the moment? A quickstart, a caveat, and a
  rationale each carry a different register.
- **Purpose.** The doc exists to move one outcome — get someone set up, understood, or
  unblocked. Do the words drive *that*, or wander off it?

## The test

When unsure whether a passage lands, read it as the reader would — or show it to someone
who isn't in your head. Where they get lost or lose interest is where the words are
redundant, buried, or missing a hook. That spot is the finding.

## Steps

1. **Scope.** Which doc(s), and who the reader is. If the reader isn't clear from the doc
   or its context, ask before reviewing.
2. **Read it as that reader would** — once, straight through, for whether it lands.
3. **Weigh the lenses** above by judgment; note anything else that hurts the phrasing.
4. **Report** (below).
5. **Fix (if asked).** The smallest change that fixes the phrasing — keep the author's
   voice, don't rewrite what already works, don't pad. A clean doc gets said so, with no
   invented findings.

## Report

Per doc, a short verdict and specific findings — no numeric score.

```
<doc> — PASS | REVISE

- <what hurts the phrasing, quoted> → <smallest fix>
```

- **PASS** — fits its reader, leads with the point, says the right thing.
- **REVISE** — specific, fixable findings (buried point, padding, wrong register, missing fact).

End with what was reviewed and the suggested next step.
