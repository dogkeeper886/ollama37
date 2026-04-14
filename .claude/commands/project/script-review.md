---
description: Review a video script for spoken delivery — hooks, number overload, transitions, pacing
argument-hint: [script-path]
---

# Script Review — Narration & Flow Audit

Review a video or demo script as if you are the viewer hearing it for the first time. The goal is to catch everything that sounds wrong *spoken aloud* — not what looks wrong on paper.

**Usage:** `/script-review $ARGUMENTS`

---

## Core Principle

A script is not a document. The viewer cannot re-read a sentence, cannot pause on a table, and will click away in 10 seconds if you don't give them a reason to stay. Review everything through that lens.

---

## Agent Instructions

### Step 1 — Read and Map

1. Read the script at `$ARGUMENTS`
2. List each section:
   - Name
   - Estimated spoken time (count ~150 words per minute for narration blocks)
   - One-sentence summary

### Step 2 — Hook Check

Evaluate the **first 3 sentences** the viewer hears:

- Does it state something interesting, surprising, or useful immediately?
- Or does it start with filler? ("Hi everyone, welcome..." / "In this video we'll..." / "Let me start by...")
- A good hook: lead with a result, a number, or a question. Weave in context after.

If the hook is weak, write a replacement.

### Step 3 — Numbers Audit

This is the most common problem in technical video scripts. For every number spoken aloud:

| Rule | Why |
|------|-----|
| **Round it.** "About 10 gigs" not "10,636 MiB" | Listeners can't parse exact numbers at speech speed |
| **Max 2 numbers per comparison.** | After the third number, the listener loses the first |
| **Say the meaning, show the number.** | "Almost 5 times faster" > "42 versus 8.6 tokens per second" |
| **Let the table carry detail.** | Say "full numbers are on screen" instead of reading every row |
| **No unit soup.** | Don't mix MiB, GB, tok/s, and percentages in one breath |

Flag every sentence that reads more than 2 numbers aloud. For each, provide a rewrite.

### Step 4 — Transition Audit

Check the **last sentence of each section** and the **first sentence of the next**:

1. **Bridge exists?** — Does the narrator connect the two topics?
2. **Bridge quality** — A good bridge:
   - Closes the current topic
   - Opens the next with a reason to keep listening
   - Sounds like natural speech

Flag missing bridges and write replacements.

**Bridge patterns:**
- Question: "Those are the numbers. But what does it actually look like running?"
- Consequence: "That's the setup. Now let's see it in action."
- Contrast: "That works for the small model. The big one is a different story."

### Step 5 — Spoken Language Check

Read each narration line as if speaking it aloud. Flag:

| Problem | Test |
|---------|------|
| **Tongue twister** | Hard consonant clusters or awkward rhythm |
| **Repeated "I" starts** | 3+ sentences starting with "I" in a section |
| **Passive voice** | "It is loaded" → "It loads" |
| **Filler opener** | "So basically..." / "Now, one thing about..." |
| **Jargon without setup** | Technical term used before the viewer knows what it means |
| **Long sentence** | Over 25 words — split it |
| **Vague qualifier** | "very interesting" / "really cool" — say what's interesting |

### Step 6 — Screen Direction Check

For each `**Screen:**` direction:

1. Does the narrator explain what the viewer is looking at?
2. Is there dead air? (Screen changes with nothing to say)
3. Is there a mismatch? (Narrator talks about X while screen shows Y)
4. Is the direction specific enough to reproduce? ("Show the page" vs "Scroll to the VRAM column in the benchmark table")

### Step 7 — Pacing Check

1. Estimate total video length from word count
2. Flag sections that are disproportionately heavy (>40% of total time in one section)
3. Flag back-to-back heavy sections (two data-dense sections with no visual break)
4. Suggest reordering if the most compelling content is buried past the halfway mark

---

## Output Format

Use this exact format. It is designed to be scanned quickly — the user reads the verdict and tables, skips to the rewrites they need.

```
## Verdict

[One paragraph: overall assessment. What works, what's the main problem, estimated video length.]

---

## Section Map

| # | Section | Est. Time | Summary | Verdict |
|---|---------|-----------|---------|---------|
| 1 | ...     | 0:00–0:45 | ...     | OK / Fix / Cut / Move |

---

## Hook

**Status:** Strong / Weak
**Current opening:** [first 2-3 sentences]
**Suggested rewrite:** (only if weak)

> [rewritten opening]

---

## Numbers Overload

| Section | Line | Original | Rewrite |
|---------|------|----------|---------|
| ...     | ~L38 | "9295 + 9288 MiB" | "about 9 gigs each" |

---

## Transitions

| Between | Status | Suggested Bridge |
|---------|--------|-----------------|
| S1 → S2 | Missing | > "..." |
| S2 → S3 | OK | — |

---

## Language Fixes

| Section | Original | Problem | Rewrite |
|---------|----------|---------|---------|
| Intro   | "I recently ported..." | I-start x3 | "This release adds..." |

---

## Screen Directions

| Section | Issue | Suggestion |
|---------|-------|------------|
| S2      | "show the release page" — too vague | "Scroll to the 'New Models' section, pause 3s" |

---

## Pacing

[Any reorder suggestions or section weight issues.]

---

## Typos & Errors

| Line | Issue |
|------|-------|
| L133 | `v2.03` should be `v2.0.3` |
```

---

## Phrase Simplification Reference

| Instead of | Say |
|------------|-----|
| "leverages" | "uses" |
| "utilizes" | "uses" |
| "facilitates" | "helps" or "lets you" |
| "what I'm really interested in" | cut — just state the comparison |
| "one thing about these models" | cut — state the thing directly |
| "very interesting" | say what's interesting |
| "let me show you" | "here's" or just show it |
| "now, the interesting part" | cut — if it's interesting, the viewer will notice |

---

## When to Use

- After writing a video script, before recording
- When a script "feels off" but you can't pinpoint why
- After revising a script, as a second-pass quality check
