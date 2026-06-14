# STORY-003: A README that explains the project at a glance and names its hardware lock

## User Story

As someone landing on the ollama37 repo for the first time,
I want a README that follows README best practices and explains each key idea with a visual,
So that I quickly understand what this project is, how to use it, and that its binary runs only on sm_37 hardware — before I invest any time.

## The Need

The README is the front door to ollama37. A newcomer should be able to grasp what
the project is, why it exists, and how to get started without digging through code or
issues. Two things make today's README fall short:

- It does not follow README best practices, so the key ideas are harder to find and
  absorb than they should be — especially for a visual reader.
- It does not make the most important constraint unmissable: this binary is locked to
  sm_37-only hardware (the Tesla K80 / Kepler target). Someone on different hardware
  can currently waste real time before discovering it won't run for them. This is not
  hypothetical — issue #223 is a user whose P100 crashed running alongside a K80,
  where the shipped build's compute-3.7 lock was the cause and documenting it in the
  README was the agreed fix.

## Success Looks Like

- A first-time reader can tell, within the first screenful, what ollama37 is and who
  it is for.
- Each key idea is paired with a visual (SVG or PNG) that explains it, not just prose.
- The sm_37-only hardware lock is stated clearly and early enough that no one is
  surprised by it after the fact.
- The README reads as following recognized README best practices rather than as an
  ad-hoc document.

## Open Questions

- Which "key ideas" each warrant their own visual, and what those visuals should show.
- Which README best-practice conventions to adopt (structure, sections, badges, etc.)
  and how strictly to follow them.
- How the visuals are produced, stored, and kept in sync with the prose.
- The exact wording and placement of the sm_37 hardware-lock notice.
- Whether a GitHub issue should track this work, and at what granularity.

## Status

- Created: 2026-06-14
- Issues: #223 (origin — README update agreed there), #229 (task)
- Plan: #228
- PR: #230 (open)
