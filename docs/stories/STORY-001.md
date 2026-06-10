# STORY-001: Tidy `.claude/commands/` and `.claude/skills/`

## User Story

As a developer working in this repo,
I want `.claude/commands/` and `.claude/skills/` organized with no leftover or duplicate files,
So that when I (or Claude) look up a command or skill, there is one obvious right answer and nothing stale to trip over.

## The Need

The `.claude/` tree had grown over time as the dev workflow evolved (e.g. the `dw-*` naming migration). The result:

- `.claude/commands/dev-workflow/` contained both the new `dw-*` commands and older `plan.md` / `implement.md` / `model-support.md` files.
- `.claude/skills/` and `.claude/references/` contained skill content that is no longer the canonical workflow.

User directive: keep only `.claude/commands/dev-workflow/` and `.claude/rules/`; remove everything else under `.claude/`.

## Success Looks Like

- Under `.claude/`, only `commands/dev-workflow/`, `rules/`, and `settings.local.json` remain.
- Inside `dev-workflow/`, only `model-support.md` plus the `dw-*` suite remain (`plan.md` and `implement.md` are gone as superseded).
- `model-support.md` has no dangling reference to a deleted skill file.
- The `dw-*` workflow is unambiguously the canonical one.

## Status

- Created: 2026-06-10
- PR: (pending)
