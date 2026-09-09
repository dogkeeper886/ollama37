# `.sessions/` — the reasoning behind the issues

An issue body is a summary, and summarising is where intent shifts. These are the raw
Claude Code session logs that produced them, copied verbatim and linked from the issue.

**The issue binds on scope, the code binds on reality, the log binds on nothing.** It is
background, never specification — read it to find out *why* a decision was made, not to
find out what is true now. An issue must stand on its own; a reader should never *need*
the log, and should almost always *have* it.

## Why this exists

#482 was filed without one. A later agent picking it up could not tell that:

- `qwen3.5:27b` is stale in the unload list because it was **deleted in a disk cleanup**
  earlier in that same session — not because it was never right;
- acceptance criterion 4 names `gemma4:26b`, `gemma3:4b` and the `gpt-oss` variants as
  missing from the models suite, but the session that wrote it meant *"loaded by other
  **suites**"*. They have no `models` testcase; the real gaps are `gemma4:12b`,
  `lfm2.5:8b`, `qwen3.6:27b` and `qwen3.6:35b`;
- the issue exists as a **precondition for the three model ports** (#479, #480, #481),
  which is what decides how it should be fixed.

All three cost a round of questions that the log answers in one read.

## Logs

| Session | Date | Produced |
|---|---|---|
| [`0cbcb148-7038-4b0f-b0b4-04fc95877a69.jsonl`](./0cbcb148-7038-4b0f-b0b4-04fc95877a69.jsonl) | 2026-09-09 | #479, #480, #481 (model ports), #482 (hardcoded model lists); the model-trim disk cleanup |

## Adding one

The host writes one file per session. For Claude Code:

```
~/.claude/projects/<project-slug>/<session-uuid>.jsonl
```

where the slug is the working directory with separators replaced —
`-home-jack-src-ollama37` for this repo.

**This repository is public. Scan before you copy.**

```bash
python3 cicd/scripts/scrub-session-log.py scan ~/.claude/projects/-home-jack-src-ollama37/<uuid>.jsonl
```

Exit `0` means clean; copy the file in unchanged. Exit `1` means it carries a credential —
`scrub` it to a redacted copy and commit that instead, or leave it out entirely:

```bash
python3 cicd/scripts/scrub-session-log.py scrub <src>.jsonl .sessions/<uuid>.jsonl
```

This is not theoretical. Any session that reads a memory file or a `-u user:pass` command
carries that secret in plain text; the session that wrote this README carried a live
Grafana password and the K80 host's private IP. See
[`../cicd/scripts/scrub-session-log.py`](../cicd/scripts/scrub-session-log.py) for what is
detected and for the local denylist that catches secrets no regex can recognise by shape.

Then link it from the issue — `Context: .sessions/<uuid>.jsonl` — and never paste the log
into the body.

## Rules

**Copy, do not render.** A markdown digest is a filter: it drops whatever it judged
uninteresting, and that judgment is the exact thing this directory removes. A readable
view can always be generated from the source, precisely because the source kept
everything.

**Raw, whole, unedited** — with credential redaction as the single exception, because it
is mechanical and the script records every line it touched. No summary, no "key
decisions" section. The moment anyone decides what mattered, the thing that mattered and
was not obvious is gone.

**A copy is a snapshot.** A live session's file is still growing when you copy it, so what
lands here is that session up to that moment, not its final state.

These files are large — a megabyte or so each, and they accumulate. That is accepted
rather than solved. Pruning is summarising with a longer interval.
