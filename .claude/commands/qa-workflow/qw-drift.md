# Check for Test Drift

```
Surface every test that no longer matches what it verifies — before a stale
test passes quietly and a green build lies.

Target: every test doc under docs/tests/ (on demand today; a CI gate once the drift port lands).

## PURPOSE

The freshness gate of the qa-workflow, and a review in its own right (it has no
paired producer — it checks the whole set). Drift is silent until something looks;
this looks, deterministically, every run.

Fits in the qa-workflow:

    … → qw-run → [human] → dw-merge   (qw-run = `npm --prefix cicd/tests test` — the cicd runner, not a slash command)
                    └──────────────► qw-drift ──► back to qw-cases when stale

---

## WORKFLOW

    /qw-drift
        │
        ├─► Run the checks (by hand for now), per case/scenario:
        │     - STALE   — `sha256sum docs/stories/STORY-XXX.md` no longer matches the
        │                 doc's `story_hash` (the story moved since the test was synced).
        │     - UNBOUND — the doc's Steps row count no longer matches the bound YAML's
        │                 `steps:` count (the binding diverged).
        │   (An automated `npm --prefix cicd/tests run drift` doing both and exiting
        │   non-zero so CI fails on drift is a deferred port — STORY-006/Phase 2.)
        │
        └─► On a finding:
            - STALE: re-read the test against the changed story. If it still holds,
              update `story_hash` (`sha256sum docs/stories/STORY-XXX.md`); if not,
              fix the test via `/qw-cases` → `/qw-review-cases`.
            - UNBOUND: fix via `/qw-bind` → `/qw-review-bind`.

---

## API Notes

- Hash-first is deterministic and needs no stack; it's a by-hand check today, and a
  build-failing CI gate once the `drift` port lands (STORY-006/Phase 2).
- A semantic, embedding-based signal (softer "drifted in meaning") is a further
  planned advisory add — the hash + row-count checks are the structural gate.
- `status` in a test doc (green | stale | unbound) reflects this gate's verdict.
- No paired producer — `qw-drift` *is* a review (the qa-workflow pairing rule).
```
