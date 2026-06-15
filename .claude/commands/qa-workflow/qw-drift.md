# Check for Test Drift

```
Surface every test that no longer matches what it verifies — before a stale
test passes quietly and a green build lies.

Target: every test doc under docs/tests/ (runs in CI and on demand).

## PURPOSE

The freshness gate of the qa-workflow, and a review in its own right (it has no
paired producer — it checks the whole set). Drift is silent until something looks;
this looks, deterministically, every run.

Fits in the qa-workflow:

    … → qw-run → [human] → dw-merge   (qw-run = `npm test` — the cicd runner, not a slash command)
                    └──────────────► qw-drift ──► back to qw-cases when stale

---

## WORKFLOW

    /qw-drift
        │
        ├─► Run the gate:  npm --prefix cicd/tests run drift
        │   Two deterministic signals, per case/scenario:
        │     - STALE   — the linked story's sha256 no longer matches the doc's
        │                 `story_hash` (the story moved since the test was synced).
        │     - UNBOUND — the doc and its bound executable diverged (reuses the
        │                 binding audit).
        │   Exits non-zero if anything is stale or unbound (so CI fails on drift).
        │
        └─► On a finding:
            - STALE: re-read the test against the changed story. If it still holds,
              update `story_hash` (`sha256sum docs/stories/STORY-XXX.md`); if not,
              fix the test via `/qw-cases` → `/qw-review-cases`.
            - UNBOUND: fix via `/qw-bind` → `/qw-review-bind`.

---

## API Notes

- Hash-first is deterministic and needs no stack; it runs in CI and on demand.
- A semantic, embedding-based signal (softer "drifted in meaning", via the store)
  is a planned advisory add — hash is the build-failing gate.
- `status` in a test doc (green | stale | unbound) reflects this gate's verdict.
- No paired producer — `qw-drift` *is* a review (the qa-workflow pairing rule).
```
