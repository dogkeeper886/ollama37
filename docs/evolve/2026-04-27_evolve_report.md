# Evolve Report — 2026-04-27

## Summary
- Time range analyzed: 90 days (2026-01-27 to 2026-04-27)
- Issues analyzed: 76 (3 open, 73 closed)
- Commits analyzed: 152 since prior report (2026-03-20)
- Insights found: 9 (High: 4, Medium: 3, Low: 2)
- Prior report: 2026-03-20 — 4 of 5 actions effective; 1 deferred

## Prior Action Effectiveness (since 2026-03-20)

| # | Prior Action | Verdict | Evidence |
|---|---|---|---|
| 1 | Trace-and-Evolve workflow section in CLAUDE.md | **Effective** | New trace doc created this period (`qwen35-flash-attention-gate.md`); /evolve ran today |
| 2 | `/judge` command (deferred) | n/a | Still no implementation; LLM judge stable so low priority |
| 3 | Verify directory-format skill remnants | **Effective** | No remnants reintroduced |
| 4 | Triage issue #39 | **Effective** | #39 closed; all open issues < 14d stale |
| 5 | Retroactively label old issues | **Effective** | Recent issues all properly labeled |

**Scorecard:** 4 effective, 0 partial, 0 ineffective, 1 deferred out of 5 total.

### Pattern-monitor scorecard

| Pattern | Target | Actual | Verdict |
|---|---|---|---|
| CLAUDE.md churn | ≤1/session | 5 edits over ~38 days | **Partial** — slowed but ongoing |
| LLM judge stability | 0 fix commits in 30d | 0 since prior report | **Effective** ✅ |
| Session summary adoption | ≥3 in 30d | 1 (this session, retroactive) | **Ineffective** — adoption stalled |
| Open issue staleness | 0 stale (>14d) | 0 stale | **Effective** ✅ |
| Anthropic API drift | Monitor | No commits to anthropic/ | **Effective** (no drift) |

## High-Confidence Insights

### 1. Workflow gap: CI YAML errors surface only at runtime (Friction Point)
- **Evidence:** `4b8a64aa` (jq `label` reserved word broke benchmark sweep partway), `c648c1b6` (shell redirect order `2>&1 > file` leaked stderr), `4b7410ba` (workflow defensive cleanup added after first run failures), `430ab602` (benchmark summary crash)
- **Confidence:** High (4 distinct cases since prior report)
- **Suggestion:** Add pre-merge lint helper. Most caught issues were locally testable in isolation but slipped because the no-local-build memory says "rely on CI." Need a portable subset of validation that doesn't need GPU.

### 2. Efficiency win: Self-review-then-fix pattern is consistently productive (Usage Pattern)
- **Evidence:** `ede5f3e8`, `821e0ab4`, `4b8a64aa`, `c648c1b6`, `96da5dcc` (5+ separate same-branch fixes from self-review during this period)
- **Confidence:** High
- **Suggestion:** Reinforce — make explicit in CLAUDE.md that self-review on a branch is expected to find ≥1 issue worth a follow-up commit.

### 3. Effective scaling pattern: Portal-of-phases cascade for multi-step work (Usage Pattern)
- **Evidence:** #105 → (#107, #108), #121 → (#122, #123, #124), with `Part of` and `Depends on` cross-refs throughout
- **Confidence:** High (2 successful end-to-end uses this session, both resulting in full closure)
- **Suggestion:** Promote from ad-hoc to documented pattern.

### 4. Knowledge gap: Engine-selection logic isn't documented anywhere (Knowledge Decay)
- **Evidence:** Phase 1 audit (#122) had wrong assumption about deepseek-r1 engine; Phase 2 (#123) empirically corrected to llama.cpp engine; the trace doc is the only place the new-vs-old engine selection is explained
- **Confidence:** High (caused a real audit error this session)
- **Suggestion:** Add a section to CLAUDE.md or a new trace doc explaining: (a) two engines exist; (b) per-arch packages indicate capability not use; (c) engine choice gated on `OllamaEngineRequired()` plus env var.

## Medium-Confidence Insights

### 5. Workflow gap: `workflow_dispatch` chicken-and-egg costs iteration speed
- **Evidence:** This session — 3 separate workflow files needed merging to main before they could be triggered against branches
- **Confidence:** Medium
- **Suggestion:** Accept as cost-of-doing-business; document.

### 6. CLAUDE.md churn slowed but didn't stop (carried-forward)
- **Evidence:** 5 edits since prior report (vs 10+ in prior period)
- **Confidence:** Medium
- **Suggestion:** Continue monitoring with relaxed target (≤3/30d).

### 7. Test framework had 2 distinct fix commits (Friction Point)
- **Evidence:** `315f81f1` (judges fail on cudaMalloc backoff), `31677d72` (LogCollector fallback)
- **Confidence:** Medium
- **Suggestion:** Edge-case handling. No structural change needed.

## Low-Confidence Observations

### 8. /session-summary adoption is below target
- **Evidence:** Only 1 new summary in 38 days
- **Suggestion:** Behavioral; can't force without a hook.

### 9. Memory-entry pattern proving its value
- **Evidence:** 4 memory entries added this session
- **Suggestion:** Continue.

## Proposed Actions

| # | Action | Category | Priority | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Add a portable pre-merge lint helper skill | Skill / CLAUDE.md | Important | Medium | None |
| 2 | Document the two-engine selection logic | Knowledge | Important | Small | None |
| 3 | Add CLAUDE.md note on self-review expectations | CLAUDE.md | Nice-to-have | Small | None |
| 4 | Promote portal-of-phases pattern | Skill / CLAUDE.md | Nice-to-have | Medium | Low |
| 5 | Update Patterns to Monitor list | Project notes | Nice-to-have | Small | None |

## Actions Applied

| # | Action | Status | Artifact |
|---|---|---|---|
| 1 | Add portable pre-merge lint helper | **Applied** | `.claude/skills/lint-ci/SKILL.md` + entry in CLAUDE.md Skill Quick Reference |
| 2 | Document the two-engine selection logic | **Applied** | `docs/traces/engine-selection.md` + new "Engine Selection" section in CLAUDE.md |
| 3 | Self-review CLAUDE.md note | Skipped | User chose not to apply |
| 4 | Portal-of-phases promotion | Skipped | User chose not to apply |
| 5 | Update Patterns to Monitor | Carried forward in this report's monitoring table | — |

## Patterns to Monitor

Carried forward + new:

| Pattern | What to check | Success criteria |
|---|---|---|
| CLAUDE.md churn | Edits per 30d | ≤3 in 30 days (relaxed — file is growing as project matures) |
| Session summary adoption | New summaries | ≥2 in next 30 days |
| Open issue staleness | Open issues with no activity >14d | Zero stale |
| **CI YAML runtime errors (NEW)** | Workflow failures from syntax-class issues (jq, shell, yaml) | Zero in next 30 days; if `lint-ci` skill is invoked in workflow PRs, this should drop |
| **Engine-selection clarity (NEW)** | Audits or reports that misidentify engine | Zero misidentifications now that `docs/traces/engine-selection.md` exists |
| LLM judge stability | Fix commits to `cicd/tests/src/judge/llm-judge.ts` | Zero fix commits in next 30 days (was effective) |
