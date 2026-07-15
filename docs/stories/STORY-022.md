# STORY-022: Record any model's context × batch fit map on the K80

## User Story

As a K80 maintainer sizing a model,
I want a repeatable way to record any model's fit map across context length and batch size —
throughput, correctness, and per-die VRAM together, in one run,
So that I choose context/batch settings from measured data instead of hand-scraping metrics.

## The Need

Choosing the qwen35moe batch tiers (#440) required a full **context × batch** map for
`qwen3.6:35b`: for each combination, whether it fits on GPU, how many dies it uses, its VRAM,
its decode speed, its long-prompt (prefill) cost, and whether the answer is still correct.

Producing that map was manual and awkward. The tool-call test reports throughput and a
correctness verdict, but **per-die VRAM had to be scraped from an external monitoring dashboard**
by hand-matching each run's timestamps. That's slow, error-prone, and not reusable — and the same
map is needed to size other models and to explain why certain models (e.g. `qwen3.6:27b`,
`qwen3-vl:30b`) spread onto an **extra die** and trip the GPU-count check.

## Success Looks Like

- A maintainer points the tooling at a model (or a few) and gets back a **committed table**: per
  context × batch — the correctness verdict, decode tok/s, prefill/long-prompt cost, **per-die
  VRAM, active-die count, and whether it is fully on GPU / spilling / OOM** — all from one run,
  **no external dashboard**.
- The measurement **bounds itself to each model's trained context** and picks a sensible
  correctness check automatically, so running a new model needs **no per-model setup**.
- A judge or infra hiccup **does not lose the fit data** — the VRAM/throughput numbers are
  recorded regardless of the correctness verdict.
- The `qwen3.6:35b` map **reproduces the hand-built one**
  (`docs/reports/qwen35moe-num-batch-context-fit-2026-07-14.md`), and the `qwen3.6:27b` /
  `qwen3-vl:30b` maps **reveal when and why** they use the extra die.

## Open Questions

- How per-die VRAM is captured inside the run, and which test vehicle carries it (the long-prompt
  tool-call test vs the throughput bench) — decided on the issue.
- Whether this **extends the existing report-sweep workflow or is a new one**.
- Default correctness-check mode (lightweight structural check vs the live grounded verifier) and
  how a model's tool-support is detected.
- Which models make a standing list vs on-demand, and the default context/batch grid.
- **K80 no-harm:** it must not change the behavior of the existing test suites.

## Status

- Created: 2026-07-15
- Plan: #444
- Issues: #445, #446, #447
