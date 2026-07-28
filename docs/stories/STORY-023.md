# STORY-023: Watch prefill and decode speed live in the container log

## User Story

As a K80 operator watching a request run,
I want the container log to show, while the model is still working, how many tokens have been
processed and how fast — separately for prefill and for decode,
So that I can see where the time is going as it happens, instead of waiting for the request to
finish and getting nothing.

## The Need

On a K80 a single request can take minutes. Today `docker logs ollama37` says nothing at all
while that happens — the only trace of a completed request is GIN's one-line total duration
(`POST "/api/generate" | 5.48s`). There is no way to tell a slow prompt from a stalled one, and
no way to see the two phases apart: a long prompt being ingested (prefill) and tokens being
produced (decode) have completely different speeds, and only their sum is visible.

The tuning work on this fork is *about* those two numbers. The batch-tier re-tunes (#458, #440)
trade prefill speed against decode speed, and every decision needs both measured. Right now the
only place either number appears is `ollama run --verbose` on the client side, after the fact —
which means the operator watching the server has to run a separate client to learn what the
server just did.

Official ollama shows this. Its log carries prefill progress, a running decode rate, and a
per-request summary of both phases, all while the request is in flight. That output comes from
llama.cpp's `llama-server`, which official ollama now runs as its child process — this fork
serves inference from its own Go runners and so gets none of it.

## Success Looks Like

- An operator tailing the container log during a long request sees, **without waiting for it to
  finish**, prefill progress: tokens ingested so far, how far through the prompt, and the prefill
  rate in tokens/second.
- During generation they see a **running decode rate** that reflects current speed, not just a
  lifetime average — so a model slowing down as its cache fills is visible while it happens.
- When the request completes, a **summary** reports both phases: tokens, elapsed, and
  tokens/second for prefill and for decode.
- Short, fast requests **do not flood the log** — the output is throttled enough that a busy
  server stays readable.
- It works for **both engines** this fork ships, so the operator does not have to know which
  engine served their model to get the numbers.
- The numbers an operator reads in the log are **trustworthy** — either they agree with what the
  API reports for the same request, or where they deliberately differ, the difference is
  documented rather than surprising.

## Open Questions

- Where in the serving path each number can be observed at all, and where it should be emitted
  from — progress while a prompt is still being ingested may not be visible everywhere the
  finished totals are.
- This fork has **two** runners (llama.cpp-engine and Go-engine) with separate batch loops and
  separate timing state. How much is shared versus duplicated per engine.
- Throttling policy: llama.cpp's own thresholds (prefill only after it has taken a few seconds;
  decode only after ~100 tokens and at most every few seconds) versus something tuned for K80
  timescales.
- Log level: always on, or behind `OLLAMA_DEBUG`. Always-on is only acceptable if throttling
  really does keep it quiet.
- Format: copy llama.cpp's text so both logs read alike, or this fork's `slog` key=value style so
  the lines are greppable and match their neighbours.
- Whether the existing API metrics have accuracy caveats worth surfacing (a warm KV cache can
  make the reported prompt rate impossible, because the token count and the measured duration
  cover different spans).
- **K80 no-harm:** logging must not perturb the batch loop's measured throughput, and the
  existing sweep numbers must be unmoved.

## Status

- Created: 2026-07-27
- Plan: #462
- Issues: #464 (merged, PR #468), #465 (PR #469 open), #466, #467 (#463 folded into #464 at task review)
