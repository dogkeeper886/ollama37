# STORY-013: Every K80 GPU run guards its own temperature and aborts before it overheats

## User Story

As a maintainer running model tests on the self-hosted K80,
I want every workflow that loads a model to abort the moment a GPU die crosses 80 °C and to record the run's peak temperature,
So that an overheating run is stopped automatically and the heat is attributable to the exact run that caused it — instead of relying on an anonymous local monitor or hardware throttling.

## The Need

The only temperature protection today is an ad-hoc `nvidia-smi` loop a human starts by hand alongside a run. It's anonymous — nothing ties the temperature to which CI run, commit, or person triggered the GPU load — and it isn't part of CI at all, so most runs have no guard whatsoever. When a run overheats, nothing stops it short of the K80's own hardware throttling, and afterward there's no record of how hot it got or which run did it. A maintainer should be able to trust that any model run on the K80 is temperature-guarded and self-documenting, without remembering to babysit it.

## Success Looks Like

- A GPU run whose temperature crosses the limit (80 °C) **fails the job** with a clear overheat error, rather than continuing to cook or depending on hardware throttling.
- Every guarded run shows its **peak GPU temperature** in its own run record (the step summary), so the heat is attributable to that exact run.
- The guard covers **all the Ollama-inference workflows** on the K80 (the MCP, throughput, flash-attention, inference, models, and runtime runs), applied consistently rather than copy-pasted into each.
- The same guard works when a maintainer runs a model test **locally**, not only in CI.
- A run that stays cool is unaffected — no change to its result, just a recorded peak.

## Open Questions

- The mechanism — a reusable wrapper script the workflows call, a composite GitHub action, or another shape — decided in planning.
- Whether the threshold and sample interval should be knobs (and their defaults), and whether there's a separate "warn" band below the hard-abort line.
- Exactly which workflows are in scope, and whether non-Ollama GPU work (e.g. MLX smoke) or build-only jobs should be guarded too.
- How the abort is wired so the job fails cleanly mid-run (killing the model process, surviving the `| tee` to the step summary).

## Status

- Created: 2026-06-20
- Plan: #331
- Issues: #332, #333, #334
