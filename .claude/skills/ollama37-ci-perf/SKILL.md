---
name: ollama37-ci-perf
description: Run a decision-grade FA/CUDA performance experiment via CI — verified throughput, self-reverting testbed
user-invocable: true
---

# Run an ollama37 performance experiment with CI

Measure a CUDA / flash-attention change's real performance through `test-throughput.yml` on a
self-hosted runner — tok/s that can actually decide whether to keep or drop the change. Build the
variant under test with `ollama37-ci-build`; apply an image with `ollama37-ci-apply`.

$ARGUMENTS

## Decision-grade, or it doesn't count

Raw tok/s from a hand `curl` loop is **not** evidence for an FA decision — it is single-run and never
checks the answer is correct; a path that is faster but emits garbage is not a win. Decision-grade
means all three:

- **Verified response** — `judge_mode=dual` runs the keyless agent judge for *meaningfulness*, not
  just "non-empty." Required for any keep/drop decision.
- **Realistic context** — `context_size` ≥ ~6.8k, not a one-line prompt. A short prompt once hid a
  7.4× flash-attention regression on the K80 behind a 22% one (`docs/porting-k80.md` §3).
- **Consistent** — the workflow recreates the container per run, so the state is fixed; compare paths
  on the same model + context.

## A "path" is (build) × (runtime env)

- **`--ref <branch>`** selects the *kernel routing* that got compiled — FA-off, all-VEC, decode-only,
  MMA-on, … — i.e. `ollama37-ci-build`'s output.
- **`flash_attention`** selects *FA vs the non-FA cuBLAS path* at runtime: `0` = cuBLAS baseline,
  `1` = FA. `kv_cache_type` is the other runtime knob.

## Run one

```bash
gh workflow run test-throughput.yml --ref <branch> \
  -f runner_label=sm75 \
  -f models=<model> \
  -f flash_attention=<0|1> \
  -f kv_cache_type='' \
  -f context_size=8192 -f num_predict=128 \
  -f judge_mode=dual
gh run watch <run-id> --exit-status
```

Read the markdown table in the run summary: **Prompt tok/s** = prefill, **Gen tok/s** = decode,
**Check** = PASS/FAIL (the judge's verdict). Collect one row per path and compare.

## The testbed reverts itself — fail is data

`test-throughput.yml` is self-contained: it applies the experiment env by recreating the container,
benchmarks, and in an `always()` step **reverts to the stable baseline (`OLLAMA_FLASH_ATTENTION=0`)**.
So a kernel that panics — e.g. tensor-core MMA on Turing under CUDA 11.4 — is a *recorded result*
("unusable"), and the testbed is still left clean for the next run. Never chase a green result by
dodging the crash; the crash is the finding.

`models` is `sm37`-only for the large models; on `sm75` size to the ~5 GiB it has.
