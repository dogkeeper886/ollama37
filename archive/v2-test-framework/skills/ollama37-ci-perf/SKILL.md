---
description: >-
  An agent measures a CUDA or flash-attention change by its own ideas. The
  agent times a hand curl loop on a one-line prompt. The single run never
  checks the answer, and a short prompt once hid a 7.4x flash-attention
  regression on the K80 behind a 22% one. The number cannot decide whether to
  keep the change. This skill asks the agent to measure the change by the
  steps below, not by its own ideas.
---

Run an ollama37 performance experiment with CI:
1. Build each path's kernel routing on its own branch with the `ollama37-ci-build` skill.
2. Run each path with `gh workflow run test-throughput.yml --ref <branch> -f runner_label=sm37 -f models=<model> -f flash_attention=<0|1> -f kv_cache_type='' -f context_size=8192 -f num_predict=128 -f judge_mode=dual`.
3. Wait for it with `gh run watch <run-id> --exit-status`.
4. Confirm the run summary shows the agent judge's verdict, not a fallback to the simple judge.
5. Record Prompt tok/s, Gen tok/s and Check for each path from the run summary.
6. Record a crashed path as unusable; the workflow reverts the testbed to `OLLAMA_FLASH_ATTENTION=0` itself.
7. Compare the paths on the same model and context.
