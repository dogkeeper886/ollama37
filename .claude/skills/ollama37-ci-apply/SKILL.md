---
description: >-
  An agent applies a freshly built ollama37 image by its own ideas. The agent
  runs docker compose on the host by hand. The hand recreate leaves no log,
  and after a build that skipped the retag it recreates the previous image.
  Every test after it silently exercises stale code. This skill asks the agent
  to apply the image by the steps below, not by its own ideas.
---

Apply the ollama37 image with CI:
1. Confirm the build run retagged the image: `gh run view <build-run> --log | grep 'retagged sha256'`.
2. Stop when the grep finds nothing, and rerun the retag with `gh workflow run test-build.yml --ref <branch> -f runner_label=sm37 -f test_id=TC-BUILD-004`.
3. Run `gh workflow run test-runtime.yml --ref <branch> -f runner_label=sm37`; it recreates the container from `dogkeeper886/ollama37:latest`, and host B's `sm61` / `sm86` run the models suite only.
4. Wait for it with `gh run watch <run-id> --exit-status`.
5. Test the applied image with the suite workflows (`test-inference.yml`, `test-models.yml`) on the same `runner_label`.
