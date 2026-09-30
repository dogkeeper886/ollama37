---
description: >-
  An agent applies a freshly built ollama37 image by its own ideas. The agent
  runs docker compose on the host by hand. The hand recreate leaves no log,
  and after a build that skipped the retag it recreates the previous image.
  Every test after it silently exercises stale code. This skill asks the agent
  to apply the image by the steps below, not by its own ideas.
---

Apply the ollama37 image with CI:
1. Confirm the build run retagged the image: `gh run download <build-run> -n suites-v2-build -D /tmp/build-junit && grep -o 'sha256:[0-9a-f]*' /tmp/build-junit/junit.xml`.
2. Stop when the grep finds nothing, and rebuild with the `ollama37-ci-build` skill. The retag runs only behind a build, so a stale `ollama37:latest` is never retagged on its own.
3. Run `gh workflow run test-suites-v2.yml --ref <branch> -f runner_label=sm37 -f suite=suites/runtime.test.ts -f restart_container=true`; it recreates the container from `dogkeeper886/ollama37:latest`, and host B's `sm61` / `sm86` run the models suite only.
4. Wait for it with `gh run watch <run-id> --exit-status`.
5. Test the applied image with `test-suites-v2.yml`, `-f suite=suites/inference.test.ts` then `-f suite=suites/models.test.ts`, on the same `runner_label`.
