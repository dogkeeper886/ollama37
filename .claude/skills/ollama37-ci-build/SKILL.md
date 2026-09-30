---
description: >-
  An agent builds the ollama37 image by its own ideas. The agent runs make on
  a dev box. The local build defaults OLLAMA_VERSION to 0.0.0, and the ollama
  registry rejects every gated model pull from it with 412. The image looks
  fine and cannot pull modern models. This skill asks the agent to build the
  image by the steps below, not by its own ideas.
---

Build the ollama37 image with CI:
1. Check the version the build injects with `gh variable list` (`OLLAMA_VERSION`).
2. Run the build suite with the build gate on, `gh workflow run test-suites-v2.yml --ref <branch> -f runner_label=sm37 -f suite=suites/build.test.ts -f build_image=true`, so `carries the tag compose reads` retags `ollama37:latest` to `dogkeeper886/ollama37:latest`.
3. Wait for it with `gh run watch <run-id> --exit-status`.
4. Confirm the retag with `gh run download <run-id> -n suites-v2-build -D /tmp/build-junit && grep -o 'sha256:[0-9a-f]*' /tmp/build-junit/junit.xml`.
5. Apply the image with the `ollama37-ci-apply` skill.
