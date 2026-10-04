---
description: >-
  ollama37 compiles with CUDA 11.4 and GCC 10 inside the ollama37-builder
  image, for the Tesla K80. An agent sees a Go repository and runs go build on
  the host. The host binary carries no CUDA libraries and version 0.0.0, so a
  test on it says nothing about what the K80 serves or what CI ships. This
  skill asks the agent to build ollama37 by the steps below.
---

Build local:
1. `systemctl --user stop ollama`
2. `OLLAMA_VERSION=<version> ./build-local.sh`
3. `systemctl --user start ollama`

Build GitHub Action:
1. `gh workflow run build.yml --ref <branch> -f build_image=true -f runner_label=sm37`
