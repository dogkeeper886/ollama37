---
description: >-
  ollama37 compiles with CUDA 11.4 and GCC 10, in the ollama37-builder image or
  on a host that has that toolchain, for the Tesla K80. An agent sees a Go
  repository and runs go build on the host. The host binary carries no CUDA
  libraries and version 0.0.0, so a test on it says nothing about what the K80
  serves or what CI ships. This skill asks the agent to build ollama37 by the
  steps below.
---

Build local in Docker (needs the ollama37-builder image):
1. `systemctl --user stop ollama`
2. `OLLAMA_VERSION=<version> ./build-local.sh`
3. `systemctl --user start ollama`

Build local on the host (needs CUDA 11.4, GCC 10 at /usr/local/bin/gcc, CMake 4 and Go):
1. `systemctl --user stop ollama`
2. From the repo root, in one shell:
   ```
   export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++
   export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
   cmake --preset "CUDA 11 K80"
   cmake --build build -j"$(nproc)"
   cmake --install build --component CPU --strip
   cmake --install build --component CUDA --strip
   mkdir -p dist/bin
   go build -ldflags "-X github.com/ollama/ollama/version.Version=<version>" -o dist/bin/ollama .
   ```
3. `systemctl --user start ollama`

Build GitHub Action:
1. `gh workflow run build.yml --ref <branch> -f build_image=true -f runner_label=sm37`
