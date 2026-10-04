#!/usr/bin/env bash
# Usage: OLLAMA_VERSION=<version> ./build-local.sh [--clean]
#   --clean  delete build/ first: CMake reconfigures and every CUDA kernel recompiles.
#            Absent: reuse build/; only changed sources recompile.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
# The version baked into the binary: any version string, e.g. 2.3.2. Required.
OLLAMA_VERSION=${OLLAMA_VERSION:?set OLLAMA_VERSION, e.g. OLLAMA_VERSION=2.3.2}
# Image with CUDA 11.4, GCC 10, CMake 4, Go: any tag built from docker/builder/Dockerfile.
BUILDER=${BUILDER:-ollama37-builder:latest}

if pgrep -f "^$ROOT/dist/bin/ollama" >/dev/null; then
  echo "build-local: $ROOT/dist/bin/ollama is running; stop it first (systemctl --user stop ollama)" >&2
  exit 1
fi
[ "${1:-}" = "--clean" ] && rm -rf "$ROOT/build"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$ROOT:$ROOT" -w "$ROOT" \
  -e HOME="$ROOT/.cache/home" \
  -e GOCACHE="$ROOT/.cache/go-build" \
  -e GOMODCACHE="$ROOT/.cache/go-mod" \
  -e OLLAMA_VERSION="$OLLAMA_VERSION" \
  "$BUILDER" bash -c '
    set -euo pipefail
    mkdir -p "$HOME"
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
    export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++
    # A configurePresets name from CMakePresets.json: Default, CPU, CUDA, CUDA 11,
    # CUDA 11 K80, CUDA 12, CUDA 13, JetPack 5, JetPack 6, ROCm, ROCm 6, Vulkan.
    # Only CUDA 11 K80 compiles native sm_37 code.
    cmake --preset "CUDA 11 K80"
    cmake --build build -j"$(nproc)"
    cmake --install build --component CPU --strip
    cmake --install build --component CUDA --strip
    mkdir -p dist/bin
    go build -ldflags "-X github.com/ollama/ollama/version.Version=${OLLAMA_VERSION}" -o dist/bin/ollama .
  '

"$ROOT/dist/bin/ollama" --version 2>&1 | tail -1
