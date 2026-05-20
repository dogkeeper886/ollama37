---
name: build
description: Provides step-by-step build instructions for ollama37. Use when you need to compile the binary from source (native build) or build a Docker runtime image.
argument-hint: [native|docker]
---

# Build

Build ollama37 from source — native binary or Docker runtime image. Use `/build` to run.

## When to use
- Compiling the Go binary and CUDA libraries from source
- Building or rebuilding the Docker runtime container
- Verifying code changes that affect the compiled binary or CUDA libraries

## Preflight — verify the toolchain first (do this before any build command)
Don't assume a local build environment exists; check it at runtime:
- `which go` / `which nvcc` (and `gcc`)
- **Present** → build locally (native or docker, below).
- **Absent** → build via `/ci` (`test-build.yml`) on the K80 runner; do not attempt a local `go build`/`make`.

This is a runtime check, not a fixed verdict — the box may or may not have the toolchain on any given day. Check, then choose.

## Environment
- GCC version: 10.5
- CUDA version: 11.4.4
- NVIDIA driver: 470
- Target GPU: Tesla K80 (compute capability 3.7)

## Build types
- **native** — Compile directly on the host using cmake and go build
- **docker** — Build a Docker runtime image via the Makefile in `docker/`

## Key references
- CMake presets: `"CUDA 11"` (all architectures 37-86), `"CUDA 11 K80"` (K80 only)
- Docker Makefile: `docker/Makefile`
- Full Docker docs: `docker/README.md`

## Build times
- Builder image: ~90 min (first time), cached thereafter
- Runtime image: ~10 min
