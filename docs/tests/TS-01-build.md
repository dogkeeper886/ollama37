---
id: TS-01
title: Build — toolchain image and runtime binary
namespace: ollama37
story: STORY-005
story_hash: 0598aff09de9ddd319d9c168f10a059e16496ea8a26a0d7dcfd1f39f12d9e0f3
status: green
---

## Why this scenario exists

ollama37 must *build* before it can run on K80: the cached builder image has to carry the exact
sm_37 toolchain (CUDA 11.4, GCC 10, Go), the runtime image has to compile from clean source
without errors, and both images have to stay within sane size bounds. This scenario is the
build half of [STORY-005](../stories/STORY-005.md).

### TC-01: Builder image verification

- **Objective:** the builder Docker image contains the required build tools (CUDA 11.4, GCC 10, Go).
- **Script:** cicd/tests/testcases/build/TC-BUILD-001.yml
- **Preconditions:** the `ollama37-builder:latest` image has been built.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check the builder image exists | `ollama37-builder:latest` is present |
| 2 | Verify the CUDA toolkit | reports `Cuda compilation tools`, release `11.4` |
| 3 | Verify the GCC version | reports `gcc` version 10 |
| 4 | Verify the Go version | reports `go1.2x` |

### TC-02: Runtime image build

- **Objective:** the ollama37 runtime image builds from local source without errors.
- **Script:** cicd/tests/testcases/build/TC-BUILD-002.yml
- **Preconditions:** the builder image exists; local source is checked out.

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Build the runtime image | prints `Runtime image built successfully`, no `error:` / `Error:` |
| 2 | Verify the runtime image exists | `ollama37:latest` is present |

### TC-03: Image size validation

- **Objective:** the builder and runtime image sizes are within expected ranges.
- **Script:** cicd/tests/testcases/build/TC-BUILD-003.yml

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check the builder image size | within range (`SIZE_OK`) |
| 2 | Check the runtime image size | within range (`SIZE_OK`) |
