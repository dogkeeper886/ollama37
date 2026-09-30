# Build Tests

Test specifications for ollama37 build verification.

---

## TC-BUILD-001: Builder Image Verification

**Importance:** High
**Execution Type:** Automated
**Timeout:** 60 seconds

**Summary:**
Verify the ollama37-builder Docker image exists and contains required build tools.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check builder image exists:<br>`docker images ollama37-builder:latest --format '{{.Repository}}:{{.Tag}}'` | Output: ollama37-builder:latest |
| 2 | Verify CUDA toolkit installed:<br>`docker run --rm ollama37-builder:latest nvcc --version` | Output contains: Cuda compilation tools, release 11.4 |
| 3 | Verify GCC version:<br>`docker run --rm ollama37-builder:latest gcc --version` | Output contains: gcc (GCC) 10 |
| 4 | Verify Go version:<br>`docker run --rm ollama37-builder:latest go version` | Output contains: go1.25 |

---

## TC-BUILD-002: Runtime Image Build

**Importance:** High
**Execution Type:** Automated
**Timeout:** 30 minutes
**Dependencies:** TC-BUILD-001

**Summary:**
Build the ollama37 runtime Docker image from source.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Build runtime image:<br>`cd /home/jack/src/ollama37/docker && make build-runtime 2>&1` | Build completes with message: "Runtime image built successfully"<br>No "error:", "Error:", or "FAILED" in output |
| 2 | Verify runtime image exists:<br>`docker images ollama37:latest --format '{{.Repository}}:{{.Tag}} {{.Size}}'` | Output: ollama37:latest with size ~30GB |

### Notes

- Build time is approximately 17-18 minutes for CUDA compilation
- Image size is ~30GB due to CUDA toolkit and compiled binaries
