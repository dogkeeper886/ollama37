---
name: ollama37-ci-build
description: Build (and publish) the ollama37 CI image on a self-hosted GPU runner — never with a local make
user-invocable: true
---

# Build the ollama37 CI image with CI

Build the fork's runtime image through the GitHub Actions build workflow on a self-hosted runner —
`gh workflow run`, never `make build-runtime-local` on a dev box. To *apply* a built image to a
testbed, see `ollama37-ci-apply`.

$ARGUMENTS

## Never build by hand — a local build is version-crippled

A local `make` build defaults `OLLAMA_VERSION=0.0.0` (`docker/Makefile`), and the ollama registry
rejects every gated model pull from an old version with `412: requires a newer version` — so the image
*looks* fine and silently cannot pull modern models (qwen3.5, gemma3, ministral-3, …). The CI build
injects `OLLAMA_VERSION` from the **repo variable** (check it: `gh variable list`), which is `>` any
`0.x` minimum and clears the gate. Only the CI image can pull those models. It also leaves logs; a hand
`make` leaves nothing.

## The build

**`TC-BUILD-002` IS the build** — `test-build.yml` runs `make build-runtime-local-no-cache`, a full
no-cache compile (preset "CUDA 11 K80", native CUBIN sm_37…sm_86, no GPU needed). It produces the local
tag **`ollama37:latest`** on the runner it ran on.

```bash
gh workflow run test-build.yml --ref <branch> -f runner_label=<sm37|sm75>
gh run watch <run-id> --exit-status        # block until done; GitHub holds the logs
```

- **`runner_label`** — which box: `sm37` (`rocky9-k80-cicd-1`, 4× K80) or `sm75` (`rocky9-2060-cicd-1`,
  RTX 2060). Default `sm37`. Never a bare `runs-on: self-hosted`.
- **`--ref <branch>`** — which code. Omit to build the default branch.
- The build normally runs on **`sm37`** (its ~15 GB builder image is cached there; on a fresh host
  `ensure-builder` rebuilds it, ~90 min). It can run on `sm75` only if `ollama37-builder:latest` is
  already present there.

## The CI image, and moving it between testbeds

`ollama37:latest` is a **local tag** on one runner's Docker daemon — `docker/docker-compose.yml` reads
exactly this tag. The two runners are **independent hosts with no network between them**, so the tag on
`sm37` names nothing on `sm75`. To test the *same bits* on both cards, publish to Docker Hub
**`dogkeeper886/ollama37-ci:ci-<sha>`** and fetch **by digest** (`@sha256:…`, immutable — a tag can be
overwritten, a digest cannot). `TC-BUILD-004` publishes; a fetch step retags the digest back to
`ollama37:latest` so compose reads it. That published, digest-addressed artifact is "the CI image."

Once built, hand off to `ollama37-ci-apply` to roll it onto the testbed.
