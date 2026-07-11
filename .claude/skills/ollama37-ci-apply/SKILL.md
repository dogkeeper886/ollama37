---
name: ollama37-ci-apply
description: Apply a built ollama37 image onto a testbed through CI — recreate the container, never docker compose by hand
user-invocable: true
---

# Apply the ollama37 image with CI

Roll a freshly built image onto a testbed through the GitHub Actions runtime workflow on a self-hosted
runner — never `docker compose up` or `docker exec` on a dev box. To *build* the image first, see
`ollama37-ci-build`.

$ARGUMENTS

## Applying = running the runtime workflow

**`test-runtime.yml` is how a built image is applied.** `TC-RUNTIME-001` runs `docker compose down`
then `docker compose up -d`, which recreates the container from the current **`ollama37:latest`** — so
it picks up whatever `ollama37-ci-build` last built. The `ollama-data` volume is `external`, so pulled
models survive the recreate. The same run then exercises the runtime checks (startup, GPU detection,
health).

```bash
gh workflow run test-runtime.yml --ref <branch> -f runner_label=<sm37|sm75>
```

- **`runner_label`** — which box: `sm37` (`rocky9-k80-cicd-1`, the reference K80 testbed) or `sm75`
  (`rocky9-2060-cicd-1`, RTX 2060). Default `sm37`. Never a bare `runs-on: self-hosted` — a K80 sweep on
  a consumer card produces numbers that look valid and are not.
- **`--ref <branch>`** — check out that branch's `docker-compose.yml` and testcases. Omit for the
  default branch.

Never `docker compose up`/`down` yourself to apply an image — run the runtime workflow so the recreate
is logged and reproducible.

## Where apply fits

`ollama37-ci-build` builds the image; this skill applies it; then the test suites
(`test-models.yml`, etc.) exercise it. `test-pipeline.yml` chains `build → runtime → inference →
models` for one `runner_label` if you want the whole flow in one run. For a CUDA change that must hold
on both cards, apply the **same published digest** on both hosts before testing — see `ollama37-ci-build`.
