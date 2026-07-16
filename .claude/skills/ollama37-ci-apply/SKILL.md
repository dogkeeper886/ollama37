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
then `docker compose up -d`, which recreates the container from **`dogkeeper886/ollama37:latest`** — the
tag `docker-compose.yml` reads. The `ollama-data` volume is `external`, so pulled models survive the
recreate. The same run then exercises the runtime checks (startup, GPU detection, health).

**Precondition — confirm the retag, or you apply a STALE image.** The build produces `ollama37:latest`;
only **`TC-BUILD-004`** retags it to `dogkeeper886/ollama37:latest` (the tag compose reads). So apply
picks up your fresh build *only if the build ran that retag*. A full `test-build.yml` run does (TC-BUILD-004 runs in the build suite) — but a single-test build
(`-f test_id=TC-BUILD-002`) **skips it**, leaving
the compose tag pointing at the previous image. **Before applying, confirm the retag actually
succeeded** — grep the build run for TC-BUILD-004's success line, not just its name (the name appears
whether it passed or failed): `gh run view <build-run> --log | grep 'retagged sha256'`. No match →
the recreate, and every test after it, silently exercises stale code.

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
models` for one `runner_label` if you want the whole flow in one run — and because the build suite
retags the fresh image to the tag compose reads (`TC-BUILD-004`, see `ollama37-ci-build`), that one
pipeline run builds *and* applies the new bits. Testing is **single-host**; the retired cross-host
by-digest path no longer applies.
