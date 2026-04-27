---
name: lint-ci
description: Pre-merge lint of CI workflow files. Use before pushing changes to .github/workflows/ to catch shell, jq, and YAML errors that would otherwise surface only at runtime.
---

# Lint CI

Catch errors in `.github/workflows/*.yml` before pushing. The dev environment can't run the full GPU build path, so CI is the only place workflow code actually executes — making syntactic-class bugs (shell redirect order, `jq` reserved words, YAML structure) expensive to discover.

## When to use
- Before pushing changes to a workflow file
- After adding inline `bash` `run:` blocks or `jq` object construction
- When iterating on a workflow that takes a long round-trip to test on the runner

## Tools and how to invoke

### `jq -n` for object templates (always available locally)

The pattern that bit us most often: bare object keys that collide with jq reserved words (`label`, `break`, `try`, `catch`, `reduce`, `foreach`, `as`, `def`, `if`, `then`, `else`, `end`, `and`, `or`, `not`, `import`, `include`, `module`, `null`, `true`, `false`).

For each `jq -n '{...}'` template in a workflow, dry-run it with placeholder values. `jq -n` (`--null-input`) takes no stdin — just the args and the object literal:

```bash
# Replace the workflow's --arg / --argjson values with placeholders, then run.
# Example: if the workflow has
#   jq -n --arg model "$MODEL" --argjson num_predict "$NUM_PREDICT" '{ model: $model, num_predict: $num_predict }'
# dry-run as:
jq -n --arg model test --argjson num_predict 1 '{ "model": $model, "num_predict": $num_predict }'
```

If jq prints the object: ✓. If it errors with `unexpected <word>`: that key is a reserved word — quote it (`"model": $model` instead of `model: $model`).

### `shellcheck` for `run:` blocks (install via package manager or docker)

Catches: redirect-order bugs (`cmd 2>&1 > file`), unquoted variables, missing error handling.

```bash
# If installed locally:
shellcheck script.sh

# Or via docker, no install:
docker run --rm -v "$PWD:/mnt" --workdir /mnt koalaman/shellcheck:stable script.sh
```

To shellcheck a workflow's `run:` blocks, the simplest path is to copy each block into a `.sh` file and run shellcheck on it. Workflow YAML doesn't have a clean automated extractor in standard tooling — `yq` (Go) or `yq` (Python) versions both work but neither is guaranteed installed. If you want automation, install `yq` from `mikefarah/yq` and run `yq '.jobs[].steps[].run // ""' workflow.yml > extracted.sh` first.

### `actionlint` for workflow structure (install or docker)

Catches: bad expressions, missing required fields, malformed `uses:` references.

```bash
# Local install (Go required):
go install github.com/rhysd/actionlint/cmd/actionlint@latest

# Or docker:
docker run --rm -v "$PWD:/repo" --workdir /repo rhysd/actionlint:latest -color
```

### YAML structural sanity (Python — usually available)

```bash
python3 -c 'import yaml; yaml.safe_load(open(".github/workflows/test-fa-k80.yml"))'
```

Doesn't catch action-specific issues but verifies basic YAML.

## Quick checklist before pushing

- [ ] Every bare key in `jq -n '{...}'` is quoted (`"key":` not `key:`)
- [ ] Every `> file 2>&1` has the redirects in that order (not `2>&1 > file`)
- [ ] No `set -u` violations — every variable expansion handles unset
- [ ] `yaml.safe_load` parses the workflow without error
- [ ] If you have actionlint or docker: a clean actionlint run

## What this skill does NOT replace

- Real CI run on the runner — only that exercises the GPU path, container orchestration, and actual model loads
- Go compile checks — those need the full toolchain (per `feedback_no_local_builds.md`)
- Functional correctness — lint catches syntax-class bugs only
