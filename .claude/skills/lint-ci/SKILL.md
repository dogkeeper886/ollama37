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

For each `jq -n '{...}'` template in a workflow, dry-run it with placeholder values:

```bash
echo '{"foo": $foo, "bar": $bar}' | jq -n --arg foo test --arg bar test '<paste object literal here>'
```

If jq prints the object: ✓. If it errors with `unexpected <word>`: that key is a reserved word — quote it (`"foo": $foo` instead of `foo: $foo`).

### `shellcheck` for `run:` blocks (install via package manager or docker)

Catches: redirect-order bugs (`cmd 2>&1 > file`), unquoted variables, missing error handling.

```bash
# If installed locally:
shellcheck path/to/extracted-script.sh

# Or via docker, no install:
docker run --rm -v "$PWD:/mnt" koalaman/shellcheck:stable path/to/extracted-script.sh
```

To run on a workflow's `run:` blocks, extract them first:

```bash
yq '.jobs[].steps[].run // empty' .github/workflows/test-fa-k80.yml > /tmp/extracted.sh
shellcheck /tmp/extracted.sh
```

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
