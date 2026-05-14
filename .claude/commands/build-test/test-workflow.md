Author or refactor a CI workflow that follows the unified test-workflow pattern. Reference `.claude/skills/test-workflow-pattern/SKILL.md` for the full contract.

The user will provide: **$ARGUMENTS** (workflow name and what it tests, e.g. `fa-k80 — flash attention validation`)

## Steps

1. **Confirm pattern fit** — This pattern is for perf benchmarks, experiments, profiling — anything that runs on the K80 runner but isn't a TC-framework correctness suite. If it's a build/runtime/inference/models correctness test, use the TC framework instead (`/add-test`).

2. **Pick the names** — `.github/workflows/test-<name>.yml` and `cicd/scripts/test-<name>.sh`. Same stem.

3. **Create the script** at `cicd/scripts/test-<name>.sh`:

```bash
#!/usr/bin/env bash
# test-<name>.sh — <one-line purpose>
#
# Exits non-zero on any validation failure. Writes structured results to
# /tmp/test-<name>-results.json. Sources shared helpers from
# cicd/scripts/lib/.

set -euo pipefail

LIB_DIR="$(dirname "$0")/lib"
# shellcheck source=lib/response_capture.sh
source "$LIB_DIR/response_capture.sh"
# shellcheck source=lib/simple_check.sh
source "$LIB_DIR/simple_check.sh"
# shellcheck source=lib/judge_response.sh   # only if you need it
source "$LIB_DIR/judge_response.sh"

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OUTPUT="${OUTPUT:-/tmp/test-<name>-results.json}"

# ... test logic ...
# capture responses, run checks, emit JSON, exit non-zero on fail

echo "$REPORT" > "$OUTPUT"
[ "$FAILED" -gt 0 ] && exit 1
```

4. **Create the workflow** at `.github/workflows/test-<name>.yml`:

```yaml
name: <Name> Test

on:
  workflow_dispatch:
    inputs:
      # ... workflow-specific inputs ...

env:
  OLLAMA_HOST: http://localhost:11434
  JUDGE_HOST: http://localhost:11435   # only if using LLM judge

jobs:
  test:
    runs-on: self-hosted
    environment: cicd-1
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # Optional standardized pre-step
      - name: Stop production ollama37
        run: |
          cd ${OLLAMA37_ROOT}/docker && docker compose down 2>/dev/null || true

      - name: Run test
        run: |
          bash cicd/scripts/test-<name>.sh

      # Always-on standardized post-step
      - name: Restart production ollama37
        if: always()
        run: |
          cd ${OLLAMA37_ROOT}/docker && docker compose up -d

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-<name>-results
          path: /tmp/test-<name>-results.json
```

5. **Verify locally before pushing**:

```bash
bash -n cicd/scripts/test-<name>.sh                    # syntax
python3 -c 'import yaml; yaml.safe_load(open(".github/workflows/test-<name>.yml"))'   # YAML
# /lint-ci skill for the full pre-merge checklist
```

6. **Trigger on the runner**:

```bash
gh workflow run test-<name>.yml --ref <branch>
gh run watch <run-id>
gh run download <run-id>   # inspect the artifact
```

## Reference implementations

- `.github/workflows/test-throughput.yml` + `cicd/scripts/benchmark-throughput.sh` — most-evolved current example. After #159 lands, the script will source from `cicd/scripts/lib/` as the canonical helpers reference.

## Do not

- Put test logic in inline workflow `run:` blocks. They orchestrate; the script does the work.
- Re-implement helpers that exist in `cicd/scripts/lib/`. Source them.
- Use the LLM judge for static / log / exit-code checks. Per CLAUDE.md → "LLM Judge Scope", the judge is for response-meaningfulness only.
- Skip the `if: always()` on the restore step. Production container left stopped = next workflow run starts broken.
