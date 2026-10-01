#!/usr/bin/env bash
# The structure #542 relies on, checked without a GPU:
#   1. Only cicd/tests/lib/ollama.ts imports the `ollama` client or names an
#      Ollama endpoint, so every model reply goes through the check and the judge.
#   2. No test script holds prompt or judge text: they live in prompts.yaml.
# Exit 1 with each offender named.
set -uo pipefail
cd "$(dirname "$0")/../tests"
failed=0

# `import type` brings in types only and cannot call anything.
offenders=$(grep -rnE "from 'ollama'|/api/(generate|chat)\b|new Ollama\(" --include='*.ts' . \
  | grep -v '^./node_modules/' | grep -v '^./lib/ollama.ts:' | grep -v '^./canary/' | grep -v 'import type ')
if [ -n "$offenders" ]; then
  echo "FAIL  only lib/ollama.ts may call the model; also calling it:"; echo "$offenders" | sed 's/^/        /'; failed=1
else
  echo "ok    only lib/ollama.ts calls the model"
fi

# A test script sends prompts by name; a literal prompt (a question mark inside a
# string) or a judge phrase in one means text escaped prompts.yaml.
offenders=$(grep -nE "'[^']*\?[^']*'|\"[^\"]*\?[^\"]*\"|yes or no" models.ts throughput.ts mcp.ts lib/*.ts \
  | grep -vE '^\S+:[0-9]+:\s*(//|\*)' | grep -v '^lib/prompts.ts:' | grep -vF '??' | grep -vF '?.')
if [ -n "$offenders" ]; then
  echo "FAIL  prompt or judge text outside prompts.yaml:"; echo "$offenders" | sed 's/^/        /'; failed=1
else
  echo "ok    no prompt or judge text outside prompts.yaml"
fi
exit $failed
