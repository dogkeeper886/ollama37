#!/usr/bin/env bash
# The structure #542 relies on, checked without a GPU:
#   1. Only cicd/tests/lib/ollama.ts imports the `ollama` client or names an
#      Ollama endpoint, so every model reply goes through the check and the judge.
#   2. No test script holds prompt or judge text: they live in prompts.yaml.
# Exit 1 with each offender named.
set -uo pipefail
cd "$(dirname "$0")/../tests"
failed=0

# `import type` brings in types only and cannot call anything. Any other import
# of the client -- single or double quotes, static, dynamic or require -- counts.
offenders=$(grep -rnE "(from|import\(|require\()[[:space:]]*['\"]ollama['\"]|/api/(generate|chat)\b|new Ollama\(" --include='*.ts' . \
  | grep -v '^./node_modules/' | grep -v '^./lib/ollama.ts:' | grep -v '^./canary/' | grep -v 'import type ')
if [ -n "$offenders" ]; then
  echo "FAIL  only lib/ollama.ts may call the model; also calling it:"; echo "$offenders" | sed 's/^/        /'; failed=1
else
  echo "ok    only lib/ollama.ts calls the model"
fi

# A test script sends prompts by name. Prompt or judge text in one -- a question
# (a word straight before "?", then a space or a closing quote; a ternary has a
# space before its "?", an optional field a colon after it), "yes or no", or an
# instruction like "Reply with ..." -- means
# text escaped prompts.yaml. `??` and `?.` are operators, removed from each line
# before the search so a prompt sharing a line with one is still found.
offenders=$(grep -nE '.' models.ts throughput.ts mcp.ts lib/*.ts \
  | grep -v '^lib/prompts.ts:' | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*|/\*)' \
  | sed -E 's/\?\?|\?\././g' \
  | grep -E "[[:alnum:]]\?([[:space:]]|['\"\`])|yes or no|['\"\`](Reply|Answer|Respond) with" )
if [ -n "$offenders" ]; then
  echo "FAIL  prompt or judge text outside prompts.yaml:"; echo "$offenders" | sed 's/^/        /'; failed=1
else
  echo "ok    no prompt or judge text outside prompts.yaml"
fi
exit $failed
