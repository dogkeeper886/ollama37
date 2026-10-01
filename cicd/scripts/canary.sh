#!/usr/bin/env bash
# Prove the judge still catches garbage (#542): a test that cannot go red proves
# nothing, and "0 failed" once hid that no reply was judged at all.
#
# Runs models.ts against a fake Ollama twice. Answering fluent word salad -- which
# passes the script check -- the judge must call it garbage. Answering a sound
# sentence, it must not. Exit 1 when either verdict is wrong or missing. The GPU
# and VRAM checks fail against a fake server by design; only the judge's verdict
# is read.
#
# Knobs: JUDGE_BASE_URL / JUDGE_MODEL / CLAUDE_CODE_OAUTH_TOKEN, as for the tests.
set -uo pipefail
cd "$(dirname "$0")/../tests"
port=18577
out=$(mktemp -d)
failed=0

for mode in salad good; do
  npx tsx canary/mock-ollama.ts "$port" "$mode" & mock=$!
  sleep 1
  env -u OLLAMA37_SERVICE -u OLLAMA37_YIELD_SERVICE -u OLLAMA37_CONTAINER OLLAMA_HOST="http://127.0.0.1:$port" \
    npx tsx models.ts --models canary:mock --output "$out/$mode.json" >/dev/null 2>&1
  kill "$mock" 2>/dev/null; wait "$mock" 2>/dev/null
  verdict=$(jq -r '.results[] | select(.prompt == "short-answer") | .judge.verdict' "$out/$mode.json" 2>/dev/null)
  # The answer that passes a reply (prompts.yaml); garbage must get the other one.
  pass=$(jq -r '.results[] | select(.prompt == "short-answer") | .judgePass' "$out/$mode.json" 2>/dev/null)
  if [ "$mode" = good ]; then want=$pass
  elif [ "$pass" = yes ]; then want=no
  else want=yes; fi
  if [ -n "$verdict" ] && [ "$verdict" = "$want" ]; then
    echo "  ok    $mode: judge said $verdict"
  else
    echo "  FAIL  $mode: judge said '${verdict:-nothing}', want '$want'"; failed=1
  fi
done
rm -rf "$out"
[ "$failed" -eq 0 ] && { echo "canary OK: the judge catches garbage and passes a sound reply"; exit 0; }
echo "canary: the judge cannot be trusted" >&2
exit 1
