#!/usr/bin/env bash
# Give the GPU back and wait for the restored server to answer (#542).
#
# Stops OLLAMA37_SERVICE and starts OLLAMA37_YIELD_SERVICE again, but only where
# the host named both and left the yielded one enabled -- so a job killed mid-run
# heals on the next one. A host that names neither is untouched. The agent judge
# runs after this, because on a shared host its server is the one the test
# borrowed the card from.
#
# Knobs: as gpu-yield.sh.
set -euo pipefail
HOST=${OLLAMA_HOST:-http://localhost:11434}

[ -n "${OLLAMA37_SERVICE:-}" ] || exit 0
systemctl --user stop "$OLLAMA37_SERVICE"
[ -n "${OLLAMA37_YIELD_SERVICE:-}" ] || exit 0
systemctl --user is-enabled --quiet "$OLLAMA37_YIELD_SERVICE" || exit 0
systemctl --user start "$OLLAMA37_YIELD_SERVICE"

for _ in $(seq 1 60); do curl -sf "$HOST/api/tags" >/dev/null && exit 0; sleep 2; done
echo "gpu-restore: $OLLAMA37_YIELD_SERVICE did not come back at $HOST" >&2
exit 1
