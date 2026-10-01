#!/usr/bin/env bash
# Hand the GPU to the server under test and wait for it to answer (#542).
#
# A host that shares its cards names its server under test in OLLAMA37_SERVICE
# (a user systemd unit) and what normally holds the card in OLLAMA37_YIELD_SERVICE.
# Both come from the runner's .env. A host that names neither is left alone: its
# server is whatever already listens on OLLAMA_HOST.
#
# Knobs:
#   OLLAMA37_SERVICE        unit to start; unset = touch nothing
#   OLLAMA37_YIELD_SERVICE  unit to stop first; unset = none
#   OLLAMA37_COMPOSE_FALLBACK  1 = on a host with no OLLAMA37_SERVICE, docker compose
#                           up the container instead of only waiting for a server;
#                           set by the setup workflow, which owns the container
#   OLLAMA37_ROOT           repo root holding docker/ (default: two levels above this script)
#   OLLAMA_HOST             default http://localhost:11434
set -euo pipefail
HOST=${OLLAMA_HOST:-http://localhost:11434}

if [ -n "${OLLAMA37_SERVICE:-}" ]; then
  systemctl --user stop 'ollama37-*' || true
  [ -z "${OLLAMA37_YIELD_SERVICE:-}" ] || systemctl --user stop "$OLLAMA37_YIELD_SERVICE"
  systemctl --user start "$OLLAMA37_SERVICE"
elif [ "${OLLAMA37_COMPOSE_FALLBACK:-}" = "1" ]; then
  root=${OLLAMA37_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
  (cd "$root/docker" && docker compose up -d)
else
  echo "gpu-yield: no OLLAMA37_SERVICE on this host; testing whatever serves $HOST"
fi

for _ in $(seq 1 60); do curl -sf "$HOST/api/tags" >/dev/null && exit 0; sleep 2; done
echo "gpu-yield: the server under test did not answer at $HOST" >&2
exit 1
