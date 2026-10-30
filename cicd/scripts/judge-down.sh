#!/usr/bin/env bash
# Remove the agent judge's server (#597), whether judge-up.sh just started it or a
# killed job left it behind. Nothing to remove is not a failure.
set -uo pipefail
NAME=ollama37-judge

docker rm -f "$NAME" >/dev/null 2>&1
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "judge-down: $NAME is still there" >&2
  exit 1
fi
