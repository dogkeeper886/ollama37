#!/usr/bin/env bash
# Start the agent judge's own server and wait for it to answer (#597).
#
# The judge is ollama37 v2.3.4, pinned by digest, so no build moves it: a CI build
# retags dogkeeper886/ollama37:latest, a local build replaces the host's own
# server, and the judge runs neither. It reads the host's model store read-only,
# never prunes it, and fails when its model is not the pinned one. runTest starts
# it after the body and removes it (judge-down.sh) before the GPU goes back.
#
# Knobs:
#   OLLAMA37_MODELS_DIR  the host's model store (the runner's .env sets it)
set -euo pipefail

NAME=ollama37-judge
IMAGE=dogkeeper886/ollama37:v2.3.4@sha256:2804a2595f9f429afcbd30d747955e7ba46ab9da74b9846485450001136aae21
MODEL=ornith-1.5:9b
DIGEST=e5df7dcdd8a263994df62d610317e07be0d6af23f96fcbd8543273058fce575e
PORT=11435
MODELS=${OLLAMA37_MODELS_DIR:?judge-up: OLLAMA37_MODELS_DIR must name the model store}

# A job killed mid-run leaves its judge behind; this one replaces it.
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --runtime=nvidia --gpus all -p "$PORT:11434" \
  -e OLLAMA_NOPRUNE=1 -v "$MODELS:/root/.ollama/models:ro" "$IMAGE" >/dev/null

tags=
for _ in $(seq 1 60); do tags=$(curl -sf "http://localhost:$PORT/api/tags") && break; sleep 2; done
[ -n "$tags" ] || { echo "judge-up: $NAME did not answer on :$PORT" >&2; exit 1; }
got=$(jq -r --arg m "$MODEL" '.models[] | select(.name == $m) | .digest' <<<"$tags")
[ "$got" = "$DIGEST" ] || { echo "judge-up: $MODEL is '${got:-missing}', want $DIGEST" >&2; exit 1; }
echo "judge-up: $MODEL on v2.3.4, :$PORT"
