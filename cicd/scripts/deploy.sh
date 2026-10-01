#!/usr/bin/env bash
# Check the ollama37 container is up, sees the GPU, and serves (#542).
#
# Setup, not a test: it leaves a healthy server or fails the job, and adds
# nothing to a test count. Carries every check the YAML runtime suite had,
# including the six the Vitest port dropped: CUDA detection in the server log,
# GPU count against nvidia-smi, marketing GPU names, and three /api/metrics
# consistency checks.
#
# Usage: deploy.sh [--restart]
#   --restart   docker compose down, then up, before checking
#
# Knobs:
#   OLLAMA37_ROOT       repo root holding docker/ (default: two levels above this script)
#   OLLAMA37_CONTAINER  default ollama37
#   OLLAMA_HOST         default http://localhost:11434
#   COMPOSE_FILE        passed through to docker compose (the runner's .env sets it)
set -uo pipefail

ROOT=${OLLAMA37_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
CONTAINER=${OLLAMA37_CONTAINER:-ollama37}
HOST=${OLLAMA_HOST:-http://localhost:11434}
failed=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; failed=1; }

if [ "${1:-}" = "--restart" ]; then
  (cd "$ROOT/docker" && docker compose down && docker compose up -d) || { echo "deploy: compose up failed" >&2; exit 1; }
fi

echo "container"
status=not_found
for _ in $(seq 1 30); do
  status=$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null) || status=not_found
  [ "$status" = healthy ] && break
  sleep 2
done
[ "$status" = healthy ] && ok "healthy" || fail "health is '$status' after 60s"
v=$(docker exec "$CONTAINER" ollama --version 2>&1 | tail -1)
[[ "$v" == *version* ]] && ok "$v" || fail "ollama --version: $v"

echo "GPU passthrough"
smi=$(docker exec "$CONTAINER" nvidia-smi 2>&1)
if [[ "$smi" == *"CUDA Version"* && "$smi" != *"NVIDIA-SMI has failed"* && "$smi" != *"No devices were found"* ]]; then
  ok "nvidia-smi: $(grep -o 'CUDA Version: [0-9.]*' <<<"$smi")"
else fail "nvidia-smi inside the container"; fi
docker exec "$CONTAINER" ldconfig -p 2>/dev/null | grep -qi cuda && ok "CUDA libraries on the loader path" || fail "no CUDA library on the loader path"
[ -e /dev/nvidia-uvm ] || sudo -n nvidia-modprobe -u -c=0 2>/dev/null || true
[ -e /dev/nvidia-uvm ] && ok "/dev/nvidia-uvm exists" || fail "/dev/nvidia-uvm missing"
# The server's own view: CUDA, a real compute capability, never a CPU fallback.
line=$(docker logs "$CONTAINER" 2>&1 | grep -i "inference compute" | tail -1)
if [[ "$line" =~ library=CUDA ]] && [[ "$line" =~ compute=[0-9]+\.[0-9]+ ]] && [[ ! "$line" =~ library=cpu ]]; then
  ok "server detects CUDA ($(grep -oE 'compute=[0-9.]+' <<<"$line" | head -1))"
else fail "server log shows no CUDA inference compute (or a CPU fallback)"; fi

echo "API"
n=$(curl -sf "$HOST/api/tags" | jq '.models | length' 2>/dev/null)
[ -n "$n" ] && ok "/api/tags lists $n model(s)" || fail "/api/tags"

echo "/api/metrics"
m=$(curl -sf "$HOST/api/metrics" 2>/dev/null)
if [ -z "$m" ] || ! jq -e . >/dev/null 2>&1 <<<"$m"; then
  fail "/api/metrics did not answer JSON"
else
  chk() { jq -e "$2" >/dev/null 2>&1 <<<"$m" && ok "$1" || fail "$1"; }
  chk "carries gpus, models, errors, totals" '(.gpus|type)=="array" and (.models|type)=="array" and (.errors|type)=="object" and (.totals|type)=="object"'
  chk "error counters present" '[.errors.load_failures, .errors.load_require_full, .errors.load_other, .errors.evictions_total] | all(type=="number")'
  chk "totals: gpu_count >= 1, loaded_models >= 0" '(.totals.gpu_count|type)=="number" and .totals.gpu_count>=1 and (.totals.loaded_models|type)=="number" and .totals.loaded_models>=0'
  chk "gpus[] length matches totals.gpu_count" '(.gpus|length) == .totals.gpu_count'
  chk "each GPU has id, name and vram_total > 0" '.gpus | length>0 and all((.id|type)=="string" and (.name|type)=="string" and (.vram_total|type)=="number" and .vram_total>0)'
  chk "GPU names are marketing names, not CUDA0-style labels (#131)" '.gpus | all((.name // "") | test("^$|^(CUDA|HIP|Vulkan)[0-9]+$") | not)'
  want=$(docker exec "$CONTAINER" nvidia-smi -L 2>/dev/null | grep -c '^GPU')
  got=$(jq '.totals.gpu_count' <<<"$m")
  [ "$want" = "$got" ] && ok "gpu_count $got matches nvidia-smi (#130)" || fail "gpu_count $got, nvidia-smi sees $want (#130)"
fi

[ "$failed" -eq 0 ] && { echo "setup OK"; exit 0; }
echo "deploy: setup checks failed" >&2
exit 1
