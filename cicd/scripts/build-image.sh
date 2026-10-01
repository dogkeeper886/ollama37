#!/usr/bin/env bash
# Check the builder toolchain, optionally build the runtime image, and retag it
# as the image compose runs (#542).
#
# Setup, not a test: it leaves a usable image or fails the job.
#
# Usage: build-image.sh [--build]
#   --build   make build-runtime-local-no-cache (about an hour of nvcc), then
#             retag ollama37:latest as dogkeeper886/ollama37:latest. Without it,
#             only the toolchain and the existing image are checked, and nothing
#             is retagged: retagging a stale image makes compose serve old code.
#
# Knobs:
#   OLLAMA_VERSION   required with --build; baked into the binary. A build without
#                    it defaults to 0.0.0 and the registry answers 412 to every gated
#                    pull, so this refuses rather than build that image.
#   OLLAMA37_ROOT    repo root holding docker/ (default: two levels above this script)
set -uo pipefail

ROOT=${OLLAMA37_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
failed=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; failed=1; }
in_builder() { docker run --rm ollama37-builder:latest "$@" 2>&1; }

echo "builder image"
docker image inspect ollama37-builder:latest >/dev/null 2>&1 && ok "ollama37-builder:latest exists" || fail "ollama37-builder:latest missing"
v=$(in_builder nvcc --version | grep -o 'release [0-9.]*'); [[ "$v" == "release 11.4" ]] && ok "CUDA $v" || fail "CUDA: '${v:-none}', want release 11.4"
v=$(in_builder gcc --version | head -1);                    [[ "$v" =~ \ 10\. ]] && ok "$v" || fail "GCC: '${v:-none}', want 10.x"
v=$(in_builder go version);                                 [[ "$v" =~ go1\.2[0-9] ]] && ok "$v" || fail "Go: '${v:-none}', want 1.20+"

if [ "${1:-}" = "--build" ]; then
  [ -n "${OLLAMA_VERSION:-}" ] || { echo "build-image: OLLAMA_VERSION is unset; refusing to build a 0.0.0 image" >&2; exit 1; }
  echo "runtime image (building ${OLLAMA_VERSION})"
  log=$(mktemp)
  if (cd "$ROOT/docker" && OLLAMA_VERSION="$OLLAMA_VERSION" make build-runtime-local-no-cache) >"$log" 2>&1 \
     && grep -q 'Runtime image built successfully' "$log" && ! grep -qE '[Ee]rror:' "$log"; then
    ok "built from local source"
  else
    fail "build failed; last lines:"; tail -20 "$log" | sed 's/^/        /'
  fi
  rm -f "$log"
else
  echo "runtime image"
fi

docker image inspect ollama37:latest >/dev/null 2>&1 && ok "ollama37:latest exists" || fail "ollama37:latest missing"
bytes=$(docker image inspect ollama37:latest --format '{{.Size}}' 2>/dev/null || echo 0)
gb=$(awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1073741824 }')
awk -v g="$gb" 'BEGIN { exit !(g >= 1 && g <= 3) }' && ok "$gb GB, without the build toolchain" || fail "$gb GB, want 1-3 GB (the toolchain must not ship)"
v=$(docker run --rm --entrypoint /usr/bin/ollama ollama37:latest --version 2>&1 | grep -o '[0-9][0-9.]*' | tail -1)
[ -n "$v" ] && [ "$v" != "0.0.0" ] && ok "baked version $v" || fail "baked version '${v:-none}'; a 0.0.0 image gets 412 on gated pulls"

if [ "${1:-}" = "--build" ] && [ "$failed" -eq 0 ]; then
  docker tag ollama37:latest dogkeeper886/ollama37:latest \
    && ok "retagged $(docker image inspect dogkeeper886/ollama37:latest --format '{{.Id}}' | cut -c1-19) as dogkeeper886/ollama37:latest" \
    || fail "retag"
fi

[ "$failed" -eq 0 ] && { echo "image OK"; exit 0; }
echo "build-image: checks failed" >&2
exit 1
