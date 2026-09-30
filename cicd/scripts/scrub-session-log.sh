#!/usr/bin/env bash
# Redact a Claude Code session log before it goes into .sessions/ (#532).
#
# Replaces scrub-session-log.py, removed in 478bf324 after it reported
# "clean, 0 findings" on a log still carrying a runner broker URL and four GPU UUIDs.
# Its denylist was per-host, written on another machine, and nothing tied it to the
# machine it ran on. Three rules follow from that:
#
#   1. Host values are DERIVED at run time, never written down: GPU UUIDs from
#      nvidia-smi, each runner's broker URL and its opaque path segment from the
#      runner's own .runner, the committer address and its local part from git
#      config. No literal lives in this file, in a denylist, or in a transcript.
#
#   2. It fails loudly when it cannot cover the host. A source that is present but
#      yields nothing -- nvidia-smi that answers empty, a .runner it cannot parse,
#      no git address -- exits non-zero instead of reporting clean. Every rule also
#      carries samples it must redact before the log is read, fragments included,
#      so a pass means the rules ran.
#
#   3. Nothing prints a secret: not this script, not the operator. Checking a scrub
#      by grepping for the value writes the value into the next session's log. Use
#      --verify, which counts by rule name.
#
# A leading fragment -- a partial UUID, a bare broker hostname, a subnet prefix,
# the first characters of the broker path -- identifies the machine as well as the
# whole value, so every rule matches those too.
#
# It redacts what it was told to look for and cannot prove the absence of a secret
# nobody listed. Read the diff before committing what it writes.
#
# Usage:
#   scrub-session-log.sh <session.jsonl> [<out-dir>]   redact into <out-dir> (default .sessions)
#   scrub-session-log.sh --verify <file>               counts per rule; exit 1 if any
#
# Knobs:
#   RUNNER_DIRS  space-separated runner folders to read .runner from
#                (default: every $HOME/actions-runner*/ that has one)
set -uo pipefail

die() { echo "scrub-session-log: $*" >&2; exit 1; }

VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && { VERIFY_ONLY=1; shift; }
SRC="${1:-}"
OUT_DIR="${2:-.sessions}"
[ -n "$SRC" ] || die "usage: $0 [--verify] <file> [<out-dir>]"
command -v jq >/dev/null 2>&1 || die "jq is required"

# ---------------------------------------------------------------- structural rules
# Shapes, not values, so they are safe to read.
RULE_NAMES=(gpu-uuid runner-broker private-ip email token)
RULE_RES=(
  'GPU-[0-9a-f]{8}(-[0-9a-f]{4}){0,3}(-[0-9a-f]{12})?'
  'pipelinesghubeus[0-9]*(\.[A-Za-z0-9.-]+)?(/[A-Za-z0-9_-]+)?'
  '(192\.168(\.[0-9]{1,3}){0,2}|172\.(1[6-9]|2[0-9]|3[01])(\.[0-9]{1,3}){0,2}|10(\.[0-9]{1,3}){3})'
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  '(gh[pousr]_|sk-ant-)[A-Za-z0-9_-]{8,}'
)
RULE_REPLS=(GPU-REDACTED BROKER-REDACTED PRIVATE-IP EMAIL-REDACTED TOKEN-REDACTED)
# Made-up values in each rule's shape. Whole values first, then the fragments a
# careless grep leaves behind.
SAMPLES=(
  'gpu-uuid|GPU-00000000-1111-2222-3333-444444444444'
  'gpu-uuid|GPU-00000000'
  'runner-broker|https://pipelinesghubeus99.actions.githubusercontent.com/AAAAAAAAAAAA/'
  'runner-broker|pipelinesghubeus99'
  'private-ip|192.168.1.2'
  'private-ip|192.168.1.'
  'email|someone@example.com'
  'token|ghp_AAAAAAAAAAAAAAAAAAAA'
)

# ------------------------------------------------------------------ derived values
# Read from their source, never typed, never echoed.
DERIVED=()
add_derived() { [ -n "${1:-}" ] && [ ${#1} -ge 6 ] && DERIVED+=("$1"); return 0; }
SOURCES=()

if command -v nvidia-smi >/dev/null 2>&1; then
  n0=${#DERIVED[@]}
  while read -r u; do add_derived "$u"; done < <(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
  [ ${#DERIVED[@]} -gt "$n0" ] || die "nvidia-smi is installed but returned no GPU UUIDs; this host is not covered"
  SOURCES+=("nvidia-smi: $(( ${#DERIVED[@]} - n0 )) value(s)")
fi

runner_dirs=${RUNNER_DIRS:-$(for f in "$HOME"/actions-runner*/.runner; do [ -f "$f" ] && dirname "$f"; done)}
for d in $runner_dirs; do
  [ -r "$d/.runner" ] || die "$d/.runner is not readable; this runner is not covered"
  n0=${#DERIVED[@]}
  while read -r u; do
    add_derived "$u"
    add_derived "$(printf '%s' "$u" | sed -E 's#^https?://[^/]+/##; s#/.*$##')"
  done < <(jq -r '.serverUrl // empty, .serverUrlV2 // empty' "$d/.runner" 2>/dev/null)
  [ ${#DERIVED[@]} -gt "$n0" ] || die "$d/.runner yields no broker URL; this runner is not covered"
  SOURCES+=("$(basename "$d")/.runner: $(( ${#DERIVED[@]} - n0 )) value(s)")
done

email=$(git config user.email 2>/dev/null || true)
[ -n "$email" ] || die "git config user.email is unset; the committer address is not covered"
n0=${#DERIVED[@]}
add_derived "$email"
add_derived "${email%%@*}"
SOURCES+=("git config: $(( ${#DERIVED[@]} - n0 )) value(s)")

# Each derived value also as its leading fragments, down to 8 characters, longest
# first so a whole value is never half-replaced by a shorter prefix of itself. Not
# a URL: its first 8 characters are "https://". The broker-shape rule covers the
# host, and the opaque path segment is derived and fragmented on its own.
is_url() { [[ "$1" =~ ^https?:// ]]; }
probe_of() { if is_url "$1"; then printf '%s' "$1"; else printf '%s' "${1:0:8}"; fi; }
FRAGS=()
for lit in "${DERIVED[@]}"; do
  if is_url "$lit" || [ ${#lit} -lt 8 ]; then FRAGS+=("$lit"); continue; fi
  for ((len = ${#lit}; len >= 8; len--)); do FRAGS+=("${lit:0:len}"); done
done
mapfile -t FRAGS < <(printf '%s\n' "${FRAGS[@]}" | awk '{ print length, $0 }' | sort -rn -k1,1 | cut -d' ' -f2- | awk '!seen[$0]++')

sed_args=()
for i in "${!RULE_NAMES[@]}"; do sed_args+=(-e "s#${RULE_RES[$i]}#${RULE_REPLS[$i]}#g"); done
for lit in "${FRAGS[@]}"; do
  esc=$(printf '%s' "$lit" | sed -e 's/[]\/$*.^[|?+(){}#]/\\&/g')
  sed_args+=(-e "s#${esc}#HOST-REDACTED#g")
done

# --------------------------------------------------------------------- 1. self-test
echo "sources"
for s in "${SOURCES[@]}"; do echo "  $s"; done
echo "self-test"
failed=0
for s in "${SAMPLES[@]}"; do
  name=${s%%|*}; sample=${s#*|}
  got=$(printf '%s' "$sample" | sed -E "${sed_args[@]}")
  if [ "$got" = "$sample" ]; then echo "  FAIL  $name: a ${#sample}-char sample was not redacted"; failed=1
  else echo "  ok    $name (${#sample} chars)"; fi
done
n=0
for lit in "${DERIVED[@]}"; do
  n=$((n + 1))
  for probe in "$lit" "$(probe_of "$lit")"; do
    got=$(printf 'x%sx' "$probe" | sed -E "${sed_args[@]}")
    case "$got" in *REDACTED*|*PRIVATE-IP*) ;; *) echo "  FAIL  derived #$n: not redacted"; failed=1 ;; esac
  done
done
echo "  ok    $n derived value(s), whole and 8-char prefix (URLs whole)"
[ "$failed" -eq 0 ] || die "self-test failed; nothing written"

# ------------------------------------------------------------------------ 2. verify
# By rule name and count. Never prints what it matched.
verify() {
  local f="$1" bad=0 c k=0
  echo "verify $f"
  for i in "${!RULE_NAMES[@]}"; do
    c=$(grep -cE -- "${RULE_RES[$i]}" "$f" 2>/dev/null || true)
    printf '  %-14s %s\n' "${RULE_NAMES[$i]}" "${c:-0}"
    [ "${c:-0}" -eq 0 ] || bad=1
  done
  for lit in "${DERIVED[@]}"; do
    k=$((k + 1))
    c=$(grep -cF -- "$(probe_of "$lit")" "$f" 2>/dev/null || true)
    printf '  %-14s %s\n' "derived #$k" "${c:-0}"
    [ "${c:-0}" -eq 0 ] || bad=1
  done
  return $bad
}

[ -f "$SRC" ] || die "no such file: $SRC"

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify "$SRC" && { echo "clean"; exit 0; }
  die "NOT CLEAN"
fi

mkdir -p "$OUT_DIR"
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
sed -E "${sed_args[@]}" "$SRC" > "$tmp"
verify "$tmp" || die "redaction incomplete; nothing written to $OUT_DIR"
jq -e . "$tmp" >/dev/null 2>&1 || die "redaction broke the JSON; nothing written"

dst="$OUT_DIR/$(basename "$SRC")"
mv "$tmp" "$dst"; chmod 644 "$dst"; trap - EXIT
echo "wrote $dst ($(wc -l < "$dst") lines)"
echo "verify it again later with: $0 --verify $dst"
