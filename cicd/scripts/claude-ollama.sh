#!/usr/bin/env bash
# Run Claude Code against a local ollama37 server instead of Anthropic's API.
#
#   cicd/scripts/claude-ollama.sh [model] [claude args...]
#   cicd/scripts/claude-ollama.sh qwen3.8:27b -p "explain llm/server.go"
#   cicd/scripts/claude-ollama.sh --acp [model]
#
# --acp starts the Claude ACP agent (the CI agent judge's default) instead of
# the claude CLI, so the judge can run on ollama. From cicd/tests:
#
#   JUDGE_MODE=dual JUDGE_AGENT="../scripts/claude-ollama.sh --acp" npm run test
#
# Both inputs are read from the environment under this script's own prefix, so
# neither collides with another program's variables — notably OLLAMA_HOST, which
# the ollama CLI reads to find a server and `ollama serve` reads to decide what
# to bind:
#
#   CLAUDE_OLLAMA_HOST   server URL, else http://localhost:11434
#   CLAUDE_OLLAMA_MODEL  model, else qwen3.8:27b (the first argument wins)
#
# The model must support tools: Claude Code sends tool definitions on every
# request, and a model without that capability is refused by the server with
# "does not support tools" (gemma3:270m, for one). Checked below rather than
# left to fail mid-session.
set -euo pipefail

ACP=""
[ "${1:-}" = "--acp" ] && { ACP=1; shift; }

HOST="${CLAUDE_OLLAMA_HOST:-http://localhost:11434}"
MODEL="${1:-${CLAUDE_OLLAMA_MODEL:-qwen3.8:27b}}"
[ $# -gt 0 ] && shift || true

# The ACP agent is an npm dependency of the test runner and runs its own
# bundled Claude binary, so it needs node rather than the claude CLI.
ACP_ENTRY="$(dirname "$0")/../tests/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
if [ -n "$ACP" ]; then
  [ -f "$ACP_ENTRY" ] || { echo "ACP agent not installed: run npm ci in cicd/tests" >&2; exit 1; }
else
  command -v claude >/dev/null || {
    echo "claude not found. Install: curl -fsSL https://claude.ai/install.sh | bash" >&2
    exit 1
  }
fi

curl -sf "${HOST}/api/version" >/dev/null || {
  echo "no ollama server at ${HOST}" >&2
  exit 1
}

caps=$(curl -s "${HOST}/api/show" -d "{\"model\":\"${MODEL}\"}" | jq -r '.capabilities // [] | join(",")')
case ",${caps}," in
  *,tools,*) ;;
  *)
    echo "${MODEL} does not support tools (capabilities: ${caps:-none}); Claude Code needs it" >&2
    echo "tools-capable models on this server:" >&2
    for m in $(curl -s "${HOST}/api/tags" | jq -r '.models[].name'); do
      curl -s "${HOST}/api/show" -d "{\"model\":\"${m}\"}" \
        | jq -e '.capabilities // [] | index("tools")' >/dev/null 2>&1 && echo "  ${m}" >&2
    done
    exit 1
    ;;
esac

echo "claude${ACP:+ (acp)} -> ${HOST} (${MODEL})" >&2

# ANTHROPIC_API_KEY must be empty, not unset: a key in the environment takes
# precedence over the base URL and the session would go to Anthropic instead.
# The haiku model is Claude Code's side-call model; ollama has no such tag.
export ANTHROPIC_BASE_URL="${HOST}"
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_API_KEY=""
export ANTHROPIC_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}"
export DISABLE_AUTOUPDATER=1

# stdout is the ACP channel in --acp mode: nothing else may write to it.
[ -n "$ACP" ] && exec node "$ACP_ENTRY"
exec claude --model "${MODEL}" "$@"
