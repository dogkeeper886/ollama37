/**
 * Agent judge configuration.
 *
 * ollama37 keeps its run config in types.ts (DEFAULT_CONFIG); this file provides
 * the CONFIG block the ported ACP agent judge (judge/agent-judge.ts) reads.
 */

/** Forward only the named env vars (comma-separated) from the parent process —
 *  the spawned MCP server's creds come from the environment, never hardcoded. */
export function pickEnv(names: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const k of names.split(',').map((s) => s.trim()).filter(Boolean)) {
    const v = process.env[k];
    if (v) out[k] = v;
  }
  return out;
}

export const CONFIG = {
  projectName: 'ollama37',

  // Agent Judge — an opt-in second opinion. The default verdict is the simple
  // (deterministic, model-free) judge; set JUDGE_MODE=dual to also run this.
  judge: {
    // 'simple' (default) = deterministic checks only. 'dual' = also run the agent judge.
    mode: process.env.JUDGE_MODE || 'simple',
    // Command that launches the ACP agent the judge talks to. Empty → the bundled
    // Claude ACP agent (@agentclientprotocol/claude-agent-acp), keyless via the
    // agent's own auth (~/.claude / CLAUDE_CODE_OAUTH_TOKEN). Set to any other ACP
    // agent's command to swap models/vendors — config, not code.
    agent: process.env.JUDGE_AGENT || '',
    timeout: 300000,
    stdoutLimit: 1000,
    stderrLimit: 500,
    logsLimit: 3000,
  },

  // MCP tool-call test (cli.ts test-mcp). Drives a REAL stdio MCP server so a
  // model's tool-calling can be checked end-to-end. testlink-mcp is only the
  // default target — point --mcp-command / --mcp-args at any stdio MCP server.
  mcp: {
    command: process.env.MCP_COMMAND || 'npx',
    args: (process.env.MCP_ARGS || '-y testlink-mcp').split(' ').filter(Boolean),
    cwd: process.env.MCP_CWD || undefined,
    prompt: process.env.MCP_PROMPT || 'List the projects.',
    // Env var names to forward to the spawned server (its creds). Values come
    // from the environment (.env locally / secret in CI). Default = testlink's.
    env: pickEnv(process.env.MCP_ENV || 'TESTLINK_URL,TESTLINK_API_KEY'),
  },
};
