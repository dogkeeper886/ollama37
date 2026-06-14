/**
 * Agent judge configuration.
 *
 * ollama37 keeps its run config in types.ts (DEFAULT_CONFIG); this file provides
 * the CONFIG block the ported ACP agent judge (judge/agent-judge.ts) reads.
 */
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
};
