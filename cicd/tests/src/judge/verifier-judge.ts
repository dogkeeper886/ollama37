/**
 * Verifier Judge — independently verifies a model's MCP answer against LIVE ground truth.
 *
 * Unlike the sandboxed AgentJudge (which opens its session with `mcpServers: []` and
 * rejects every tool call), the verifier spawns the same ACP agent WITH the test's MCP
 * server attached and lets it call the tools itself to check the answer. Tool access is
 * restricted to a per-case allow-list (read-only by default) enforced two ways:
 *
 *   1. Claude SDK `disallowedTools` — every server tool outside the allow-list is denied
 *      by the agent's own SDK, before execution (hard).
 *   2. our `requestPermission` backstop — allow only allow-listed tools (matched on the
 *      real tool name in `toolCall._meta.claudeCode.toolName`), else reject.
 *
 * This file does NOT touch `agent-judge.ts` — the generic judge stays sandboxed.
 */

import { spawn, type ChildProcess, type StdioOptions } from 'node:child_process';
import { Readable, Writable } from 'node:stream';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import {
  ClientSideConnection,
  ndJsonStream,
  PROTOCOL_VERSION,
  type Client,
  type SessionNotification,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
} from '@agentclientprotocol/sdk';
import type { McpServerConfig } from '../mcp/host.js';
import type { Judgment } from '../types.js';
import { CONFIG } from '../config.js';

/** One model's answer to verify against the live server. */
export interface VerifyTarget {
  testId: string;
  prompt: string;
  answer: string;
}

export interface VerifierConfig {
  /** The stdio MCP server the host used (same command/args/env). */
  server: McpServerConfig;
  /** Rule prefix: tools are addressed as `mcp__<serverName>__<tool>`. */
  serverName: string;
  /** Every tool the server exposes — used to derive the deny-list (deny-by-default). */
  toolNames: string[];
  /** Allow-list predicate over the BARE tool name. Default: read-only. */
  isAllowedTool?: (bareToolName: string) => boolean;
}

/** Default per-case policy: read-only — `list_`, `read_`, `get_` prefixes. */
export const READ_ONLY = (name: string): boolean => (/^(list_|read_|get_)/).test(name);

export class VerifierJudge {
  private agentCmd: string;
  private cwd: string;
  private child?: ChildProcess;
  private conn?: ClientSideConnection;
  private sessionId?: string;
  private turnText = '';

  private readonly cfg: VerifierConfig;
  private readonly allow: (bare: string) => boolean;
  /** mcp__<server>__<tool> rules for every NON-allowed tool. */
  private readonly disallowedTools: string[];

  constructor(cfg: VerifierConfig, agentCmd: string = CONFIG.judge.agent, cwd: string = process.cwd()) {
    this.cfg = cfg;
    this.agentCmd = agentCmd;
    this.cwd = cwd;
    this.allow = cfg.isAllowedTool ?? READ_ONLY;
    this.disallowedTools = cfg.toolNames
      .filter((t) => !this.allow(t))
      .map((t) => `mcp__${cfg.serverName}__${t}`);
    process.once('exit', () => this.kill());
  }

  /** `mcp__server__tool` (or a bare name) → the bare tool name. */
  private bareName(name: string): string {
    const parts = name.split('__');
    return parts.length >= 3 ? parts.slice(2).join('__') : name;
  }

  private async withTimeout<T>(p: Promise<T>, label: string): Promise<T> {
    let timer: NodeJS.Timeout;
    const timeout = new Promise<never>((_, rej) => {
      timer = setTimeout(() => rej(new Error(`${label} exceeded ${CONFIG.judge.timeout}ms`)), CONFIG.judge.timeout);
    });
    try {
      return await Promise.race([p, timeout]);
    } finally {
      clearTimeout(timer!);
    }
  }

  /** Spawn the configured ACP agent as a stdio child (same logic as AgentJudge). */
  private spawnAgent(): ChildProcess {
    const env = { ...process.env };
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_ENTRYPOINT;
    delete env.CLAUDE_CODE_SSE_PORT;
    const stdio: StdioOptions = ['pipe', 'pipe', 'inherit'];

    if (this.agentCmd) {
      return spawn(this.agentCmd, { cwd: this.cwd, stdio, env, shell: true });
    }
    const require = createRequire(import.meta.url);
    const pkg = require.resolve('@agentclientprotocol/claude-agent-acp/package.json');
    const entry = resolve(dirname(pkg), 'dist/index.js');
    return spawn(process.execPath, [entry], { cwd: this.cwd, stdio, env });
  }

  private async ensureStarted(): Promise<void> {
    if (this.sessionId) return;

    const child = this.spawnAgent();
    this.child = child;

    const stream = ndJsonStream(
      Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>,
      Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>,
    );

    const client: Client = {
      sessionUpdate: async (params: SessionNotification): Promise<void> => {
        const u = params.update;
        if (u.sessionUpdate === 'agent_message_chunk' && u.content.type === 'text') {
          this.turnText += u.content.text;
        }
      },
      // The backstop: allow only allow-listed tools, else reject. The real tool
      // name is on toolCall._meta.claudeCode.toolName; disallowedTools has already
      // hard-denied writes at the SDK layer, so this only ever sees allowed tools —
      // but we re-check by name and default to reject if anything is unexpected.
      requestPermission: async (params: RequestPermissionRequest): Promise<RequestPermissionResponse> => {
        const meta = (params.toolCall as { _meta?: { claudeCode?: { toolName?: string } }; title?: string });
        const rawName = meta?._meta?.claudeCode?.toolName ?? meta?.title ?? '';
        const allowed = this.allow(this.bareName(String(rawName)));
        const want = allowed ? 'allow' : 'reject';
        const opt =
          params.options.find((o) => o.kind?.startsWith(want)) ??
          params.options.find((o) => o.kind?.startsWith('reject'));
        return opt
          ? { outcome: { outcome: 'selected', optionId: opt.optionId } }
          : { outcome: { outcome: 'cancelled' } };
      },
      readTextFile: async () => { throw new Error('filesystem access disabled for the verifier'); },
      writeTextFile: async () => { throw new Error('filesystem access disabled for the verifier'); },
    };

    this.conn = new ClientSideConnection(() => client, stream);
    try {
      await this.withTimeout(this.conn.initialize({
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: `${CONFIG.projectName}-verifier`, version: '1.0.0' },
      }), 'verifier initialize');

      const env = Object.entries(this.cfg.server.env ?? {}).map(([name, value]) => ({ name, value }));
      const session = await this.withTimeout(
        this.conn.newSession({
          cwd: this.cwd,
          // Populate the server (the generic judge keeps this empty) and hard-deny
          // every non-allowed tool via the agent's own SDK config.
          mcpServers: [{ name: this.cfg.serverName, command: this.cfg.server.command, args: this.cfg.server.args, env }],
          _meta: { claudeCode: { options: { disallowedTools: this.disallowedTools } } },
        }),
        'verifier session/new',
      );
      this.sessionId = session.sessionId;
    } catch (e) {
      this.kill();
      throw e;
    }
  }

  private kill(): void {
    try { this.child?.kill('SIGKILL'); } catch { /* already gone */ }
    this.child = undefined;
    this.conn = undefined;
    this.sessionId = undefined;
  }

  /** Probe: spawn, open a session (with the server attached), one throwaway turn. */
  async isAvailable(): Promise<boolean> {
    try {
      await this.ensureStarted();
      const reply = await this.promptAgent('Reply with exactly: ok');
      if (!reply.trim()) throw new Error('verifier agent produced no output on probe turn');
      return true;
    } catch (error) {
      process.stderr.write(`  [verify] Agent not reachable: ${error}\n`);
      this.kill();
      return false;
    }
  }

  private buildPrompt(t: VerifyTarget): string {
    return JSON.stringify({
      role: `You independently verify another model's answer for ${CONFIG.projectName}. You have READ-ONLY access to the ${this.cfg.serverName} MCP tools.`,
      task: 'Call the tools yourself to find the real data, then judge whether the model\'s answer is correct and grounded in that real data. Do NOT trust the answer — check it.',
      user_prompt: t.prompt,
      model_answer: t.answer,
      rules: [
        'Use the read-only tools to retrieve the ground truth.',
        'pass=true only if the answer matches the real data you retrieved.',
        'If the answer states facts the tools contradict or do not contain, pass=false.',
      ],
      respond: {
        format: 'Respond with a single JSON object and nothing else',
        fields: {
          testId: t.testId,
          pass: 'true if the answer matches the live tool data, false otherwise',
          reason: 'Brief explanation',
          evidence: 'The tool result you verified against (required if pass is false)',
        },
      },
    }, null, 2);
  }

  private extractJson(text: string): string | null {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start === -1 || end === -1 || end < start) return null;
    return text.substring(start, end + 1);
  }

  private async promptAgent(prompt: string): Promise<string> {
    this.turnText = '';
    try {
      await this.withTimeout(
        this.conn!.prompt({ sessionId: this.sessionId!, prompt: [{ type: 'text', text: prompt }] }),
        'verifier turn',
      );
    } catch (e) {
      this.kill();
      throw e;
    }
    return this.turnText;
  }

  private async verifyOne(t: VerifyTarget): Promise<Judgment> {
    const responseText = await this.promptAgent(this.buildPrompt(t));
    if (!responseText) {
      return { testId: t.testId, pass: false, reason: 'Verifier returned empty response' };
    }
    const json = this.extractJson(responseText);
    if (!json) {
      return { testId: t.testId, pass: false, reason: `No JSON in verifier response: ${responseText.substring(0, 200)}` };
    }
    try {
      const judgment = JSON.parse(json) as Judgment;
      judgment.testId = t.testId;
      if (typeof judgment.pass === 'string') {
        judgment.pass = (judgment.pass as unknown as string).toLowerCase() === 'true';
      }
      if (typeof judgment.pass !== 'boolean') {
        return { testId: t.testId, pass: false, reason: `Verifier response missing "pass": ${responseText.substring(0, 200)}` };
      }
      if (!judgment.reason) judgment.reason = judgment.pass ? 'Verified (no reason provided)' : 'Failed (no reason provided)';
      return judgment;
    } catch {
      return { testId: t.testId, pass: false, reason: `Failed to parse verifier response: ${responseText.substring(0, 200)}` };
    }
  }

  /** Verify each target over one reused session; tears the agent down when done. */
  async verify(targets: VerifyTarget[]): Promise<Judgment[]> {
    const out: Judgment[] = [];
    try {
      for (let i = 0; i < targets.length; i++) {
        const t = targets[i];
        process.stderr.write(`  [verify] Verifying ${i + 1}/${targets.length}: ${t.testId} (deny ${this.disallowedTools.length} write/non-read tools)...\n`);
        try {
          await this.ensureStarted();
          const judgment = await this.verifyOne(t);
          out.push(judgment);
          process.stderr.write(`  [verify] ${t.testId}: ${judgment.pass ? 'PASS' : 'FAIL'} — ${judgment.reason}\n`);
        } catch (error) {
          process.stderr.write(`  [verify] Failed to verify ${t.testId}: ${error}\n`);
          out.push({ testId: t.testId, pass: false, reason: 'Verifier failed: ' + String(error) });
        }
      }
    } finally {
      this.kill();
    }
    return out;
  }
}
