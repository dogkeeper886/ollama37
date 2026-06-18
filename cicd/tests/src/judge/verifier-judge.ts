/**
 * Verifier Judge — independently verifies a model's MCP answer against LIVE ground truth.
 *
 * Unlike the sandboxed AgentJudge (which opens its session with `mcpServers: []` and
 * rejects every tool call), the verifier spawns the same ACP agent WITH the test's MCP
 * server attached and lets it call the tools itself to check the answer. Tool access is
 * restricted to a per-case **exact allow-list** of tool names (read-only by intent),
 * enforced two ways:
 *
 *   1. `requestPermission` backstop — the gate. For an MCP tool the permission request's
 *      `toolCall.title` IS the `mcp__<server>__<tool>` id (the bundled claude-agent-acp
 *      sets it there; `_meta.claudeCode.toolName` is NOT populated on permission requests).
 *      We strip it to the bare name and ALLOW only if it's in the exact allow-list, else
 *      REJECT. Built-in tools (Bash/Edit/Write/…) carry a prose title that is never in the
 *      allow-list, so they are rejected too; an empty/unknown name also rejects (fail-closed).
 *   2. Claude SDK `disallowedTools` (via newSession `_meta.claudeCode.options`) — defence in
 *      depth: every server tool NOT in the allow-list is also denied by the agent's own SDK
 *      before it reaches permission.
 *
 * Read-only is therefore by *exact allow-list*, never a name prefix (a prefix would let a
 * `get_and_purge` through). This file does NOT touch `agent-judge.ts` — the generic judge
 * stays sandboxed.
 *
 * NOTE: that the SDK actually hard-denies, and that the agent does call the tools, is proven
 * live by the forced-write-refusal test in the validation task (#281) before this is trusted
 * against a real backend.
 */

import { spawn, type ChildProcess, type StdioOptions } from 'node:child_process';
import { Readable, Writable } from 'node:stream';
import { createRequire } from 'node:module';
import { dirname, resolve, join } from 'node:path';
import { mkdirSync, writeFileSync, existsSync, symlinkSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
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

/** VERIFY_DEBUG=1 → reveal every ACP event (sessionUpdate kinds, tool calls, permission/trust
 *  requests) so nothing the adapter emits is swallowed silently. */
const DEBUG = !!process.env.VERIFY_DEBUG;

/** One model's answer to verify against the live server. */
export interface VerifyTarget {
  testId: string;
  prompt: string;
  answer: string;
}

export interface VerifierConfig {
  /** The stdio MCP server the host used (same command/args/env). */
  server: McpServerConfig;
  /** Rule prefix: the server's tools are `mcp__<serverName>__<tool>`. */
  serverName: string;
  /** Every tool the server exposes — used to derive the SDK deny-list (deny-by-default). */
  toolNames: string[];
  /** EXACT bare tool names the verifier may call (read-only by intent). Anything not here is
   *  rejected. Empty ⇒ the verifier can call nothing (fail-closed). */
  allowTools: string[];
}

export class VerifierJudge {
  private agentCmd: string;
  private isolatedCfgDir?: string;
  private isolatedCwd?: string;
  private child?: ChildProcess;
  private conn?: ClientSideConnection;
  private sessionId?: string;
  private turnText = '';
  /** Raw live tool result(s) the agent received this turn — captured from `tool_call_update`
   *  events, independent of what the agent says it saw. Surfaced as Judgment.evidence. */
  private toolEvidence = '';
  /** `toolCallId`s of allow-listed tool calls seen this turn — gates evidence capture so the report
   *  only ever shows ground truth from a tool we permit (not a denied/built-in call). Correlated by
   *  id because a `tool_call_update` carries the id but not the tool title. */
  private allowedCallIds = new Set<string>();

  private readonly cfg: VerifierConfig;
  private readonly allow: Set<string>;
  /** `mcp__<server>__<tool>` rules for every tool NOT in the allow-list (SDK hard-deny). */
  private readonly disallowedTools: string[];

  constructor(cfg: VerifierConfig, agentCmd: string = CONFIG.judge.agent) {
    this.cfg = cfg;
    this.agentCmd = agentCmd;
    this.allow = new Set(cfg.allowTools);
    this.disallowedTools = cfg.toolNames
      .filter((t) => !this.allow.has(t))
      .map((t) => `mcp__${cfg.serverName}__${t}`);
    process.once('exit', () => this.kill());
  }

  /** `mcp__server__tool` (or a bare name) → the bare tool name. */
  private bareName(name: string): string {
    const parts = name.split('__');
    return parts.length >= 3 ? parts.slice(2).join('__') : name;
  }

  /** Pull the actual result text out of a `tool_call_update.content` envelope
   *  (`[{ type:'content', content:{ type:'text', text } }]`) — not the wrapper JSON. */
  private toolResultText(content: unknown): string {
    if (!Array.isArray(content)) return '';
    const parts: string[] = [];
    for (const block of content) {
      const b = block as { content?: { type?: string; text?: string }; text?: string };
      if (b?.content?.type === 'text' && typeof b.content.text === 'string') parts.push(b.content.text);
      else if (typeof b?.text === 'string') parts.push(b.text);
    }
    return parts.join('\n');
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

  /** A PERMANENT CLAUDE_CONFIG_DIR holding only keyless creds + a pre-trusted, connector-free
   *  `.claude.json`, plus a clean cwd with no project `.claude/`. This isolates the spawned client
   *  from the operator's account connectors AND this repo's project config — the two confirmed
   *  sources that flood the agent's toolset and bury the injected MCP server.
   *
   *  Built ONCE and then left completely alone (build-once-if-absent — every file is only written
   *  when missing). The connectors-off guarantee lives in the per-spawn `ENABLE_CLAUDEAI_MCP_SERVERS=0`
   *  env (see spawnAgent), NOT in this file staying pristine — the agent writes its own state back
   *  here over time and that is fine. Fixed path keeps it outside the git tree. Credentials are a
   *  symlink to the live `~/.claude/.credentials.json` (never an API key), so the token stays fresh
   *  and is never duplicated on disk. Override the path with VERIFY_CONFIG_DIR. In CI no
   *  `~/.claude/.credentials.json` exists, so the SDK reads the CLAUDE_CODE_OAUTH_TOKEN secret. */
  private prepareIsolation(): { cfgDir: string; cwd: string } {
    if (this.isolatedCfgDir && this.isolatedCwd) return { cfgDir: this.isolatedCfgDir, cwd: this.isolatedCwd };
    const base = process.env.VERIFY_CONFIG_DIR || join(homedir(), '.cache', 'ollama37-verify');
    const cfgDir = join(base, 'config');
    const work = join(base, 'work');
    mkdirSync(cfgDir, { recursive: true });
    mkdirSync(work, { recursive: true });
    const cred = join(homedir(), '.claude', '.credentials.json');
    const credDest = join(cfgDir, '.credentials.json');
    // SYMLINK (not copy) to the live credentials, so the verifier always uses the current,
    // auto-refreshing OAuth token and we never duplicate the secret on disk. A frozen copy
    // expires within ~1h and 401s. Self-heal a prior run's stale copy/link. In CI the source is
    // absent, so we skip it and the SDK reads CLAUDE_CODE_OAUTH_TOKEN from the environment.
    if (existsSync(cred)) {
      rmSync(credDest, { force: true });
      symlinkSync(cred, credDest);
    }
    // Pre-trust the clean cwd so no trust-folder prompt blocks; zero connectors, zero project servers.
    const cfgJson = join(cfgDir, '.claude.json');
    if (!existsSync(cfgJson)) writeFileSync(cfgJson, JSON.stringify({
      hasCompletedOnboarding: true,
      mcpServers: {},
      projects: { [work]: { hasTrustDialogAccepted: true, mcpServers: {}, enabledMcpjsonServers: [], allowedTools: [] } },
    }));
    const settings = join(cfgDir, 'settings.json');
    if (!existsSync(settings)) writeFileSync(settings, '{}');
    if (DEBUG) process.stderr.write(`  [verify:isolate] CLAUDE_CONFIG_DIR=${cfgDir} cwd=${work}\n`);
    this.isolatedCfgDir = cfgDir;
    this.isolatedCwd = work;
    return { cfgDir, cwd: work };
  }

  /** Spawn the configured ACP agent as a stdio child (same logic as AgentJudge), in an isolated
   *  config dir + clean cwd, with claude.ai connectors and tool-search disabled. */
  private spawnAgent(): ChildProcess {
    const { cfgDir, cwd } = this.prepareIsolation();
    const env = { ...process.env };
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_ENTRYPOINT;
    delete env.CLAUDE_CODE_SSE_PORT;
    env.CLAUDE_CONFIG_DIR = cfgDir;
    env.ENABLE_CLAUDEAI_MCP_SERVERS = '0';
    env.ENABLE_TOOL_SEARCH = '0';
    const stdio: StdioOptions = ['pipe', 'pipe', 'inherit'];

    if (this.agentCmd) {
      return spawn(this.agentCmd, { cwd, stdio, env, shell: true });
    }
    const require = createRequire(import.meta.url);
    const pkg = require.resolve('@agentclientprotocol/claude-agent-acp/package.json');
    const entry = resolve(dirname(pkg), 'dist/index.js');
    return spawn(process.execPath, [entry], { cwd, stdio, env });
  }

  /** The read-only gate: allow only exact-allow-listed tools, else reject (fail-closed). */
  private decide(params: RequestPermissionRequest): RequestPermissionResponse {
    const tc = params.toolCall as { _meta?: { claudeCode?: { toolName?: string } }; title?: string };
    // For MCP tools `title` is the `mcp__server__tool` id; `_meta.claudeCode.toolName` is not
    // set on permission requests, so prefer title and treat both as best-effort.
    const raw = tc?.title ?? tc?._meta?.claudeCode?.toolName ?? '';
    const allowed = this.allow.has(this.bareName(String(raw)));
    const want = allowed ? 'allow' : 'reject';
    const opt =
      params.options.find((o) => o.kind?.startsWith(want)) ??
      params.options.find((o) => o.kind?.startsWith('reject'));
    // The security boundary — always log every decision (allow and reject), not just DEBUG.
    process.stderr.write(`  [verify] ${allowed ? 'ALLOWED' : 'DENIED'} tool permission: "${String(raw).slice(0, 80)}"\n`);
    return opt
      ? { outcome: { outcome: 'selected', optionId: opt.optionId } }
      : { outcome: { outcome: 'cancelled' } };
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
        const x = u as Record<string, unknown>;
        if (DEBUG) {
          const extra =
            u.sessionUpdate === 'tool_call' || u.sessionUpdate === 'tool_call_update'
              ? ` id=${JSON.stringify(x.toolCallId)} title=${JSON.stringify(x.title)} status=${String(x.status)}`
              : '';
          process.stderr.write(`  [verify:event] ${u.sessionUpdate}${extra}\n`);
        }
        if (u.sessionUpdate === 'agent_message_chunk' && u.content.type === 'text') {
          this.turnText += u.content.text;
        }
        // Remember which calls are allow-listed (the `tool_call` carries the title); a later
        // `tool_call_update` for the same id is then known to be ground truth we permit.
        if (u.sessionUpdate === 'tool_call' && typeof x.toolCallId === 'string' &&
            this.allow.has(this.bareName(String(x.title ?? '')))) {
          this.allowedCallIds.add(x.toolCallId);
        }
        // Bucket #3: capture the live tool result the agent received, independent of its prose —
        // hard ground-truth evidence for the report. Only a COMPLETED call we allow-listed counts
        // (a denied/built-in call is not ground truth); extract the inner text from the
        // ToolCallContent envelope; total-cap so a chatty server can't blow the field.
        if (u.sessionUpdate === 'tool_call_update' && typeof x.toolCallId === 'string' &&
            this.allowedCallIds.has(x.toolCallId) && String(x.status) === 'completed') {
          const text = this.toolResultText(x.content);
          if (text) this.toolEvidence = (this.toolEvidence + text).slice(0, 2000);
        }
      },
      requestPermission: async (params: RequestPermissionRequest): Promise<RequestPermissionResponse> => {
        if (DEBUG) process.stderr.write(`  [verify:event] requestPermission ${JSON.stringify(params).slice(0, 500)}\n`);
        return this.decide(params);
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
          cwd: this.prepareIsolation().cwd,
          // Populate the server (the generic judge keeps this empty); hard-deny every
          // non-allow-listed server tool via the agent's own SDK config (defence in depth).
          mcpServers: [{ name: this.cfg.serverName, command: this.cfg.server.command, args: this.cfg.server.args, env }],
          // settingSources:[] loads NO filesystem settings (no project/local `.claude/`); the
          // injected mcpServers above survive regardless.
          _meta: { claudeCode: { options: { settingSources: [], disallowedTools: this.disallowedTools } } },
        }),
        'verifier session/new',
      );
      this.sessionId = session.sessionId;
      if (DEBUG) {
        // Diagnostic warm-up: ask the agent to list its actual toolset, so we can see whether the
        // injected mcp__<server>__* tools surfaced and whether the flood is gone.
        this.turnText = '';
        await this.withTimeout(
          this.conn.prompt({ sessionId: this.sessionId, prompt: [{ type: 'text', text: 'List the exact names of every tool you can call right now, one per line. Nothing else.' }] }),
          'verifier warm-up',
        ).catch((e) => process.stderr.write(`  [verify:warmup] error: ${e}\n`));
        process.stderr.write(`  [verify:warmup] tools:\n${this.turnText}\n`);
      }
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

  /** Probe agent reachability — without the MCP server, so it tests the agent, not the backend. */
  async isAvailable(): Promise<boolean> {
    let child: ChildProcess | undefined;
    try {
      child = this.spawnAgent();
      const stream = ndJsonStream(
        Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>,
        Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>,
      );
      let text = '';
      const client: Client = {
        sessionUpdate: async (p: SessionNotification) => {
          const u = p.update;
          if (u.sessionUpdate === 'agent_message_chunk' && u.content.type === 'text') text += u.content.text;
        },
        requestPermission: async () => ({ outcome: { outcome: 'cancelled' } } as RequestPermissionResponse),
        readTextFile: async () => { throw new Error('disabled'); },
        writeTextFile: async () => { throw new Error('disabled'); },
      };
      const conn = new ClientSideConnection(() => client, stream);
      await this.withTimeout(conn.initialize({
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: `${CONFIG.projectName}-verifier`, version: '1.0.0' },
      }), 'verifier probe initialize');
      const s = await this.withTimeout(conn.newSession({ cwd: this.prepareIsolation().cwd, mcpServers: [] }), 'verifier probe session');
      await this.withTimeout(conn.prompt({ sessionId: s.sessionId, prompt: [{ type: 'text', text: 'Reply with exactly: ok' }] }), 'verifier probe turn');
      if (!text.trim()) throw new Error('verifier agent produced no output on probe turn');
      return true;
    } catch (error) {
      process.stderr.write(`  [verify] Agent not reachable: ${error}\n`);
      return false;
    } finally {
      try { child?.kill('SIGKILL'); } catch { /* gone */ }
    }
  }

  private buildPrompt(t: VerifyTarget): string {
    return JSON.stringify({
      role: `You independently verify another model's answer for ${CONFIG.projectName}. You have access to the ${this.cfg.serverName} MCP tools.`,
      task: 'You MUST call the available tools yourself to retrieve the real data, then judge whether the model\'s answer is correct and grounded in that real data. Do NOT trust the answer — check it. If you cannot call any tool, set pass=false and say so.',
      allowed_tools: [...this.allow],
      user_prompt: t.prompt,
      model_answer: t.answer,
      rules: [
        'Call the allowed tools to retrieve the ground truth before deciding.',
        'pass=true only if the answer matches the real data you retrieved.',
        'If the answer states facts the tools contradict or do not contain, pass=false.',
      ],
      respond: {
        format: 'Respond with a single JSON object and nothing else',
        fields: {
          testId: t.testId,
          pass: 'true if the answer matches the live tool data, false otherwise',
          reason: 'Brief explanation, naming the tool(s) you called',
          evidence: 'The tool result you verified against',
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
    this.toolEvidence = '';
    this.allowedCallIds.clear();
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
      // Prefer the raw tool result we captured ourselves over the agent's self-reported evidence.
      if (this.toolEvidence) judgment.evidence = this.toolEvidence;
      return judgment;
    } catch {
      return { testId: t.testId, pass: false, reason: `Failed to parse verifier response: ${responseText.substring(0, 200)}` };
    }
  }

  /** Verify each target in its OWN fresh session (no cross-target context bleed). */
  async verify(targets: VerifyTarget[]): Promise<Judgment[]> {
    const out: Judgment[] = [];
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      process.stderr.write(`  [verify] Verifying ${i + 1}/${targets.length}: ${t.testId} (allow ${this.allow.size} tools, deny ${this.disallowedTools.length})...\n`);
      try {
        await this.ensureStarted();
        const judgment = await this.verifyOne(t);
        out.push(judgment);
        process.stderr.write(`  [verify] ${t.testId}: ${judgment.pass ? 'PASS' : 'FAIL'} — ${judgment.reason}\n`);
      } catch (error) {
        process.stderr.write(`  [verify] Failed to verify ${t.testId}: ${error}\n`);
        out.push({ testId: t.testId, pass: false, reason: 'Verifier failed: ' + String(error) });
      } finally {
        // Fresh session per target — tear down so the next target re-spawns clean.
        this.kill();
      }
    }
    return out;
  }
}
