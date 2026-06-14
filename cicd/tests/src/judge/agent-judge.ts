/**
 * Agent Judge - Semantic analysis of test results via an ACP agent.
 *
 * Uses a reasoning model to evaluate test execution logs against criteria,
 * catching silent failures that exit-code checking misses.
 *
 * The judge is an Agent Client Protocol (ACP) client: it spawns a configured
 * agent process (JUDGE_AGENT; default the bundled Claude ACP agent) over stdio
 * and drives one prompt turn per test through @agentclientprotocol/sdk
 * (initialize → session/new → session/prompt), then parses a JSON verdict from
 * the agent's reply. Auth is the agent's concern — the default Claude agent runs
 * keyless on a subscription (~/.claude locally, CLAUDE_CODE_OAUTH_TOKEN in CI),
 * no ANTHROPIC_API_KEY required. Swapping models/vendors is a JUDGE_AGENT config
 * change, not a code change.
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
import { TestResult, Judgment } from '../types.js';
import { CONFIG } from '../config.js';

export class AgentJudge {
  private agentCmd: string;
  private cwd: string;
  private child?: ChildProcess;
  private conn?: ClientSideConnection;
  private sessionId?: string;
  /** Accumulates the current turn's agent text (reset before each prompt). */
  private turnText = '';

  constructor(agentCmd: string = CONFIG.judge.agent, cwd: string = process.cwd()) {
    this.agentCmd = agentCmd;
    this.cwd = cwd;
    // Register the orphan-guard once per instance (not per spawn) — the session
    // is re-spawned on timeouts, and re-registering here would leak listeners.
    process.once('exit', () => this.kill());
  }

  /**
   * Await a promise but reject if it outruns CONFIG.judge.timeout — so a hung
   * handshake or turn fails the judge rather than wedging the whole run.
   */
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

  /**
   * Spawn the configured ACP agent as a stdio child. Clears the CLAUDECODE
   * markers so the bundled Claude agent (which wraps the `claude` CLI) runs
   * standalone rather than refusing to launch nested. Auth flows from the
   * inherited environment (~/.claude / CLAUDE_CODE_OAUTH_TOKEN / any key the
   * agent honours) — the judge sets none itself.
   */
  private spawnAgent(): ChildProcess {
    const env = { ...process.env };
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_ENTRYPOINT;
    delete env.CLAUDE_CODE_SSE_PORT;
    const stdio: StdioOptions = ['pipe', 'pipe', 'inherit'];

    if (this.agentCmd) {
      // Any agent command, via the shell — "add a model = config, not code".
      return spawn(this.agentCmd, { cwd: this.cwd, stdio, env, shell: true });
    }
    // Default: the bundled Claude ACP agent. Resolve its entry from package.json
    // and run it under the current node — avoids .bin/PATH/shebang quirks.
    const require = createRequire(import.meta.url);
    const pkg = require.resolve('@agentclientprotocol/claude-agent-acp/package.json');
    const entry = resolve(dirname(pkg), 'dist/index.js');
    return spawn(process.execPath, [entry], { cwd: this.cwd, stdio, env });
  }

  /**
   * Spawn + initialize + open a session, once. Idempotent — a second call is a
   * no-op once a session exists.
   */
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
      // A judge must not execute anything — refuse every tool-permission request.
      requestPermission: async (
        params: RequestPermissionRequest,
      ): Promise<RequestPermissionResponse> => {
        const reject = params.options.find((o) => o.kind?.startsWith('reject'));
        return reject
          ? { outcome: { outcome: 'selected', optionId: reject.optionId } }
          : { outcome: { outcome: 'cancelled' } };
      },
      // fs is disabled in clientCapabilities, so these should never be called.
      readTextFile: async () => { throw new Error('filesystem access disabled for the judge'); },
      writeTextFile: async () => { throw new Error('filesystem access disabled for the judge'); },
    };

    this.conn = new ClientSideConnection(() => client, stream);
    try {
      await this.withTimeout(this.conn.initialize({
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: `${CONFIG.projectName}-judge`, version: '1.0.0' },
      }), 'agent initialize');
      const session = await this.withTimeout(
        this.conn.newSession({ cwd: this.cwd, mcpServers: [] }),
        'agent session/new',
      );
      this.sessionId = session.sessionId;
    } catch (e) {
      // A spawnable-but-silent agent would otherwise leave a live child and a
      // pending handshake. Tear it down so isAvailable() fails over cleanly.
      this.kill();
      throw e;
    }
  }

  /** Tear down the agent process. Safe to call more than once. */
  private kill(): void {
    try { this.child?.kill('SIGKILL'); } catch { /* already gone */ }
    this.child = undefined;
    this.conn = undefined;
    this.sessionId = undefined;
  }

  /**
   * Probe the configured agent: spawn, open a session, and run one bounded
   * throwaway turn. The warmup turn matters — a successful handshake doesn't
   * prove the agent can actually answer (e.g. broken/expired auth surfaces only
   * at prompt time), so without it a misauthed agent would pass the probe and
   * every test would be marked FAIL instead of falling back to the simple judge.
   * Returns false on any failure. On success the session is kept open for
   * judgeResults().
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.ensureStarted();
      const reply = await this.promptAgent('Reply with exactly: ok');
      if (!reply.trim()) throw new Error('agent produced no output on probe turn');
      return true;
    } catch (error) {
      process.stderr.write(`  [judge] Agent not reachable: ${error}\n`);
      this.kill();
      return false;
    }
  }

  /**
   * Truncate a string to a maximum length.
   */
  private truncate(text: string, limit: number): string {
    if (text.length <= limit) return text;
    return text.substring(0, limit) + '... (truncated)';
  }

  /**
   * Build structured JSON prompt for evaluation of a single test.
   */
  private buildPrompt(result: TestResult): string {
    const r = result;

    const steps = r.steps.map((step, j) => {
      const stepDef = r.testCase.steps[j];
      return {
        name: step.name,
        command: step.command.trim(),
        exit_code: step.exitCode,
        duration_ms: step.duration,
        timeout_ms: stepDef?.timeout || r.testCase.timeout,
        stdout: this.truncate(step.stdout, CONFIG.judge.stdoutLimit),
        stderr: this.truncate(step.stderr, CONFIG.judge.stderrLimit),
      };
    });

    const promptData = {
      role: `You are a test result evaluator for ${CONFIG.projectName}. Analyze the test execution data and determine if the test passed or failed.`,
      rules: [
        'Check step stdout for error responses (e.g. {"error":"..."} means FAIL)',
        'Errors with exit code 0 are still FAIL',
        'For AI-generated text, accept reasonable variations',
        'Long durations within timeout are acceptable',
        'Focus on semantic correctness, not formatting differences',
      ],
      test: {
        id: r.testCase.id,
        name: r.testCase.name,
        suite: r.testCase.suite,
        goal: r.testCase.goal || r.testCase.name,
        criteria: r.testCase.criteria,
        timeout_ms: r.testCase.timeout,
        duration_ms: r.totalDuration,
      },
      steps,
      container_logs: this.truncate(r.logs, CONFIG.judge.logsLimit),
      respond: {
        format: 'Respond with a single JSON object and nothing else',
        fields: {
          testId: r.testCase.id,
          pass: 'true if test meets all criteria, false otherwise',
          reason: 'Brief explanation of your verdict',
          evidence: 'Required if pass is false — the exact stdout content or log line that caused failure',
        },
      },
    };

    const prompt = JSON.stringify(promptData, null, 2);

    // Log prompt stats
    const totalStdout = r.steps.reduce((sum, s) => sum + s.stdout.length, 0);
    const totalStderr = r.steps.reduce((sum, s) => sum + s.stderr.length, 0);
    process.stderr.write(`  [judge] Prompt for ${r.testCase.id}: logs ${r.logs.length} chars, stdout ${totalStdout} chars, stderr ${totalStderr} chars\n`);
    process.stderr.write(`  [judge] Prompt size: ${prompt.length} chars\n`);

    return prompt;
  }

  /**
   * Extract the first JSON object from a model response, tolerating prose or
   * markdown fences around it.
   */
  private extractJson(text: string): string | null {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start === -1 || end === -1 || end < start) return null;
    return text.substring(start, end + 1);
  }

  /**
   * Run one prompt turn through the agent and return its raw reply text.
   * Bounded by CONFIG.judge.timeout. On timeout/failure the turn isn't actually
   * cancelled — it keeps running on the session and its late chunks would bleed
   * into the next test's reply (which shares this.turnText). So we tear the
   * agent down here; the next test re-spawns a clean session via ensureStarted.
   */
  private async promptAgent(prompt: string): Promise<string> {
    this.turnText = '';
    try {
      await this.withTimeout(
        this.conn!.prompt({ sessionId: this.sessionId!, prompt: [{ type: 'text', text: prompt }] }),
        'agent turn',
      );
    } catch (e) {
      this.kill();
      throw e;
    }
    return this.turnText;
  }

  /**
   * Judge a single test result.
   */
  private async judgeOne(result: TestResult): Promise<Judgment> {
    const prompt = this.buildPrompt(result);
    const testId = result.testCase.id;

    const responseText = await this.promptAgent(prompt);

    // Handle empty response
    if (!responseText) {
      process.stderr.write(`  [judge] WARNING: Empty response for ${testId}\n`);
      return {
        testId,
        pass: false,
        reason: 'Agent returned empty response',
      };
    }

    process.stderr.write(`  [judge] Raw response for ${testId} (${responseText.length} chars): ${responseText.substring(0, 500)}\n`);

    const json = this.extractJson(responseText);
    if (!json) {
      process.stderr.write(`  [judge] WARNING: No JSON object in response for ${testId}\n`);
      return {
        testId,
        pass: false,
        reason: `No JSON object in agent response: ${responseText.substring(0, 200)}`,
      };
    }

    try {
      const judgment = JSON.parse(json) as Judgment;

      // Validate testId matches
      if (judgment.testId !== testId) {
        process.stderr.write(`  [judge] WARNING: Response testId "${judgment.testId}" doesn't match expected "${testId}"\n`);
        judgment.testId = testId;
      }

      // Coerce string "true"/"false" to boolean (models often return strings)
      if (typeof judgment.pass === 'string') {
        judgment.pass = (judgment.pass as unknown as string).toLowerCase() === 'true';
      }

      // Validate required fields
      if (typeof judgment.pass !== 'boolean') {
        process.stderr.write(`  [judge] WARNING: Response missing "pass" field for ${testId}\n`);
        return {
          testId,
          pass: false,
          reason: `Agent response missing "pass" field: ${responseText.substring(0, 200)}`,
        };
      }

      if (!judgment.reason) {
        judgment.reason = judgment.pass ? 'Passed (no reason provided)' : 'Failed (no reason provided)';
      }

      return judgment;
    } catch {
      process.stderr.write(`  [judge] WARNING: Failed to parse JSON for ${testId}\n`);
      process.stderr.write(`  [judge] Full response: ${responseText}\n`);
      return {
        testId,
        pass: false,
        reason: `Failed to parse agent response: ${responseText.substring(0, 200)}`,
      };
    }
  }

  /**
   * Judge all test results, one at a time, over a single reused agent session.
   * Tears the agent down when done.
   */
  async judgeResults(results: TestResult[]): Promise<Judgment[]> {
    const allJudgments: Judgment[] = [];

    try {
      for (let i = 0; i < results.length; i++) {
        const result = results[i];
        process.stderr.write(
          `  [judge] Judging ${i + 1}/${results.length}: ${result.testCase.id}...\n`
        );

        try {
          // Idempotent when the session is alive; re-spawns a clean one if a
          // prior test's timeout tore the agent down.
          await this.ensureStarted();
          const judgment = await this.judgeOne(result);
          allJudgments.push(judgment);
          process.stderr.write(`  [judge] ${result.testCase.id}: ${judgment.pass ? 'PASS' : 'FAIL'} — ${judgment.reason}\n`);
        } catch (error) {
          process.stderr.write(`  [judge] Failed to judge ${result.testCase.id}: ${error}\n`);
          allJudgments.push({
            testId: result.testCase.id,
            pass: false,
            reason: 'Agent judgment failed: ' + String(error),
          });
        }
      }
    } finally {
      this.kill();
    }

    return allJudgments;
  }
}
