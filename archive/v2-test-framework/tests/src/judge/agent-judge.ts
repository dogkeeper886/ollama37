/**
 * Agent Judge - Semantic analysis of test results via an ACP agent.
 *
 * Uses a reasoning model to evaluate test execution logs against criteria,
 * catching silent failures that exit-code checking misses.
 *
 * The judge is an Agent Client Protocol (ACP) client: it spawns a configured
 * agent process (JUDGE_AGENT; default the bundled Claude ACP agent) over stdio
 * and drives one prompt turn per test through @agentclientprotocol/sdk
 * (initialize once, then session/new → session/prompt → session/close per
 * test), then parses a JSON verdict from
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
  /**
   * The current turn's agent messages, in arrival order (reset before each prompt).
   *
   * Kept per message rather than as one string, because ContentChunk carries a
   * `messageId` and the schema says a change in it starts a NEW message. A turn can
   * therefore hold several: this agent answers "Yes" with no messageId, then repeats
   * "Yes" under `msg_...`. Concatenating across that boundary produced "YesYes", and
   * before the answer was one word it produced the whole JSON verdict twice.
   */
  private messages: { id: string | undefined; text: string }[] = [];
  /** The current turn's streamed thinking and reply, timestamped, for the loop guard. */
  private streamed: { at: number; text: string }[] = [];
  /** Set by the loop guard when it cancels a turn; the turn then fails with this reason. */
  private loopReason?: string;

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
   * Spawn + initialize the agent, once. Idempotent — a second call is a no-op while
   * the agent is up. Sessions are opened per test by openSession().
   */
  private async ensureStarted(): Promise<void> {
    if (this.conn) return;

    const child = this.spawnAgent();
    this.child = child;

    const stream = ndJsonStream(
      Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>,
      Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>,
    );

    const client: Client = {
      sessionUpdate: async (params: SessionNotification): Promise<void> => {
        const u = params.update;
        // Thinking streams separately from the reply, and a loop can live entirely in the
        // thinking (the agent never reaches its answer), so the guard watches both.
        if ((u.sessionUpdate === 'agent_message_chunk' || u.sessionUpdate === 'agent_thought_chunk')
            && u.content.type === 'text') {
          this.streamed.push({ at: Date.now(), text: u.content.text });
          if (u.sessionUpdate === 'agent_message_chunk') {
            // ACP types this MessageId | null, so an explicit null must not read as a
          // different message from an absent field — that would split one reply in two.
          const id = (u as { messageId?: string | null }).messageId ?? undefined;
            const last = this.messages[this.messages.length - 1];
            // Same message -> same chunk stream, so append. A different id is a new
            // message and must not be glued onto the previous one.
            if (last && last.id === id) last.text += u.content.text;
            else this.messages.push({ id, text: u.content.text });
          }
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
    } catch (e) {
      // A spawnable-but-silent agent would otherwise leave a live child and a
      // pending handshake. Tear it down so isAvailable() fails over cleanly.
      this.kill();
      throw e;
    }
  }

  /**
   * Open a fresh session for one judgement. One session per test, rather than one per
   * run: a reused session carries every earlier test and verdict into each prompt, and
   * on a 19-case run it grew from 23k to 64k tokens — the whole window of the judge
   * model — with each test slower to judge than the last. A fresh session costs the
   * same every time, and no verdict can see another test.
   *
   * tools: [] drops Claude Code's built-in tools (~18k tokens of definitions per
   * request). The judge only reads and answers; replaying 19 real judge inputs gave
   * identical verdicts with and without them, and it never asked for one. Agents other
   * than the bundled Claude one ignore this _meta.
   */
  private async openSession(): Promise<void> {
    const session = await this.withTimeout(
      this.conn!.newSession({
        cwd: this.cwd,
        mcpServers: [],
        _meta: { claudeCode: { options: { tools: [] } } },
      }),
      'agent session/new',
    );
    this.sessionId = session.sessionId;
  }

  /** Close the current session so its resources go now, not at the end of the run. */
  private closeSession(): void {
    if (!this.conn || !this.sessionId) return;
    // Best-effort: an agent without session/close support just keeps the session.
    this.conn.closeSession({ sessionId: this.sessionId }).catch(() => { /* not supported */ });
    this.sessionId = undefined;
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
   * Returns false on any failure. On success the agent is kept running for
   * judgeResults(); the probe's own session is closed.
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.ensureStarted();
      await this.openSession();
      const reply = await this.promptAgent('Reply with exactly: ok');
      if (!reply.trim()) throw new Error('agent produced no output on probe turn');
      this.closeSession();
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
  /**
   * One question, and the text to answer it about.
   *
   * The old payload sent a role, five rules, a goal, a criteria paragraph, the
   * evidence and a response schema -- three instruction blocks owned by two files,
   * arriving together with no precedence and contradicting each other, and never a
   * question. A judge that wrote "this is coherent, readable English" and then
   * returned pass:false was answering something nobody had asked.
   *
   * A script decides everything a script can: simpleContentCheck rejects empty
   * output, output with no letters or digits, and one short unit repeated to fill
   * the reply, and only what passes it reaches here. What is left needs a reader,
   * and it is one question: fluent words in an order that means nothing, which no
   * pattern catches and which is not random at the character level.
   *
   * The question names what "readable" means and the two shapes a healthy reply
   * takes that the bare question read as failure. Asked "Is this paragraph
   * readable language?", gpt-oss-64k:20b said "no" to a clean Lincoln rewrite 5 of
   * 5 times and coin-flipped on a bulleted `thinking` cut off by num_predict — the
   * throughput replies all end on budget, and a list is not a paragraph (#524).
   * Probed 5 votes each on four real replies and three garbage texts (word salad,
   * token soup, mixed junk): 7/20 readable passed before, 17/20 after, and the
   * garbage failed 15/15 both times. Dropping the word-order clause passed 20/20
   * readable but let word salad through once in five.
   */
  private buildPrompt(paragraph: string): string {
    return (
      'Is this text readable language, with words in an order that means something? ' +
      'It may stop mid-sentence or be a list; that is fine. yes or no\n\n' +
      `"${this.truncate(paragraph, CONFIG.judge.stdoutLimit)}"`
    );
  }

  /**
   * Run one prompt turn through the agent and return its raw reply text.
   * Bounded by CONFIG.judge.timeout. On timeout/failure the turn isn't actually
   * cancelled — it keeps running on the session and its late chunks would bleed
   * into the next test's reply (which shares this.messages). So we tear the
   * agent down here; the next test re-spawns a clean session via ensureStarted.
   */
  private async promptAgent(prompt: string): Promise<string> {
    this.messages = [];
    this.streamed = [];
    this.loopReason = undefined;
    const guard = setInterval(() => this.checkForLoop(), CONFIG.judge.loopCheckMs);
    try {
      await this.withTimeout(
        this.conn!.prompt({ sessionId: this.sessionId!, prompt: [{ type: 'text', text: prompt }] }),
        'agent turn',
      );
    } catch (e) {
      this.kill();
      throw this.loopReason ? new Error(this.loopReason) : e;
    } finally {
      clearInterval(guard);
    }
    if (this.loopReason) {
      // The cancelled turn left loop text in the session; start the next test clean.
      this.kill();
      throw new Error(this.loopReason);
    }
    // The answer is the agent's last message. Earlier ones are supersedes -- a
    // partial answer it then restated -- not content to be concatenated.
    // Every message, joined. ACP says a new messageId starts a new message, and
    // consecutive messages are sequential parts of one turn — so an agent that
    // answers "Yes." and then adds a closing remark keeps its verdict here. The
    // reader below takes the LAST yes/no, which also absorbs a doubled answer.
    return this.messages.map((m) => m.text).join('\n');
  }

  /**
   * Loop guard, run every loopCheckMs during a turn. Counts 3-word phrases across the last
   * loopWindowMs of streamed thinking and reply; if one reaches loopLimit, the agent is
   * repeating itself — it copies repeated text back and cannot stop — so cancel the turn.
   * A phrase rather than a sentence is the unit because the loops seen ("4 4 4 …",
   * "the the the …") never end a sentence.
   */
  private checkForLoop(): void {
    if (this.loopReason || !this.conn || !this.sessionId) return;
    const since = Date.now() - CONFIG.judge.loopWindowMs;
    this.streamed = this.streamed.filter((c) => c.at >= since);
    const words = this.streamed.map((c) => c.text).join('').toLowerCase().split(/\s+/).filter(Boolean);
    const counts = new Map<string, number>();
    let top = '';
    let topCount = 0;
    for (let i = 0; i + 2 < words.length; i++) {
      const phrase = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
      const n = (counts.get(phrase) ?? 0) + 1;
      counts.set(phrase, n);
      if (n > topCount) { top = phrase; topCount = n; }
    }
    if (topCount < CONFIG.judge.loopLimit) return;
    this.loopReason = `judge looped: "${top.slice(0, 40)}" repeated ${topCount}x in ${CONFIG.judge.loopWindowMs / 1000}s; turn cancelled`;
    process.stderr.write(`  [judge] ${this.loopReason}\n`);
    this.conn.cancel({ sessionId: this.sessionId }).catch(() => { /* the kill after the turn ends covers it */ });
  }

  /**
   * Judge a single test result.
   */
  private async judgeOne(result: TestResult): Promise<Judgment> {
    const testId = result.testCase.id;

    // The response, or the thinking when the response is empty -- the same two
    // fields in the same order as simpleContentCheck, which already rejected the
    // rows where both are empty. A response means the model finished, so the
    // thinking behind it is working notes and a second paragraph would only drag
    // the verdict; an empty response means the budget ran out mid-turn, and the
    // thinking is then the only language the model produced.
    //
    // stdout stays out. It is the step's whole console output -- jq lines and
    // sentinels around the prose -- and asking whether that is readable language
    // earns a truthful "no" (TC-MODELS-003, run 36597608237).
    const step = result.steps.find((s) => s.reply);
    const paragraph = step?.reply?.response?.trim() || step?.reply?.thinking?.trim() || '';

    // No step asked a model at all: build, models and inference drive the CLI and
    // check sentinels. This judge has no paragraph, so it abstains and the simple
    // judge, which reads exit codes and patterns, decides alone.
    if (!paragraph) {
      return { testId, pass: true, reason: 'judge: no model reply to read' };
    }

    const responseText = await this.promptAgent(this.buildPrompt(paragraph));

    // Handle empty response
    if (!responseText) {
      process.stderr.write(`  [judge] WARNING: Empty response for ${testId}\n`);
      return {
        testId,
        pass: false,
        reason: 'Agent returned empty response',
      };
    }

    process.stderr.write(`  [judge] Raw response for ${testId} (${responseText.length} chars): ${responseText.substring(0, 200)}\n`);

    // "yes" means readable, which passes. Asked the other way round -- "is this
    // gibberish?" -- the affirmative answer would be the failing one, which is the
    // arrangement a careless parse inverts.
    //
    // \1? absorbs a doubled answer. The agent returns its turn text twice --
    // "yesyes", "nono", and before this it returned the whole JSON verdict twice --
    // so a match anchored on both sides finds nothing. Allowing the repeat is
    // narrower than dropping the trailing \b, which would match "no" inside
    // "nonsense" and fail a readable paragraph.
    // The LAST yes/no in the reply, not the first: an agent that reasons before
    // answering ("There is no gibberish here — yes, it reads fine") would
    // otherwise be read off its reasoning. \b on both sides keeps "no" out of
    // "nonsense".
    const found = [...responseText.toLowerCase().matchAll(/\b(yes|no)\b/g)];
    if (found.length === 0) {
      return { testId, pass: false, reason: `No yes/no in agent response: ${responseText.substring(0, 200)}` };
    }
    const readable = found[found.length - 1][1] === 'yes';
    return {
      testId,
      pass: readable,
      reason: readable ? 'judge: readable language' : 'judge: not readable language',
    };
  }

  /**
   * Judge all test results, one at a time, each in its own session on one agent
   * process. Tears the agent down when done.
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
          // Idempotent while the agent is up; re-spawns it if a prior test's timeout
          // or loop guard tore it down.
          await this.ensureStarted();
          await this.openSession();
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
        } finally {
          this.closeSession();
        }
      }
    } finally {
      this.kill();
    }

    return allJudgments;
  }
}
