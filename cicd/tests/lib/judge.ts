/**
 * The agent judge (#542). Used only by lib/ollama.ts, through lib/run.ts.
 *
 * One job: put a question from prompts.yaml to an agent and read back yes or no.
 * The agent is an ACP agent (default: the bundled Claude one) spawned over stdio,
 * with one fresh session per question so no verdict can see another. With
 * JUDGE_BASE_URL set, the agent talks to that ollama server and JUDGE_MODEL on it,
 * so the judge runs off the box under test; unset, it uses Claude on
 * CLAUDE_CODE_OAUTH_TOKEN or ~/.claude.
 *
 * Anything short of a clear yes or no -- the agent will not start, times out,
 * loops, or answers without a yes/no -- comes back as `abstain`, and the run
 * counts an abstain as a failure. A reply nobody read never passes.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { Readable, Writable } from 'node:stream';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import {
  ClientSideConnection,
  ndJsonStream,
  PROTOCOL_VERSION,
  type Client,
  type SessionNotification,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
} from '@agentclientprotocol/sdk';
import { warmupQuestion } from './prompts.js';

export interface Judgment {
  verdict: 'yes' | 'no' | 'abstain';
  reason: string;
  /** The judge model's reasoning before its answer, as streamed; empty when it emits none. */
  thinking?: string;
}

/** Bounds on one question. A loop is a 3-word phrase seen LOOP_LIMIT times in LOOP_WINDOW_MS. */
const TIMEOUT_MS = 300_000;
/** The first question also starts the agent and cold-loads the judge model on its server. */
const WARMUP_MS = 900_000;
const LOOP_WINDOW_MS = 30_000;
const LOOP_CHECK_MS = 5_000;
const LOOP_LIMIT = 100;
/** Longest reply text put in front of the judge. */
const REPLY_LIMIT = 1000;
/**
 * Longest tool result put in front of a grounded judge. Larger than the reply cap,
 * since an answer can rest on any part of the result; bounded because the judge
 * model runs at its default 4k context and a judgement already takes ~2k tokens in
 * and up to ~1.2k out (2,500 characters is ~650 tokens). A larger result needs a
 * larger judge context.
 */
const RESULT_LIMIT = 2500;

/**
 * The judge's working directory: empty, outside any repository. Claude Code puts
 * the git status and recent commits of its working directory into the system
 * prompt; run inside this repo, the judge read our commits about the judge and
 * reasoned from them ("a test case for a judge ... mentioned in the git history").
 */
const JUDGE_CWD = mkdtempSync(join(tmpdir(), 'ollama37-judge-'));

/** The agent's environment: JUDGE_BASE_URL/JUDGE_MODEL become the Anthropic-compatible settings. */
function agentEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.CLAUDECODE;
  delete env.CLAUDE_CODE_ENTRYPOINT;
  delete env.CLAUDE_CODE_SSE_PORT;
  if (env.JUDGE_BASE_URL) {
    // The auth token has to follow the base URL; set one without the other and the
    // fallback to Claude breaks instead of degrading.
    env.ANTHROPIC_BASE_URL = env.JUDGE_BASE_URL;
    env.ANTHROPIC_AUTH_TOKEN = 'ollama';
    if (env.JUDGE_MODEL) {
      env.ANTHROPIC_MODEL = env.JUDGE_MODEL;
      env.ANTHROPIC_DEFAULT_HAIKU_MODEL = env.JUDGE_MODEL; // Claude Code's side calls; ollama has no haiku
    }
  }
  return env;
}

class Agent {
  private child?: ChildProcess;
  private conn?: ClientSideConnection;
  /** This turn's agent messages; a new messageId starts a new one (ACP). */
  private messages: { id: string | undefined; text: string }[] = [];
  /** This turn's reasoning, kept for the verdict. */
  thoughts = '';
  private streamed: { at: number; text: string }[] = [];
  private loopReason?: string;

  /** Deadline for the next turn: WARMUP_MS until a warm-up has been answered, then TIMEOUT_MS. */
  limit = TIMEOUT_MS;

  private async within<T>(p: Promise<T>, label: string): Promise<T> {
    const ms = this.limit;
    let timer: NodeJS.Timeout;
    const t = new Promise<never>((_, rej) => { timer = setTimeout(() => rej(new Error(`${label} exceeded ${ms}ms`)), ms); });
    try { return await Promise.race([p, t]); } finally { clearTimeout(timer!); }
  }

  private spawnAgent(): ChildProcess {
    const stdio: ['pipe', 'pipe', 'inherit'] = ['pipe', 'pipe', 'inherit'];
    if (process.env.JUDGE_AGENT) return spawn(process.env.JUDGE_AGENT, { cwd: JUDGE_CWD, stdio, env: agentEnv(), shell: true });
    const req = createRequire(import.meta.url);
    const entry = resolve(dirname(req.resolve('@agentclientprotocol/claude-agent-acp/package.json')), 'dist/index.js');
    return spawn(process.execPath, [entry], { cwd: JUDGE_CWD, stdio, env: agentEnv() });
  }

  private async start(): Promise<void> {
    if (this.conn) return;
    this.child = this.spawnAgent();
    const stream = ndJsonStream(
      Writable.toWeb(this.child.stdin!) as WritableStream<Uint8Array>,
      Readable.toWeb(this.child.stdout!) as ReadableStream<Uint8Array>,
    );
    const client: Client = {
      sessionUpdate: async (params: SessionNotification) => {
        const u = params.update;
        if ((u.sessionUpdate === 'agent_message_chunk' || u.sessionUpdate === 'agent_thought_chunk') && u.content.type === 'text') {
          this.streamed.push({ at: Date.now(), text: u.content.text });
          if (u.sessionUpdate === 'agent_thought_chunk') this.thoughts += u.content.text;
          if (u.sessionUpdate === 'agent_message_chunk') {
            const id = (u as { messageId?: string | null }).messageId ?? undefined;
            const last = this.messages[this.messages.length - 1];
            if (last && last.id === id) last.text += u.content.text;
            else this.messages.push({ id, text: u.content.text });
          }
        }
      },
      // A judge executes nothing.
      requestPermission: async (p: RequestPermissionRequest): Promise<RequestPermissionResponse> => {
        const reject = p.options.find((o) => o.kind?.startsWith('reject'));
        return reject ? { outcome: { outcome: 'selected', optionId: reject.optionId } } : { outcome: { outcome: 'cancelled' } };
      },
      readTextFile: async () => { throw new Error('the judge has no filesystem'); },
      writeTextFile: async () => { throw new Error('the judge has no filesystem'); },
    };
    this.conn = new ClientSideConnection(() => client, stream);
    try {
      await this.within(this.conn.initialize({
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: 'ollama37-judge', version: '3.0.0' },
      }), 'agent initialize');
    } catch (e) { this.kill(); throw e; }
  }

  private checkForLoop(sessionId: string): void {
    if (this.loopReason || !this.conn) return;
    const since = Date.now() - LOOP_WINDOW_MS;
    this.streamed = this.streamed.filter((c) => c.at >= since);
    const words = this.streamed.map((c) => c.text).join('').toLowerCase().split(/\s+/).filter(Boolean);
    const counts = new Map<string, number>();
    let top = '', topN = 0;
    for (let i = 0; i + 2 < words.length; i++) {
      const p = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
      const n = (counts.get(p) ?? 0) + 1;
      counts.set(p, n);
      if (n > topN) { top = p; topN = n; }
    }
    if (topN < LOOP_LIMIT) return;
    this.loopReason = `judge looped on "${top.slice(0, 40)}"`;
    this.conn.cancel({ sessionId }).catch(() => {});
  }

  /** One question in a fresh session; the joined text of every message in the turn. */
  async ask(question: string): Promise<string> {
    await this.start();
    // No tools at all: tools: [] drops Claude Code's built-in tools, and
    // strictMcpConfig keeps the user's own MCP servers out (only mcpServers, here
    // none, are loaded). Without it a server added to ~/.claude.json -- playwright,
    // once -- rode along on every question and pushed it past the judge's 4k context.
    const { sessionId } = await this.within(
      this.conn!.newSession({ cwd: JUDGE_CWD, mcpServers: [], _meta: { claudeCode: { options: { tools: [], strictMcpConfig: true } } } }),
      'agent session/new',
    );
    this.messages = [];
    this.streamed = [];
    this.thoughts = '';
    this.loopReason = undefined;
    const guard = setInterval(() => this.checkForLoop(sessionId), LOOP_CHECK_MS);
    try {
      await this.within(this.conn!.prompt({ sessionId, prompt: [{ type: 'text', text: question }] }), 'agent turn');
    } catch (e) {
      this.kill(); // a timed-out turn keeps running and would bleed into the next question
      throw this.loopReason ? new Error(this.loopReason) : e;
    } finally {
      clearInterval(guard);
    }
    if (this.loopReason) { this.kill(); throw new Error(this.loopReason); }
    this.conn?.closeSession({ sessionId }).catch(() => {});
    return this.messages.map((m) => m.text).join('\n');
  }

  kill(): void {
    try { this.child?.kill('SIGKILL'); } catch { /* gone */ }
    this.child = undefined;
    this.conn = undefined;
  }
}

let agent: Agent | undefined;

/**
 * Put `question` (a judge template from prompts.yaml) to the judge, with
 * `{reply}` and `{result}` filled in, and read its verdict: the answer must OPEN
 * with yes or no, past markdown and quotes. The reasoning streams separately (the
 * thinking), so a sound answer leads with its verdict ("**Yes** -- this is random
 * words") and the explanation after it is full of "no" that must not count.
 * Anything else -- a hedge, another language, a verdict buried mid-answer -- is
 * an abstain, which fails the reply. (?:\1)? absorbs a doubled turn ("yesyes").
 */
export async function judge(template: string, reply: string, result = ''): Promise<Judgment> {
  // A cut is said out loud, so the judge does not read "absent from what it was
  // shown" as "absent from the result".
  const cut = (s: string, limit: number) =>
    s.length > limit ? `${s.slice(0, limit)}... (cut here; the full text is ${s.length} characters)` : s;
  const question = template.replaceAll('{reply}', cut(reply, REPLY_LIMIT)).replaceAll('{result}', cut(result, RESULT_LIMIT));
  if (!agent) {
    agent = new Agent();
    // A throwaway turn with a long deadline, so the cold start (agent spawn, judge
    // model load) is not charged to the first real question. Failing here still
    // leaves the question below to abstain on its own.
    agent.limit = WARMUP_MS;
    await agent.ask(warmupQuestion()).catch(() => {});
    agent.limit = TIMEOUT_MS;
  }
  let answer: string;
  try {
    answer = await agent.ask(question);
  } catch (e) {
    return { verdict: 'abstain', reason: `judge could not answer: ${e instanceof Error ? e.message : e}` };
  }
  const thinking = agent.thoughts.trim();
  const first = answer.toLowerCase().replace(/^[\s*_`#>"'“”‘’]+/, '').match(/^(yes|no)(?:\1)?\b/);
  if (!first) return { verdict: 'abstain', reason: `no yes/no in: ${answer.slice(0, 120)}`, thinking };
  const v = first[1] as 'yes' | 'no';
  return { verdict: v, reason: `judge said ${v}`, thinking };
}

/** Stop the agent process at the end of a run. */
export function closeJudge(): void {
  agent?.kill();
  agent = undefined;
}
