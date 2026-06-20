/**
 * Server-agnostic MCP host: let an Ollama model drive a real stdio MCP server's
 * tools end-to-end (no mock).
 *
 * Connect to a stdio MCP server (any one — the spawn config is passed in),
 * list its tools, translate them to Ollama's /api/chat `tools[]` format, then
 * run the tool loop: chat → on tool_calls, execute each against the *real*
 * server via callTool → feed results back → chat again → final answer.
 *
 * What's being tested is the MODEL: given a real menu, does it pick the right
 * tool with valid args and use the result? A model whose template lacks tool
 * support is reported `supported:false` — a clean verdict, not a crash (mirrors
 * the catch-returns-FAIL shape in perf/fa.ts). The server choice + its
 * credentials are config, not code.
 */
import http from 'node:http';
import https from 'node:https';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';

/** How to spawn the stdio MCP server. testlink-mcp is just one such config. */
export interface McpServerConfig {
  /** Identifies the server when several share one menu — e.g. the verifier picks
   *  its read-only server by this name. Defaults to 'mcp' when unset. */
  name?: string;
  command: string;
  args: string[];
  cwd?: string;
  /** Extra env the server needs (e.g. TESTLINK_URL / TESTLINK_API_KEY) — merged
   *  over a minimal default environment; never hardcoded in this module. */
  env?: Record<string, string>;
}

export interface McpHostOptions {
  /** Ollama host, e.g. http://localhost:11434 */
  host: string;
  model: string;
  prompt: string;
  /** One or more stdio MCP servers whose tools are merged into a single menu the
   *  model picks from. Two+ servers make "tool selection" a real choice (the model
   *  must reach for the right one); a tool call routes back to its owning server. */
  servers: McpServerConfig[];
  numCtx?: number;
  /** Max chat↔tool rounds before giving up (guards against a tool loop). */
  maxIters?: number;
  /** Per-call Ollama response timeout (ms). Default 600000 (10 min) — heavy models on slow
   *  hardware (the K80) need more than fetch's implicit ~5-min default. */
  timeoutMs?: number;
}

export interface ToolCallRecord {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultRecord {
  name: string;
  content: string;
  isError: boolean;
}

export interface McpTrajectory {
  model: string;
  /** false ⇢ the model's template can't do tools (Ollama rejected `tools`). */
  supported: boolean;
  /** Tool names the MCP server(s) exposed (merged across servers). */
  toolNames: string[];
  /** Required arg names per tool, from each tool's inputSchema.required. */
  toolRequired: Record<string, string[]>;
  /** Which server each tool came from (toolName → server name) — lets a caller
   *  pick out one server's tools (e.g. the verifier's read-only server). */
  toolServer: Record<string, string>;
  toolCalls: ToolCallRecord[];
  toolResults: ToolResultRecord[];
  finalAnswer: string;
  /** Ollama generation perf, summed across the chat↔tool rounds. */
  inTokens: number;
  /** Largest single round's prompt tokens — the value Ollama checks against num_ctx
   *  (the summed inTokens spans rounds and can't be compared to the window). */
  maxPromptTokens: number;
  outTokens: number;
  totalDurationS: number;
  evalTps: number;
  error?: string;
}

/** MCP tool {name, description?, inputSchema} → Ollama tools[] entry (near 1:1). */
export function mcpToOllamaTools(tools: Array<{ name: string; description?: string; inputSchema?: unknown }>): unknown[] {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description ?? '',
      parameters: t.inputSchema ?? { type: 'object', properties: {} },
    },
  }));
}

/** POST /api/chat over node:http so the per-call timeout is honored end-to-end.
 *  fetch() (undici) imposes its own ~300s headers/body timeout that AbortSignal
 *  can't lift — it silently kills slow K80 generations at 5 min. node:http has no
 *  such cap, so timeoutMs (via AbortSignal) is the only deadline. */
async function chat(host: string, model: string, messages: unknown[], tools: unknown[], numCtx: number, timeoutMs: number): Promise<any> {
  const body = JSON.stringify({ model, messages, tools, stream: false, options: { temperature: 0, seed: 42, num_ctx: numCtx } });
  const url = new URL(`${host}/api/chat`);
  const transport = url.protocol === 'https:' ? https : http;
  return new Promise((resolve, reject) => {
    const req = transport.request(
      url,
      { method: 'POST', headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) } },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        // The response is a separate emitter from req — a reset mid-body (a real risk on a
        // slow/OOMing K80 backend) errors here, not on req. Without this it's an uncaught throw.
        res.on('error', reject);
        res.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
          } catch {
            reject(new Error('ollama /api/chat: invalid JSON response'));
          }
        });
      },
    );
    const signal = AbortSignal.timeout(timeoutMs);
    const onAbort = () => req.destroy(new Error(`ollama /api/chat timed out after ${timeoutMs}ms`));
    signal.addEventListener('abort', onAbort, { once: true });
    req.on('close', () => signal.removeEventListener('abort', onAbort));
    req.on('error', reject);
    req.end(body);
  });
}

/** Evict a model from Ollama (keep_alive: 0). Best-effort: a benchmark loops over models
 *  sequentially, so each must be released before the next loads — otherwise Ollama's default
 *  keep-alive leaves the previous one resident and two models contend for the GPU (skewing
 *  timings, risking OOM). A failed unload must never fail the test. */
export async function unloadModel(host: string, model: string): Promise<void> {
  try {
    await fetch(`${host}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, keep_alive: 0 }),
      signal: AbortSignal.timeout(30000),
    });
  } catch {
    /* best-effort */
  }
}

const round2 = (n: number): number => Math.round(n * 100) / 100;
const tps = (tokens: number, durNs: number): number => (durNs > 0 ? round2(tokens / (durNs / 1e9)) : 0);

/** Join an MCP CallToolResult's content blocks into a single text payload. */
function resultText(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content
    .map((c: any) => (c?.type === 'text' && typeof c.text === 'string' ? c.text : JSON.stringify(c)))
    .join('\n');
}

/** Ollama is documented to return tool-call arguments as an object, but some
 *  model templates emit a JSON string — normalize both to an object. */
function asArgs(raw: unknown): Record<string, unknown> {
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw);
    } catch {
      return {};
    }
  }
  return (raw ?? {}) as Record<string, unknown>;
}

export async function runMcpHost(opts: McpHostOptions): Promise<McpTrajectory> {
  const numCtx = opts.numCtx ?? 4096;
  const maxIters = opts.maxIters ?? 5;
  // Clamp to a sane (1s … 1h) window: rejects NaN/0/negative (→ default) and caps huge values that
  // would overflow the 32-bit timer and abort instantly. Bad --timeout can't silently fail every model.
  const reqMs = Number(opts.timeoutMs);
  const timeoutMs = Number.isFinite(reqMs) && reqMs > 0 ? Math.min(reqMs, 3_600_000) : 600000;
  let totalDurNs = 0;
  let evalDurNs = 0;

  // One client per server; their tools are merged into a single menu, and a tool
  // call routes back to the server that owns the name (toolToClient).
  const clients = opts.servers.map((s) => ({
    name: s.name ?? 'mcp',
    client: new Client({ name: 'ollama37-mcp-host', version: '1.0.0' }),
    transport: new StdioClientTransport({
      command: s.command,
      args: s.args,
      cwd: s.cwd,
      env: { ...getDefaultEnvironment(), ...(s.env ?? {}) },
    }),
  }));
  const toolToClient = new Map<string, Client>();

  const traj: McpTrajectory = {
    model: opts.model,
    supported: true,
    toolNames: [],
    toolRequired: {},
    toolServer: {},
    toolCalls: [],
    toolResults: [],
    finalAnswer: '',
    inTokens: 0,
    maxPromptTokens: 0,
    outTokens: 0,
    totalDurationS: 0,
    evalTps: 0,
  };

  const messages: any[] = [{ role: 'user', content: opts.prompt }];

  try {
    // Connect every server, list its tools, and merge into one menu. A tool name
    // exposed by two servers can't be routed unambiguously — fail fast rather than
    // silently send the call to the wrong one.
    const merged: Array<{ name: string; description?: string; inputSchema?: unknown }> = [];
    for (const { name: serverName, client, transport } of clients) {
      await client.connect(transport);
      const listed = await client.listTools();
      for (const t of listed.tools) {
        if (toolToClient.has(t.name)) {
          throw new Error(`tool name collision: "${t.name}" exposed by "${traj.toolServer[t.name]}" and "${serverName}"`);
        }
        toolToClient.set(t.name, client);
        traj.toolServer[t.name] = serverName;
        traj.toolRequired[t.name] = Array.isArray((t.inputSchema as any)?.required) ? (t.inputSchema as any).required : [];
        merged.push(t);
      }
    }
    traj.toolNames = merged.map((t) => t.name);
    const tools = mcpToOllamaTools(merged);

    for (let i = 0; i < maxIters; i++) {
      const raw = await chat(opts.host, opts.model, messages, tools, numCtx, timeoutMs);

      if (!raw || typeof raw !== 'object') {
        throw new Error('ollama /api/chat: no/invalid response');
      }
      // Ollama returns {error: "...does not support tools"} when the model's
      // template can't do tool calling — a clean capability verdict, not a crash.
      if (raw.error) {
        if (/does not support tools/i.test(String(raw.error))) traj.supported = false;
        traj.error = String(raw.error);
        return traj;
      }
      // Accumulate generation perf across the rounds (Ollama durations are ns).
      traj.inTokens += raw.prompt_eval_count ?? 0;
      traj.maxPromptTokens = Math.max(traj.maxPromptTokens, raw.prompt_eval_count ?? 0);
      traj.outTokens += raw.eval_count ?? 0;
      totalDurNs += raw.total_duration ?? 0;
      evalDurNs += raw.eval_duration ?? 0;
      traj.totalDurationS = round2(totalDurNs / 1e9);
      traj.evalTps = tps(traj.outTokens, evalDurNs);

      const msg = raw.message;
      const toolCalls: any[] = msg?.tool_calls ?? [];

      if (toolCalls.length === 0) {
        traj.finalAnswer = msg?.content ?? '';
        return traj;
      }

      // Echo the assistant's tool-call turn (normalized), then run each call.
      messages.push({ role: 'assistant', content: msg?.content ?? '', tool_calls: toolCalls });
      for (const tc of toolCalls) {
        const name = tc.function?.name ?? '';
        const args = asArgs(tc.function?.arguments);
        traj.toolCalls.push({ name, arguments: args });

        // A bad tool name / args throws — but "the model picked wrong" is exactly
        // the verdict under test, so capture it and feed it back, don't crash.
        let content: string;
        let isError: boolean;
        const owner = toolToClient.get(name);
        if (!owner) {
          content = `tool call failed: unknown tool "${name}"`;
          isError = true;
        } else {
          try {
            const result: any = await owner.callTool({ name, arguments: args });
            content = resultText(result?.content);
            isError = Boolean(result?.isError);
          } catch (e) {
            content = `tool call failed: ${e instanceof Error ? e.message : String(e)}`;
            isError = true;
          }
        }
        traj.toolResults.push({ name, content, isError });
        messages.push({ role: 'tool', content, tool_name: name });
      }
    }

    // Ran out of iterations still calling tools — report what we have.
    traj.error = `no final answer within ${maxIters} tool rounds`;
    return traj;
  } catch (e) {
    // Connect/list/transport failure or an Ollama HTTP/JSON failure: return a
    // trajectory carrying the error rather than throwing (mirrors perf/fa.ts).
    traj.error = e instanceof Error ? e.message : String(e);
    return traj;
  } finally {
    for (const { client } of clients) await client.close().catch(() => {});
  }
}
