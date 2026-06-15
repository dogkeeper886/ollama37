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
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';

/** How to spawn the stdio MCP server. testlink-mcp is just one such config. */
export interface McpServerConfig {
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
  server: McpServerConfig;
  numCtx?: number;
  /** Max chat↔tool rounds before giving up (guards against a tool loop). */
  maxIters?: number;
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
  /** Tool names the MCP server exposed. */
  toolNames: string[];
  /** Required arg names per tool, from each tool's inputSchema.required. */
  toolRequired: Record<string, string[]>;
  toolCalls: ToolCallRecord[];
  toolResults: ToolResultRecord[];
  finalAnswer: string;
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

async function chat(host: string, model: string, messages: unknown[], tools: unknown[], numCtx: number): Promise<any> {
  const res = await fetch(`${host}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages, tools, stream: false, options: { temperature: 0, seed: 42, num_ctx: numCtx } }),
  });
  return res.json();
}

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

  const transport = new StdioClientTransport({
    command: opts.server.command,
    args: opts.server.args,
    cwd: opts.server.cwd,
    env: { ...getDefaultEnvironment(), ...(opts.server.env ?? {}) },
  });
  const client = new Client({ name: 'ollama37-mcp-host', version: '1.0.0' });

  const traj: McpTrajectory = {
    model: opts.model,
    supported: true,
    toolNames: [],
    toolRequired: {},
    toolCalls: [],
    toolResults: [],
    finalAnswer: '',
  };

  const messages: any[] = [{ role: 'user', content: opts.prompt }];

  try {
    await client.connect(transport);
    const listed = await client.listTools();
    traj.toolNames = listed.tools.map((t) => t.name);
    for (const t of listed.tools) {
      const req = (t.inputSchema as any)?.required;
      traj.toolRequired[t.name] = Array.isArray(req) ? req : [];
    }
    const tools = mcpToOllamaTools(listed.tools);

    for (let i = 0; i < maxIters; i++) {
      const raw = await chat(opts.host, opts.model, messages, tools, numCtx);

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
        try {
          const result: any = await client.callTool({ name, arguments: args });
          content = resultText(result?.content);
          isError = Boolean(result?.isError);
        } catch (e) {
          content = `tool call failed: ${e instanceof Error ? e.message : String(e)}`;
          isError = true;
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
    await client.close().catch(() => {});
  }
}
