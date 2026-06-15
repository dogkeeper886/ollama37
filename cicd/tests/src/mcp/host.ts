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
    toolCalls: [],
    toolResults: [],
    finalAnswer: '',
  };

  try {
    await client.connect(transport);
    const listed = await client.listTools();
    traj.toolNames = listed.tools.map((t) => t.name);
    const tools = mcpToOllamaTools(listed.tools);

    const messages: any[] = [{ role: 'user', content: opts.prompt }];

    for (let i = 0; i < maxIters; i++) {
      const raw = await chat(opts.host, opts.model, messages, tools, numCtx);

      // Ollama returns {error: "...does not support tools"} when the model's
      // template can't do tool calling — a clean capability verdict, not a crash.
      if (raw?.error) {
        if (/does not support tools/i.test(String(raw.error))) {
          traj.supported = false;
          traj.error = String(raw.error);
          return traj;
        }
        throw new Error(`ollama /api/chat: ${raw.error}`);
      }

      const msg = raw?.message;
      const toolCalls: any[] = msg?.tool_calls ?? [];

      if (toolCalls.length === 0) {
        traj.finalAnswer = msg?.content ?? '';
        return traj;
      }

      // Echo the assistant's tool-call turn, then run each call for real.
      messages.push(msg);
      for (const tc of toolCalls) {
        const name = tc.function?.name ?? '';
        const args = (tc.function?.arguments ?? {}) as Record<string, unknown>;
        traj.toolCalls.push({ name, arguments: args });

        const result: any = await client.callTool({ name, arguments: args });
        const content = resultText(result?.content);
        traj.toolResults.push({ name, content, isError: Boolean(result?.isError) });

        messages.push({ role: 'tool', content, tool_name: name });
      }
    }

    // Ran out of iterations still calling tools — report what we have.
    traj.error = `no final answer within ${maxIters} tool rounds`;
    return traj;
  } finally {
    await client.close().catch(() => {});
  }
}
