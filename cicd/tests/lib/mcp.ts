/**
 * Real stdio MCP servers as one tool menu (#542). Never calls the model: the
 * chat-and-tool loop is lib/ollama.ts's converse(), which runs tools through this.
 *
 * Several servers merge into one menu, so picking the right tool is a real choice;
 * a call routes back to the server that owns the name. Which servers, and the
 * credentials they need, are configuration, not code.
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import type { Tool } from 'ollama';

export interface ServerConfig {
  name: string;
  command: string;
  args: string[];
  /** Extra environment for the server (its credentials), over a minimal default. */
  env?: Record<string, string>;
}

export interface ToolResult {
  name: string;
  content: string;
  isError: boolean;
}

export class Menu {
  /** Ollama tools[] entries, merged across servers. */
  tools: Tool[] = [];
  /** Required argument names per tool, from its inputSchema. */
  required: Record<string, string[]> = {};
  private owner = new Map<string, Client>();
  /** In-process tools: name → fixed result. */
  private fixed = new Map<string, string>();
  private clients: Client[] = [];

  static async open(servers: ServerConfig[]): Promise<Menu> {
    const menu = new Menu();
    try {
      for (const s of servers) {
        const client = new Client({ name: 'ollama37-mcp', version: '3.0.0' });
        menu.clients.push(client);
        await client.connect(new StdioClientTransport({
          command: s.command,
          args: s.args,
          env: { ...getDefaultEnvironment(), ...(s.env ?? {}) },
        }));
        for (const t of (await client.listTools()).tools) {
          // One name on two servers cannot be routed; fail rather than guess.
          if (menu.owner.has(t.name)) throw new Error(`tool "${t.name}" is exposed by two servers`);
          menu.owner.set(t.name, client);
          const schema = (t.inputSchema ?? { type: 'object', properties: {} }) as { required?: string[] };
          menu.required[t.name] = Array.isArray(schema.required) ? schema.required : [];
          menu.tools.push({
            type: 'function',
            function: { name: t.name, description: t.description ?? '', parameters: schema as Tool['function']['parameters'] },
          });
        }
      }
    } catch (e) {
      await menu.close();
      throw e;
    }
    return menu;
  }

  /** Run one tool call. A wrong name or bad arguments is the model's mistake under test: reported, not thrown. */
  async call(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    const result = this.fixed.get(name);
    if (result !== undefined) return { name, content: result, isError: false };
    const client = this.owner.get(name);
    if (!client) return { name, content: `unknown tool "${name}"`, isError: true };
    try {
      const r = (await client.callTool({ name, arguments: args })) as { content?: unknown; isError?: boolean };
      const text = Array.isArray(r.content)
        ? r.content.map((c: { type?: string; text?: string }) => (c?.type === 'text' && typeof c.text === 'string' ? c.text : JSON.stringify(c))).join('\n')
        : '';
      return { name, content: text, isError: Boolean(r.isError) };
    } catch (e) {
      return { name, content: `tool call failed: ${e instanceof Error ? e.message : String(e)}`, isError: true };
    }
  }

  /** A menu of in-process tools, each answering a fixed result (prompts.yaml `tools`). */
  static local(tools: { name: string; description: string; result: string }[]): Menu {
    const menu = new Menu();
    for (const t of tools) {
      menu.required[t.name] = [];
      menu.fixed.set(t.name, t.result);
      menu.tools.push({ type: 'function', function: { name: t.name, description: t.description, parameters: { type: 'object', properties: {} } } });
    }
    return menu;
  }

  async close(): Promise<void> {
    for (const c of this.clients) await c.close().catch(() => {});
  }
}
