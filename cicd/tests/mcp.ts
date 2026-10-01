/**
 * mcp: can each model drive a real MCP server's tools -- pick the right one from a
 * menu, pass valid arguments, and answer from what it returned (#542)?
 *
 *   npx tsx mcp.ts [--models "a b"] [--num-ctx N] [--num-batch N] [--output results/mcp.json]
 *
 * The tool-call prompt and its grounded judge come from prompts.yaml. A wrong
 * answer is a failure here: the judge asks whether the answer uses the tool
 * result, and the calls themselves are checked (a tool called, known, given its
 * required arguments, run without error).
 *
 * Servers are configuration (the runner's .env or the environment):
 *   MCP_COMMAND / MCP_ARGS     the server under test   (default: testlink-mcp in docker)
 *   MCP_ENV                    comma-separated env names forwarded to it (default TESTLINK_URL,TESTLINK_API_KEY)
 *   MCP_DISTRACTOR_COMMAND / MCP_DISTRACTOR_ARGS   an optional second server whose tools
 *                              join the menu as distractors (e.g. npx @playwright/mcp@latest)
 */
import { converse, ensureModel, load, unload } from './lib/ollama.js';
import { arg, forEachModel, modelsArg, runTest } from './lib/run.js';
import { Menu, type ServerConfig } from './lib/mcp.js';

const split = (s: string | undefined) => (s ?? '').split(/\s+/).filter(Boolean);
const forward = (names: string) =>
  Object.fromEntries(names.split(',').map((n) => n.trim()).filter((n) => process.env[n]).map((n) => [n, process.env[n]!]));

const servers: ServerConfig[] = [{
  name: 'server',
  command: process.env.MCP_COMMAND ?? 'docker',
  args: split(process.env.MCP_ARGS ?? 'run --rm -i -e TESTLINK_URL -e TESTLINK_API_KEY dogkeeper886/testlink-mcp:latest'),
  env: forward(process.env.MCP_ENV ?? 'TESTLINK_URL,TESTLINK_API_KEY'),
}];
if (process.env.MCP_DISTRACTOR_COMMAND) {
  servers.push({ name: 'distractor', command: process.env.MCP_DISTRACTOR_COMMAND, args: split(process.env.MCP_DISTRACTOR_ARGS) });
}

const int = (name: string) => { const v = arg(name); if (v && !/^[1-9][0-9]*$/.test(v)) throw new Error(`--${name} must be a positive integer`); return v ? Number(v) : undefined; };
const numCtx = int('num-ctx');
const numBatch = int('num-batch');
const options = { ...(numCtx ? { num_ctx: numCtx } : {}), ...(numBatch ? { num_batch: numBatch } : {}) };

await runTest('mcp', async () => {
  const menu = await Menu.open(servers);
  process.stderr.write(`menu: ${menu.tools.length} tool(s) from ${servers.map((s) => s.name).join(' + ')}\n`);
  try {
    await forEachModel(modelsArg('OLLAMA37_MCP_MODELS'), async (model) => {
      await ensureModel(model);
      await load(model, 'tool-call', { options });
      const r = await converse(model, 'tool-call', menu, { options });
      if (r.metrics.saturated) r.checks.window = { pass: false, reason: `a round filled the context window (max prompt ${r.metrics.maxPrompt})` };
      await unload(model);
    });
  } finally {
    await menu.close();
  }
});
