/**
 * Verifier Judge — checks a model's MCP answer against INDEPENDENT live ground truth.
 *
 * Rather than trust the tool result the model captured (which could be stale or mis-captured),
 * the verifier makes its OWN read-only call to the live MCP server and grades the model's answer
 * against THAT fresh result. Grading is done by the sandboxed, keyless `AgentJudge`
 * (`mcpServers: []`, rejects every tool call) — the grading agent only reads text, it is never
 * exposed to the live tool, so there is no account-connector flood and no write path.
 *
 * Read-only is enforced by an exact allow-list of tool names: only those are ever called, and a
 * name not on the server's menu is skipped. Empty allow-list ⇒ verifies nothing (fail-closed).
 *
 * Why not have the *agent* call the tool? The Claude ACP adapters either don't surface a
 * client-injected stdio MCP server (claude-agent-acp) or flood the agent with the operator's
 * claude.ai connectors that can only be disabled with an API key (claude-code-acp). Both keyless
 * paths are exhausted (see #285). Calling the tool ourselves keeps the independent-live-check
 * property — a fresh second call to truth, not the model's captured result — with none of that.
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import { AgentJudge } from './agent-judge.js';
import { CONFIG } from '../config.js';
import type { McpServerConfig } from '../mcp/host.js';
import type { Judgment, TestResult } from '../types.js';

/** One model's answer to verify against the live server. */
export interface VerifyTarget {
  testId: string;
  prompt: string;
  answer: string;
}

export interface VerifierConfig {
  /** The stdio MCP server the host used (same command/args/env). */
  server: McpServerConfig;
  /** Rule prefix the host reported tools under (kept for report symmetry). */
  serverName: string;
  /** Every tool the server exposes (informational). */
  toolNames: string[];
  /** EXACT read-only tool names the verifier may call. Empty ⇒ verifies nothing (fail-closed). */
  allowTools: string[];
}

interface GroundTruth {
  tool: string;
  result: string;
  isError: boolean;
}

/** Join an MCP CallToolResult's content blocks into one text payload (mirrors host.ts). */
function resultText(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content
    .map((c: any) => (c?.type === 'text' && typeof c.text === 'string' ? c.text : JSON.stringify(c)))
    .join('\n');
}

const VERIFY_CRITERIA =
  'You are given the user PROMPT, the model ANSWER to verify, and LIVE GROUND TRUTH independently ' +
  'fetched from the real tool. PASS only if the answer is consistent with the ground truth. FAIL ' +
  'if the answer states facts the ground truth contradicts or does not contain (wrong or invented ' +
  'names, ids, counts, or data), or if the answer is empty.';

export class VerifierJudge {
  private readonly cfg: VerifierConfig;
  private readonly allow: string[];
  private readonly judge: AgentJudge;

  constructor(cfg: VerifierConfig, agentCmd: string = CONFIG.judge.agent, cwd: string = process.cwd()) {
    this.cfg = cfg;
    this.allow = [...new Set(cfg.allowTools.filter(Boolean))];
    // The verdict is graded by the sandboxed agent judge (text-only, keyless).
    this.judge = new AgentJudge(agentCmd, cwd);
  }

  /** Available iff the (keyless, sandboxed) grading agent is reachable. */
  async isAvailable(): Promise<boolean> {
    return this.judge.isAvailable();
  }

  /** One independent read-only call per allow-listed tool against the LIVE server. */
  private async fetchGroundTruth(): Promise<GroundTruth[]> {
    if (this.allow.length === 0) return [];
    const transport = new StdioClientTransport({
      command: this.cfg.server.command,
      args: this.cfg.server.args,
      cwd: this.cfg.server.cwd,
      env: { ...getDefaultEnvironment(), ...(this.cfg.server.env ?? {}) },
    });
    const client = new Client({ name: 'ollama37-verifier', version: '1.0.0' });
    const out: GroundTruth[] = [];
    try {
      await client.connect(transport);
      const onMenu = new Set((await client.listTools()).tools.map((t) => t.name));
      for (const tool of this.allow) {
        if (!onMenu.has(tool)) continue; // never call a tool the server doesn't expose
        try {
          const r: any = await client.callTool({ name: tool, arguments: {} });
          out.push({ tool, result: resultText(r?.content), isError: Boolean(r?.isError) });
        } catch (e) {
          out.push({ tool, result: `tool call failed: ${e instanceof Error ? e.message : String(e)}`, isError: true });
        }
      }
    } finally {
      await client.close().catch(() => {});
    }
    return out;
  }

  /** Synthetic TestResult so the sandboxed judge grades the answer vs the live ground truth. */
  private toTestResult(t: VerifyTarget, truth: GroundTruth[]): TestResult {
    const gt = truth.map((x) => `${x.tool} ->\n${x.result.slice(0, 1000)}`).join('\n\n');
    const stdout =
      `PROMPT: ${t.prompt}\n\nMODEL ANSWER (verify this): ${t.answer}\n\n` +
      `LIVE GROUND TRUTH (independently fetched from the real tool):\n${gt}`;
    return {
      testCase: {
        id: t.testId,
        name: `verify:${t.testId}`,
        suite: 'inference',
        priority: 1,
        timeout: 60000,
        dependencies: [],
        goal: 'Decide whether the model answer matches the live ground truth',
        steps: [{ name: 'verify', command: '(independent live tool call)' }],
        criteria: VERIFY_CRITERIA,
      },
      steps: [{ name: 'verify', command: '(independent live tool call)', stdout, stderr: '', exitCode: 0, duration: 0 }],
      totalDuration: 0,
      logs: '',
      logFile: '',
    };
  }

  /** Verify every target against ONE independent fetch of live ground truth. */
  async verify(targets: VerifyTarget[]): Promise<Judgment[]> {
    let truth: GroundTruth[];
    try {
      truth = await this.fetchGroundTruth();
    } catch (e) {
      const reason = 'verifier could not reach the live server: ' + (e instanceof Error ? e.message : String(e));
      return targets.map((t) => ({ testId: t.testId, pass: false, reason }));
    }
    const usable = truth.filter((x) => !x.isError && x.result.trim());
    if (usable.length === 0) {
      const reason =
        this.allow.length === 0
          ? 'verifier allow-list empty — verified nothing (fail-closed)'
          : 'verifier got no usable result from any allowed read-only tool (fail-closed)';
      return targets.map((t) => ({ testId: t.testId, pass: false, reason }));
    }
    process.stderr.write(`  [verify] live ground truth from: ${usable.map((x) => x.tool).join(', ')}\n`);
    const verdicts = await this.judge.judgeResults(targets.map((t) => this.toTestResult(t, usable)));
    const evidence = usable.map((x) => `${x.tool}: ${x.result.replace(/\s+/g, ' ').slice(0, 300)}`).join(' | ');
    return verdicts.map((v) => ({ ...v, evidence: v.evidence ?? evidence }));
  }
}
