/**
 * Per-model MCP tool-call test (the cli.ts test-mcp subcommand).
 *
 * For each model: drive a REAL stdio MCP server through the host (host.ts) and
 * grade the trajectory. The simple check is structural (did the model call a
 * real tool, did the call succeed, did it produce a final answer?); with
 * --judge the keyless AgentJudge adds the semantic check (did the final answer
 * correctly use the tool result?). A model whose template can't do tools is
 * reported "no tool support" — a clean verdict, not a failure of the harness.
 *
 * Mirrors perf/throughput.ts: markdown summary on stdout (CI step summary) and,
 * with --output, a JSON report. Exit code is non-zero if any model fails.
 */
import { writeFileSync } from 'node:fs';
import { runMcpHost, type McpServerConfig, type McpTrajectory } from './host.js';
import { AgentJudge } from '../judge/index.js';
import { TestResult, Judgment } from '../types.js';

const JUDGE_CRITERIA =
  'Given the user prompt and the tool result(s) the model received, the final answer must ' +
  'correctly use the tool result to answer the prompt. Reject empty answers, answers that ' +
  'ignore or contradict the tool result, or data the tool result does not contain.';

export interface McpTestOptions {
  models: string[];
  prompt: string;
  server: McpServerConfig;
  host: string;
  numCtx: number;
  judge: boolean;
  output?: string;
}

interface McpSimpleVerdict {
  pass: boolean;
  reason: string;
}

interface McpModelResult {
  model: string;
  supported: boolean;
  tool_names: string[];
  tool_calls: { name: string; arguments: Record<string, unknown> }[];
  tool_results: { name: string; isError: boolean }[];
  final_answer_preview: string;
  error?: string;
  check: { overall_pass: boolean; simple: McpSimpleVerdict; agent: Judgment | null };
}

/** Structural check: the model must have called a real tool, the call must have
 *  succeeded, and it must have produced a non-empty final answer. */
function simpleMcpCheck(t: McpTrajectory): McpSimpleVerdict {
  if (!t.supported) return { pass: false, reason: 'model template does not support tools' };
  if (t.error) return { pass: false, reason: t.error };
  if (t.toolCalls.length === 0) return { pass: false, reason: 'model produced no tool call' };
  const unknown = t.toolCalls.find((c) => !t.toolNames.includes(c.name));
  if (unknown) return { pass: false, reason: `called unknown tool "${unknown.name}"` };
  // Args vs schema: every required arg of the called tool must be present.
  for (const c of t.toolCalls) {
    const missing = (t.toolRequired[c.name] ?? []).filter((k) => !(k in c.arguments));
    if (missing.length) return { pass: false, reason: `tool "${c.name}" called without required arg(s): ${missing.join(', ')}` };
  }
  const errored = t.toolResults.find((r) => r.isError);
  if (errored) return { pass: false, reason: `tool "${errored.name}" call failed (bad args or server error)` };
  if (!t.finalAnswer.trim()) return { pass: false, reason: 'empty final answer' };
  return { pass: true, reason: `called ${t.toolCalls.map((c) => c.name).join(', ')} and produced a final answer` };
}

/** Build a synthetic TestResult so the AgentJudge can grade groundedness.
 *  Lead with the prompt + final answer: the judge truncates step stdout, so a
 *  large tool result must not push the answer out of the judge's window. Each
 *  tool result is capped to keep enough grounding context within that window. */
function toTestResult(t: McpTrajectory, prompt: string): TestResult {
  const toolLog = t.toolCalls
    .map((c, i) => `- ${c.name}(${JSON.stringify(c.arguments)}) -> ${(t.toolResults[i]?.content ?? '').slice(0, 500)}`)
    .join('\n');
  const stdout = `PROMPT: ${prompt}\n\nFINAL ANSWER: ${t.finalAnswer}\n\nTOOL CALLS AND RESULTS:\n${toolLog}`;
  return {
    testCase: {
      id: t.model,
      name: `mcp:${t.model}`,
      suite: 'inference',
      priority: 1,
      timeout: 60000,
      dependencies: [],
      goal: 'Use the MCP tool result to answer the prompt',
      steps: [{ name: 'tool-call', command: '(captured MCP trajectory)' }],
      criteria: JUDGE_CRITERIA,
    },
    steps: [{ name: 'tool-call', command: '(captured MCP trajectory)', stdout, stderr: '', exitCode: 0, duration: 0 }],
    totalDuration: 0,
    logs: '',
    logFile: '',
  };
}

export async function runMcpTest(opts: McpTestOptions): Promise<number> {
  const results: McpModelResult[] = [];
  const trajByModel = new Map<string, McpTrajectory>();

  for (const model of opts.models) {
    process.stderr.write(`--- ${model} ---\n`);
    const traj = await runMcpHost({ host: opts.host, model, prompt: opts.prompt, server: opts.server, numCtx: opts.numCtx });
    trajByModel.set(model, traj);
    const simple = simpleMcpCheck(traj);
    results.push({
      model,
      supported: traj.supported,
      tool_names: traj.toolNames,
      tool_calls: traj.toolCalls,
      tool_results: traj.toolResults.map((r) => ({ name: r.name, isError: r.isError })),
      final_answer_preview: traj.finalAnswer.slice(0, 200),
      error: traj.error,
      check: { overall_pass: simple.pass, simple, agent: null },
    });
    process.stderr.write(`  supported=${traj.supported} calls=${traj.toolCalls.length} simple=${simple.pass}\n`);
  }

  // Agent judge (dual mode): one batch over a single reused session, only for
  // models that passed the structural check (no point grading a non-call).
  if (opts.judge) {
    const eligible = results.filter((r) => r.check.simple.pass);
    if (eligible.length > 0) {
      const agentJudge = new AgentJudge();
      if (await agentJudge.isAvailable()) {
        const byModel = new Map(results.map((r) => [r.model, r]));
        const verdicts = await agentJudge.judgeResults(eligible.map((r) => toTestResult(trajByModel.get(r.model)!, opts.prompt)));
        for (const v of verdicts) {
          const r = byModel.get(v.testId);
          if (r) {
            r.check.agent = v;
            r.check.overall_pass = r.check.simple.pass && v.pass;
          }
        }
      } else {
        process.stderr.write('[WARN] agent judge not available — simple check only\n');
      }
    }
  }

  // "No tool support" is an informational capability verdict, not a harness
  // failure — only a supported model that failed its check drives a non-zero exit.
  const failed = results.filter((r) => r.supported && !r.check.overall_pass).length;

  if (opts.output) {
    const full = results.map((r) => ({ ...r, trajectory: trajByModel.get(r.model) }));
    writeFileSync(
      opts.output,
      JSON.stringify(
        { timestamp: new Date().toISOString(), server: { command: opts.server.command, args: opts.server.args }, prompt: opts.prompt, judge: opts.judge ? 'dual' : 'simple', results: full },
        null,
        2
      )
    );
    process.stderr.write(`Results written to ${opts.output}\n`);
  }

  printSummary(opts, results);
  return failed > 0 ? 1 : 0;
}

function printSummary(opts: McpTestOptions, results: McpModelResult[]): void {
  const out: string[] = [];
  out.push('## MCP tool-call test');
  out.push('');
  out.push(`**Server:** \`${opts.server.command} ${opts.server.args.join(' ')}\` · **Prompt:** ${opts.prompt} · **Judge:** ${opts.judge ? 'dual' : 'simple'}`);
  out.push('');
  out.push('| Model | Verdict | Tool calls | Answer | Reason |');
  out.push('|---|---|---|---|---|');
  for (const r of results) {
    const verdict = r.check.overall_pass ? 'PASS' : r.supported ? 'FAIL' : 'NO TOOL SUPPORT';
    const calls = r.tool_calls.map((c) => c.name).join(', ') || '—';
    const answer = r.final_answer_preview.trim() ? 'yes' : 'no';
    const reason = (r.check.agent?.reason ?? r.check.simple.reason ?? '').replace(/[\n|]/g, ' ').slice(0, 200);
    out.push(`| ${r.model} | ${verdict} | ${calls} | ${answer} | ${reason} |`);
  }
  process.stdout.write(out.join('\n') + '\n');
}
