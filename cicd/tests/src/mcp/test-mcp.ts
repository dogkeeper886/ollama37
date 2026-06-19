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
import { AgentJudge, VerifierJudge } from '../judge/index.js';
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
  /** Opt in the live verifier: the judge calls the server's read-only tools itself
   *  to check the answer against live ground truth (supersedes --judge). */
  verifyLive?: boolean;
  /** Exact tool names the verifier may call. Fail-closed: if empty/unset, the verifier
   *  verifies nothing — pass an explicit allow-list (no prefix heuristic across servers). */
  verifyAllow?: string[];
  /** mcp__<name>__<tool> label the verifier registers the server under. */
  verifyServerName?: string;
  /** Per-model Ollama response timeout (ms). Default 600000 (10 min). */
  timeoutMs?: number;
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
  /** Ollama generation perf for this model. */
  out_tokens: number;
  total_duration_s: number;
  eval_tps: number;
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
    const traj = await runMcpHost({ host: opts.host, model, prompt: opts.prompt, server: opts.server, numCtx: opts.numCtx, timeoutMs: opts.timeoutMs });
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
      out_tokens: traj.outTokens,
      total_duration_s: traj.totalDurationS,
      eval_tps: traj.evalTps,
      check: { overall_pass: simple.pass, simple, agent: null },
    });
    process.stderr.write(`  supported=${traj.supported} calls=${traj.toolCalls.length} simple=${simple.pass} · ${traj.outTokens} tok @ ${traj.evalTps} tok/s in ${traj.totalDurationS}s\n`);
  }

  // Agent-level check, only for models that passed the structural check. Two modes:
  //  --verify-live : the verifier calls the server's read-only tools itself to check
  //                  the answer against LIVE truth (the stronger check; supersedes --judge).
  //  --judge       : the sandboxed agent judge grades groundedness over the captured trajectory.
  const eligible = results.filter((r) => r.check.simple.pass);
  const byModel = new Map(results.map((r) => [r.model, r]));
  const applyVerdicts = (verdicts: Judgment[]) => {
    for (const v of verdicts) {
      const r = byModel.get(v.testId);
      if (r) {
        r.check.agent = v;
        r.check.overall_pass = r.check.simple.pass && v.pass;
      }
    }
  };

  if (eligible.length > 0 && opts.verifyLive) {
    // Full server tool list (same server for every model) → the verifier's allow/deny.
    // Fail-closed: only an explicit --verify-allow opens tools. No prefix heuristic —
    // it can't tell a read-only `get_x` from a destructive `get_and_purge` on an
    // arbitrary server, and the verifier's contract is exact-name allow-listing.
    const allToolNames = Array.from(new Set(eligible.flatMap((r) => r.tool_names)));
    const allowTools = opts.verifyAllow ?? [];
    const verifier = new VerifierJudge(
      { server: opts.server, serverName: opts.verifyServerName ?? 'mcp', toolNames: allToolNames, allowTools },
      '',
    );
    if (await verifier.isAvailable()) {
      applyVerdicts(
        await verifier.verify(eligible.map((r) => ({ testId: r.model, prompt: opts.prompt, answer: trajByModel.get(r.model)!.finalAnswer, toolCalls: r.tool_calls }))),
      );
    } else {
      // Fail closed: a verify-live run whose verifier can't even start has NOT verified anything,
      // so it must not green-light on the structural check alone. Mark every eligible model FAIL.
      process.stderr.write('[ERROR] verify-live requested but the verifier agent is unavailable (auth/availability) — failing closed\n');
      applyVerdicts(eligible.map((r) => ({
        testId: r.model,
        pass: false,
        reason: 'verify-live could not run: verifier agent unavailable (auth/availability)',
        evidenceStatus: 'verifier-unavailable' as const,
      })));
    }
  } else if (eligible.length > 0 && opts.judge) {
    const agentJudge = new AgentJudge();
    if (await agentJudge.isAvailable()) {
      applyVerdicts(await agentJudge.judgeResults(eligible.map((r) => toTestResult(trajByModel.get(r.model)!, opts.prompt))));
    } else {
      process.stderr.write('[WARN] agent judge not available — simple check only\n');
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
        { timestamp: new Date().toISOString(), server: { command: opts.server.command, args: opts.server.args }, prompt: opts.prompt, judge: judgeLabel(opts), results: full },
        null,
        2
      )
    );
    process.stderr.write(`Results written to ${opts.output}\n`);
  }

  printSummary(opts, results);
  return failed > 0 ? 1 : 0;
}

/** Turn an empty evidence cell into a one-glance reason instead of a bare "—" (STORY-010). */
function evidenceWhy(status: Judgment['evidenceStatus']): string {
  switch (status) {
    case 'denied': return '— (tool denied)';
    case 'not-called': return '— (no tool called)';
    case 'no-data': return '— (tool returned no data)';
    case 'verifier-unavailable': return '— (verifier did not run)';
    default: return '—';
  }
}

const tick = (b: boolean): string => (b ? '✅' : '❌');
const judgeLabel = (opts: McpTestOptions): string => (opts.verifyLive ? 'verify-live' : opts.judge ? 'dual' : 'simple');
/** Escape so model/server-controlled text can't break out of the Markdown/<details>/code block. */
const mdSafe = (s: string): string => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function printSummary(opts: McpTestOptions, results: McpModelResult[]): void {
  const supported = results.filter((r) => r.supported);
  const passed = supported.filter((r) => r.check.overall_pass).length;
  const failed = supported.length - passed;
  const sha = process.env.GITHUB_SHA?.slice(0, 8);
  const out: string[] = [];

  // Banner — a glance tells you the outcome. ⚪ when nothing was actually verified (don't fake a green).
  const icon = failed > 0 ? '❌' : passed > 0 ? '✅' : '⚪';
  out.push(`## 🧪 MCP tool-call test — ${icon} ${passed} passed · ${failed} failed`);
  out.push('');
  out.push(`**Server:** \`${opts.server.command} ${opts.server.args.join(' ')}\``);
  out.push(`**Prompt:** ${opts.prompt} · **Judge:** ${judgeLabel(opts)}${sha ? ` · **Commit:** \`${sha}\`` : ''}`);
  out.push('');

  // Scannable table — judge stages + cross-check, no raw JSON in the cells.
  out.push('| Model | Verdict | Tool call(s) | Judge stages (tool·query·content) | Cross-check | Time (s) | Out tok | tok/s |');
  out.push('|---|---|---|---|---|--:|--:|--:|');
  for (const r of results) {
    const verdict = r.check.overall_pass ? '✅ PASS' : r.supported ? '❌ FAIL' : '⚪ NO TOOL SUPPORT';
    const calls = r.tool_calls.map((c) => c.name).join(', ') || '—';
    const s = r.check.agent?.stages;
    const stages = s ? `${tick(s.tool)} · ${tick(s.query)} · ${tick(s.content)}` : '—';
    const cc = r.check.agent?.crossCheckUnsupported;
    const cross = cc === undefined ? '—' : cc.length ? `❌ ${cc.join(', ').slice(0, 60)}` : '✅ grounded';
    out.push(`| ${r.model} | ${verdict} | ${calls} | ${stages} | ${cross} | ${r.total_duration_s || '—'} | ${r.out_tokens || '—'} | ${r.eval_tps || '—'} |`);
  }

  // Per-model detail (reasoning + raw evidence) tucked behind a toggle — proof without the clutter.
  for (const r of results) {
    const reason = (r.check.agent?.reason ?? r.check.simple.reason ?? '').trim();
    const evidence = r.check.agent?.evidence ?? '';
    out.push('');
    out.push(`<details><summary>Judge detail — ${mdSafe(r.model)}</summary>`);
    out.push('');
    if (reason) out.push(`**Reasoning:** ${mdSafe(reason.replace(/\n/g, ' '))}`);
    out.push('');
    if (evidence) {
      // HTML-escaped <pre> (not a ``` fence) so a ``` or </details> in the result can't break out.
      out.push('**Live evidence:**');
      out.push(`<pre>${mdSafe(evidence.slice(0, 4000))}</pre>`);
    } else {
      out.push(`**Live evidence:** ${evidenceWhy(r.check.agent?.evidenceStatus)}`);
    }
    out.push('</details>');
  }
  process.stdout.write(out.join('\n') + '\n');
}
