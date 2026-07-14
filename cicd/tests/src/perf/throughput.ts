/**
 * Throughput benchmark (port of cicd/scripts/benchmark-throughput.sh).
 *
 * For each model: ensure pulled → warmup → benchmark generate (tok/s, durations)
 * → GPU offload/VRAM → output check. The output check is `simple` (non-empty)
 * always, plus the keyless `AgentJudge` when `--judge` is set ("dual"). Perf
 * numbers without a coherence check are misleading — a model can emit 128 tokens
 * of garbage at 50 tok/s — so a passing result means fast AND meaningful.
 *
 * Emits a markdown summary on stdout (for the CI step summary) and, with
 * --output, a JSON report artifact. Exit code is non-zero if any model fails.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFileSync } from 'node:fs';
import { captureResponse } from './capture.js';
import { gpuInfo, gpuOffload, type GpuRow } from './gpu.js';
import { simpleContentCheck } from './content-check.js';
import { AgentJudge } from '../judge/index.js';
import { TestResult, Judgment } from '../types.js';

const execFileAsync = promisify(execFile);

const PROMPT = 'Explain how a computer works to a curious 10-year-old. Be fun and use analogies.';
const JUDGE_CRITERIA =
  'The response must be a coherent, on-topic answer to the prompt in the right language. ' +
  'Reject empty, garbled, repetitive nonsense, off-topic, or error-message output.';

export interface ThroughputOptions {
  models: string[];
  numPredict: number;
  numCtx: number;
  /** Micro-batch size (num_batch). Undefined = model/server default (512). */
  numBatch?: number;
  judge: boolean;
  host: string;
  output?: string;
}

interface ModelResult {
  model: string;
  in_tokens: number;
  out_tokens: number;
  prompt_eval_tps: number;
  eval_tps: number;
  gpu_offload_pct: number;
  vram_used_mib: number[];
  done_reason: string;
  response_preview: string;
  check: { overall_pass: boolean; simple: ReturnType<typeof simpleContentCheck>; agent: Judgment | null };
}

async function gitSha(): Promise<string> {
  try {
    const { stdout } = await execFileAsync('git', ['rev-parse', '--short', 'HEAD']);
    return stdout.trim();
  } catch {
    return 'unknown';
  }
}

async function ensureModel(host: string, model: string): Promise<void> {
  const show = await fetch(`${host}/api/show`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: model }),
  }).catch(() => null);
  if (show && show.ok) return;
  await fetch(`${host}/api/pull`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: model, stream: false }),
  });
}

/** Build a synthetic TestResult so the AgentJudge can grade the captured output. */
function toTestResult(model: string, response: string, thinking: string): TestResult {
  const output = [thinking, response].filter((s) => s.trim()).join('\n\n');
  return {
    testCase: {
      id: model,
      name: `throughput:${model}`,
      suite: 'inference',
      priority: 1,
      timeout: 60000,
      dependencies: [],
      goal: 'Produce a coherent answer to the prompt',
      steps: [{ name: 'generate', command: '(captured /api/generate response)' }],
      criteria: JUDGE_CRITERIA,
    },
    steps: [{ name: 'generate', command: '(captured /api/generate response)', stdout: output, stderr: '', exitCode: 0, duration: 0 }],
    totalDuration: 0,
    logs: '',
    logFile: '',
  };
}

export async function runThroughput(opts: ThroughputOptions): Promise<number> {
  const { host, models, numPredict, numCtx, numBatch, judge } = opts;
  const sha = await gitSha();
  const gpuBefore = await gpuInfo();

  const results: ModelResult[] = [];
  // Full captured text per model — the report keeps only a preview, but the
  // agent judge must see the complete response + thinking.
  const fullText = new Map<string, { response: string; thinking: string }>();

  for (const model of models) {
    process.stderr.write(`--- ${model} ---\n`);
    await ensureModel(host, model);

    let cap;
    try {
      cap = await captureResponse(host, model, PROMPT, numPredict, numCtx, numBatch);
    } catch (e) {
      process.stderr.write(`  ERROR: ${e instanceof Error ? e.message : e}\n`);
      results.push({
        model, in_tokens: 0, out_tokens: 0, prompt_eval_tps: 0, eval_tps: 0,
        gpu_offload_pct: 0, vram_used_mib: [], done_reason: 'error', response_preview: '',
        check: { overall_pass: false, simple: { pass: false, reason: 'capture failed', source: 'none' }, agent: null },
      });
      continue;
    }

    const offload = await gpuOffload(host, model);
    const loaded = await gpuInfo();
    const simple = simpleContentCheck(cap.response, cap.thinking);
    fullText.set(model, { response: cap.response, thinking: cap.thinking });

    results.push({
      model,
      in_tokens: cap.inTokens,
      out_tokens: cap.outTokens,
      prompt_eval_tps: cap.promptEvalTps,
      eval_tps: cap.evalTps,
      gpu_offload_pct: offload,
      vram_used_mib: loaded.map((g) => g.usedMib),
      done_reason: cap.doneReason,
      response_preview: cap.response.slice(0, 120),
      check: { overall_pass: simple.pass, simple, agent: null },
    });
    process.stderr.write(`  ${cap.outTokens} tok @ ${cap.evalTps} tok/s · offload ${offload}% · simple=${simple.pass}\n`);
  }

  // Agent judge (dual mode): one batch over a single reused session, only for
  // models that passed the simple check (an empty output is already a fail).
  let judgeFellBack = false;
  if (judge) {
    const eligible = results.filter((r) => r.check.simple.pass);
    if (eligible.length > 0) {
      const agentJudge = new AgentJudge();
      if (await agentJudge.isAvailable()) {
        const byModel = new Map(results.map((r) => [r.model, r]));
        const trs = eligible.map((r) => {
          const ft = fullText.get(r.model) ?? { response: r.response_preview, thinking: '' };
          return toTestResult(r.model, ft.response, ft.thinking);
        });
        const verdicts = await agentJudge.judgeResults(trs);
        for (const v of verdicts) {
          const r = byModel.get(v.testId);
          if (r) {
            r.check.agent = v;
            r.check.overall_pass = r.check.simple.pass && v.pass;
          }
        }
      } else {
        judgeFellBack = true;
        process.stderr.write('[WARN] agent judge not available — simple check only\n');
      }
    }
  }

  const gpuAfter = await gpuInfo();
  const failed = results.filter((r) => !r.check.overall_pass).length;

  // JSON report artifact
  if (opts.output) {
    const report = {
      git_sha: sha,
      timestamp: new Date().toISOString(),
      config: { num_predict: numPredict, num_ctx: numCtx, num_batch: numBatch ?? null },
      gpu: { before: gpuBefore, after: gpuAfter },
      results,
    };
    writeFileSync(opts.output, JSON.stringify(report, null, 2));
    process.stderr.write(`Results written to ${opts.output}\n`);
  }

  // Markdown summary on stdout (CI appends this to the step summary). When dual was asked for but
  // the judge couldn't run, say so — don't claim "dual" for a simple-only result (STORY-010).
  const judgeMode = judge ? (judgeFellBack ? 'dual → simple (judge unavailable)' : 'dual') : 'simple';
  printSummary(sha, gpuBefore, numCtx, judgeMode, results);
  return failed > 0 ? 1 : 0;
}

function printSummary(sha: string, gpu: GpuRow[], numCtx: number, mode: string, results: ModelResult[]): void {
  const gpuName = gpu[0]?.name ?? 'unknown';
  const gpuTotal = gpu[0]?.totalMib ?? '?';
  const out: string[] = [];
  out.push('## Throughput Benchmark');
  out.push('');
  out.push(`**Commit:** \`${sha}\` | **GPU:** ${gpu.length}x ${gpuName} | **VRAM:** ${gpuTotal} MiB each | **Context:** ${numCtx} | **Judge:** ${mode}`);
  out.push('');
  out.push('| Model | Check | IN tok | OUT tok | Prompt tok/s | Gen tok/s | GPU% | VRAM used (MiB) |');
  out.push('|---|---|---|---|---|---|---|---|');
  for (const r of results) {
    const check = r.check.overall_pass ? 'PASS' : 'FAIL';
    out.push(`| ${r.model} | ${check} | ${r.in_tokens} | ${r.out_tokens} | ${r.prompt_eval_tps} | ${r.eval_tps} | ${r.gpu_offload_pct}% | ${r.vram_used_mib.join(' / ') || 'n/a'} |`);
  }
  const failures = results.filter((r) => !r.check.overall_pass);
  if (failures.length > 0) {
    out.push('');
    out.push('### Failed output checks');
    out.push('');
    for (const r of failures) {
      const reason = r.check.agent?.reason ?? r.check.simple.reason;
      out.push(`- **${r.model}**: ${reason}`);
    }
  }
  process.stdout.write(out.join('\n') + '\n');
}
