/**
 * Long-context benchmark for the flash-attention path comparison.
 *
 * Unlike `bench-throughput` (a fast, short-prompt K80 model-regression tool), this
 * primes a long DETERMINISTIC prompt so prefill is realistic and the KV cache is
 * full before decode — the only regime where flash attention's decode benefit and
 * prefill cost are visible (docs/porting-k80.md §3). A buried "needle" makes
 * long-context correctness checkable *deterministically*: a broken FA path emits
 * fluent-but-wrong output, which a coherence judge passes and the needle catches.
 *
 * Per (model, context): prime ~context tokens → generate → measure prefill/decode
 * tok/s → verdict = simple ∧ needle ∧ agent(dual). Deterministic prompt ⇒ stable
 * numbers run-to-run. Calibrate against the FA=0 (cuBLAS) baseline: a model that
 * fails the needle under cuBLAS is just weak at retrieval — exclude it — so an
 * FA-only needle failure convicts the FA path.
 */
import { writeFileSync } from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { captureResponse } from './capture.js';
import { gpuInfo, gpuOffload, type GpuRow } from './gpu.js';
import { simpleContentCheck } from './content-check.js';
import { AgentJudge } from '../judge/index.js';
import { TestResult, Judgment } from '../types.js';

const execFileAsync = promisify(execFile);

const NEEDLE_CODE = '7492';
const NEEDLE_SENTENCE = `CRITICAL FACT: the secret launch code is ${NEEDLE_CODE}. Remember this exact number.`;
const TASK =
  '\n\nThe passage above is repetitive filler; ignore it except for the one CRITICAL FACT. In about ' +
  '100 words, explain what flash attention is and why it helps large language model inference on GPUs. ' +
  'Then on its own final line write the secret launch code from the passage exactly like: ' +
  'LAUNCH CODE: <code>.';
const JUDGE_CRITERIA =
  'The response must be a coherent, on-topic English explanation of flash attention — a real GPU/ML ' +
  'technique (streaming the KV cache, tiling, avoiding materializing the full attention matrix, etc.). ' +
  'Reject empty, garbled, repetitive, off-topic, or error-message output. (The launch-code line is ' +
  'checked separately and is not your concern.)';

const FILLER_WORDS =
  'flash attention kernel tensor core memory bandwidth throughput latency prefill decode softmax matmul ' +
  'cublas turing ampere kepler compute capability toolchain codegen register warp shuffle transpose ' +
  'quantization inference context window batch sequence token cache'.split(' ');

/** Deterministic long prompt: fixed filler + a needle at ~30% depth + the task. Same target ⇒ same bytes. */
function buildPrompt(targetTokens: number): string {
  const targetWords = Math.max(64, Math.floor(targetTokens * 0.8)); // ~0.8 words/token for English filler
  const needleAtWord = Math.floor(targetWords * 0.3);
  const parts: string[] = [];
  let i = 0;
  let words = 0;
  let needleDone = false;
  while (words < targetWords) {
    if (!needleDone && words >= needleAtWord) {
      parts.push(NEEDLE_SENTENCE);
      needleDone = true;
    }
    const chunk: string[] = [];
    for (let k = 0; k < 9; k++) {
      chunk.push(FILLER_WORDS[(i + 7 * k) % FILLER_WORDS.length]);
      i += 63;
    }
    parts.push('The ' + chunk.join(' ') + ' determines ' + FILLER_WORDS[i % FILLER_WORDS.length] + ' performance.');
    i += 1;
    words += 11;
  }
  if (!needleDone) parts.push(NEEDLE_SENTENCE);
  return parts.join(' ') + TASK;
}

function needleCheck(response: string): { pass: boolean; reason: string } {
  const found = response.includes(NEEDLE_CODE);
  return {
    pass: found,
    reason: found ? `needle ${NEEDLE_CODE} present` : `needle ${NEEDLE_CODE} MISSING — long-context retrieval failed`,
  };
}

export interface ContextOptions {
  models: string[];
  numPredict: number;
  context: number; // target prompt tokens
  judge: boolean;
  host: string;
  output?: string;
}

interface CtxResult {
  model: string;
  in_tokens: number;
  out_tokens: number;
  prompt_eval_tps: number;
  eval_tps: number;
  gpu_offload_pct: number;
  vram_used_mib: number[];
  done_reason: string;
  response_preview: string;
  needle: { pass: boolean; reason: string };
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

/** Wrap the captured output so the AgentJudge can grade its coherence. */
function toTestResult(model: string, response: string, thinking: string): TestResult {
  const output = [thinking, response].filter((s) => s.trim()).join('\n\n');
  return {
    testCase: {
      id: model,
      name: `context:${model}`,
      suite: 'inference',
      priority: 1,
      timeout: 120000,
      dependencies: [],
      goal: 'Coherent explanation of flash attention',
      steps: [{ name: 'generate', command: '(captured /api/generate response)' }],
      criteria: JUDGE_CRITERIA,
    },
    steps: [{ name: 'generate', command: '(captured /api/generate response)', stdout: output, stderr: '', exitCode: 0, duration: 0 }],
    totalDuration: 0,
    logs: '',
    logFile: '',
  };
}

export async function runContext(opts: ContextOptions): Promise<number> {
  const { host, models, numPredict, context, judge } = opts;
  const sha = await gitSha();
  const gpuBefore = await gpuInfo();
  const prompt = buildPrompt(context);
  const numCtx = context + numPredict + 512; // headroom for the primed prompt + generation

  const results: CtxResult[] = [];
  const fullText = new Map<string, { response: string; thinking: string }>();

  for (const model of models) {
    process.stderr.write(`--- ${model} (ctx target ${context}) ---\n`);
    await ensureModel(host, model);

    let cap;
    try {
      cap = await captureResponse(host, model, prompt, numPredict, numCtx);
    } catch (e) {
      process.stderr.write(`  ERROR: ${e instanceof Error ? e.message : e}\n`);
      results.push({
        model, in_tokens: 0, out_tokens: 0, prompt_eval_tps: 0, eval_tps: 0, gpu_offload_pct: 0, vram_used_mib: [],
        done_reason: 'error', response_preview: '', needle: { pass: false, reason: 'no response' },
        check: { overall_pass: false, simple: { pass: false, reason: 'capture failed', source: 'none' }, agent: null },
      });
      continue;
    }

    const offload = await gpuOffload(host, model);
    const loaded = await gpuInfo();
    const simple = simpleContentCheck(cap.response, cap.thinking);
    const needle = needleCheck(cap.response);
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
      needle,
      check: { overall_pass: simple.pass && needle.pass, simple, agent: null },
    });
    process.stderr.write(
      `  in ${cap.inTokens} · out ${cap.outTokens} @ ${cap.evalTps} tok/s · prefill ${cap.promptEvalTps} tok/s · needle=${needle.pass} · simple=${simple.pass}\n`,
    );
  }

  // Agent judge (dual): only models that already passed simple AND needle — a
  // wrong-context answer is already a fail, so don't spend the judge on it.
  let judgeFellBack = false;
  if (judge) {
    const eligible = results.filter((r) => r.check.simple.pass && r.needle.pass);
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
            r.check.overall_pass = r.check.simple.pass && r.needle.pass && v.pass;
          }
        }
      } else {
        judgeFellBack = true;
        process.stderr.write('[WARN] agent judge not available — simple + needle only\n');
      }
    }
  }

  const gpuAfter = await gpuInfo();
  const failed = results.filter((r) => !r.check.overall_pass).length;

  if (opts.output) {
    const report = {
      git_sha: sha,
      timestamp: new Date().toISOString(),
      config: { num_predict: numPredict, context, num_ctx: numCtx, needle: NEEDLE_CODE },
      gpu: { before: gpuBefore, after: gpuAfter },
      results,
    };
    writeFileSync(opts.output, JSON.stringify(report, null, 2));
    process.stderr.write(`Results written to ${opts.output}\n`);
  }

  const judgeMode = judge ? (judgeFellBack ? 'dual → simple+needle (judge unavailable)' : 'dual') : 'simple+needle';
  printSummary(sha, gpuBefore, context, judgeMode, results);
  return failed > 0 ? 1 : 0;
}

function printSummary(sha: string, gpu: GpuRow[], context: number, mode: string, results: CtxResult[]): void {
  const gpuName = gpu[0]?.name ?? 'unknown';
  const out: string[] = [];
  out.push('## Long-context Benchmark');
  out.push('');
  out.push(`**Commit:** \`${sha}\` | **GPU:** ${gpuName} | **Context target:** ${context} tok | **Judge:** ${mode}`);
  out.push('');
  out.push('| Model | Check | Needle | IN tok | OUT tok | Prefill tok/s | Decode tok/s | GPU% | VRAM (MiB) |');
  out.push('|---|---|:--:|--:|--:|--:|--:|--:|--:|');
  for (const r of results) {
    const check = r.check.overall_pass ? 'PASS' : 'FAIL';
    const needle = r.needle.pass ? '✓' : '✗';
    out.push(
      `| ${r.model} | ${check} | ${needle} | ${r.in_tokens} | ${r.out_tokens} | ${r.prompt_eval_tps} | ${r.eval_tps} | ${r.gpu_offload_pct}% | ${r.vram_used_mib.join(' / ') || 'n/a'} |`,
    );
  }
  const failures = results.filter((r) => !r.check.overall_pass);
  if (failures.length > 0) {
    out.push('', '### Failed checks', '');
    for (const r of failures) {
      const reason = !r.needle.pass ? r.needle.reason : r.check.agent?.reason ?? r.check.simple.reason;
      out.push(`- **${r.model}**: ${reason}`);
    }
  }
  process.stdout.write(out.join('\n') + '\n');
}
