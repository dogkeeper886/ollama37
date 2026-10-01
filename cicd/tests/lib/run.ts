/**
 * A test run, start to finish (#542).
 *
 *   runTest('models', body)
 *     ├─ yield GPU          cicd/scripts/gpu-yield.sh
 *     ├─ body: generate(model, name) → reply → check → recorded
 *     ├─ restore GPU        cicd/scripts/gpu-restore.sh
 *     ├─ judge every recorded reply
 *     └─ report + exit      "N models, N judged, K failed"
 *
 * The judge runs after the GPU is back because on a shared host its server is the
 * one the test borrowed the card from. A test cannot skip it: generate() records
 * into the run, and runTest judges every record before it reports. The run fails
 * on a failed check, a "no", an abstain, a model that errored, or a run that
 * recorded no reply at all -- "0 failed" means every reply was checked and judged.
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readable, type Reply, type Verdict } from './check.js';
import { judge, closeJudge, type Judgment } from './judge.js';

export interface Result {
  model: string;
  prompt: string;
  judgeName: string;
  /** The judge's answer that passes the reply (prompts.yaml). */
  judgePass: 'yes' | 'no';
  reply: Reply;
  /** `content` is the script check generate() ran; a test may add its own (never remove). */
  checks: Record<string, Verdict>;
  metrics: { inTokens: number; outTokens: number; prefillTps: number; decodeTps: number } & Record<string, unknown>;
  /** What the reply must rest on, for a grounded judge: the tool result the model was given. */
  groundedOn?: string;
  /** `not-judged`: the script check already failed it, so it was never sent. Never passes. */
  judge: Judgment | { verdict: 'pending' | 'not-judged'; reason: string };
  pass: boolean;
}

type Recorded = Omit<Result, 'checks' | 'judge' | 'pass'> & { judgeTemplate: string; check: Verdict };

interface Run { test: string; results: Result[]; templates: Map<Result, string>; errors: { model: string; error: string }[] }
let current: Run | undefined;

/** Called by generate() before it asks the model: outside a run no one would judge the reply. */
export function assertInRun(): Run {
  if (!current) throw new Error('generate() only runs inside runTest(): outside it no reply would be judged');
  return current;
}

/** Called by generate() with the reply. */
export function record(r: Recorded): Result {
  const run = assertInRun();
  const { judgeTemplate, check, ...rest } = r;
  const result: Result = { ...rest, checks: { content: check }, judge: { verdict: 'pending', reason: 'not judged yet' }, pass: false };
  run.results.push(result);
  run.templates.set(result, judgeTemplate);
  return result;
}

/** Run `each` for every model, recording a model that throws as a failure rather than ending the run. */
export async function forEachModel(models: string[], each: (model: string) => Promise<void>): Promise<void> {
  for (const model of models) {
    process.stderr.write(`--- ${model} ---\n`);
    try {
      await each(model);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      process.stderr.write(`  ERROR: ${error}\n`);
      current!.errors.push({ model, error });
    }
  }
}

const SCRIPTS = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'scripts');
function shell(script: string): void {
  const r = spawnSync('bash', [resolve(SCRIPTS, script)], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`${script} exited ${r.status}`);
}

/** Models from --models "a b c" (or comma-separated); the fallback is the runner's list. */
export function modelsArg(fallbackEnv: string): string[] {
  const i = process.argv.indexOf('--models');
  const raw = i > 0 ? process.argv[i + 1] : process.env[fallbackEnv];
  const models = (raw ?? '').split(/[,\s]+/).filter(Boolean);
  if (models.length === 0) throw new Error(`no models: pass --models "…" or set ${fallbackEnv}`);
  return models;
}

/** The value after --name on the command line. */
export function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : undefined;
}

export async function runTest(test: string, body: () => Promise<void>): Promise<never> {
  current = { test, results: [], templates: new Map(), errors: [] };
  const run = current;
  let bodyError: string | undefined;

  shell('gpu-yield.sh');
  try {
    await body();
  } catch (e) {
    bodyError = e instanceof Error ? e.message : String(e);
    process.stderr.write(`RUN ERROR: ${bodyError}\n`);
  } finally {
    shell('gpu-restore.sh');
  }

  // Judge every reply the run recorded. A reply the script check failed is not
  // sent -- it fails anyway, and a judge asked to quote a loop loops with it.
  for (const r of run.results) {
    if (!r.checks.content.pass) {
      r.judge = { verdict: 'not-judged', reason: `not judged: ${r.checks.content.reason}` };
    } else if (r.judgeName === 'grounded' && !r.groundedOn) {
      r.judge = { verdict: 'abstain', reason: 'grounded judge has no tool result to check against' };
    } else {
      process.stderr.write(`  [judge] ${r.model} (${r.prompt})...\n`);
      r.judge = await judge(run.templates.get(r)!, readable(r.reply), r.groundedOn);
    }
    r.pass = Object.values(r.checks).every((c) => c.pass) && r.judge.verdict === r.judgePass;
  }
  closeJudge();

  const failed = run.results.filter((r) => !r.pass).length + run.errors.length;
  const judged = run.results.filter((r) => r.judge.verdict === 'yes' || r.judge.verdict === 'no').length;
  const abstained = run.results.filter((r) => r.judge.verdict === 'abstain').length;
  const notJudged = run.results.filter((r) => r.judge.verdict === 'not-judged').length;
  const models = new Set([...run.results.map((r) => r.model), ...run.errors.map((e) => e.model)]).size;
  const empty = run.results.length === 0;

  const lines = [
    `## ${test}`,
    '',
    `**${models} model(s), ${run.results.length} reply(s), ${judged} judged, ${abstained} abstained, ${notJudged} failed the check, ${failed} failed**` +
      (empty ? ' — no reply was recorded, so nothing was tested' : ''),
    '',
    '| Model | Prompt | Checks | Judge | Pass | Prefill tok/s | Decode tok/s | In / out tok | More | Reply |',
    '|---|---|---|---|---|---|---|---|---|---|',
    ...run.results.map((r) => {
      const checks = Object.entries(r.checks).map(([k, v]) => `${k}: ${v.pass ? 'ok' : v.reason}`).join('; ');
      const text = readable(r.reply).replace(/\s+/g, ' ').replace(/\|/g, '\\|').slice(0, 80);
      const { inTokens, outTokens, prefillTps, decodeTps, ...more } = r.metrics;
      const extra = Object.entries(more).map(([k, v]) => `${k}=${v}`).join(' ');
      return `| ${r.model} | ${r.prompt} | ${checks} | ${r.judge.reason} | ${r.pass ? 'PASS' : 'FAIL'} | ${prefillTps} | ${decodeTps} | ${inTokens} / ${outTokens} | ${extra} | ${text} |`;
    }),
    ...run.errors.map((e) => `| ${e.model} | — | error: ${e.error.replace(/\|/g, '\\|').slice(0, 120)} | — | FAIL | — | — | — | — | — |`),
    ...(bodyError ? ['', `Run error: ${bodyError}`] : []),
    '',
  ];
  const md = lines.join('\n');
  process.stdout.write(md + '\n');
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, md + '\n');

  const out = arg('output');
  if (out) {
    mkdirSync(dirname(resolve(out)), { recursive: true });
    writeFileSync(out, JSON.stringify({ test, models, judged, abstained, notJudged, failed, results: run.results, errors: run.errors, bodyError }, null, 2));
  }
  process.exit(failed > 0 || empty || bodyError ? 1 : 0);
}
