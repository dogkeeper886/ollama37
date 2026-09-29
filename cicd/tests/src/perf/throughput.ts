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
import { createHash } from 'node:crypto';
import { captureResponse, unloadModel } from './capture.js';
import { gpuInfo, gpuOffload, type GpuRow } from './gpu.js';
import { simpleContentCheck } from './content-check.js';
import { AgentJudge } from '../judge/index.js';
import { TestResult, Judgment } from '../types.js';

const execFileAsync = promisify(execFile);

/**
 * One request measures both numbers: the speech is the prefill (~900 tokens of real
 * prose), the edit is the decode. An open-ended question was the wrong task — asked to
 * explain something to a 10-year-old, a thinking model plans tone and analogies and
 * spends the whole budget reasoning, which left 12 of 22 models with an empty
 * `response` in run 36445237157. Editing a text given in the prompt leaves nothing to
 * plan.
 *
 * Lincoln's Second Inaugural, 1865: public domain, and about 700 words. Churchill's
 * "we shall fight on the beaches" was the first choice, but his estate holds literary
 * copyright until 2035 and this repository is public.
 *
 * Transcribed here rather than fetched, so the prompt is deterministic and the sweep
 * needs no network. Worth checking against an authoritative text once.
 */
const SPEECH = `Fellow countrymen: At this second appearing to take the oath of the presidential office, there is less occasion for an extended address than there was at the first. Then a statement, somewhat in detail, of a course to be pursued, seemed fitting and proper. Now, at the expiration of four years, during which public declarations have been constantly called forth on every point and phase of the great contest which still absorbs the attention and engrosses the energies of the nation, little that is new could be presented.

The progress of our arms, upon which all else chiefly depends, is as well known to the public as to myself; and it is, I trust, reasonably satisfactory and encouraging to all. With high hope for the future, no prediction in regard to it is ventured.

On the occasion corresponding to this four years ago, all thoughts were anxiously directed to an impending civil war. All dreaded it, all sought to avert it. While the inaugural address was being delivered from this place, devoted altogether to saving the Union without war, insurgent agents were in the city seeking to destroy it without war, seeking to dissolve the Union and divide effects by negotiation. Both parties deprecated war, but one of them would make war rather than let the nation survive, and the other would accept war rather than let it perish, and the war came.

One eighth of the whole population were colored slaves, not distributed generally over the Union, but localized in the southern part of it. These slaves constituted a peculiar and powerful interest. All knew that this interest was somehow the cause of the war. To strengthen, perpetuate, and extend this interest was the object for which the insurgents would rend the Union even by war, while the government claimed no right to do more than to restrict the territorial enlargement of it.

Neither party expected for the war the magnitude or the duration which it has already attained. Neither anticipated that the cause of the conflict might cease with, or even before, the conflict itself should cease. Each looked for an easier triumph, and a result less fundamental and astounding. Both read the same Bible and pray to the same God, and each invokes His aid against the other. It may seem strange that any men should dare to ask a just God's assistance in wringing their bread from the sweat of other men's faces, but let us judge not, that we be not judged. The prayers of both could not be answered. That of neither has been answered fully. The Almighty has His own purposes. "Woe unto the world because of offences; for it must needs be that offences come, but woe to that man by whom the offence cometh." If we shall suppose that American slavery is one of those offences which, in the providence of God, must needs come, but which, having continued through His appointed time, He now wills to remove, and that He gives to both North and South this terrible war as the woe due to those by whom the offence came, shall we discern therein any departure from those divine attributes which the believers in a living God always ascribe to Him?

Fondly do we hope, fervently do we pray, that this mighty scourge of war may speedily pass away. Yet, if God wills that it continue until all the wealth piled by the bondsman's two hundred and fifty years of unrequited toil shall be sunk, and until every drop of blood drawn with the lash shall be paid by another drawn with the sword, as was said three thousand years ago, so still it must be said "the judgments of the Lord are true and righteous altogether."

With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in, to bind up the nation's wounds, to care for him who shall have borne the battle and for his widow and his orphan, to do all which may achieve and cherish a just and lasting peace among ourselves and with all nations.`;

/**
 * The task. Drawn from the reviewing-phrasing discipline: cut verbal tics and filler,
 * put each sentence's main character first, make negative sentences affirmative, and
 * replace a phrase with the one word that means it.
 */
const PROMPT = `${SPEECH}

Rewrite the speech above in plainer English. Cut every verbal tic and filler word. Put each sentence's main character first. Turn each negative sentence affirmative. Replace any phrase that one exact word can replace. Output only the rewritten speech.`;
// Only what a script cannot decide. simpleContentCheck already rejects empty output,
// output with no letters or digits, and one short unit repeated to fill the reply, and
// the agent judge only ever sees results that passed it. Asking the judge for those
// again buys nothing and gives it extra grounds to fail on.
const JUDGE_CRITERIA =
  'Text no person could read as language fails: word salad, or a fragment that is ' +
  'not a reply. Judge the text itself -- the prompt is not in this payload. ' +
  'Judge `response`. When `response` is empty, judge `thinking` instead: a model that ' +
  'spent the budget reasoning still produced language. ' +
  'Generation stops at a token budget, so a truncated reply passes. ' +
  'A wrong answer passes. The benchmark measures speed.';


export interface ThroughputOptions {
  models: string[];
  numPredict: number;
  /** Undefined = let the model keep its own context window. */
  numCtx?: number;
  /** Micro-batch size (num_batch). Undefined = model/server default (512). */
  numBatch?: number;
  judge: boolean;
  host: string;
  output?: string;
}

export interface ModelResult {
  model: string;
  in_tokens: number;
  out_tokens: number;
  prompt_eval_tps: number;
  eval_tps: number;
  gpu_offload_pct: number;
  vram_used_mib: number[];
  done_reason: string;
  /** One request per model. The prompt is a ~950-token speech, long enough that
   *  prefill escapes the fixed per-request cost, so `prompt_eval_tps`/`in_tokens`
   *  and `eval_tps`/`out_tokens` all come from the same call -- as do the response
   *  and the verdict. Decode is therefore measured at that depth, not at a shallow
   *  context. */
  response_preview: string;
  /** Full captured text. `judge-throughput` runs in a later step, in a new process,
   *  and has no other source for it — judging the preview would grade 120 characters
   *  as if they were the whole answer. */
  response: string;
  thinking: string;
  /** The prompt filled --context, so prefill was timed over a prompt the caller did
   *  not send. Fails the row, as it does in context.ts. */
  truncated: boolean;
  check: { overall_pass: boolean; simple: ReturnType<typeof simpleContentCheck>; agent: Judgment | null };
}

/** The JSON report `--output` writes, and `judge-throughput` reads back. */
export interface ThroughputReport {
  git_sha: string;
  timestamp: string;
  /** `prompt` identifies the workload, not just its size. Decode moved from a
   *  31-token question to a ~870-token speech and num_predict from 400 to 100 in
   *  one commit; without this, two reports measuring different work are
   *  indistinguishable in their own metadata, and docs/reports/README.md tells the
   *  reader to diff snapshots to see what a build changed. */
  config: {
    num_predict: number;
    num_ctx: number | null;
    num_batch: number | null;
    prompt: { chars: number; sha256: string };
  };
  gpu: { before: GpuRow[]; after: GpuRow[] };
  results: ModelResult[];
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

/**
 * Strip control markers models leak into their own text. Seen in run 36445237157:
 * gemma4:31b opened `thinking` with `<|channel>thought`, deepseek-r1:8b embedded
 * `<think>`. A judge told to accept only language a person can read may fail those
 * as garbled, which would trade one wrong verdict for another. Stripped here in the
 * caller rather than in a parser: the fork's job is running models on the K80.
 */
function stripMarkers(s: string | undefined): string {
  return (s ?? '').replace(/<\|[^|>]*\|?>|<\/?think>/g, '').trim();
}

/** Build a synthetic TestResult so the AgentJudge can grade the captured output.
 *  The reply travels as fields, not as one joined string: 15 of 22 models in run
 *  36428282153 returned an empty `response` with every token in `thinking`, and a
 *  judge handed the two glued together grades reasoning text as if it were the
 *  answer. `done_reason` goes with them so the judge can see the budget ended it. */
function toTestResult(r: ModelResult): TestResult {
  const model = r.model;
  return {
    testCase: {
      id: model,
      name: `throughput:${model}`,
      suite: 'inference',
      priority: 1,
      timeout: 60000,
      dependencies: [],
      // The goal states the judgement; the criteria state only the tolerances, so
      // neither repeats the other. buildPrompt falls back to testCase.name when
      // goal is unset, which would put "throughput:<model>" in the prompt.
      // It must never demand a produced answer -- num_predict truncates every
      // reply, so such a goal fails them all before the criteria are read.
      goal: 'Judge whether the reply is meaningful language',
      steps: [{ name: 'generate', command: '(captured /api/generate response)' }],
      criteria: JUDGE_CRITERIA,
    },
    steps: [
      {
        name: 'generate',
        command: '(captured /api/generate response)',
        stdout: '',
        stderr: '',
        exitCode: 0,
        duration: 0,
        reply: {
          response: stripMarkers(r.response),
          thinking: stripMarkers(r.thinking),
          doneReason: r.done_reason,
          evalCount: r.out_tokens,
        },
      },
    ],
    totalDuration: 0,
    logs: '',
    logFile: '',
  };
}

/**
 * Agent judge (dual mode): one batch over a single reused session, only for models
 * that passed the simple check (an empty output is already a fail). Verdicts are
 * written back into each result's `check`.
 *
 * Returns true when dual was asked for but nothing could be graded — the caller
 * then reports "simple", never "dual" (STORY-010).
 */
export async function judgeThroughputResults(results: ModelResult[]): Promise<boolean> {
  const eligible = results.filter((r) => !r.truncated && r.check.simple.pass);
  if (eligible.length === 0) return false;

  // A report written before the full text was persisted carries only the 120-char
  // preview. Grading that would report "dual" on a fifth of a sentence.
  if (eligible.some((r) => typeof r.response !== 'string')) {
    process.stderr.write('[WARN] report has no captured response — rerun the benchmark to judge it\n');
    return true;
  }

  const agentJudge = new AgentJudge();
  if (!(await agentJudge.isAvailable())) {
    process.stderr.write('[WARN] agent judge not available — simple check only\n');
    return true;
  }

  const byModel = new Map(results.map((r) => [r.model, r]));
  const verdicts = await agentJudge.judgeResults(
    eligible.map((r) => toTestResult(r))
  );
  for (const v of verdicts) {
    const r = byModel.get(v.testId);
    if (r) {
      r.check.agent = v;
      r.check.overall_pass = !r.truncated && r.check.simple.pass && v.pass;
    }
  }
  return false;
}

export async function runThroughput(opts: ThroughputOptions): Promise<number> {
  const { host, models, numPredict, numCtx, numBatch, judge } = opts;
  const sha = await gitSha();
  const gpuBefore = await gpuInfo();

  const results: ModelResult[] = [];

  for (const model of models) {
    process.stderr.write(`--- ${model} ---\n`);
    await ensureModel(host, model);

    // One request, then one explicit unload after the GPU snapshot below -- the
    // snapshot has to read the model while it is still resident. The capture's own
    // failure path releases the weights; a throw from gpuOffload or gpuInfo between
    // the two would leave them resident into the next iteration, which no caller
    // has hit and nothing currently guards.
    let cap;
    try {
      // keepLoaded: the GPU snapshot below must read the model while it is resident.
      cap = await captureResponse(host, model, PROMPT, numPredict, numCtx, numBatch, true);
    } catch (e) {
      process.stderr.write(`  ERROR: ${e instanceof Error ? e.message : e}\n`);
      await unloadModel(host, model);
      results.push({
        model, in_tokens: 0, out_tokens: 0, prompt_eval_tps: 0, eval_tps: 0,
        gpu_offload_pct: 0, vram_used_mib: [], done_reason: 'error', response_preview: '',
        response: '', thinking: '',
        truncated: false,
        check: { overall_pass: false, simple: { pass: false, reason: 'capture failed', source: 'none' }, agent: null },
      });
      continue;
    }

    // context.ts:288 guards the same case. The old prefill prompt was sized "short
    // enough to fit the smallest context the sweep uses"; the speech is ~870 tokens
    // and test-report-sweep.yml passes --context from a ladder, so a small value
    // truncates the prompt and prefill is then timed over a prompt nobody sent.
    const truncated = Boolean(numCtx && cap.inTokens >= numCtx);

    const offload = await gpuOffload(host, model);
    const loaded = await gpuInfo();
    await unloadModel(host, model);
    // Stripped, because the judge reads stripped text (toTestResult): a reply of
    // only control markers has letters, so the raw text would pass this check and
    // then reach the judge as an empty payload. The report keeps the raw reply.
    const simple = simpleContentCheck(stripMarkers(cap.response), stripMarkers(cap.thinking));

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
      response: cap.response,
      thinking: cap.thinking,
      truncated,
      check: { overall_pass: !truncated && simple.pass, simple, agent: null },
    });
    process.stderr.write(
      `  prefill ${cap.promptEvalTps} tok/s @ ${cap.inTokens} tok · decode ${cap.evalTps} tok/s @ ${cap.outTokens} tok · offload ${offload}% · ${truncated ? 'TRUNCATED ' : ''}simple=${simple.pass}\n`
    );
  }

  const judgeFellBack = judge ? await judgeThroughputResults(results) : false;

  const gpuAfter = await gpuInfo();
  const failed = results.filter((r) => !r.check.overall_pass).length;

  // JSON report artifact
  if (opts.output) {
    const report: ThroughputReport = {
      git_sha: sha,
      timestamp: new Date().toISOString(),
      config: {
        num_predict: numPredict,
        num_ctx: numCtx ?? null,
        num_batch: numBatch ?? null,
        prompt: { chars: PROMPT.length, sha256: createHash('sha256').update(PROMPT).digest('hex').slice(0, 12) },
      },
      gpu: { before: gpuBefore, after: gpuAfter },
      results,
    };
    writeFileSync(opts.output, JSON.stringify(report, null, 2));
    process.stderr.write(`Results written to ${opts.output}\n`);
  }

  // Markdown summary on stdout (CI appends this to the step summary).
  printSummary(sha, gpuBefore, numCtx, judgeModeLabel(judge, judgeFellBack), results);
  return failed > 0 ? 1 : 0;
}

/** When dual was asked for but the judge couldn't run, say so — don't claim "dual"
 *  for a simple-only result (STORY-010). */
export function judgeModeLabel(judge: boolean, fellBack: boolean): string {
  if (!judge) return 'simple';
  return fellBack ? 'dual → simple (judge unavailable)' : 'dual';
}

export function printSummary(sha: string, gpu: GpuRow[], numCtx: number | null | undefined, mode: string, results: ModelResult[]): void {
  const gpuName = gpu[0]?.name ?? 'unknown';
  const gpuTotal = gpu[0]?.totalMib ?? '?';
  const out: string[] = [];
  out.push('## Throughput Benchmark');
  out.push('');
  out.push(`**Commit:** \`${sha}\` | **GPU:** ${gpu.length}x ${gpuName} | **VRAM:** ${gpuTotal} MiB each | **Context:** ${numCtx ?? 'model default'} | **Judge:** ${mode}`);
  out.push('');
  // Both numbers come from one request (see ModelResult), so each sits beside the
  // token count it was measured over: prefill over the prompt, decode over the reply.
  out.push('| Model | Check | Prefill tok/s | in tok | Decode tok/s | out tok | GPU% | VRAM used (MiB) |');
  out.push('|---|---|---|---|---|---|---|---|');
  for (const r of results) {
    const check = r.check.overall_pass ? 'PASS' : 'FAIL';
    out.push(`| ${r.model} | ${check} | ${r.prompt_eval_tps} | ${r.in_tokens} | ${r.eval_tps} | ${r.out_tokens} | ${r.gpu_offload_pct}% | ${r.vram_used_mib.join(' / ') || 'n/a'} |`);
  }
  const failures = results.filter((r) => !r.check.overall_pass);
  if (failures.length > 0) {
    out.push('');
    out.push('### Failed output checks');
    out.push('');
    for (const r of failures) {
      const reason = r.truncated
        ? `TRUNCATED — prompt (${r.in_tokens} tok) filled the context window; result invalid`
        : r.check.agent?.reason ?? r.check.simple.reason;
      out.push(`- **${r.model}**: ${reason}`);
    }
  }
  process.stdout.write(out.join('\n') + '\n');
}
