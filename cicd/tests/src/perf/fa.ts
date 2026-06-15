/**
 * K80 flash-attention regression + benchmark (port of cicd/scripts/test-fa-k80.sh).
 *
 * Boots a fresh ollama37 container per configuration (FA off/on × KV cache type),
 * runs a deterministic inference, captures VRAM + KV cache size + tok/s + logs,
 * then judges each response (simple content check + the keyless AgentJudge). The
 * judge is the point: an FA bug on K80 shows up as non-empty-but-garbled output,
 * which only a semantic judge catches.
 *
 * Container lifecycle of the *production* ollama37 (stop/start) is the caller's
 * job (the workflow), exactly as with the bash original. This needs exclusive use
 * of port 11434.
 *
 * Emits the benchmark + judge markdown tables on stdout (→ CI step summary) and,
 * with --output, an aggregate JSON report. Exit code is non-zero if any inference
 * fails or any judge verdict is FAIL.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFileSync, mkdirSync } from 'node:fs';
import { gpuInfo } from './gpu.js';
import { simpleContentCheck } from './content-check.js';
import { AgentJudge } from '../judge/index.js';
import { TestResult, Judgment } from '../types.js';

const execFileAsync = promisify(execFile);
const HOST = 'http://localhost:11434';
const IMAGE = 'ollama37:latest';
const ARTIFACT_DIR = '/tmp/fa-test';

const JUDGE_CRITERIA =
  'The response must be a coherent, on-topic answer to the prompt in the right language. ' +
  'Reject empty, garbled, repetitive nonsense, off-topic, or error-message output.';

export interface FaOptions {
  model: string;
  prompt: string;
  numPredict: number;
  numCtx: number;
  benchmark: boolean;
  kvCacheType: string;
  baselineCompare: boolean;
  output?: string;
}

interface ConfigSpec { label: string; enableFa: boolean; kvType: string; }

interface ConfigMetrics {
  config: string;
  enable_fa: boolean;
  kv_cache_type: string;
  num_ctx: number;
  num_predict: number;
  eval_count: number;
  tokens_per_sec: number;
  vram_used_mib: number;
  kv_cache_mib: string;
  response: string;
  thinking: string;
  ok: boolean;
  check?: { simple: ReturnType<typeof simpleContentCheck>; agent: Judgment | null; pass: boolean };
}

async function docker(args: string[]): Promise<string> {
  const { stdout } = await execFileAsync('docker', args, { maxBuffer: 64 * 1024 * 1024 });
  return stdout;
}

async function waitForApi(timeoutS: number): Promise<boolean> {
  for (let i = 0; i < timeoutS; i++) {
    try {
      const r = await fetch(`${HOST}/`);
      if (r.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((res) => setTimeout(res, 1000));
  }
  return false;
}

async function waitVramRelease(thresholdMib: number, timeoutS: number): Promise<void> {
  for (let i = 0; i < timeoutS; i++) {
    const gpus = await gpuInfo();
    if ((gpus[0]?.usedMib ?? 0) < thresholdMib) return;
    await new Promise((res) => setTimeout(res, 1000));
  }
}

/** Boot a container with the given FA/KV settings, run one deterministic
 *  inference, capture metrics + logs, then tear the container down. */
async function runInference(spec: ConfigSpec, opts: FaOptions): Promise<ConfigMetrics> {
  const { label, enableFa, kvType } = spec;
  const cname = `ollama37-fa-test-${label}`;
  const outDir = `${ARTIFACT_DIR}/${label}`;
  mkdirSync(outDir, { recursive: true });

  await docker(['rm', '-f', cname]).catch(() => {});

  const env = [
    '-e', `OLLAMA_FLASH_ATTENTION=${enableFa ? '1' : '0'}`,
    '-e', 'OLLAMA_DEBUG=1',
    '-e', 'NVIDIA_VISIBLE_DEVICES=all',
    '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
    '-e', 'OLLAMA_HOST=0.0.0.0:11434',
  ];
  if (kvType) env.push('-e', `OLLAMA_KV_CACHE_TYPE=${kvType}`);

  await docker([
    'run', '-d', '--rm', '--name', cname, '--runtime', 'nvidia', '--gpus', 'all',
    ...env, '-v', 'ollama-data:/root/.ollama', '-p', '11434:11434', IMAGE,
  ]);

  const fail = (reason: string): ConfigMetrics => ({
    config: label, enable_fa: enableFa, kv_cache_type: kvType || 'f16',
    num_ctx: opts.numCtx, num_predict: opts.numPredict, eval_count: 0,
    tokens_per_sec: 0, vram_used_mib: 0, kv_cache_mib: 'unknown',
    response: '', thinking: '', ok: false,
    check: { simple: { pass: false, reason, source: 'none' }, agent: null, pass: false },
  });

  try {
    if (!(await waitForApi(60))) {
      await docker(['logs', cname]).then((l) => writeFileSync(`${outDir}/container-logs.log`, l)).catch(() => {});
      return fail('API never came up');
    }

    const res = await fetch(`${HOST}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: opts.model, prompt: opts.prompt, stream: false,
        options: { num_predict: opts.numPredict, num_ctx: opts.numCtx, temperature: 0, seed: 42 },
      }),
    });
    const raw: any = await res.json();

    const vram = (await gpuInfo())[0]?.usedMib ?? 0;
    const logs = await docker(['logs', cname]).catch(() => '');
    writeFileSync(`${outDir}/container-logs.log`, logs);
    writeFileSync(`${outDir}/response.json`, JSON.stringify(raw, null, 2));

    if (!raw || typeof raw.response !== 'string') {
      return fail(`inference returned no response field${raw?.error ? `: ${raw.error}` : ''}`);
    }

    const kvMatch = logs.match(/"kv cache" device=\S+ size="([0-9.]+) MiB"/);
    const evalCount = raw.eval_count ?? 0;
    const evalDur = raw.eval_duration ?? 0;

    return {
      config: label, enable_fa: enableFa, kv_cache_type: kvType || 'f16',
      num_ctx: opts.numCtx, num_predict: opts.numPredict,
      eval_count: evalCount,
      tokens_per_sec: evalDur > 0 ? Math.round((evalCount / (evalDur / 1e9)) * 100) / 100 : 0,
      vram_used_mib: vram,
      kv_cache_mib: kvMatch ? kvMatch[1] : 'unknown',
      response: raw.response ?? '',
      thinking: raw.thinking ?? '',
      ok: true,
    };
  } catch (e) {
    // A crash mid-generation (FA bug, OOM, CUDA error → dropped connection) is
    // a FAIL for this config, not a reason to abort the whole sweep.
    return fail(`inference error: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    await docker(['stop', '--time', '30', cname]).catch(() => {});
    await waitVramRelease(1024, 30);
  }
}

function toTestResult(label: string, response: string, thinking: string, criteria: string): TestResult {
  const output = [thinking, response].filter((s) => s.trim()).join('\n\n');
  return {
    testCase: {
      id: label, name: `fa:${label}`, suite: 'inference', priority: 1, timeout: 60000,
      dependencies: [], goal: 'Produce coherent output under this FA config', steps: [{ name: 'generate', command: '(captured)' }],
      criteria,
    },
    steps: [{ name: 'generate', command: '(captured)', stdout: output, stderr: '', exitCode: 0, duration: 0 }],
    totalDuration: 0, logs: '', logFile: '',
  };
}

export async function runFa(opts: FaOptions): Promise<number> {
  mkdirSync(ARTIFACT_DIR, { recursive: true });

  const configs: ConfigSpec[] = opts.benchmark
    ? [
        { label: 'off-f16', enableFa: false, kvType: '' },
        { label: 'on-f16', enableFa: true, kvType: '' },
        { label: 'on-q8_0', enableFa: true, kvType: 'q8_0' },
      ]
    : [
        ...(opts.baselineCompare ? [{ label: 'baseline', enableFa: false, kvType: opts.kvCacheType }] : []),
        { label: 'fa', enableFa: true, kvType: opts.kvCacheType },
      ];

  const metrics: ConfigMetrics[] = [];
  for (const spec of configs) {
    process.stderr.write(`\n=== Run: ${spec.label} (fa=${spec.enableFa}, kv=${spec.kvType || 'f16'}) ===\n`);
    metrics.push(await runInference(spec, opts));
  }

  // Judge each successful config (simple → agent), one batched agent session.
  const judgeable = metrics.filter((m) => m.ok);
  for (const m of metrics) {
    if (!m.ok) continue;
    m.check = { simple: simpleContentCheck(m.response, m.thinking), agent: null, pass: false };
    m.check.pass = m.check.simple.pass; // updated by the agent below
  }
  const eligible = judgeable.filter((m) => m.check!.simple.pass);
  if (eligible.length > 0) {
    const agentJudge = new AgentJudge();
    if (await agentJudge.isAvailable()) {
      const byLabel = new Map(metrics.map((m) => [m.config, m]));
      const verdicts = await agentJudge.judgeResults(
        eligible.map((m) => toTestResult(m.config, m.response, m.thinking, JUDGE_CRITERIA))
      );
      for (const v of verdicts) {
        const m = byLabel.get(v.testId);
        if (m && m.check) {
          m.check.agent = v;
          m.check.pass = m.check.simple.pass && v.pass;
        }
      }
    } else {
      process.stderr.write('[WARN] agent judge not available — simple check only\n');
    }
  }

  const failed = metrics.filter((m) => !m.ok || (m.check && !m.check.pass)).length;

  // Single-mode --baseline-compare: does enabling FA change the output? Exact
  // match means FA is numerically identical to the f16 baseline.
  let comparison: { match: boolean } | undefined;
  if (!opts.benchmark && opts.baselineCompare) {
    const base = metrics.find((m) => m.config === 'baseline');
    const fa = metrics.find((m) => m.config === 'fa');
    if (base?.ok && fa?.ok) {
      comparison = { match: base.response === fa.response };
      process.stderr.write(comparison.match ? '::notice::baseline and FA outputs match exactly\n' : '::warning::baseline and FA outputs differ\n');
    }
  }

  if (opts.output) {
    writeFileSync(
      opts.output,
      JSON.stringify(
        { mode: opts.benchmark ? 'benchmark' : 'single', model: opts.model, num_ctx: opts.numCtx, num_predict: opts.numPredict, configs: metrics, ...(comparison ? { comparison } : {}) },
        null,
        2
      )
    );
  }

  printTables(opts, metrics, comparison);
  return failed > 0 ? 1 : 0;
}

function printTables(opts: FaOptions, metrics: ConfigMetrics[], comparison?: { match: boolean }): void {
  const out: string[] = [];
  if (opts.benchmark) {
    out.push('## K80 FA benchmark');
    out.push('');
    out.push(`**Model**: ${opts.model} · **num_ctx**: ${opts.numCtx} · **num_predict**: ${opts.numPredict}`);
    out.push('');
    out.push('| Config | KV cache (MiB) | Total VRAM (MiB) | tokens/sec |');
    out.push('|---|---|---|---|');
    for (const m of metrics) {
      out.push(`| ${m.config} | ${m.kv_cache_mib} | ${m.vram_used_mib} | ${m.tokens_per_sec} |`);
    }
    out.push('');
  }
  out.push('## K80 FA judge verdicts');
  out.push('');
  out.push('| Config | Verdict | Reason |');
  out.push('|---|---|---|');
  for (const m of metrics) {
    const pass = m.ok && m.check?.pass;
    const reason = (m.check?.agent?.reason ?? m.check?.simple.reason ?? 'no judgment').replace(/[\n|]/g, ' ').slice(0, 200);
    out.push(`| ${m.config} | ${pass ? 'PASS' : 'FAIL'} | ${reason} |`);
  }
  if (comparison) {
    out.push('');
    out.push(`**Baseline vs FA output:** ${comparison.match ? 'match exactly' : 'differ'}`);
  }
  process.stdout.write(out.join('\n') + '\n');
}
