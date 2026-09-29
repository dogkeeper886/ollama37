/**
 * Per-model smoke test: does this build load the model on a K80, answer in
 * language, keep its weights in VRAM, and give the dies back?
 *
 * Replaces testcases/models/TC-MODELS-*.yml — nineteen files that differed only
 * in a model name. The reply arrives here as a typed GenerateResponse rather
 * than as characters in a shell variable, so `response`, `thinking` and
 * `done_reason` are readable as fields and the content rules live in one place
 * (simpleContentCheck) instead of being written twice, once more in jq (#535).
 */
import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import { execa } from 'execa';
import { type GenerateResponse } from 'ollama';
import { simpleContentCheck } from '../src/perf/content-check.js';
import { ollamaClient } from '../src/ollama-client.js';

const HOST = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const CONTAINER = process.env.OLLAMA37_CONTAINER;
const ollama = ollamaClient(HOST);

/** A K80 die holds 11441 MiB. A model may span dies; it may not waste one. */
const DIE_MIB = 11441;

const PROMPT = 'What is 2+2? Answer in one short sentence.';

/** The runner names the models that fit its card; unset falls back to the K80 fleet. */
const MODELS = (process.env.OLLAMA37_MODELS?.trim()
  ? process.env.OLLAMA37_MODELS.trim().split(/[,\s]+/)
  : [
      'deepseek-r1:1.5b',
      'deepseek-r1:8b',
      'gemma3:270m',
      'gemma3:27b',
      'gemma3n:e2b',
      'gemma4:12b',
      'gemma4:31b',
      'gemma4:e2b',
      'gpt-oss:20b',
      'lfm2.5:8b',
      'lfm2.5-thinking:1.2b',
      'llama3.1:8b',
      'ministral-3:3b',
      'ornith:35b',
      'qwen3.5:9b',
      'qwen3.6:35b',
      'qwen3.8:27b',
      'qwen3-vl:2b',
      'qwen3-vl:30b',
    ]);

interface DieUsage {
  /** Dies this server holds any allocation on. */
  active: number;
  totalMib: number;
}

/**
 * VRAM this server's processes hold, per die. Attributed by process, not by
 * whole-GPU usage: another tenant's allocation is not this model's footprint.
 */
async function dieUsage(): Promise<DieUsage> {
  const { stdout } = await execa('nvidia-smi', [
    '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
    '--format=csv,noheader,nounits',
  ]);
  const rows = stdout
    .split('\n')
    .map((l) => l.split(',').map((c) => c.trim()))
    .filter((c) => c.length === 4 && /ollama/.test(c[3]))
    .map((c) => ({ uuid: c[0], usedMib: Number(c[2]) }));
  return {
    active: new Set(rows.map((r) => r.uuid)).size,
    totalMib: rows.reduce((s, r) => s + r.usedMib, 0),
  };
}

/** One die, or a footprint that genuinely needs the dies it is spread over. */
const fits = (u: DieUsage): boolean => u.active <= 1 || u.totalMib > (u.active - 1) * DIE_MIB;

const generate = (model: string): Promise<GenerateResponse> =>
  ollama.generate({
    model,
    prompt: PROMPT,
    stream: false,
    options: { temperature: 0, seed: 0, num_predict: 400 },
  });

describe.each(MODELS)('%s', (model) => {
  let reply: GenerateResponse;

  beforeAll(async () => {
    reply = await generate(model);
  });

  afterAll(async () => {
    await ollama.generate({ model, prompt: '', keep_alive: 0 }).catch(() => {});
  });

  test('the request completes', async ({ annotate }) => {
    await annotate(`done_reason=${reply.done_reason} eval_count=${reply.eval_count}`);
    expect(reply.done).toBe(true);
  });

  test('the reply is readable language', async ({ annotate }) => {
    const verdict = simpleContentCheck(reply.response ?? '', reply.thinking ?? '');
    await annotate(`source=${verdict.source}: ${verdict.reason}`);
    expect(verdict.pass, verdict.reason).toBe(true);
  });

  test('the weights are resident on the GPU', async ({ annotate }) => {
    const usage = await dieUsage();
    await annotate(`${usage.totalMib} MiB across ${usage.active} die(s)`);
    expect(usage.totalMib).toBeGreaterThan(0);
  });

  test('no more dies than the footprint needs', async ({ annotate }) => {
    // Placement is not deterministic: now and then a model spills a small
    // allocation onto one die more than it needs. Reload once before calling it
    // a regression; overshooting twice is one.
    let usage = await dieUsage();
    if (!fits(usage)) {
      await ollama.generate({ model, prompt: '', keep_alive: 0 }).catch(() => {});
      await generate(model);
      usage = await dieUsage();
      await annotate('reloaded once');
    }
    await annotate(`${usage.totalMib} MiB across ${usage.active} die(s)`);
    expect(fits(usage), `${usage.totalMib} MiB fits in ${usage.active - 1} dies but uses ${usage.active}`).toBe(true);
  });
});
