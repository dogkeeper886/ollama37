/**
 * throughput: how fast does each model prefill and decode on this card, and is
 * what it decodes still language (#542)?
 *
 *   npx tsx throughput.ts [--models "a b"] [--prompt NAME] [--num-ctx N] [--num-batch N]
 *                         [--output results/throughput.json]
 *
 * Default: the speech-rewrite prompt (~870 tokens in, 100 out). --prompt
 * farewell-summary runs a long context instead (~8k tokens in, 100 out), with the
 * window it needs set in prompts.yaml. Per model: pull if missing → load with the
 * options that decide the VRAM reservation → generate → speed, GPU share and
 * per-die VRAM → unload. Every reply is checked and judged by the run.
 */
import { ensureModel, generate, load, offload, unload } from './lib/ollama.js';
import { arg, forEachModel, modelsArg, runTest } from './lib/run.js';
import { prompt, promptNames } from './lib/prompts.js';
import { serverVram } from './lib/gpu.js';

const int = (name: string): number | undefined => {
  const v = arg(name);
  if (v === undefined) return undefined;
  if (!/^[1-9][0-9]*$/.test(v)) throw new Error(`--${name} must be a positive integer, got "${v}"`);
  return Number(v);
};

const name = arg('prompt') ?? 'speech-rewrite';
if (!promptNames().includes(name)) throw new Error(`--prompt must be one of ${promptNames().join(', ')}, got "${name}"`);
const numBatch = int('num-batch');
// A prompt that sets its own window is a long context; --num-ctx overrides it.
const ownCtx = Number(prompt(name).options.num_ctx ?? 0) || undefined;
const numCtx = int('num-ctx') ?? ownCtx;
const options = { ...(numCtx ? { num_ctx: numCtx } : {}), ...(numBatch ? { num_batch: numBatch } : {}) };

await runTest(name === 'speech-rewrite' ? 'throughput' : `throughput (${name})`, () =>
  forEachModel(modelsArg('OLLAMA37_THROUGHPUT_MODELS'), async (model) => {
    await ensureModel(model);
    // The request that loads a model reserves its compute graph, so the options
    // that size it go on the load, not only on the timed request.
    await load(model, name, { options });
    const r = await generate(model, name, { options });
    const gpu = await offload(model);
    const vram = serverVram();
    r.metrics.gpuPct = gpu;
    r.metrics.vramMib = `${vram.totalMib}/${vram.dies}die`;
    if (numCtx) {
      // A prompt that filled the window was cut, so prefill was timed over a prompt nobody sent.
      r.checks.window = { pass: r.metrics.inTokens < numCtx, reason: `${r.metrics.inTokens} of ${numCtx} tokens` };
    }
    if (ownCtx) {
      // A long window spilled to CPU measures the CPU, not the card.
      r.checks.gpu = { pass: gpu === 100, reason: `${gpu}% on GPU` };
    }
    await unload(model);
  }),
);
