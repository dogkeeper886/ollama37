/**
 * throughput: how fast does each model prefill and decode on this card, and is
 * what it decodes still language (#542)?
 *
 *   npx tsx throughput.ts [--models "a b"] [--context 8192] [--num-ctx N] [--num-batch N]
 *                         [--output results/throughput.json]
 *
 * Default: the speech-rewrite prompt (~870 tokens in, 100 out). --context N runs
 * the long-context prompt instead: filler to N tokens with one fact planted ~30%
 * in, which the reply must repeat. Per model: pull if missing → load with the
 * options that decide the VRAM reservation → generate → speed, GPU share and
 * per-die VRAM → unload. Every reply is checked and judged by the run.
 */
import { ensureModel, generate, load, offload, unload } from './lib/ollama.js';
import { arg, forEachModel, modelsArg, runTest } from './lib/run.js';
import { prompt } from './lib/prompts.js';
import { serverVram } from './lib/gpu.js';

const int = (name: string): number | undefined => {
  const v = arg(name);
  if (v === undefined) return undefined;
  if (!/^[1-9][0-9]*$/.test(v)) throw new Error(`--${name} must be a positive integer, got "${v}"`);
  return Number(v);
};

const context = int('context');
const numBatch = int('num-batch');
const name = context ? 'long-context' : 'speech-rewrite';
const numPredict = Number(prompt(name, context ?? 1).options.num_predict ?? 0);
// Long context: room for the prompt, the reply, and tokenizer variance between models.
const numCtx = context ? context + numPredict + 1024 : int('num-ctx');
const options = { ...(numCtx ? { num_ctx: numCtx } : {}), ...(numBatch ? { num_batch: numBatch } : {}) };

await runTest(context ? `throughput (long context ${context})` : 'throughput', () =>
  forEachModel(modelsArg('OLLAMA37_THROUGHPUT_MODELS'), async (model) => {
    await ensureModel(model);
    // The request that loads a model reserves its compute graph, so the options
    // that size it go on the load, not only on the timed request.
    await load(model, name, { tokens: context, options });
    const r = await generate(model, name, { tokens: context, options });
    const gpu = await offload(model);
    const vram = serverVram();
    r.metrics.gpuPct = gpu;
    r.metrics.vramMib = `${vram.totalMib}/${vram.dies}die`;
    if (numCtx) {
      // A prompt that filled the window was cut, so prefill was timed over a prompt nobody sent.
      r.checks.window = { pass: r.metrics.inTokens < numCtx, reason: `${r.metrics.inTokens} of ${numCtx} tokens` };
    }
    if (context) {
      // A long window spilled to CPU measures the CPU, not the card.
      r.checks.gpu = { pass: gpu === 100, reason: `${gpu}% on GPU` };
    }
    await unload(model);
  }),
);
