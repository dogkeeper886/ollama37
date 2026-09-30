/**
 * models: does this build load each model, answer in language, and use the GPU as
 * it should (#542)? One model is an inference test; the runner's list is the fleet.
 *
 *   npx tsx models.ts [--models "a b c"] [--output results/models.json]
 *
 * Per model: pull if missing → generate('short-answer') → check + judge (by the
 * run) → weights resident, no die wasted → unload. A wrong answer passes; a reply
 * that is not language does not.
 */
import { ensureModel, generate, load, unload } from './lib/ollama.js';
import { forEachModel, modelsArg, runTest } from './lib/run.js';
import { serverVram } from './lib/gpu.js';

/** A K80 die. A model may span dies; it may not waste one. */
const DIE_MIB = Number(process.env.OLLAMA37_DIE_MIB ?? 11441);

/**
 * Share of a die a model may fill before one more die is expected rather than
 * wasted; 1 unless listed. ornith:35b (~22.8 GB) is ~99.5% of two dies, too tight
 * for KV growth, so three is correct (TC-MODELS-019; run 36710119123).
 */
const HEADROOM: Record<string, number> = { 'ornith:35b': 0.9 };

const fits = (dies: number, mib: number, model: string) =>
  dies <= 1 || mib > (dies - 1) * DIE_MIB * (HEADROOM[model] ?? 1);

await runTest('models', () =>
  forEachModel(modelsArg('OLLAMA37_MODELS'), async (model) => {
    const had = await ensureModel(model);
    const r = await generate(model, 'short-answer');
    let vram = serverVram();
    // Placement is not deterministic: now and then a model spills onto one die more
    // than it needs. Reload once before calling it a regression.
    if (!fits(vram.dies, vram.totalMib, model)) {
      await unload(model);
      await load(model);
      vram = serverVram();
    }
    r.checks.resident = { pass: vram.totalMib > 0, reason: `${vram.totalMib} MiB across ${vram.dies} die(s)` };
    r.checks.dies = { pass: fits(vram.dies, vram.totalMib, model), reason: `${vram.totalMib} MiB across ${vram.dies} die(s)` };
    r.metrics.model = had;
    await unload(model);
  }),
);
