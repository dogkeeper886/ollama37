/**
 * models: does this build load each model, answer in language, and use the GPU as
 * it should (#542)? One model is an inference test; the runner's list is the fleet.
 *
 *   npx tsx models.ts [--models "a b c"] [--output results/models.json]
 *
 * Per model: pull if missing → generate('short-answer') → check + judge (by the
 * run) → weights resident, no die wasted → the model's own paths, if prompts.yaml
 * names any (gemma4:12b's image and audio, qwen3.8:27b's tool call) → unload. A
 * wrong answer passes; a reply that is not language, or a refusal, does not.
 */
import { converse, ensureModel, generate, load, unload } from './lib/ollama.js';
import { prompt, promptsFor } from './lib/prompts.js';
import { Menu } from './lib/mcp.js';
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
      await load(model, 'short-answer');
      vram = serverVram();
    }
    r.checks.resident = { pass: vram.totalMib > 0, reason: `${vram.totalMib} MiB across ${vram.dies} die(s)` };
    r.checks.dies = { pass: fits(vram.dies, vram.totalMib, model), reason: `${vram.totalMib} MiB across ${vram.dies} die(s)` };
    r.metrics.model = had;
    // Model-specific paths, while the weights are resident.
    for (const name of promptsFor(model)) {
      const p = prompt(name);
      if (p.tools.length) await converse(model, name, Menu.local(p.tools));
      else await generate(model, name);
    }
    await unload(model);
  }),
);
