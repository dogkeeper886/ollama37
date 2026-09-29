/**
 * The REST path itself: is the model there, does /api/generate answer, are the
 * weights in VRAM, does it unload?
 *
 * Replaces testcases/inference/TC-INFERENCE-001.yml and -002.yml. Those drove
 * `docker exec … ollama show/pull/list` and `curl` from bash and asserted on the
 * text that came back — `expectPatterns: ["response"]` passes on the word
 * "response" appearing anywhere in a thousand bytes of JSON. Here the client
 * parses it, so the assertion is on the field (#535).
 */
import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import { execa } from 'execa';
import { Ollama, type GenerateResponse } from 'ollama';

const HOST = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const MODEL = process.env.TEST_MODEL ?? 'gemma3:4b';
const ollama = new Ollama({ host: HOST });

/** MiB held on any die by this server's processes. */
async function ollamaVramMib(): Promise<number> {
  const { stdout } = await execa('nvidia-smi', [
    '--query-compute-apps=pid,used_memory,process_name',
    '--format=csv,noheader,nounits',
  ]);
  return stdout
    .split('\n')
    .map((l) => l.split(',').map((c) => c.trim()))
    .filter((c) => c.length === 3 && /ollama/.test(c[2]))
    .reduce((sum, c) => sum + Number(c[1]), 0);
}

describe(`inference: ${MODEL}`, () => {
  test('the model is available', async ({ annotate }) => {
    // Confirm presence first and consult the registry only when absent. A
    // locally-imported model (`ollama create`) is pulled even though it is not
    // in the registry, so a forced pull would 404 on it.
    try {
      await ollama.show({ model: MODEL });
      await annotate('already pulled');
    } catch {
      await annotate('absent — pulling from the registry');
      await ollama.pull({ model: MODEL, stream: false });
    }
    const listed = await ollama.list();
    expect(listed.models.map((m) => m.name)).toContain(MODEL);
  });

  describe('/api/generate', () => {
    let reply: GenerateResponse;

    beforeAll(async () => {
      reply = await ollama.generate({
        model: MODEL,
        prompt: 'What is 2+2? Answer with just the number.',
        stream: false,
      });
    });

    afterAll(async () => {
      await ollama.generate({ model: MODEL, prompt: '', keep_alive: 0 }).catch(() => {});
    });

    test('answers with a response', async ({ annotate }) => {
      await annotate(`done_reason=${reply.done_reason} eval_count=${reply.eval_count}`);
      expect(reply.done).toBe(true);
      expect(reply.response).toBeTruthy();
    });

    test('the weights are resident on the GPU', async ({ annotate }) => {
      const mib = await ollamaVramMib();
      await annotate(`${mib} MiB`);
      expect(mib).toBeGreaterThan(0);
    });
  });

  test('the model unloads', async () => {
    const unloaded = await ollama.generate({ model: MODEL, prompt: '', keep_alive: 0 });
    expect(unloaded.done_reason).toBe('unload');
  });
});
