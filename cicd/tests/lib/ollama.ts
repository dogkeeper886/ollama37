/**
 * The only code that talks to the model under test (#542).
 *
 *   generate(model, 'short-answer')
 *     └─ prompts.yaml['short-answer'] → text, options, judge question
 *          └─ ollama generate → reply
 *               └─ check(reply)
 *                    └─ recorded for the judge, who runs after the GPU is back
 *                         └─ Result { reply, check, judge, pass }
 *
 * `generate` is the one export that returns model text, and it only runs inside
 * runTest, so every reply is checked and every reply is judged. The client is
 * private. The other calls here produce no model text: pull a missing model,
 * load one (an empty prompt), read its GPU share, unload one.
 */
import http from 'node:http';
import https from 'node:https';
import { Ollama, type Fetch, type GenerateResponse } from 'ollama';
import { prompt } from './prompts.js';
import { check, type Reply } from './check.js';
import { assertInRun, record, type Result } from './run.js';

export const HOST = process.env.OLLAMA_HOST ?? 'http://localhost:11434';

/**
 * Node's global fetch caps a request at ~300s, and with `stream: false` Ollama
 * sends no headers until generation ends, so a slow K80 generate would be killed
 * mid-flight. node:http has no cap; this deadline is the only one.
 */
function nodeFetch(timeoutMs: number): Fetch {
  return (input, init) =>
    new Promise<Response>((resolve, reject) => {
      const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
      const body = init?.body as string | undefined;
      const headers: Record<string, string> = {};
      new Headers(init?.headers).forEach((v, k) => (headers[k] = v));
      if (body !== undefined) headers['content-length'] = String(Buffer.byteLength(body));
      const req = (url.protocol === 'https:' ? https : http).request(url, { method: init?.method ?? 'GET', headers }, (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        res.on('error', reject);
        res.on('end', () => {
          const h = new Headers();
          for (const [k, v] of Object.entries(res.headers)) if (v !== undefined) h.set(k, Array.isArray(v) ? v.join(', ') : v);
          resolve(new Response(Buffer.concat(chunks), { status: res.statusCode ?? 502, statusText: res.statusMessage ?? '', headers: h }));
        });
      });
      req.setTimeout(timeoutMs, () => req.destroy(new Error(`no response in ${timeoutMs}ms`)));
      req.on('error', reject);
      if (body !== undefined) req.write(body);
      req.end();
    });
}

const client = new Ollama({ host: HOST, fetch: nodeFetch(1_200_000) });

/** Pull the model when the server does not have it. A locally created model is present without being in a registry. */
export async function ensureModel(model: string): Promise<'present' | 'pulled'> {
  try {
    await client.show({ model });
    return 'present';
  } catch {
    await client.pull({ model, stream: false });
    return 'pulled';
  }
}

/** Load the weights without generating (an empty prompt), with the options that decide the reservation. */
export async function load(model: string, options: Record<string, unknown> = {}): Promise<void> {
  await client.generate({ model, prompt: '', stream: false, options });
}

/** Percent of the loaded model resident in VRAM, per /api/ps; 0 when it is not loaded. */
export async function offload(model: string): Promise<number> {
  const { models } = await client.ps();
  const m = models.find((x) => x.name === model || x.model === model);
  return m && m.size ? Math.round((m.size_vram / m.size) * 100) : 0;
}

/** Release the weights. */
export async function unload(model: string): Promise<void> {
  await client.generate({ model, prompt: '', keep_alive: 0 }).catch(() => {});
}

/**
 * Back-to-back models on the K80 now and then crash a cold load ("llama runner
 * process has terminated"). Retry that one error, 5 tries 8s apart; anything else
 * is the model's.
 */
async function withLoadRetry<T>(f: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await f();
    } catch (e) {
      if (attempt === 5 || !String(e).includes('llama runner process has terminated')) throw e;
      await new Promise((r) => setTimeout(r, 8000));
    }
  }
}

export interface GenerateOptions {
  /** Size of a `long` prompt, in tokens. */
  tokens?: number;
  /** Extra Ollama options, over the prompt's own (e.g. num_ctx, num_batch). */
  options?: Record<string, unknown>;
}

const perSec = (count?: number, ns?: number) => (count && ns ? +(count / (ns / 1e9)).toFixed(2) : 0);

/** Send the named prompt to `model`, check the reply, and record it for the judge. */
export async function generate(model: string, name: string, opts: GenerateOptions = {}): Promise<Result> {
  assertInRun();
  const p = prompt(name, opts.tokens);
  const options = { ...p.options, ...(opts.options ?? {}) };
  const res: GenerateResponse = await withLoadRetry(() =>
    client.generate({ model, prompt: p.text, stream: false, options }),
  );
  const reply: Reply = {
    response: res.response ?? '',
    thinking: res.thinking ?? '',
    doneReason: res.done_reason ?? '',
    evalCount: res.eval_count ?? 0,
  };
  return record({
    model,
    prompt: name,
    judgeName: p.judgeName,
    judgePass: p.judgePass,
    judgeTemplate: p.judge,
    reply,
    check: check(reply, p.expect),
    metrics: {
      inTokens: res.prompt_eval_count ?? 0,
      outTokens: res.eval_count ?? 0,
      prefillTps: perSec(res.prompt_eval_count, res.prompt_eval_duration),
      decodeTps: perSec(res.eval_count, res.eval_duration),
    },
  });
}
