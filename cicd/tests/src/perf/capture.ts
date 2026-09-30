/**
 * Call ollama /api/generate through the official `ollama` client and project an
 * enriched, typed perf record (port of cicd/scripts/lib/response_capture.sh).
 *
 * Sequence: warmup (1-token prime → loads model) → deterministic benchmark
 * generate (temperature 0, seed 42) → unload (keep_alive:0). `response` and
 * `thinking` are kept separate so a thinking model with an empty `response`
 * is still judged on its real output.
 */
import { type GenerateResponse } from 'ollama';
import { ollamaClient } from '../ollama-client.js';

export interface CaptureResult {
  model: string;
  inTokens: number;
  outTokens: number;
  promptEvalTps: number;
  evalTps: number;
  totalDurationS: number;
  loadDurationS: number;
  doneReason: string;
  response: string;
  thinking: string;
}

/** Round to 2 decimals, matching the bash `* 100 | round / 100`. */
function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** tokens / (duration_ns / 1e9), or 0 when duration is non-positive. */
function tps(count: number, durationNs: number): number {
  return durationNs > 0 ? round2(count / (durationNs / 1e9)) : 0;
}

/** Release a model's VRAM. Safe when nothing is loaded; failures are ignored. */
export async function unloadModel(host: string, model: string): Promise<void> {
  await ollamaClient(host).generate({ model, prompt: '', keep_alive: 0 }).catch(() => {});
}

export async function captureResponse(
  host: string,
  model: string,
  prompt: string,
  numPredict = 400,
  numCtx?: number,
  numBatch?: number,
  /** Leave the model resident on return, for a caller about to make a second
   *  request against the same weights. That caller owns the unload. Default
   *  false, so every existing caller keeps today's load-run-unload behaviour. */
  keepLoaded = false
): Promise<CaptureResult> {
  const ollama = ollamaClient(host);

  // num_batch must be set on the warmup too — Ollama reserves the compute graph
  // (the Q·Kᵀ score buffer) at load time, so the batch that decides VRAM is the
  // one on the request that first loads the model. Omit entirely when unset so
  // default behavior is unchanged.
  const batchOpt = numBatch ? { num_batch: numBatch } : {};
  // Same for num_ctx: omitted unless asked for, so the model keeps the window it
  // chose rather than one this harness asserted for it.
  const ctxOpt = numCtx ? { num_ctx: numCtx } : {};

  // Warmup: load the model + prime caches (ignore failures).
  await ollama
    .generate({ model, prompt: 'Hi', stream: false, options: { num_predict: 1, ...ctxOpt, ...batchOpt } })
    .catch(() => {});

  // Benchmark call (deterministic). The client throws on an API error, so an
  // unknown or unloadable model surfaces as a rejection rather than an
  // all-zeros record; name the model and host, which the client's message does not.
  let raw: GenerateResponse;
  try {
    raw = await ollama.generate({
      model,
      prompt,
      stream: false,
      options: { temperature: 0, seed: 42, num_predict: numPredict, ...ctxOpt, ...batchOpt },
    });
  } catch (err) {
    throw new Error(`captureResponse: ${model} at ${host} — ${err instanceof Error ? err.message : String(err)}`);
  }

  // Every field of api.Metrics is `omitempty` (api/types.go:383): a fully cached
  // prompt or a reply with no tokens omits the count rather than sending 0, while
  // the client types it as a required number. Guard each one or the record goes NaN.
  const result: CaptureResult = {
    model,
    inTokens: raw.prompt_eval_count ?? 0,
    outTokens: raw.eval_count ?? 0,
    promptEvalTps: tps(raw.prompt_eval_count ?? 0, raw.prompt_eval_duration ?? 0),
    evalTps: tps(raw.eval_count ?? 0, raw.eval_duration ?? 0),
    totalDurationS: round2((raw.total_duration ?? 0) / 1e9),
    loadDurationS: round2((raw.load_duration ?? 0) / 1e9),
    doneReason: raw.done_reason ?? '',
    response: raw.response ?? '',
    thinking: raw.thinking ?? '',
  };

  // Unload so the next caller starts clean (ignore failures).
  if (!keepLoaded) await unloadModel(host, model);

  return result;
}
