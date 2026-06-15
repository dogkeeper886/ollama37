/**
 * Call ollama /api/generate and project an enriched, typed perf record
 * (port of cicd/scripts/lib/response_capture.sh).
 *
 * Sequence: warmup (1-token prime → loads model) → deterministic benchmark
 * generate (temperature 0, seed 42) → unload (keep_alive:0). `response` and
 * `thinking` are kept separate so a thinking model with an empty `response`
 * is still judged on its real output.
 */
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

async function generate(host: string, body: Record<string, unknown>): Promise<any> {
  const res = await fetch(`${host}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function captureResponse(
  host: string,
  model: string,
  prompt: string,
  numPredict = 128,
  numCtx = 2048
): Promise<CaptureResult> {
  // Warmup: load the model + prime caches (ignore failures).
  await generate(host, { model, prompt: 'Hi', stream: false, options: { num_predict: 1, num_ctx: numCtx } }).catch(() => {});

  // Benchmark call (deterministic).
  const raw = await generate(host, {
    model,
    prompt,
    stream: false,
    options: { temperature: 0, seed: 42, num_predict: numPredict, num_ctx: numCtx },
  });

  if (!raw || typeof raw !== 'object') {
    throw new Error(`captureResponse: no response from ${model} at ${host}`);
  }

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
  await generate(host, { model, keep_alive: 0 }).catch(() => {});

  return result;
}
