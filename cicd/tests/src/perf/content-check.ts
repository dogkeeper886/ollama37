/**
 * Deterministic non-empty-output check (port of cicd/scripts/lib/simple_check.sh).
 *
 * Returns pass:true if either the response or the thinking field is non-empty
 * after a whitespace trim. An empty model output is a hard fail regardless of
 * any semantic judge — this gates whether the agent judge is even worth running.
 *
 * Thinking models (qwen3.x, qwen3-vl, deepseek-r1, gemma4) can put coherent
 * output entirely in `thinking` while `response` is empty when num_predict caps
 * generation early, so either field counts.
 */
export interface ContentVerdict {
  pass: boolean;
  reason: string;
  source: 'response' | 'thinking' | 'none';
}

export function simpleContentCheck(response: string, thinking: string): ContentVerdict {
  if (response.trim()) {
    return { pass: true, reason: 'non-empty response', source: 'response' };
  }
  if (thinking.trim()) {
    return { pass: true, reason: 'non-empty thinking (response empty)', source: 'thinking' };
  }
  return { pass: false, reason: 'both response and thinking are empty/whitespace', source: 'none' };
}
