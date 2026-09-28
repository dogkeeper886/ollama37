/**
 * Deterministic output check (port of cicd/scripts/lib/simple_check.sh, plus the
 * two jq sentinels the models suite carries in TC-MODELS-003).
 *
 * Three ways to fail, all cheap and model-free: empty output, output with no
 * letters or digits in it (REPLY_NO_TEXT), and one short unit repeated to fill
 * the reply (REPLY_REPEAT). A failure here gates whether the agent judge is even
 * worth running — and the repeat case must never reach it, because a judge asked
 * to quote repeated text back loops on it.
 *
 * Thinking models (qwen3.x, qwen3-vl, deepseek-r1, gemma4) can put coherent
 * output entirely in `thinking` while `response` is empty when num_predict caps
 * generation early, so either field counts; the checks run on whichever is used.
 */
export interface ContentVerdict {
  pass: boolean;
  reason: string;
  source: 'response' | 'thinking' | 'none';
}

/**
 * True when `s` is its own first `u` characters repeated, for some unit length
 * 1..50. Short strings are exempt (a reply must be at least max(100, 5*u) long)
 * so a brief genuine answer is not read as a loop. Mirrors the jq in
 * cicd/tests/testcases/models/TC-MODELS-003.yml.
 */
function isRepeatedUnit(s: string): boolean {
  const n = s.length;
  for (let u = 1; u <= 50; u++) {
    if (n < Math.max(100, 5 * u)) continue;
    if (s.slice(0, u).repeat(Math.ceil(n / u) + 1).slice(0, n) === s) return true;
  }
  return false;
}

export function simpleContentCheck(response: string, thinking: string): ContentVerdict {
  let text: string;
  let source: 'response' | 'thinking';
  if (response.trim()) {
    text = response.trim();
    source = 'response';
  } else if (thinking.trim()) {
    text = thinking.trim();
    source = 'thinking';
  } else {
    return { pass: false, reason: 'both response and thinking are empty/whitespace', source: 'none' };
  }

  if (!/[\p{L}\p{N}]/u.test(text)) {
    return { pass: false, reason: 'no letters or digits in the reply', source };
  }
  if (isRepeatedUnit(text)) {
    return { pass: false, reason: 'one short unit repeated to fill the reply', source };
  }

  return {
    pass: true,
    reason: source === 'response' ? 'non-empty response' : 'non-empty thinking (response empty)',
    source,
  };
}
