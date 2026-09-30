/**
 * The script check on a reply (#542). Used only by lib/ollama.ts.
 *
 * Decides what a script can, before the judge is asked anything: the reply has
 * text (letters or digits), is not one short unit tiled to its length, and holds
 * the string the prompt planted, if it planted one. It cannot tell language from
 * fluent word salad -- that is the judge's question, and every reply goes on to it.
 */
export interface Reply {
  /** The answer field. Empty when a thinking model spent its budget reasoning. */
  response: string;
  /** Reasoning, when the model emits it separately. */
  thinking: string;
  /** `stop`, `length` (the budget ended it), ... */
  doneReason: string;
  evalCount: number;
}

export interface Verdict {
  pass: boolean;
  reason: string;
}

/** The text a reader should see: the answer, else the reasoning when that is all there is. */
export const readable = (r: Reply): string => r.response.trim() || r.thinking.trim();

/** True when `s` is its first `u` characters repeated, for a unit of 1..50. Short replies are exempt. */
function isLoop(s: string): boolean {
  for (let u = 1; u <= 50; u++) {
    if (s.length < Math.max(100, 5 * u)) continue;
    if (s.slice(0, u).repeat(Math.ceil(s.length / u) + 1).slice(0, s.length) === s) return true;
  }
  return false;
}

export function check(r: Reply, expect?: string): Verdict {
  const text = readable(r);
  if (!text) return { pass: false, reason: 'no text in response or thinking' };
  if (!/[\p{L}\p{N}]/u.test(text)) return { pass: false, reason: 'no letters or digits' };
  if (isLoop(text)) return { pass: false, reason: 'one short unit repeated to fill the reply' };
  if (expect && !r.response.includes(expect) && !r.thinking.includes(expect)) {
    return { pass: false, reason: `expected "${expect}" is missing` };
  }
  return { pass: true, reason: expect ? `has text; contains "${expect}"` : 'has text' };
}
