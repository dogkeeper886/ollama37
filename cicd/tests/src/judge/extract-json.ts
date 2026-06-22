/**
 * Extract a JSON object from an agent/verifier reply.
 *
 * Scans from the first '{' tracking string state to find the matching close. The
 * ACP agent sometimes ends a turn one token short of its trailing '}' (or the final
 * streamed chunk isn't captured), so a complete-but-unclosed object would otherwise
 * be discarded — flipping a genuine verdict to a "No JSON" FAIL. Tolerate that by
 * appending the missing closers, but only when the truncation is OUTSIDE a string
 * (a value cut mid-string is genuinely unrecoverable → null).
 *
 * Shared by agent-judge.ts and verifier-judge.ts so the recovery can't drift between
 * them — it did once: the fix (#327) landed in verifier-judge.ts while agent-judge.ts
 * kept the old indexOf/lastIndexOf version and still FAILed genuine verdicts on
 * truncated responses.
 */
export function extractJson(text: string): string | null {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
    } else if (c === '"') inStr = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return text.substring(start, i + 1);
  }
  return depth > 0 && !inStr ? text.substring(start) + '}'.repeat(depth) : null;
}
