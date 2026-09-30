/**
 * #234 — Validate the keyless ACP agent judge against REAL K80 inference.
 *
 * Hits the live ollama37 stack (/api/generate) for two cases against the same
 * criteria, then judges the real model output with the agent judge:
 *   - GOOD : a correct answer  -> expect PASS
 *   - WRONG: a real-but-wrong answer for the criteria -> expect FAIL
 *
 * Proves the judge reads the model's actual output and grades it (not rubber-
 * stamping), keyless via ~/.claude. Writes evidence to docs/traces/.
 *
 * Run from cicd/tests:  npx tsx scripts/validate-agent-judge.ts
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AgentJudge } from '../src/judge/agent-judge.js';
import type { TestResult, Judgment } from '../src/types.js';

const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';
const MODEL = process.env.VALIDATE_MODEL || 'gemma3:4b';
const CRITERIA =
  'The response must correctly identify Paris as the capital of France. ' +
  'Any other city is incorrect and must FAIL.';

async function generate(prompt: string): Promise<string> {
  const res = await fetch(`${OLLAMA}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, prompt, stream: false }),
  });
  const json: any = await res.json();
  return (json.response ?? '').trim();
}

function wrap(id: string, prompt: string, response: string): TestResult {
  const command = `curl /api/generate -d '{"model":"${MODEL}","prompt":${JSON.stringify(prompt)}}'`;
  return {
    testCase: {
      id, name: id, suite: 'inference', priority: 1, timeout: 60000,
      dependencies: [], goal: 'Identify the capital of France',
      steps: [{ name: 'generate', command }],
      criteria: CRITERIA,
    },
    steps: [{ name: 'generate', command, stdout: response, stderr: '', exitCode: 0, duration: 0 }],
    totalDuration: 0, logs: '', logFile: '',
  };
}

const GOOD_PROMPT = 'What is the capital of France? Answer in one word.';
const WRONG_PROMPT = 'What is the capital of Japan? Answer in one word.';

console.error(`[validate] model=${MODEL} stack=${OLLAMA}`);
console.error(`[validate] ANTHROPIC_API_KEY set? ${!!process.env.ANTHROPIC_API_KEY} (expect false — keyless)`);

const goodAns = await generate(GOOD_PROMPT);
const wrongAns = await generate(WRONG_PROMPT);
console.error(`[validate] GOOD real answer:  ${JSON.stringify(goodAns)}`);
console.error(`[validate] WRONG real answer: ${JSON.stringify(wrongAns)}`);

const judge = new AgentJudge();
if (!(await judge.isAvailable())) {
  console.error('[validate] agent judge not available (~/.claude / CLAUDE_CODE_OAUTH_TOKEN?) — aborting');
  process.exit(2);
}

const verdicts: Judgment[] = await judge.judgeResults([
  wrap('VALIDATE-GOOD', GOOD_PROMPT, goodAns),
  wrap('VALIDATE-WRONG', WRONG_PROMPT, wrongAns),
]);

const good = verdicts.find((v) => v.testId === 'VALIDATE-GOOD')!;
const wrong = verdicts.find((v) => v.testId === 'VALIDATE-WRONG')!;
const ok = good.pass === true && wrong.pass === false;

console.log('\n=== VERDICTS ===');
console.log(`GOOD  (answer ${JSON.stringify(goodAns)}):  ${good.pass ? 'PASS' : 'FAIL'} — ${good.reason}`);
console.log(`WRONG (answer ${JSON.stringify(wrongAns)}): ${wrong.pass ? 'PASS' : 'FAIL'} — ${wrong.reason}`);
console.log(`\n=== RESULT: ${ok ? 'OK — good->PASS, wrong->FAIL' : 'UNEXPECTED'} ===`);

const stamp = new Date().toISOString();
const evidence = `# Evidence — ACP agent judge on real K80 inference (#234)

- When: ${stamp}
- Model: \`${MODEL}\` on the live stack (\`${OLLAMA}\`)
- Auth: keyless — \`ANTHROPIC_API_KEY\` unset, agent spawned via \`~/.claude\`
- Criteria (both cases): ${CRITERIA}

## GOOD case
- Prompt: ${GOOD_PROMPT}
- Real model output: \`${goodAns}\`
- Verdict: **${good.pass ? 'PASS' : 'FAIL'}** — ${good.reason}

## WRONG case
- Prompt: ${WRONG_PROMPT}
- Real model output: \`${wrongAns}\` (correct for the prompt, but wrong for the criteria)
- Verdict: **${wrong.pass ? 'PASS' : 'FAIL'}**${wrong.evidence ? ` — evidence: ${wrong.evidence}` : ''} — ${wrong.reason}

## Result
${ok ? '✅ The agent judge passed the correct answer and failed the wrong one — it grades real output semantically, keyless.' : '❌ Unexpected verdicts — see above.'}
`;

const traces = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'docs', 'traces');
mkdirSync(traces, { recursive: true });
const out = resolve(traces, 'agent-judge-validation.md');
writeFileSync(out, evidence);
console.log(`\n[validate] evidence written to ${out}`);
process.exit(ok ? 0 : 1);
