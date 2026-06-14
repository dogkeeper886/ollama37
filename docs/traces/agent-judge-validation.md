# Evidence — ACP agent judge on real K80 inference (#234)

- When: 2026-06-14T23:51:22.629Z
- Model: `gemma3:4b` on the live stack (`http://localhost:11434`)
- Auth: keyless — `ANTHROPIC_API_KEY` unset, agent spawned via `~/.claude`
- Criteria (both cases): The response must correctly identify Paris as the capital of France. Any other city is incorrect and must FAIL.

## GOOD case
- Prompt: What is the capital of France? Answer in one word.
- Real model output: `Paris`
- Verdict: **PASS** — The generate step exited with code 0 and returned 'Paris', which correctly identifies the capital of France per the test criteria. No error responses present in stdout.

## WRONG case
- Prompt: What is the capital of Japan? Answer in one word.
- Real model output: `Tokyo` (correct for the prompt, but wrong for the criteria)
- Verdict: **FAIL** — evidence: Tokyo — The criteria requires the response to correctly identify Paris as the capital of France. The step returned 'Tokyo', which is the capital of Japan, not Paris. Any city other than Paris must FAIL.

## Result
✅ The agent judge passed the correct answer and failed the wrong one — it grades real output semantically, keyless.
