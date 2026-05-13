/**
 * LLM Judge - Semantic evaluation of an LLM's response against test criteria.
 *
 * Scope: the judge decides whether the model's response (step stdout) meets the
 * criteria. It does NOT analyze ollama container logs, GPU detection state, or
 * CUDA error patterns - those are SimpleJudge's domain (expect/reject patterns
 * on logs). Keeping the two judges orthogonal stops them from second-guessing
 * each other.
 *
 * Sends a structured JSON prompt and uses Ollama JSON mode for reliable parsing.
 */

import axios from 'axios';
import { TestResult, Judgment } from '../types.js';

export class LLMJudge {
  private ollamaUrl: string;
  private model: string;

  constructor(ollamaUrl: string = 'http://localhost:11435', model: string = 'gemma3:12b-judge') {
    this.ollamaUrl = ollamaUrl;
    this.model = model;
  }

  /**
   * Check if the LLM judge is available.
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.ollamaUrl}/api/tags`, {
        timeout: 5000,
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Truncate a string to a maximum length.
   */
  private truncate(text: string, limit: number): string {
    if (text.length <= limit) return text;
    return text.substring(0, limit) + '... (truncated)';
  }

  /**
   * Build structured JSON prompt for LLM evaluation of a single test.
   *
   * The prompt carries the test's criteria plus each step's stdout/stderr
   * (where the model's response lives for inference-style tests). Container
   * logs and infra-error rules are intentionally absent - SimpleJudge handles
   * those via pattern matches.
   */
  private buildPrompt(result: TestResult): string {
    const r = result;
    const stdoutLimit = 1000;
    const stderrLimit = 500;

    const steps = r.steps.map((step, j) => {
      const stepDef = r.testCase.steps[j];
      return {
        name: step.name,
        command: step.command.trim(),
        exit_code: step.exitCode,
        duration_ms: step.duration,
        timeout_ms: stepDef?.timeout || r.testCase.timeout,
        stdout: this.truncate(step.stdout, stdoutLimit),
        stderr: this.truncate(step.stderr, stderrLimit),
      };
    });

    const promptData = {
      role: 'You evaluate an LLM-generated response against a set of test criteria. Look only at the response (in step stdout) and decide whether it meets the criteria. Do not infer pass/fail from environment, GPU state, or container logs - those are checked separately.',
      rules: [
        'Judge only the response text against the criteria.',
        'An API-level error response such as {"error":"model not found"} in stdout is FAIL - the model did not produce an answer.',
        'Empty, garbled, or repetitive nonsense output is FAIL.',
        'Off-topic content that does not address the prompt is FAIL.',
        'Truncation at the token limit is acceptable as long as the visible text is coherent and on-topic.',
        'For AI-generated text, accept reasonable variations (e.g. "4", "four", "The answer is 4").',
      ],
      test: {
        id: r.testCase.id,
        name: r.testCase.name,
        suite: r.testCase.suite,
        goal: r.testCase.goal || r.testCase.name,
        criteria: r.testCase.criteria,
      },
      steps,
      respond: {
        format: 'Respond with a single JSON object',
        fields: {
          testId: r.testCase.id,
          pass: 'true if the response meets the criteria, false otherwise',
          reason: 'Brief explanation of your verdict',
          evidence: 'Required if pass is false - the exact stdout content that caused failure',
        },
      },
    };

    const prompt = JSON.stringify(promptData, null, 2);

    // Log prompt stats
    const totalStdout = r.steps.reduce((sum, s) => sum + s.stdout.length, 0);
    const totalStderr = r.steps.reduce((sum, s) => sum + s.stderr.length, 0);
    process.stderr.write(`  [LLM] Prompt for ${r.testCase.id}: stdout ${totalStdout} chars, stderr ${totalStderr} chars\n`);

    return prompt;
  }

  /**
   * Judge a single test result.
   */
  private async judgeOne(result: TestResult): Promise<Judgment> {
    const prompt = this.buildPrompt(result);
    const testId = result.testCase.id;

    process.stderr.write(`  [LLM] Prompt size: ${prompt.length} chars\n`);

    const response = await axios.post(
      `${this.ollamaUrl}/api/generate`,
      {
        model: this.model,
        prompt,
        stream: false,
        format: 'json',
        options: {
          temperature: 0.1,
          num_predict: 1024,
        },
      },
      {
        timeout: 300000,
      }
    );

    const responseText = response.data.response;
    const promptTokens = response.data.prompt_eval_count ?? '?';
    const responseTokens = response.data.eval_count ?? '?';
    process.stderr.write(`  [LLM] Tokens for ${testId}: prompt=${promptTokens}, response=${responseTokens}\n`);

    // Log raw response
    if (!responseText) {
      process.stderr.write(`  [LLM] WARNING: Empty response for ${testId}\n`);
      return {
        testId,
        pass: false,
        reason: 'LLM returned empty response',
      };
    }

    process.stderr.write(`  [LLM] Raw response for ${testId} (${responseText.length} chars): ${responseText.substring(0, 500)}\n`);

    try {
      const judgment = JSON.parse(responseText) as Judgment;

      // Validate testId matches
      if (judgment.testId !== testId) {
        process.stderr.write(`  [LLM] WARNING: Response testId "${judgment.testId}" doesn't match expected "${testId}"\n`);
        judgment.testId = testId;
      }

      // Coerce string "true"/"false" to boolean (LLMs often return strings)
      if (typeof judgment.pass === 'string') {
        judgment.pass = (judgment.pass as unknown as string).toLowerCase() === 'true';
      }

      // Validate required fields
      if (typeof judgment.pass !== 'boolean') {
        process.stderr.write(`  [LLM] WARNING: Response missing "pass" field for ${testId}\n`);
        return {
          testId,
          pass: false,
          reason: `LLM response missing "pass" field: ${responseText.substring(0, 200)}`,
        };
      }

      if (!judgment.reason) {
        judgment.reason = judgment.pass ? 'Passed (no reason provided)' : 'Failed (no reason provided)';
      }

      return judgment;
    } catch {
      process.stderr.write(`  [LLM] WARNING: Failed to parse JSON for ${testId}\n`);
      process.stderr.write(`  [LLM] Full response: ${responseText}\n`);
      return {
        testId,
        pass: false,
        reason: `Failed to parse LLM response: ${responseText.substring(0, 200)}`,
      };
    }
  }

  /**
   * Judge all test results, one at a time.
   */
  async judgeResults(results: TestResult[]): Promise<Judgment[]> {
    const allJudgments: Judgment[] = [];

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      process.stderr.write(
        `  [LLM] Judging ${i + 1}/${results.length}: ${result.testCase.id}...\n`
      );

      try {
        const judgment = await this.judgeOne(result);
        allJudgments.push(judgment);
        process.stderr.write(`  [LLM] ${result.testCase.id}: ${judgment.pass ? 'PASS' : 'FAIL'} — ${judgment.reason}\n`);
      } catch (error) {
        process.stderr.write(`  [LLM] Failed to judge ${result.testCase.id}: ${error}\n`);
        allJudgments.push({
          testId: result.testCase.id,
          pass: false,
          reason: 'LLM judgment failed: ' + String(error),
        });
      }
    }

    return allJudgments;
  }

  /**
   * Unload the judge model from VRAM.
   */
  async unloadModel(): Promise<void> {
    try {
      process.stderr.write(`  [LLM] Unloading judge model ${this.model}...\n`);
      await axios.post(
        `${this.ollamaUrl}/api/generate`,
        {
          model: this.model,
          keep_alive: 0,
        },
        {
          timeout: 30000,
        }
      );
      process.stderr.write(`  [LLM] Judge model unloaded.\n`);
    } catch {
      process.stderr.write(`  [LLM] Warning: Failed to unload judge model\n`);
    }
  }
}
