/**
 * LLM Judge - Semantic analysis of test results using Ollama.
 *
 * Uses a language model to evaluate test execution logs against criteria.
 * Catches silent failures that exit code checking misses:
 * - CUDA errors that don't cause exit failures
 * - GPU fallback to CPU mode
 * - Memory allocation warnings
 * - Incorrect but non-crashing output
 *
 * Sends structured JSON prompt and uses Ollama JSON mode for reliable parsing.
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
   */
  private buildPrompt(result: TestResult): string {
    const r = result;
    const stdoutLimit = 1000;
    const stderrLimit = 500;
    const logsLimit = 3000;

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
      role: 'You are a test result evaluator for ollama37 (Ollama fork for Tesla K80 GPU, CUDA compute 3.7). Analyze the test execution data and determine if the test passed or failed.',
      rules: [
        'Check step stdout for error responses (e.g. {"error":"model not found"} means FAIL)',
        'CUBLAS_STATUS_* in step stdout → FAIL',
        'cudaMalloc failed followed by "applying backoff" in container logs is NORMAL (iterative memory fitting) → NOT an error',
        'cudaMalloc failed in step stdout without backoff context → FAIL',
        'library=cpu in logs means GPU detection failed → FAIL',
        'CUDA errors with exit code 0 → still FAIL',
        'flash attention warnings on K80 are acceptable, NOT errors',
        'For AI-generated text, accept reasonable variations (e.g. "4", "four", "The answer is 4")',
        'Long durations within timeout are acceptable',
        'Build times are expected to be long for CUDA compilation',
        'GPU_COUNT_EXCEEDED in step output means the model is using more GPUs than its total VRAM requires (each K80 has 11441 MiB) → FAIL',
      ],
      test: {
        id: r.testCase.id,
        name: r.testCase.name,
        suite: r.testCase.suite,
        goal: r.testCase.goal || r.testCase.name,
        criteria: r.testCase.criteria,
        timeout_ms: r.testCase.timeout,
        duration_ms: r.totalDuration,
      },
      steps,
      container_logs: this.truncate(r.logs, logsLimit),
      respond: {
        format: 'Respond with a single JSON object',
        fields: {
          testId: r.testCase.id,
          pass: 'true if test meets all criteria, false otherwise',
          reason: 'Brief explanation of your verdict',
          evidence: 'Required if pass is false — the exact stdout content or log line that caused failure',
        },
      },
    };

    const prompt = JSON.stringify(promptData, null, 2);

    // Log prompt stats
    const totalStdout = r.steps.reduce((sum, s) => sum + s.stdout.length, 0);
    const totalStderr = r.steps.reduce((sum, s) => sum + s.stderr.length, 0);
    process.stderr.write(`  [LLM] Prompt for ${r.testCase.id}: logs ${r.logs.length} chars, stdout ${totalStdout} chars, stderr ${totalStderr} chars\n`);

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
