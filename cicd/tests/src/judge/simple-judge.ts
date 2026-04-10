/**
 * Simple Judge - Fast, deterministic verification based on:
 * 1. Exit codes (all steps must return 0)
 * 2. Pattern matching (expected patterns found, rejected patterns absent)
 * 3. CUDA error detection (no CUBLAS/CUDA errors in logs)
 */

import { TestResult, Judgment } from '../types.js';

/**
 * Known CUDA/K80 error patterns that indicate failures.
 */
const CUDA_ERROR_PATTERNS = [
  /CUBLAS_STATUS_/i,
  /CUDA error/i,
  /cudaMalloc failed/i,
  /out of memory/i,
  /OOM/,
  /library=cpu/i, // GPU detection failed - fell back to CPU
  /NVIDIA-SMI has failed/i,
];

export class SimpleJudge {
  /**
   * Judge a single test result.
   */
  judge(result: TestResult): Judgment {
    const reasons: string[] = [];
    let pass = true;

    // Check 1: All steps exit code 0
    const failedSteps = result.steps.filter((s) => s.exitCode !== 0);
    if (failedSteps.length > 0) {
      pass = false;
      reasons.push(
        `${failedSteps.length} step(s) failed with non-zero exit code: ${failedSteps.map((s) => `${s.name}(${s.exitCode})`).join(', ')}`
      );
    }

    // Check 2: Expected patterns found
    for (const step of result.steps) {
      if (step.patternMatches) {
        const missing = step.patternMatches.expected.filter((p) => !p.found);
        if (missing.length > 0) {
          pass = false;
          reasons.push(
            `Step "${step.name}" missing expected patterns: ${missing.map((p) => p.pattern).join(', ')}`
          );
        }

        // Check 3: Rejected patterns absent
        const found = step.patternMatches.rejected.filter((p) => p.found);
        if (found.length > 0) {
          pass = false;
          reasons.push(
            `Step "${step.name}" found rejected patterns: ${found.map((p) => p.pattern).join(', ')}`
          );
        }
      }
    }

    // Check 4: No CUDA errors in step output
    // Only check step stdout/stderr, not container logs — container logs may contain
    // expected cudaMalloc failures from Ollama's iterative memory backoff process.
    const stepOutput = result.steps.map((s) => s.stdout + s.stderr).join('\n');
    for (const pattern of CUDA_ERROR_PATTERNS) {
      if (pattern.test(stepOutput)) {
        pass = false;
        reasons.push(`CUDA error pattern detected: ${pattern.source}`);
        break; // Only report first CUDA error
      }
    }

    return {
      testId: result.testCase.id,
      pass,
      reason: pass
        ? 'All steps passed with exit code 0, patterns matched, no CUDA errors'
        : reasons.join('; '),
    };
  }

  /**
   * Judge multiple test results.
   */
  judgeAll(results: TestResult[]): Judgment[] {
    return results.map((r) => this.judge(r));
  }
}
