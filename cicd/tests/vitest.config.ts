import { defineConfig } from 'vitest/config';

/**
 * One GPU, so one test at a time. Vitest parallelises by default: two models
 * would load at once, contend for VRAM, and report timings neither of them earned.
 */
export default defineConfig({
  test: {
    include: ['suites/**/*.test.ts'],
    // false pins maxWorkers to 1, so the files run one after another in one process.
    fileParallelism: false,
    sequence: { concurrent: false },
    // A 27b model on a K80 loads for minutes before it emits a token. The 5s
    // default fails every test here; 15 minutes matches the YAML testcases'
    // own timeout.
    testTimeout: 900_000,
    hookTimeout: 900_000,
    reporters: process.env.GITHUB_ACTIONS ? ['default', 'github-actions', 'junit'] : ['default'],
    outputFile: { junit: 'results/junit.xml' },
  },
});
