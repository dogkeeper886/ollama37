/**
 * JSON Reporter - Outputs test results as JSON files.
 */

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { hostname } from 'os';
import path from 'path';
import { TestResult, TestReport, TestSummary, Judgment, StepReportEntry } from '../types.js';
import { parseMemoryProfile } from '../log-parser.js';

export class JsonReporter {
  private outputDir: string;

  constructor(outputDir: string) {
    this.outputDir = outputDir;
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true });
    }
  }

  /**
   * Generate reports from test results and judgments.
   */
  generateReports(
    results: TestResult[],
    simpleJudgments: Judgment[],
    llmJudgments: Judgment[],
    startTime: Date,
    suite: string
  ): { summary: TestSummary; reports: TestReport[] } {
    const endTime = new Date();
    const duration = endTime.getTime() - startTime.getTime();

    // Create judgment maps for quick lookup
    const simpleMap = new Map(simpleJudgments.map((j) => [j.testId, j]));
    const llmMap = new Map(llmJudgments.map((j) => [j.testId, j]));

    // Generate individual test reports
    const reports: TestReport[] = results.map((result) => {
      const simple = simpleMap.get(result.testCase.id) || {
        testId: result.testCase.id,
        pass: false,
        reason: 'No judgment',
      };
      const llm = llmMap.get(result.testCase.id) || {
        testId: result.testCase.id,
        pass: false,
        reason: 'No judgment',
      };

      // Both must pass in dual mode
      const pass = simple.pass && llm.pass;

      // Convert StepResult[] to StepReportEntry[]
      const steps: StepReportEntry[] = result.steps.map((step) => ({
        name: step.name,
        command: step.command,
        exitCode: step.exitCode,
        duration: step.duration,
        stdout: step.stdout,
        stderr: step.stderr,
        pass: step.exitCode === 0,
      }));

      const memoryProfile = parseMemoryProfile(result.logs) || undefined;

      return {
        testId: result.testCase.id,
        name: result.testCase.name,
        suite: result.testCase.suite,
        pass,
        reason: pass
          ? 'Both judges passed'
          : `Simple: ${simple.reason}; LLM: ${llm.reason}`,
        duration: result.totalDuration,
        steps,
        logFile: result.logFile,
        simpleJudge: simple,
        llmJudge: llm,
        memoryProfile,
      };
    });

    // Calculate summary
    const simplePassed = simpleJudgments.filter((j) => j.pass).length;
    const llmPassed = llmJudgments.filter((j) => j.pass).length;
    const passed = reports.filter((r) => r.pass).length;

    const summary: TestSummary = {
      runId: startTime.toISOString(),
      suite,
      timestamp: startTime.toISOString(),
      duration,
      total: results.length,
      passed,
      failed: results.length - passed,
      simple: {
        passed: simplePassed,
        failed: results.length - simplePassed,
      },
      llm: {
        passed: llmPassed,
        failed: results.length - llmPassed,
      },
      environment: {
        hostname: hostname(),
        nodeVersion: process.version,
      },
      tests: results.map((r) => r.testCase.id),
    };

    return { summary, reports };
  }

  /**
   * Write reports to JSON files.
   */
  writeReports(summary: TestSummary, reports: TestReport[]): void {
    // Write summary
    const summaryPath = path.join(this.outputDir, 'summary.json');
    writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

    // Write individual test reports
    for (const report of reports) {
      const reportPath = path.join(this.outputDir, `${report.testId}.json`);
      writeFileSync(reportPath, JSON.stringify(report, null, 2));
    }

    process.stderr.write(`\n[JSON] Results written to ${this.outputDir}/\n`);
  }

  /**
   * Output summary to stdout as JSON.
   */
  outputSummary(summary: TestSummary, reports: TestReport[]): void {
    const output = {
      summary,
      results: reports.map((r) => ({
        testId: r.testId,
        name: r.name,
        suite: r.suite,
        pass: r.pass,
        reason: r.reason,
        duration: r.duration,
        steps: r.steps,
        simpleJudge: r.simpleJudge,
        llmJudge: r.llmJudge,
        memoryProfile: r.memoryProfile,
      })),
    };
    console.log(JSON.stringify(output, null, 2));
  }
}
