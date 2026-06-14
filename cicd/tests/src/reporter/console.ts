/**
 * Console Reporter - Outputs test results to terminal with formatting.
 */

import chalk from 'chalk';
import { TestReport, TestSummary } from '../types.js';
import { formatMemoryProfile } from '../log-parser.js';

export class ConsoleReporter {
  /**
   * Format duration for display.
   */
  private formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}min`;
  }

  /**
   * Output test results to console.
   */
  report(summary: TestSummary, reports: TestReport[]): void {
    console.log('\n' + '='.repeat(60));
    console.log(chalk.bold('Test Results'));
    console.log('='.repeat(60));

    // Individual test results
    for (const report of reports) {
      const status = report.pass
        ? chalk.green.bold('[PASS]')
        : chalk.red.bold('[FAIL]');

      console.log(`\n${status} ${report.testId}: ${report.name}`);
      console.log(`  Suite: ${report.suite}`);
      console.log(`  Duration: ${this.formatDuration(report.duration)}`);

      // Show both judge results
      const simpleStatus = report.simpleJudge.pass
        ? chalk.green('PASS')
        : chalk.red('FAIL');
      const agentStatus = report.agentJudge.pass
        ? chalk.green('PASS')
        : chalk.red('FAIL');

      console.log(`  Simple Judge: ${simpleStatus} - ${report.simpleJudge.reason}`);
      console.log(`  Agent Judge: ${agentStatus} - ${report.agentJudge.reason}`);
      if (!report.agentJudge.pass && report.agentJudge.evidence) {
        console.log(`  ${chalk.yellow('Evidence:')} ${report.agentJudge.evidence}`);
      }

      if (report.memoryProfile) {
        console.log(chalk.dim('  Memory:'));
        for (const line of formatMemoryProfile(report.memoryProfile)) {
          console.log(chalk.dim(`    ${line}`));
        }
      }

      if (report.logFile) {
        console.log(`  Log file: ${report.logFile}`);
      }
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log(chalk.bold('Summary'));
    console.log('='.repeat(60));

    const passColor = summary.passed === summary.total ? chalk.green : chalk.yellow;
    console.log(
      `Total: ${summary.total} | ` +
        passColor(`Passed: ${summary.passed}`) +
        ` | ` +
        chalk.red(`Failed: ${summary.failed}`)
    );

    console.log(
      `  Simple Judge: ${summary.simple.passed}/${summary.total} passed`
    );
    console.log(`  Agent Judge: ${summary.agent.passed}/${summary.total} passed`);
    console.log(`Duration: ${this.formatDuration(summary.duration)}`);
    console.log(`Output: ${summary.runId}`);

    // Final status
    console.log('\n' + '='.repeat(60));
    if (summary.failed === 0) {
      console.log(chalk.green.bold('All tests passed!'));
    } else {
      console.log(
        chalk.red.bold(`${summary.failed} test(s) failed.`)
      );
    }
    console.log('='.repeat(60) + '\n');
  }
}
