#!/usr/bin/env node
/**
 * CLI for the ollama37 test framework.
 *
 * Usage:
 *   npx tsx src/cli.ts run [options]
 *   npx tsx src/cli.ts list [options]
 */

import 'dotenv/config'; // load cicd/tests/.env into process.env before config.ts reads it
import { Command } from 'commander';
import path from 'path';
import { mkdirSync, existsSync, writeFileSync, readFileSync } from 'fs';
import { TestLoader } from './loader.js';
import { TestExecutor } from './executor.js';
import { SimpleJudge, AgentJudge } from './judge/index.js';
import { JsonReporter, ConsoleReporter } from './reporter/index.js';
import { RunConfig, TestResult, TestSummary } from './types.js';
import { CONFIG, pickEnv } from './config.js';
import {
  runThroughput,
  judgeThroughputResults,
  judgeModeLabel,
  printSummary,
  type ThroughputReport,
} from './perf/throughput.js';
import { runContext } from './perf/context.js';
import { runMcpTest } from './mcp/test-mcp.js';
import { modelBounds } from './perf/model-bounds.js';

/**
 * Judge executed results and write the reports. `run` calls it straight after the
 * tests; `judge` calls it later on results a `run` saved, once the judge's server is up.
 */
async function judgeAndReport(
  results: TestResult[],
  judgeMode: RunConfig['judgeMode'],
  outputDir: string,
  outputFormat: RunConfig['outputFormat'],
  startTime: Date,
  suiteName: string
): Promise<TestSummary> {
  // Run judges
  process.stderr.write('\n[JUDGE] Running simple judge...\n');
  const simpleJudge = new SimpleJudge();
  const simpleJudgments = simpleJudge.judgeAll(results);

  let agentJudgments = simpleJudgments.map((j) => ({
    ...j,
    reason: judgeMode === 'dual' ? j.reason : 'Agent judge disabled (simple mode)',
  }));

  if (judgeMode === 'dual') {
    // A test the simple judge already failed is failed whatever the agent says (both
    // must pass), so don't ask it. That also keeps flagged replies — REPLY_REPEAT above
    // all — away from the agent, which loops when it quotes repeated text back.
    const simpleFailed = new Map(simpleJudgments.filter((j) => !j.pass).map((j) => [j.testId, j]));
    const toJudge = results.filter((r) => !simpleFailed.has(r.testCase.id));
    const skipped = [...simpleFailed.values()].map((j) => ({
      testId: j.testId,
      pass: false,
      reason: `Skipped — simple judge already failed: ${j.reason}`,
    }));
    agentJudgments = skipped;

    if (toJudge.length === 0) {
      process.stderr.write('[JUDGE] Agent judge skipped: every test already failed the simple judge\n');
    } else {
      process.stderr.write(`[JUDGE] Running agent judge on ${toJudge.length} test(s), ${skipped.length} skipped...\n`);
      const agentJudge = new AgentJudge();

      const available = await agentJudge.isAvailable();
      if (available) {
        agentJudgments = [...skipped, ...(await agentJudge.judgeResults(toJudge))];
      } else {
        process.stderr.write('[WARN] Agent judge not available, using simple judge results\n');
        agentJudgments = simpleJudgments;
      }
    }
  }

  // Generate and output reports
  const jsonReporter = new JsonReporter(outputDir);
  const { summary, reports } = jsonReporter.generateReports(
    results,
    simpleJudgments,
    agentJudgments,
    startTime,
    suiteName
  );

  // Write JSON files regardless of format
  jsonReporter.writeReports(summary, reports);

  // Console output
  if (outputFormat === 'console') {
    const consoleReporter = new ConsoleReporter();
    consoleReporter.report(summary, reports);
  } else if (outputFormat === 'json') {
    jsonReporter.outputSummary(summary, reports);
  }

  return summary;
}

const program = new Command();

program
  .name('ollama37-test')
  .description('Test framework for ollama37 CUDA 3.7 CI/CD validation')
  .version('2.0.0');

/**
 * Run command - execute tests
 */
program
  .command('run')
  .description('Run test cases')
  .option('-s, --suite <suite>', 'Run only tests from this suite (build, runtime, inference, models)')
  .option('-i, --id <id>', 'Run only the test with this ID')
  .option('--dry-run', 'Show what would run without executing', false)
  .option('-o, --output-dir <dir>', 'Output directory for results')
  .option('-f, --format <format>', 'Output format (console, json)', 'console')
  .action(async (options) => {
    const startTime = new Date();

    // Resolve paths
    const testsDir = path.dirname(new URL(import.meta.url).pathname);
    const projectRoot = path.resolve(testsDir, '..', '..', '..');
    const testcasesDir = path.join(testsDir, '..', 'testcases');

    // Generate output directory with timestamp
    const timestamp = startTime.toISOString().replace(/[:.]/g, '-').substring(0, 19);
    const suiteName = options.suite || 'all';
    const outputDir = options.outputDir || path.join(testsDir, '..', '..', 'results', `${timestamp}_${suiteName}`);

    // Ensure output directory exists
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true });
    }

    // Agent judge runs in 'dual' mode — enabled via the JUDGE_MODE env var.
    const judgeMode: RunConfig['judgeMode'] = CONFIG.judge.mode === 'dual' ? 'dual' : 'simple';

    const config: RunConfig = {
      suite: options.suite as RunConfig['suite'],
      testId: options.id,
      dryRun: options.dryRun,
      judgeMode,
      outputDir,
      outputFormat: options.format as RunConfig['outputFormat'],
      workingDir: projectRoot,
    };

    process.stderr.write(`\n[CONFIG] Project root: ${projectRoot}\n`);
    process.stderr.write(`[CONFIG] Server: ${process.env.OLLAMA37_CONTAINER} at ${process.env.OLLAMA_HOST}\n`);
    process.stderr.write(`[CONFIG] Testcases: ${testcasesDir}\n`);
    process.stderr.write(`[CONFIG] Output: ${outputDir}\n`);
    process.stderr.write(`[CONFIG] Agent Judge: ${config.judgeMode === 'dual' ? 'enabled (dual)' : 'disabled (simple only)'}\n`);

    // Load test cases
    const loader = new TestLoader(testcasesDir);
    const allTestCases = await loader.loadAll();

    if (allTestCases.length === 0) {
      process.stderr.write('[ERROR] No test cases found\n');
      process.exit(1);
    }

    // Apply user filters
    let filteredTestCases = allTestCases;

    // Filter by suite
    if (config.suite) {
      filteredTestCases = filteredTestCases.filter((tc) => tc.suite === config.suite);
    }

    // Filter by the host's subset, from the runner's .env
    if (process.env.OLLAMA37_TEST_IDS) {
      const ids = process.env.OLLAMA37_TEST_IDS.split(',').map((s) => s.trim());
      filteredTestCases = filteredTestCases.filter((tc) => ids.includes(tc.id));
    }

    // Filter by ID
    if (config.testId) {
      filteredTestCases = filteredTestCases.filter((tc) => tc.id === config.testId);
    }

    if (filteredTestCases.length === 0) {
      process.stderr.write('[ERROR] No matching test cases found\n');
      process.exit(1);
    }

    // Resolve cross-suite dependencies
    const { tests: resolvedTestCases, autoIncluded } = loader.resolveDependencies(
      filteredTestCases,
      allTestCases
    );

    if (autoIncluded.length > 0) {
      process.stderr.write(`[INFO] Auto-included ${autoIncluded.length} dependency test(s): ${autoIncluded.join(', ')}\n`);
    }

    // Sort by dependencies
    const testCases = loader.sortByDependencies(resolvedTestCases);

    process.stderr.write(`[INFO] Found ${testCases.length} test(s) to run\n`);

    // Dry run - just show what would run
    if (config.dryRun) {
      process.stderr.write('\n[DRY RUN] Would execute:\n');
      for (const tc of testCases) {
        process.stderr.write(`  - ${tc.id}: ${tc.name} (${tc.suite})\n`);
        for (const step of tc.steps) {
          process.stderr.write(`      Step: ${step.name}\n`);
        }
      }
      process.exit(0);
    }

    // Execute tests
    const executor = new TestExecutor(config);
    const results = await executor.executeAll(testCases);

    // Save the raw results, so `judge` can judge them in a later step
    writeFileSync(path.join(outputDir, 'results.json'), JSON.stringify(results));

    const summary = await judgeAndReport(
      results,
      config.judgeMode,
      outputDir,
      config.outputFormat,
      startTime,
      suiteName
    );

    // Exit with appropriate code
    process.exit(summary.failed > 0 ? 1 : 0);
  });

/**
 * Judge command - judge the results a `run` saved
 */
program
  .command('judge <resultsDir>')
  .description('Judge the results a run saved in <resultsDir>, and rewrite its reports')
  .option('-f, --format <format>', 'Output format (console, json)', 'console')
  .action(async (resultsDir, options) => {
    const results: TestResult[] = JSON.parse(
      readFileSync(path.join(resultsDir, 'results.json'), 'utf-8')
    );
    const judgeMode: RunConfig['judgeMode'] = CONFIG.judge.mode === 'dual' ? 'dual' : 'simple';
    const suiteName = [...new Set(results.map((r) => r.testCase.suite))].join('+') || 'all';
    process.stderr.write(`[CONFIG] Judging ${results.length} result(s) from ${resultsDir}\n`);
    const summary = await judgeAndReport(
      results,
      judgeMode,
      resultsDir,
      options.format as RunConfig['outputFormat'],
      new Date(),
      suiteName
    );
    process.exit(summary.failed > 0 ? 1 : 0);
  });

/**
 * List command - show available tests
 */
program
  .command('list')
  .description('List available test cases')
  .option('-s, --suite <suite>', 'Filter by suite')
  .action(async (options) => {
    const testsDir = path.dirname(new URL(import.meta.url).pathname);
    const testcasesDir = path.join(testsDir, '..', 'testcases');

    const loader = new TestLoader(testcasesDir);
    let testCases = await loader.loadAll();

    if (options.suite) {
      testCases = testCases.filter((tc) => tc.suite === options.suite);
    }

    testCases = loader.sortByDependencies(testCases);
    const groups = loader.groupBySuite(testCases);

    console.log('\nAvailable Test Cases:');
    console.log('='.repeat(60));

    for (const [suite, cases] of groups) {
      console.log(`\n${suite.toUpperCase()} SUITE (${cases.length} tests):`);
      for (const tc of cases) {
        console.log(`  ${tc.id}: ${tc.name}`);
        console.log(`    Priority: ${tc.priority}, Timeout: ${tc.timeout}ms`);
        if (tc.dependencies.length > 0) {
          console.log(`    Depends on: ${tc.dependencies.join(', ')}`);
        }
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`Total: ${testCases.length} test(s)`);
  });

/**
 * bench-throughput — measure tok/s across models and validate the output.
 *
 * Ports benchmark-throughput.sh into the TS framework: perf metrics always,
 * plus a coherence check (simple, and the keyless agent judge with --judge).
 * Prints a markdown summary to stdout and writes a JSON report with --output.
 */
program
  .command('bench-throughput')
  .description('Benchmark model throughput (tok/s) + validate output')
  .argument('<models...>', 'One or more model names to benchmark')
  .option('-n, --num-predict <n>', 'Max tokens to generate; 400 matches the models suite', '400')
  .option('-c, --context <n>', 'Context window size; empty = the model\'s own window')
  .option('-b, --num-batch <n>', 'Micro-batch size (num_batch); empty = model default (512)')
  .option('--judge', 'Also run the agent judge on each response (dual mode)', false)
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST)
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (models: string[], options) => {
    const code = await runThroughput({
      models,
      numPredict: Number(options.numPredict),
      numCtx: options.context ? Number(options.context) : undefined,
      numBatch: options.numBatch ? Number(options.numBatch) : undefined,
      judge: options.judge,
      host: options.host,
      output: options.output,
    });
    // Flush stdout before forcing exit — the markdown summary is piped to tee
    // in CI, and process.exit() can truncate buffered pipe output.
    await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
    process.exit(code);
  });

/**
 * judge-throughput — judge what a bench-throughput run already captured.
 *
 * The benchmark holds the GPU; the judge must not. Running it here, from the saved
 * report, lets CI give the card back before any judging starts (test-models.yml's
 * Yield → run → Restore → judge order). Reads the full captured text, which is why
 * the report persists it rather than only a preview.
 */
program
  .command('judge-throughput <json>')
  .description('Judge the responses a bench-throughput run saved in <json>, and rewrite it')
  .action(async (jsonPath: string) => {
    const report: ThroughputReport = JSON.parse(readFileSync(jsonPath, 'utf-8'));
    process.stderr.write(`[CONFIG] Judging ${report.results.length} result(s) from ${jsonPath}\n`);

    const fellBack = await judgeThroughputResults(report.results);
    writeFileSync(jsonPath, JSON.stringify(report, null, 2));

    printSummary(
      report.git_sha,
      report.gpu.before,
      report.config.num_ctx,
      judgeModeLabel(true, fellBack),
      report.results
    );
    const failed = report.results.filter((r) => !r.check.overall_pass).length;
    // Flush stdout before forcing exit — the markdown summary is piped to tee
    // in CI, and process.exit() can truncate buffered pipe output.
    await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
    process.exit(failed > 0 ? 1 : 0);
  });

/**
 * bench-context — long-context throughput + correctness for the FA path comparison.
 *
 * Primes a long DETERMINISTIC prompt (so prefill is realistic and the KV cache is
 * full before decode) with a buried needle, then measures prefill/decode tok/s and
 * a per-model verdict of simple ∧ needle ∧ agent(dual). Unlike bench-throughput's
 * short prompt, this is the regime where flash attention's cost/benefit shows.
 */
program
  .command('bench-context')
  .description('Long-context benchmark (prefill/decode tok/s) with a deterministic needle + output validation')
  .argument('<models...>', 'One or more model names to benchmark')
  .option('-n, --num-predict <n>', 'Max tokens to generate (room for a thinking model to reason + answer)', '1024')
  .option('-c, --context <n>', 'Target primed-prompt length in tokens', '4096')
  .option('--judge', 'Also run the agent judge on each response (dual mode)', false)
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST)
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (models: string[], options) => {
    const code = await runContext({
      models,
      numPredict: Number(options.numPredict),
      context: Number(options.context),
      judge: options.judge,
      host: options.host,
      output: options.output,
    });
    await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
    process.exit(code);
  });

/**
 * test-mcp — can a model drive a REAL MCP server's tools end-to-end?
 *
 * Connects to a stdio MCP server (default testlink-mcp; override with
 * --mcp-command/--mcp-args), lists + translates its tools, and runs the model
 * through the tool loop. Reports a per-model verdict: structural check always,
 * plus the keyless agent judge with --judge. Server-agnostic — no mock.
 */
program
  .command('test-mcp')
  .description("Test whether models can drive a real MCP server's tools")
  .argument('<models...>', 'One or more model names to test')
  .option('--prompt <text>', 'Prompt that should trigger a tool call', CONFIG.mcp.prompt)
  .option('-c, --num-ctx <n>', 'Context window size (8192 fits a merged multi-server tool menu; one server fits in 4096)', '8192')
  .option('-b, --num-batch <n>', 'Micro-batch size (num_batch); empty = model default (512)')
  .option('--judge', 'Also run the agent judge on the final answer (dual mode)', false)
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST)
  .option('--mcp-command <cmd>', 'Command to launch the stdio MCP server', CONFIG.mcp.command)
  .option('--mcp-args <args>', 'Args for the MCP server (space-separated; no spaces within a single arg)', CONFIG.mcp.args.join(' '))
  .option('--mcp-env <names>', 'Comma-separated env var names to forward to the server as creds (overrides MCP_ENV)', '')
  .option('--distractor-command <cmd>', 'Optional second (menu-only) MCP server, e.g. playwright — its tools join the menu as distractors the model must NOT pick; the verifier never touches it')
  .option('--distractor-args <args>', 'Args for the distractor MCP server (space-separated; no spaces within a single arg)', '')
  .option('--verify-live', 'Verify the answer against LIVE truth: the judge calls the server\'s read-only tools itself (supersedes --judge)', false)
  .option('--verify-allow <names>', 'Comma-separated exact tool names the verifier may call (fail-closed: empty verifies nothing)', '')
  .option('--verify-server-name <name>', 'Name of the server the verifier spawns (the primary server is registered under this name); its tools are mcp__<name>__<tool>', 'mcp')
  .option('--timeout <seconds>', 'Per-call Ollama response timeout (a multi-server menu on the K80 takes ~18 min/round; a single server is far faster)', '1800')
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (models: string[], options) => {
    const code = await runMcpTest({
      models,
      prompt: options.prompt,
      numCtx: Number(options.numCtx),
      numBatch: options.numBatch ? Number(options.numBatch) : undefined,
      timeoutMs: Number(options.timeout) * 1000,
      judge: options.judge,
      verifyLive: options.verifyLive,
      verifyAllow: options.verifyAllow ? String(options.verifyAllow).split(',').map((s: string) => s.trim()).filter(Boolean) : undefined,
      verifyServerName: options.verifyServerName,
      host: options.host,
      // Primary server (the verifier's read-only target) is named after --verify-server-name
      // so the verifier can find it. An optional distractor server joins the model's menu only.
      servers: [
        {
          name: options.verifyServerName,
          command: options.mcpCommand,
          args: String(options.mcpArgs).split(' ').filter(Boolean),
          cwd: CONFIG.mcp.cwd,
          env: options.mcpEnv ? pickEnv(options.mcpEnv) : CONFIG.mcp.env,
        },
        ...(options.distractorCommand
          ? [{
              name: 'distractor',
              command: options.distractorCommand,
              args: String(options.distractorArgs).split(' ').filter(Boolean),
            }]
          : []),
      ],
      output: options.output,
    });
    await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
    process.exit(code);
  });

/**
 * model-bounds — resolve a model's native context + tool capability for the fit-map sweep.
 *
 * Prints ONE line to stdout: `<nativeCtx> <tools 0|1> <arch>` — so the sweep can
 *   read NATIVE_CTX TOOLS ARCH <<< "$(cli.ts model-bounds "$M")"
 * to bound its context ladder and pick the judge. All diagnostics go to stderr.
 */
program
  .command('model-bounds')
  .description("Resolve a model's native context length + tool support from /api/show")
  .argument('<model>', 'Model name to inspect')
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST)
  .action(async (model: string, options) => {
    try {
      const b = await modelBounds(options.host, model);
      process.stderr.write(`${model}: native_ctx=${b.nativeCtx} tools=${b.tools} arch=${b.arch}\n`);
      process.stdout.write(`${b.nativeCtx} ${b.tools ? 1 : 0} ${b.arch}\n`);
      process.exit(0);
    } catch (e) {
      process.stderr.write(`model-bounds ${model}: ${e instanceof Error ? e.message : e}\n`);
      process.stdout.write('0 0 \n');
      process.exit(1);
    }
  });

program.parse();
