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
import { mkdirSync, existsSync } from 'fs';
import { TestLoader } from './loader.js';
import { TestExecutor } from './executor.js';
import { SimpleJudge, AgentJudge } from './judge/index.js';
import { JsonReporter, ConsoleReporter } from './reporter/index.js';
import { RunConfig } from './types.js';
import { CONFIG, pickEnv } from './config.js';
import { runThroughput } from './perf/throughput.js';
import { runFa } from './perf/fa.js';
import { runMcpTest } from './mcp/test-mcp.js';

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
    const dockerDir = path.join(projectRoot, 'docker');

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
      dockerComposePath: dockerDir,
    };

    process.stderr.write(`\n[CONFIG] Project root: ${projectRoot}\n`);
    process.stderr.write(`[CONFIG] Docker compose: ${dockerDir}\n`);
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

    // Run judges
    process.stderr.write('\n[JUDGE] Running simple judge...\n');
    const simpleJudge = new SimpleJudge();
    const simpleJudgments = simpleJudge.judgeAll(results);

    let agentJudgments = simpleJudgments.map((j) => ({
      ...j,
      reason: config.judgeMode === 'dual' ? j.reason : 'Agent judge disabled (simple mode)',
    }));

    if (config.judgeMode === 'dual') {
      process.stderr.write('[JUDGE] Running agent judge...\n');
      const agentJudge = new AgentJudge();

      const available = await agentJudge.isAvailable();
      if (available) {
        agentJudgments = await agentJudge.judgeResults(results);
      } else {
        process.stderr.write('[WARN] Agent judge not available, using simple judge results\n');
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
    if (config.outputFormat === 'console') {
      const consoleReporter = new ConsoleReporter();
      consoleReporter.report(summary, reports);
    } else if (config.outputFormat === 'json') {
      jsonReporter.outputSummary(summary, reports);
    }

    // Exit with appropriate code
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
  .option('-n, --num-predict <n>', 'Max tokens to generate', '128')
  .option('-c, --context <n>', 'Context window size', '2048')
  .option('--judge', 'Also run the agent judge on each response (dual mode)', false)
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST || 'http://localhost:11434')
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (models: string[], options) => {
    const code = await runThroughput({
      models,
      numPredict: Number(options.numPredict),
      numCtx: Number(options.context),
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
 * test-fa — K80 flash-attention regression + benchmark.
 *
 * Ports test-fa-k80.sh: boots a fresh container per FA/KV config, runs a
 * deterministic inference, captures VRAM/KV/tok-s, and judges coherence with
 * the keyless agent judge. Needs exclusive use of port 11434 — the caller (the
 * workflow) stops/starts production ollama37 around it.
 */
program
  .command('test-fa')
  .description('K80 flash-attention regression + benchmark')
  .option('--model <model>', 'Model to test', 'gemma3:4b')
  .option('--prompt <text>', 'Inference prompt (deterministic; temp=0)', 'Write a detailed multi-paragraph explanation of how large language models work. Cover tokenization, attention mechanisms, training procedures, and inference. Be thorough.')
  .option('-n, --num-predict <n>', 'Tokens to generate', '256')
  .option('-c, --num-ctx <n>', 'Context window size', '4096')
  .option('--benchmark', 'Run all three configs (off-f16, on-f16, on-q8_0)', false)
  .option('--kv-cache-type <type>', '(single mode) OLLAMA_KV_CACHE_TYPE (empty = f16)', '')
  .option('--baseline-compare', '(single mode) run an FA-off baseline first', false)
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (options) => {
    const code = await runFa({
      model: options.model,
      prompt: options.prompt,
      numPredict: Number(options.numPredict),
      numCtx: Number(options.numCtx),
      benchmark: options.benchmark,
      kvCacheType: options.kvCacheType,
      baselineCompare: options.baselineCompare,
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
  .option('-c, --num-ctx <n>', 'Context window size', '4096')
  .option('--judge', 'Also run the agent judge on the final answer (dual mode)', false)
  .option('-H, --host <url>', 'Ollama host', process.env.OLLAMA_HOST || 'http://localhost:11434')
  .option('--mcp-command <cmd>', 'Command to launch the stdio MCP server', CONFIG.mcp.command)
  .option('--mcp-args <args>', 'Args for the MCP server (space-separated; no spaces within a single arg)', CONFIG.mcp.args.join(' '))
  .option('--mcp-env <names>', 'Comma-separated env var names to forward to the server as creds (overrides MCP_ENV)', '')
  .option('--verify-live', 'Verify the answer against LIVE truth: the judge calls the server\'s read-only tools itself (supersedes --judge)', false)
  .option('--verify-allow <names>', 'Comma-separated exact tool names the verifier may call (fail-closed: empty verifies nothing)', '')
  .option('--verify-server-name <name>', 'mcp__<name>__<tool> label the verifier registers the server under', 'mcp')
  .option('--timeout <seconds>', 'Per-model Ollama response timeout (heavy models need more)', '600')
  .option('-o, --output <file>', 'Write the JSON report to this file')
  .action(async (models: string[], options) => {
    const code = await runMcpTest({
      models,
      prompt: options.prompt,
      numCtx: Number(options.numCtx),
      timeoutMs: Number(options.timeout) * 1000,
      judge: options.judge,
      verifyLive: options.verifyLive,
      verifyAllow: options.verifyAllow ? String(options.verifyAllow).split(',').map((s: string) => s.trim()).filter(Boolean) : undefined,
      verifyServerName: options.verifyServerName,
      host: options.host,
      server: {
        command: options.mcpCommand,
        args: String(options.mcpArgs).split(' ').filter(Boolean),
        cwd: CONFIG.mcp.cwd,
        env: options.mcpEnv ? pickEnv(options.mcpEnv) : CONFIG.mcp.env,
      },
      output: options.output,
    });
    await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
    process.exit(code);
  });

program.parse();
