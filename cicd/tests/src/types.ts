/**
 * TypeScript interfaces for the ollama37 test framework v2.
 */

import type { MemoryProfile } from './log-parser.js';

// ============================================
// Test Case Definitions (from YAML)
// ============================================

/**
 * A single step within a test case.
 */
export interface TestStep {
  /** Human-readable step name */
  name: string;
  /** Shell command to execute */
  command: string;
  /** Step-specific timeout in ms (overrides test case timeout) */
  timeout?: number;
  /** Regex patterns that MUST appear in stdout/stderr */
  expectPatterns?: string[];
  /** Regex patterns that must NOT appear in stdout/stderr */
  rejectPatterns?: string[];
  /** Signal LogCollector to reconnect after this step (e.g. after container restart) */
  reconnectLogs?: boolean;
}

/**
 * Test design intent — the canonical record of WHY this test exists.
 * Read by humans / AI agents to understand purpose; not consumed by the
 * runner for execution decisions. The single design authority for the test.
 */
export interface Intent {
  /** User story or imperative goal: what value this test delivers. */
  userStory: string;
  /** What must be true for this test to be considered correct (human-readable). */
  acceptance?: string[];
  /** Free-form notes: prerequisites, gotchas, acceptable warnings. */
  notes?: string;
}

/**
 * A complete test case definition.
 */
export interface TestCase {
  /** Unique test case ID (e.g., TC-BUILD-001) */
  id: string;
  /** Human-readable test name */
  name: string;
  /** Test suite (build, runtime, inference, models) */
  suite: 'build' | 'runtime' | 'inference' | 'models';
  /** Execution priority (lower = runs first) */
  priority: number;
  /** Default timeout for all steps in ms */
  timeout: number;
  /** Test IDs that must pass before this test runs */
  dependencies: string[];
  /** GitHub issue number this test traces to */
  issue?: number;
  /** Design intent for this test (user story + acceptance + notes). */
  intent?: Intent;
  /** One-line objective, retained for LLM judge context and backwards compat with pre-intent YAMLs. */
  goal?: string;
  /** Test steps to execute */
  steps: TestStep[];
  /** Human-readable criteria for LLM judge evaluation */
  criteria: string;
}

// ============================================
// Execution Results
// ============================================

/**
 * Pattern matching result for a single pattern.
 */
export interface PatternMatch {
  pattern: string;
  found: boolean;
}

/**
 * Result of executing a single test step.
 */
export interface StepResult {
  /** Step name */
  name: string;
  /** Command that was executed */
  command: string;
  /** Captured stdout */
  stdout: string;
  /** Captured stderr */
  stderr: string;
  /** Process exit code */
  exitCode: number;
  /** Execution duration in ms */
  duration: number;
  /** Pattern matching results (if patterns were defined) */
  patternMatches?: {
    expected: PatternMatch[];
    rejected: PatternMatch[];
  };
}

/**
 * Result of executing an entire test case.
 */
export interface TestResult {
  /** The test case that was executed */
  testCase: TestCase;
  /** Results for each step */
  steps: StepResult[];
  /** Total execution duration in ms */
  totalDuration: number;
  /** Extracted logs for this test (from LogCollector) */
  logs: string;
  /** Path to the full log file */
  logFile: string;
}

// ============================================
// Judge System
// ============================================

/**
 * A judge's verdict on a test result.
 */
export interface Judgment {
  /** Test case ID */
  testId: string;
  /** Pass/fail verdict */
  pass: boolean;
  /** Explanation of the verdict */
  reason: string;
  /** Evidence log line (required when pass=false) */
  evidence?: string;
  /** Why the evidence is what it is — so an empty cell explains itself instead of a bare "—":
   *  'captured' (live data), 'denied' (a tool was refused), 'not-called' (no tool was called),
   *  'no-data' (called but returned nothing), 'verifier-unavailable' (the verifier never ran). */
  evidenceStatus?: 'captured' | 'denied' | 'not-called' | 'no-data' | 'verifier-unavailable';
}

// ============================================
// Reports
// ============================================

/**
 * Structured step result for JSON output.
 */
export interface StepReportEntry {
  /** Step name */
  name: string;
  /** Command executed */
  command: string;
  /** Exit code */
  exitCode: number;
  /** Duration in ms */
  duration: number;
  /** Captured stdout */
  stdout: string;
  /** Captured stderr */
  stderr: string;
  /** Whether step passed */
  pass: boolean;
}

/**
 * Complete report for a single test.
 */
export interface TestReport {
  /** Test case ID */
  testId: string;
  /** Test name */
  name: string;
  /** Test suite */
  suite: string;
  /** Final pass/fail (both judges must pass in dual mode) */
  pass: boolean;
  /** Final reason (combined from both judges) */
  reason: string;
  /** Execution duration in ms */
  duration: number;
  /** Structured step results */
  steps: StepReportEntry[];
  /** Path to full log file */
  logFile: string;
  /** Simple judge verdict */
  simpleJudge: Judgment;
  /** Agent judge verdict */
  agentJudge: Judgment;
  /** Parsed memory profile from container logs (if available) */
  memoryProfile?: MemoryProfile;
}

/**
 * Summary of a test run.
 */
export interface TestSummary {
  /** Run identifier (ISO timestamp) */
  runId: string;
  /** Suite that was run (or 'all') */
  suite: string;
  /** When the run started */
  timestamp: string;
  /** Total duration in ms */
  duration: number;
  /** Total number of tests */
  total: number;
  /** Number of passing tests */
  passed: number;
  /** Number of failing tests */
  failed: number;
  /** Simple judge breakdown */
  simple: {
    passed: number;
    failed: number;
  };
  /** Agent judge breakdown */
  agent: {
    passed: number;
    failed: number;
  };
  /** Environment info */
  environment: {
    hostname: string;
    nodeVersion: string;
    dockerVersion?: string;
  };
  /** List of test IDs in execution order */
  tests: string[];
}

// ============================================
// Configuration
// ============================================

/**
 * CLI run configuration.
 */
export interface RunConfig {
  /** Filter by suite */
  suite?: 'build' | 'runtime' | 'inference' | 'models';
  /** Filter by specific test ID */
  testId?: string;
  /** Show what would run without executing */
  dryRun: boolean;
  /** Judge mode: 'simple' (deterministic only) or 'dual' (also run the agent judge).
   *  Defaults to 'simple'; opt in via the JUDGE_MODE=dual env var. */
  judgeMode: 'simple' | 'dual';
  /** Output directory for results */
  outputDir: string;
  /** Output format */
  outputFormat: 'console' | 'json' | 'junit';
  /** Working directory (project root) */
  workingDir: string;
  /** Path to docker-compose.yml for test subject */
  dockerComposePath: string;
}

/**
 * Default configuration values.
 */
export const DEFAULT_CONFIG: Partial<RunConfig> = {
  dryRun: false,
  judgeMode: 'simple',
  outputFormat: 'console',
};
