/**
 * TypeScript interfaces for the ollama37 test framework v2.
 */

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
  /** One-line objective. No testcase sets it; the perf tools do. */
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
  /** The model's reply, when this step produced one. Absent for a step that
   *  ran no model, which is why every consumer must keep its stdout path. */
  reply?: ModelReply;
}

/**
 * An Ollama reply, as fields rather than as text.
 *
 * `response` and `thinking` mean different things and a thinking model can fill
 * one while leaving the other empty, so flattening them loses which field an
 * assertion or a judge actually read. `doneReason` is here because `length`
 * means the reply stopped on its token budget rather than mid-thought, and a
 * judge told only "this is the output" reads that truncation as incoherence.
 *
 * Every field is optional: a caller fills what its endpoint returns.
 * `/api/generate` has no `tool_calls`, and `perf/capture.ts` throws on `error`
 * rather than returning it, so those two arrive only from other callers.
 */
export interface ModelReply {
  /** The answer field. Empty when a thinking model spent its budget reasoning. */
  response?: string;
  /** Reasoning, when the model emits it separately. */
  thinking?: string;
  toolCalls?: unknown[];
  error?: string;
  /** `stop`, `length`, … — `length` means the token budget ended it. */
  doneReason?: string;
  evalCount?: number;
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
   *  'no-data' (called but returned nothing), 'verifier-unavailable' (the verifier produced no
   *  verdict — couldn't start, or errored/timed out mid-run). */
  evidenceStatus?: 'captured' | 'denied' | 'not-called' | 'no-data' | 'verifier-unavailable';
  /** verify-live: the verifier's per-stage rubric flags (tool selection / query / interpretation),
   *  if it graded them — so the report can show *judge info*, not just the final pass/fail. */
  stages?: { tool: boolean; query: boolean; content: boolean };
  /** verify-live deterministic cross-check: the claimed facts NOT found in the live data
   *  (empty = all grounded). Undefined when there was no captured evidence to check against. */
  crossCheckUnsupported?: string[];
}
