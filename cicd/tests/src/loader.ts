/**
 * Test case loader - reads YAML test definitions and provides
 * filtering, sorting, and grouping capabilities.
 */

import { readFileSync } from 'fs';
import { glob } from 'glob';
import yaml from 'js-yaml';
import path from 'path';
import { Intent, TestCase, TestStep } from './types.js';

/**
 * Loads and manages test case definitions from YAML files.
 */
export class TestLoader {
  private testcasesDir: string;

  constructor(testcasesDir: string) {
    this.testcasesDir = testcasesDir;
  }

  /**
   * Load all test cases from the testcases directory.
   */
  async loadAll(): Promise<TestCase[]> {
    const pattern = path.join(this.testcasesDir, '**/*.yml');
    const files = await glob(pattern);

    const testCases: TestCase[] = [];

    for (const file of files) {
      try {
        const content = readFileSync(file, 'utf-8');
        const raw = yaml.load(content) as Record<string, unknown>;
        const testCase = this.validateAndNormalize(raw, file);
        if (testCase) {
          testCases.push(testCase);
        }
      } catch (error) {
        console.error(`Failed to load ${file}:`, error);
      }
    }

    return testCases;
  }

  /**
   * Load test cases filtered by suite.
   */
  async loadBySuite(suite: string): Promise<TestCase[]> {
    const all = await this.loadAll();
    return all.filter((tc) => tc.suite === suite);
  }

  /**
   * Load a single test case by ID.
   */
  async loadById(id: string): Promise<TestCase | undefined> {
    const all = await this.loadAll();
    return all.find((tc) => tc.id === id);
  }

  /**
   * Sort test cases by dependencies using topological sort.
   * Tests with lower priority numbers run first within dependency constraints.
   */
  sortByDependencies(testCases: TestCase[]): TestCase[] {
    const sorted: TestCase[] = [];
    const visited = new Set<string>();
    const idMap = new Map(testCases.map((tc) => [tc.id, tc]));

    const visit = (tc: TestCase) => {
      if (visited.has(tc.id)) return;
      visited.add(tc.id);

      // Visit dependencies first
      for (const depId of tc.dependencies) {
        const dep = idMap.get(depId);
        if (dep) visit(dep);
      }

      sorted.push(tc);
    };

    // Sort by priority first, then visit to apply dependency ordering
    const byPriority = [...testCases].sort((a, b) => a.priority - b.priority);
    for (const tc of byPriority) {
      visit(tc);
    }

    return sorted;
  }

  /**
   * Resolve all dependencies for a filtered set of test cases.
   * Handles cross-suite dependencies by looking up from the full test set.
   *
   * @param filteredTests - The tests selected by user filters
   * @param allTests - The complete set of all available test cases
   * @returns Object with expanded test list and auto-included dependency IDs
   */
  resolveDependencies(
    filteredTests: TestCase[],
    allTests: TestCase[]
  ): { tests: TestCase[]; autoIncluded: string[] } {
    const filteredIds = new Set(filteredTests.map((tc) => tc.id));
    const allTestsMap = new Map(allTests.map((tc) => [tc.id, tc]));
    const result = new Map<string, TestCase>();
    const autoIncluded: string[] = [];

    // Recursive function to collect a test and all its dependencies
    const collectWithDeps = (testId: string) => {
      // Already processed
      if (result.has(testId)) return;

      const test = allTestsMap.get(testId);
      if (!test) {
        process.stderr.write(`[WARN] Dependency ${testId} not found\n`);
        return;
      }

      // Collect dependencies first (recursive)
      for (const depId of test.dependencies) {
        collectWithDeps(depId);
      }

      result.set(testId, test);

      // Track if this was auto-included (not in original filter)
      if (!filteredIds.has(testId)) {
        autoIncluded.push(testId);
      }
    };

    // Process each filtered test
    for (const test of filteredTests) {
      collectWithDeps(test.id);
    }

    return { tests: Array.from(result.values()), autoIncluded };
  }

  /**
   * Group test cases by suite.
   * Returns groups in execution order: build -> runtime -> inference
   */
  groupBySuite(testCases: TestCase[]): Map<string, TestCase[]> {
    const groups = new Map<string, TestCase[]>();
    const suiteOrder = ['build', 'runtime', 'inference', 'models'];

    // Initialize groups in order
    for (const suite of suiteOrder) {
      groups.set(suite, []);
    }

    // Distribute test cases to groups
    for (const tc of testCases) {
      const suite = tc.suite;
      if (!groups.has(suite)) {
        groups.set(suite, []);
      }
      groups.get(suite)!.push(tc);
    }

    // Remove empty groups
    for (const [suite, cases] of groups) {
      if (cases.length === 0) {
        groups.delete(suite);
      }
    }

    return groups;
  }

  /**
   * Validate and normalize a raw YAML object into a TestCase.
   */
  private validateAndNormalize(
    raw: Record<string, unknown>,
    filePath: string
  ): TestCase | null {
    // Required fields
    if (!raw.id || typeof raw.id !== 'string') {
      console.error(`${filePath}: missing or invalid 'id' field`);
      return null;
    }
    if (!raw.name || typeof raw.name !== 'string') {
      console.error(`${filePath}: missing or invalid 'name' field`);
      return null;
    }
    if (!raw.suite || !['build', 'runtime', 'inference', 'models'].includes(raw.suite as string)) {
      console.error(`${filePath}: missing or invalid 'suite' field`);
      return null;
    }
    if (!raw.steps || !Array.isArray(raw.steps) || raw.steps.length === 0) {
      console.error(`${filePath}: missing or empty 'steps' array`);
      return null;
    }

    // Validate and normalize steps
    const steps: TestStep[] = [];
    for (const step of raw.steps) {
      if (!step.name || typeof step.name !== 'string') {
        console.error(`${filePath}: step missing 'name' field`);
        return null;
      }
      if (!step.command || typeof step.command !== 'string') {
        console.error(`${filePath}: step '${step.name}' missing 'command' field`);
        return null;
      }

      steps.push({
        name: step.name,
        command: step.command,
        timeout: typeof step.timeout === 'number' ? step.timeout : undefined,
        expectPatterns: Array.isArray(step.expectPatterns) ? step.expectPatterns : undefined,
        rejectPatterns: Array.isArray(step.rejectPatterns) ? step.rejectPatterns : undefined,
        reconnectLogs: step.reconnectLogs === true ? true : undefined,
      });
    }

    // Parse intent block. Warn (not fail) if missing so migration can be
    // incremental — pre-intent YAMLs still load via the legacy goal: field.
    let intent: Intent | undefined;
    if (raw.intent && typeof raw.intent === 'object' && !Array.isArray(raw.intent)) {
      const rawIntent = raw.intent as Record<string, unknown>;
      const userStory =
        typeof rawIntent.user_story === 'string' ? rawIntent.user_story : '';
      if (!userStory) {
        console.warn(
          `[WARN] ${filePath}: intent block present but missing 'user_story' field`
        );
      } else {
        intent = {
          userStory,
          acceptance: Array.isArray(rawIntent.acceptance)
            ? rawIntent.acceptance.filter((x): x is string => typeof x === 'string')
            : undefined,
          notes: typeof rawIntent.notes === 'string' ? rawIntent.notes : undefined,
        };
      }
    } else if (raw.intent === undefined) {
      // Only warn if the test also lacks goal: (so pre-intent YAMLs that still
      // have goal: are quietly tolerated during the #156 migration window).
      if (typeof raw.goal !== 'string') {
        console.warn(
          `[WARN] ${filePath}: missing 'intent:' block (and no legacy goal:) — see #156`
        );
      }
    }

    return {
      id: raw.id as string,
      name: raw.name as string,
      suite: raw.suite as 'build' | 'runtime' | 'inference' | 'models',
      priority: typeof raw.priority === 'number' ? raw.priority : 1,
      timeout: typeof raw.timeout === 'number' ? raw.timeout : 60000,
      dependencies: Array.isArray(raw.dependencies) ? raw.dependencies : [],
      issue: typeof raw.issue === 'number' ? raw.issue : undefined,
      intent,
      goal: typeof raw.goal === 'string' ? raw.goal : undefined,
      steps,
      criteria: typeof raw.criteria === 'string' ? raw.criteria : '',
    };
  }
}
