---
name: ci-run
description: Execute test cases with the simple judge by default, or opt in the agent judge
user-invocable: true
---

# Run Test Cases

Execute test cases and evaluate results. The simple (deterministic) judge is the
default verdict; the agent judge is an opt-in second opinion (`JUDGE_MODE=dual`),
keyless on a Claude Code subscription.

```
$ARGUMENTS

## PURPOSE

Run YAML test cases by executing commands and evaluating results against patterns and criteria.

---

## AGENT WORKFLOW

### Step 1: Load Test Cases

Input can be:
- A test ID (e.g., `TC-RUNTIME-001`) — run that specific test
- A suite name (`build`, `runtime`, `inference`, or `models`) — run all tests in that suite
- Empty — run all tests

Read YAML test case files from `cicd/tests/testcases/`.

### Step 2: Resolve Dependencies

Sort tests by:
1. Dependencies (tests that depend on others run after)
2. Priority (lower number = runs first)

If running a specific test that has dependencies, auto-include them.

### Step 3: Execute Each Test

For each test case, for each step:

1. **Substitute variables** — replace `{{varName}}` with captured values from previous steps (falls back to `process.env`)
2. **Execute the command** specified in `command`
3. **Capture the output** (stdout and stderr)
4. **Check expectPatterns** — each regex must match somewhere in the output
5. **Check rejectPatterns** — none of these regexes should match
6. **Capture variables** — if `capture` is defined, extract values from JSON output
7. **Record result**: PASS or FAIL with evidence

If a step fails, continue with remaining steps but note the failure.

### Step 4: Judge Results

For each test case, determine overall verdict:
- **PASS** — all steps passed, all patterns matched, no errors detected
- **FAIL** — any step failed, with details on which and why

If a test has `criteria` and `goal` fields, evaluate whether the output semantically satisfies them.

### Step 5: Report

Output a summary table:

```
Test Results
============
TC-RUNTIME-001   Container Startup     PASS  (12.4s)
TC-RUNTIME-002   GPU Detection         FAIL  (step 4: expected "library=CUDA" not found)
TC-INFERENCE-002 API Inference Test    PASS  (8.1s)

Summary: 2/3 passed
Duration: 28.6s
```

If any test failed, show:
- Which step failed
- Expected vs actual output (truncated)
- Suggested fix or investigation

### Alternative: Use the CLI

Instead of agent-based execution, use the built-in CLI:

```bash
cd cicd/tests
npm test                         # All tests (simple judge — fast, no model)
npm test -- --suite runtime      # Specific suite (build|runtime|inference|models)
npm test -- --id TC-RUNTIME-001  # Specific test
npm test -- --dry-run            # Preview only
JUDGE_MODE=dual npm test         # Opt in the agent judge (env, not a flag)
```

**Environment variables for CI:**
- `JUDGE_MODE` — `simple` (default) or `dual` (opt in the agent judge)
- `JUDGE_AGENT` — Command for the ACP agent the judge drives; unset uses the bundled Claude ACP agent (keyless). Set it to swap model/vendor
- `CLAUDE_CODE_OAUTH_TOKEN` — Authenticates the bundled Claude agent on a GitHub-hosted runner; unneeded on a self-hosted runner logged into Claude Code (`~/.claude`)

---

## WHICH TESTBED

Two self-hosted runners. They are **independent hosts with no network between them**, so a
Docker tag on one names nothing on the other. Select the host with `runner_label`; move the
image through a registry, by digest.

```
                    ┌──────────────────────────────────────────────┐
                    │  Docker Hub — public, pull needs no auth      │
                    │  dogkeeper886/ollama37-ci:ci-<sha>            │
                    │                      @sha256:<digest>        │
                    └───────▲──────────────────────────┬───────────┘
                    publish │                          │ fetch, by digest
                   (sm37 only)                         │ (BOTH hosts)
   ┌────────────────────────┴──────┐      ┌────────────▼──────────────────┐
   │ sm37   rocky9-k80-cicd-1      │      │ sm75   rocky9-2060-cicd-1     │
   │ 4× Tesla K80 · cc 3.7         │      │ RTX 2060 · cc 7.5             │
   │ driver 470 · 11.4 GiB per die │      │ driver 580 · 5.1 GiB usable   │
   │ GPU_TEMP_LIMIT=80             │      │ GPU_TEMP_LIMIT=85             │
   │ THE REFERENCE TESTBED         │      │ the only tensor-core testbed  │
   ├───────────────────────────────┤      ├───────────────────────────────┤
   │ build      the build itself   │      │ build      no — hours, no cache│
   │ runtime    yes                │      │ runtime    yes                │
   │ inference  yes                │      │ inference  see note below     │
   │ models     yes (16 cases)     │      │ models     never — VRAM       │
   │ perf       yes                │      │ perf       models that fit    │
   └───────────────────────────────┘      └───────────────────────────────┘

   fetch on BOTH, including the host that built. Testing a local build on sm37
   while sm75 tests a pulled image compares two artifacts, not one.
```

**Why a digest, never a tag.** A tag can be overwritten; a digest cannot. That is the only
thing making "sm37 says no-harm" and "sm75 says fixed" statements about the same bits.

**`runner_label`.** Today only `test-runtime.yml` takes it (`workflow_dispatch` and
`workflow_call`, default `sm37`). Every other workflow is still pinned to `sm37`. Never use a
bare `runs-on: self-hosted` — a K80 sweep landing on a small consumer card produces numbers
that look valid and are not. See `.claude/rules/workflow-patterns.md`.

```bash
gh workflow run test-runtime.yml -f runner_label=sm75
```

**Per-host knobs live on the host.** `cicd/scripts/gpu-temp-guard.sh` reads `GPU_TEMP_LIMIT`
from the environment; each runner sets it in `~/actions-runner/.env`. Don't change the script
default, and don't add a workflow input for it.

**What a suite actually runs.** `docker/docker-compose.yml` reads the local tag
`ollama37:latest`. To test a specific build, pull it by digest and retag to that name — no
compose change is needed. The publish and fetch steps belong in **testcases** (see
`ci-testcase`), driven by these workflows with `--id`; they are not yet written.

**`inference` on sm75 is currently blocked.** `TC-INFERENCE-001` pulls `gemma3:4b`, an
architecture on ollama's flash-attention whitelist. On cc ≥ 7.5 the stock compose auto-enables
FA and the runner panics — issue #385. Once that lands, `test-inference` on sm75 becomes its
regression test.

---

## OUTPUT

Test results summary with pass/fail status for each test case.
