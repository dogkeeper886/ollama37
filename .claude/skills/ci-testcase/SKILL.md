---
name: ci-testcase
description: Generate YAML test cases from requirements or acceptance criteria
user-invocable: true
---

# Create Test Case

Generate a YAML test case file from requirements or acceptance criteria.

$ARGUMENTS

## PURPOSE

Read requirements (story, ticket, or description) and generate YAML test case(s) that verify the acceptance criteria.

---

## AGENT WORKFLOW

### Step 1: Understand Requirements

Input can be:
- A story/requirement file path — read and analyze it
- A description of what to test — use it directly
- A ticket ID — look for related docs or ask the user

Identify:
- What functionality to test
- What commands or tools to call
- What output to expect
- What would indicate failure

### Step 2: Review Existing Tests

Read existing test cases in `cicd/tests/testcases/` to:
- Find the next available ID number per suite
- Follow existing naming and pattern conventions
- Avoid duplicating existing coverage

### Step 3: Generate Test Case YAML

Create test case file(s) in `cicd/tests/testcases/<suite>/` using this format:

```yaml
id: TC-[SUITE]-[NUMBER]
name: Descriptive test name
suite: build|runtime|inference|models
priority: 1-10 (lower = runs first)
timeout: 30000
dependencies: []
tags: [feature-name]

intent:                    # the design authority — why this test exists
  user_story: |
    What value this test delivers
  acceptance:              # optional — observable criteria
    - "..."
  notes: |                 # optional — prerequisites, gotchas, acceptable warnings
    ...

steps:
  - name: Step description
    command: <shell command>
    timeout: 5000              # Optional
    expectPatterns:
      - "regex pattern"
    rejectPatterns:
      - "error"
    capture:                   # Optional: extract values for later steps
      varName: "json.path"

criteria: |
  Human-readable criteria for agent judge evaluation.
```

**Suite guidelines:**
- `build` — toolchain image, runtime image build, image sizes. **Also where the CI image is
  published and fetched**: a testcase may have side effects (`TC-BUILD-002` runs
  `make build-runtime-local-no-cache`). `sm37` only.
- `runtime` — container startup, GPU detection, health, `/api/metrics`. Runs on either testbed.
- `inference` — model pull + GPU inference smoke.
- `models` — per-model regression. `sm37` only: the models exceed any other card's VRAM.

**ID format:** `TC-BUILD-XXX`, `TC-RUNTIME-XXX`, `TC-INFERENCE-XXX`, `TC-MODELS-XXX`

**Tags:** the `tags:` field is optional metadata; the runner selects by `--suite`/`--id` (there is no `--tag` filter).

**For shell command testing:**
```yaml
command: curl -s http://localhost:3000/api/endpoint
```

**For multi-step with variable capture:**
```yaml
steps:
  - name: Create resource
    command: <command that returns JSON>
    capture:
      resourceId: "id"
  - name: Use captured value
    command: <command using {{resourceId}}>
```

Variables resolve from captured step output first, then fall back to `process.env`.

---

### Writing for two testbeds

The suites run on **`sm37`** (4× Tesla K80, cc 3.7, driver 470) and **`sm75`** (RTX 2060,
cc 7.5, driver 580). A testcase is plain shell, executed on whichever runner the workflow
selected. It cannot know which one — so don't let it assume.

**Assert the property, not the device.** A test that names one testbed's hardware fails on
the other while testing nothing more.

```yaml
# WRONG — TC-RUNTIME-002 does this today, and it fails on sm75
expectPatterns:
  - "Tesla K80"        # the device
  - "470\\."           # its driver
  - "compute=3\\.7"    # its compute capability

# RIGHT — the intent is "a CUDA GPU was detected and is in use"
expectPatterns:
  - "library=CUDA"
  - "compute=[0-9]+\\.[0-9]+"
rejectPatterns:
  - "library=cpu"
```

The exact card, driver and compute capability are already recorded per run by the
`identify-host` action. A testcase does not need to restate them.

**Host-specific values come from the environment, never a literal.** Variables fall back to
`process.env`, so a workflow passes a CI image digest, a temperature limit, or a model name
in — the YAML stays host-agnostic.

**Size the test to the smallest testbed it may run on**, or scope it to one suite. `sm75` has
~5.1 GiB of usable VRAM; the `models` suite exceeds that, which is why it is `sm37` only.

**A testcase cannot emit a workflow output.** It prints to stdout and the results JSON. If a
later workflow run needs a value it produced — an image digest, say — that handoff is the
workflow's problem, not the testcase's.

### Step 4: Report

Show the user:
- Test case files created
- What each test validates
- Suggest: `/ci-run` to execute tests

---

## OUTPUT

Paths to created test case files.
