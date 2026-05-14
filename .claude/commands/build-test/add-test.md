Add a test case to the ollama37 test framework. Reference `.claude/skills/add-test/SKILL.md` for context.

The user will provide: **$ARGUMENTS** (suite name and description of what to test)

## Steps

1. **Identify user story** — Ask the user for the GitHub issue number, or create one if needed.

2. **Determine the suite** — build, runtime, inference, or models.

3. **Find next ID** — Check existing test cases to auto-increment:

```bash
ls cicd/tests/testcases/<suite>/
```

4. **Create the YAML test case** at `cicd/tests/testcases/<suite>/TC-<SUITE>-<NNN>.yml`:

```yaml
id: TC-<SUITE>-<NNN>
name: <Descriptive Name>
suite: <suite>
priority: <1-3>
timeout: <milliseconds>
dependencies: []
issue: <github-issue-number>

intent:
  user_story: |
    <What value this test delivers, in plain prose.>
  acceptance:
    - <What must be true for the test to be considered correct>
  notes: |
    <Optional: prerequisites, gotchas, acceptable warnings>

steps:
  - name: <step description>
    command: <bash command>
    expectPatterns:
      - "<regex pattern>"
    rejectPatterns:
      - "<regex pattern>"

criteria: |
  <Description for LLM judge — only used when running with --llm>

  Expected:
  - <condition 1>
  - <condition 2>
```

5. **Verify** — Run the new test:

```bash
cd cicd/tests && npm run test -- --suite <suite>
```
