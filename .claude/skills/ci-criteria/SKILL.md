---
name: ci-criteria
description: Creating or editing a testcase under cicd/tests/testcases/build/*.yml means writing its criteria block, the part a reader and the judge use to tell what the test asserts. Each agent writes that block from its own idea of what matters, so the blocks come out in different shapes. A reviewer cannot tell what the test asserts without reading its steps, and the block drifts toward restated commands, bare ranges and notes. Use this skill for the one formula every criteria block follows.
user-invocable: true
---

# Criteria formula

- `Verify <subject> <claim A> and <claim B>:`
- `- <requirement> (<probe> shows "<literal>")`
- `- <requirement> (<probe> shows "<literal>")`
