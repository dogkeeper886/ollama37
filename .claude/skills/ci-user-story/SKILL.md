---
name: ci-user-story
description: Creating or editing a testcase under cicd/tests/testcases/build/*.yml means writing its intent.user_story, the one line that says what the test is for. Each agent writes that line from its own idea of what matters, so the lines come out in different shapes. A reader cannot tell the test apart from its neighbours, and the line drifts toward mechanism, rationale and multi-line notes. Use this skill for the one formula every user_story follows.
user-invocable: true
---

# User story formula

- `<verb> <subject> <claim> <qualifier>`
- `<verb>`: `Verify` when the test observes, the action verb itself when the test performs
- `<qualifier>`: what would make it fail — `(<what must be present>)` or `<the outcome it must reach>`
- one line, present tense, no trailing period, no probe, no command, no reason
