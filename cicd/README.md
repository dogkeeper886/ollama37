# cicd

The test harness is being rebuilt (#542): three scripts, one call path to the model that always runs the check and the agent judge, and one file for every prompt and judge question.

The previous harness (the Vitest suites, the `cli.ts` commands, their workflows, specs and skills) is in [`archive/v2-test-framework/`](../archive/v2-test-framework/). Nothing runs it.

What is still live:

| Path | What |
|---|---|
| `scripts/scrub-session-log.sh` | session-log gate (#532): redact a log before it goes into `.sessions/` |
| `scripts/test-mlx-smoke.sh` | MLX-on-K80 probe, run by `.github/workflows/test-mlx-smoke.yml` |
| `scripts/claude-ollama.sh` | run Claude Code against a local ollama37 server |
| `infrastructure/` | runner and host notes |
| `.github/actions/identify-host`, `yield-gpu`, `restore-gpu` | host attribution and GPU hand-over, kept for #542's shell layer |
| `.github/workflows/release-docker.yml` | builds and publishes the image on a release |
