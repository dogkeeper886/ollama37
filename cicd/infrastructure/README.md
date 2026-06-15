# CI/CD Infrastructure

The test framework's semantic judge is the **keyless ACP agent judge** — it spawns a
Claude agent over the Agent Client Protocol and needs **no separate container or model**.

## Agent judge auth

Authentication is keyless — no `ANTHROPIC_API_KEY`:

- **Local runs:** the agent uses the runner's `~/.claude` login.
- **CI:** set `CLAUDE_CODE_OAUTH_TOKEN` as a secret in the `cicd-1` environment; the
  workflows pass it to the judging steps.

Enable the judge per run with `JUDGE_MODE=dual` (the structured suites) or the perf
subcommands' `--judge` flag. With it off (the default), tests run the deterministic
SimpleJudge only.

## History

The old reference judge — a separate Ollama instance running a custom judge model, defined
by `docker-compose.judge.yml` + `Modelfile.judge` — was removed in #237. The agent judge
replaces it.
