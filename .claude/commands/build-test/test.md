Run the ollama37 test framework locally. Reference `.claude/skills/test/SKILL.md` for context.

If the user specifies a suite, run that suite. Otherwise ask which suite or run all.

## Run tests

```bash
# Install dependencies (first time)
cd cicd/tests && npm install

# Run all tests
npm run test

# Run specific suite
npm run test -- --suite build
npm run test -- --suite runtime
npm run test -- --suite inference
npm run test -- --suite models

# Run with LLM judge enabled (default is simple judge only)
npm run test -- --llm

# List available tests
npm run list
```

## LLM judge setup (first time)

```bash
# Start the judge container (stable reference Ollama on port 11435)
cd cicd/infrastructure
docker compose -f docker-compose.judge.yml up -d

# Pull the judge model
curl -X POST http://localhost:11435/api/pull -d '{"name": "gemma3:4b"}'
```

## Stop LLM judge

```bash
cd cicd/infrastructure
docker compose -f docker-compose.judge.yml down
```
