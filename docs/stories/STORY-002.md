# STORY-002: Trust the keyless agent judge against real inference output

## User Story

As a maintainer of ollama37 (and of agent-workflows-runner),
I want the new keyless, vendor-neutral ACP agent judge to grade real LLM inference output from this repo,
So that I can trust it to semantically judge model responses without depending on the old hardcoded Ollama reference instance.

## The Need

ollama37 produces genuine LLM inference output, but it is currently graded by the
old judge, which is wired to one specific Ollama reference instance. The
agent-workflows-runner project has a newer ACP agent judge
that is keyless and vendor-neutral, and we need confidence it actually works before
relying on it. Because ollama37 emits real model output, it is the natural proving
ground — this is where dogkeeper886/agent-workflows-runner#46 gets validated against
something real rather than a mock.

## Success Looks Like

- Real model output from this repo is graded by the keyless agent judge.
- The judge **passes** a genuinely good response.
- The judge **fails** a deliberately wrong response — proving it is judging meaning,
  not rubber-stamping.
- There is captured evidence of both verdicts that a human can look at and believe.

## Open Questions

- How the stack is brought up so real inference output is available to grade
  (Docker stack / K80, `/api/generate` reachable).
- How the agent judge and its ACP dependencies are ported into this repo and run
  keyless.
- Which inference test(s) to point at a real prompt, and how dual-mode grading is
  exercised.
- What form the captured evidence takes.

## Status

- Created: 2026-06-14
- Issues: #227
- Plan: #231
