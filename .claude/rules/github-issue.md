# Writing a GitHub issue

An issue is read by someone with **no memory of the conversation that produced it** — a
teammate months later, or an agent with a fresh context window. Write for that reader.

## Mirror the template

`gh issue create --body-file` **bypasses** `.github/ISSUE_TEMPLATE/*.yml`. The template's
sections are still required — reproduce them by hand, reading the fields from the template
itself rather than from memory:

| Kind | Template |
|---|---|
| Something is broken | `bug.yml` |
| Change existing behaviour | `enhancement.yml` |
| New capability | `feature.yml` |
| Take something out | `removal.yml` |

Don't restate a template's fields here — read the file. It is the source of truth.

## Add, on top of the template

**Goal.** One sentence: what "done" looks like. The outcome, not the fix.

**Plain words first.** The opening paragraphs must land for a reader who doesn't know CUDA.
Jargon and `file:line` belong further down, not in the first thing they read.

**The path.** Name the **model**, the **hardware**, and the **config**, then trace the chain to
the failure with `file:line` at each hop. A problem statement without a path is a rumour; a path
can be fixed by anyone.

```
# example — the real one lives in #385
Model: <tag> (arch <name>)   Hardware: <card>, cc <x.y>   Config: <what was set>
  <entry point> → … → <dispatch> → <failing call> → <observed failure>
```

**Observed vs predicted.** Label every claim; never let them blur.
- *Observed* — you ran it and read the output or the log.
- *Predicted* — you read the source and reasoned. Say so, and say what would confirm it.

**Blast radius.** Which cards, which architectures, which engine. Be exact — a compute-capability
threshold is a number, not a vibe. State what is **untested** rather than implying it is safe.

**Scope / Non-goals.** What this issue does *not* do. The recurring ones here:
- Don't edit upstream gates (`fs/ggml/`, `ml/`) — the next re-sync overwrites them.
- Vendored `llama/llama.cpp/**` changes are a **port** into the numbered patch series
  (`llama/patches/`), never a cherry-pick or a vendor bump. See `docs/porting-k80.md` §1.
- Don't adapt the toolchain to the code. The CUDA version is load-bearing: it is what keeps
  sm_37 alive.

**Acceptance criteria.** Checkboxes. Always include the **K80 no-harm gate** — the K80 is the only
hardware-validated target, so any change must leave its sweep unmoved. Use the tolerance and
methodology recorded in `docs/reports/k80-vram-by-family-*.md`; don't invent a number here. If a
change *should* be a no-op there, say so and make it a criterion.

**Unverified.** An honest list of what you did not prove. An issue that claims more certainty than
it earned is worse than one that admits a gap.

**Links.** Parent issue, the comment holding the evidence, any plan or research doc.

## The two failures this prevents

1. **A confident wrong claim.** Benchmark before asserting an optimization helps
   (`docs/porting-k80.md` §2). Label a hunch as a hunch. Flash attention on the K80 was assumed a
   win and measured a large loss — see #337, #346.
2. **A rumour with no path.** "FA is broken on Turing" is unactionable. The same claim with a
   model, a card, and a call chain is a fix waiting to happen.

## Right-size it

A typo or a dead link does not need this. Use judgment. A crash, a silent wrong answer, or anything
someone will act on months from now does.
