# Reading prefill and decode speed in the server log

While a request is still running, the server reports how much of the prompt it has read and
how fast it is producing tokens. On a K80 a single request can take minutes, so this is often
the only way to tell a slow request from a stuck one.

Everything below appears in `docker logs ollama37`.

## The four lines

```
prefill progress seq=0 tokens=1024 remaining=7168 progress=0.13 elapsed=3.346s  tps=306.05
prefill done     seq=0 tokens=8192 elapsed=35.251s tps=232.39
decode progress  seq=0 tokens=40 elapsed=3.007s tps=13.3 tps_recent=13.3
completion seq=0 reason=length prefill_tokens=8192 prefill_elapsed=35.251s prefill_tps=232.39 decode_tokens=150 decode_elapsed=11.391s decode_tps=13.17
```

**`prefill progress`** — the model is reading the prompt. Nothing is generated yet.
`tokens` is how much has been read, `remaining` how much is left, `progress` the fraction
done, `elapsed` how long it has taken, and `tps` the reading rate.

**`prefill done`** — the prompt has landed. Generation starts next.

**`decode progress`** — tokens are being produced. `tps` is the average since generation
started; `tps_recent` covers only the last few seconds, so a model slowing down as it fills
its cache shows up here first.

**`completion`** — one line per request, whatever ended it. `reason` is `stop` (the model
finished), `length` (hit the token limit), or `closed` (the client hung up mid-request).

`seq` identifies the request. It is always `0` unless `OLLAMA_NUM_PARALLEL` is raised above
its default of 1.

## Silence is not a stall

Progress lines are throttled: nothing is reported until a phase has run for **3 seconds**,
and then at most one line every 3 seconds. A fast request emits only the `completion`
line. That is deliberate — without it a long prompt would print a line for every chunk it reads.

## `tokens` counts work done, not prompt size

When part of a prompt is already in the cache from an earlier request, that part costs no
computation. The log counts only what actually went through the GPU, so on a repeat request
the numbers look surprisingly small. That is correct.

Measured on a K80, the same 2410-token prompt sent twice:

```
  the prompt the client sent: 2410 tokens
  ┌───────────────────────────────────────────────┬───┐
  │  2409 already cached — no work this request    │ 1 │
  └───────────────────────────────────────────────┴───┘
                                                    ▲
                                          only this was computed
                                             elapsed = 128 ms

  the API says   prompt_eval_count    = 2410  ┐
                 prompt_eval_duration = 128ms ├─▶ 18,800 tokens/s  ← impossible
                                              ┘

  the log says   prefill_tokens  = 1          ┐
                 prefill_elapsed = 128ms      ├─▶ 7.8 tokens/s     ← what happened
                                              ┘
```

The API pairs a whole-prompt count with an uncached-work duration, because the count
is taken before the cached part is subtracted. Dividing one by the other gives a rate no
hardware could reach.

**Which to trust:** the log's `prefill_tokens` for speed, the API's `prompt_eval_count` for
how large the prompt was. They answer different questions and the field names differ on
purpose.

## Cache reuse depends on the model

Some models cannot reuse a cached prompt at all, so every request pays full price. Observed
on a K80:

| Model family | Reuses a cached prompt? |
|---|---|
| `gpt-oss` | Yes — a 25k conversation read only the 1–1500 new tokens each turn |
| `deepseek-r1` | Yes |
| `gemma3`, `gemma4` | No — a repeated 2417-token prompt was re-read in full |
| `qwen3.5`, `qwen3.6`, `ornith` | No — a repeated 2475-token prompt took 26.3s again, against 26.5s the first time |

For the families that cannot reuse, a long conversation re-reads its whole history every
turn. At the ~50 tokens/s a 20B model manages on a K80, a 25k-token history costs about
**eight minutes per turn**. If a run feels far slower than the generation rate suggests, this
is the first thing to check: watch whether `prefill_tokens` stays large turn after turn.

## The two engines measure prefill differently

This fork runs two inference engines, and `prefill_elapsed` does not mean the same thing in
each. Which engine serves a model is fixed by its architecture — see
`OllamaEngineRequired` in `fs/ggml/ggml.go`.

```
                    request arrives
                          │
                          │  waiting for a free slot
                          │  (excluded by both engines)
                          ▼
                  reading starts
                          │
     ┌────────┐  gap  ┌────────┐  gap  ┌────────┐
     │ chunk1 │═══════│ chunk2 │═══════│ chunk3 │ → first token
     └────────┘       └────────┘       └────────┘

  llama.cpp engine  ▓▓▓▓▓▓        ▓▓▓▓▓▓        ▓▓▓▓▓▓
                    └─ adds up the reading only, gaps excluded ─┘

  Go engine         ▓▓▓▓▓▓═══════▓▓▓▓▓▓═══════▓▓▓▓▓▓
                    └─ clock time until the first token, gaps included ─┘
```

The difference is small when one request runs at a time, and grows when several compete. This is not a discrepancy to fix. Each engine reports exactly what it also reports through
the API. And the two are built differently: one reads the prompt in steps that finish before
the next begins, so each can be timed on its own; the other overlaps reading with preparing
the next chunk, leaving no separate interval to time.

**In practice:** a model always uses one engine, so comparing one model across settings is
safe. Comparing `prefill_tps` between two models on different engines compares two
definitions.

## Smaller caveats

**Concurrent requests read low.** Every request in flight is charged the full time of each
shared step, so with several at once each one's `tps` understates the hardware. Only
`OLLAMA_NUM_PARALLEL=1` — the default, and the only setting these lines have been checked
against — is free of this.

**Short replies overstate the rate.** `decode_tps` covers one step fewer than
`decode_tokens` counts, because the first token's work is charged to prefill. A one-token
reply therefore reads `decode_tokens=1 decode_elapsed=0s decode_tps=0`. This matches what
the API reports, which is why it is left alone.

**A request can end without a client.** `reason=closed` means the caller went away. The
server notices only once it is generating, so a client that leaves while the prompt is still
being read is not noticed until the reading finishes. Until then the work continues, holding
the GPU for a request nobody is waiting for.

## Where these numbers come from

`runner/common/progress.go` emits the lines; both runners call it. The durations are the same
values the API returns as `prompt_eval_duration` and `eval_duration`, so the log and an API
response for the same request agree.

For benchmarking guidance, see [Porting upstream code to the K80](./porting-k80.md).
