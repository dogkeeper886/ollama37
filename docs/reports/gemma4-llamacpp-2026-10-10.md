# gemma4:12b on the llama.cpp engine (#568)

| | |
|---|---|
| Build | branch `study-568-gemma4-llamacpp` on `176ee542`, version 2.3.3-568 |
| Baseline | CI run 37924854227 (#573's code, the same as `main`), same host |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470.256.02 |
| Measured | 2026-10-10 |

gemma4:12b ships an audio projector (`gemma4ua`), which Ollama's engine can't serve, so gemma4:12b alone
runs on the vendored llama.cpp. Three bugs on that path hit it, and none was K80-specific.

## 1. Tokenizer: every space was its own token

gemma4's tokenizer is SPM-style BPE: its `tokenizer.json` replaces spaces with `▁`, merges on raw UTF-8,
and falls back to `<0xHH>` byte tokens. The Ollama engine does that. llama.cpp marked gemma4 this way but
its BPE session still ran GPT-2 byte-level encoding on regex-split words, so each space became the token
`Ġ` and words lost their `▁` prefix:

```
before  "Fellow" "Ġ" "country" "men" ":" "Ġ" "At" "Ġ" "this" ...   1,698 tokens
after   "Fellow" " countrymen" ":" " At" " this" ...                 857 tokens
```

**Fix** (`llama-vocab.cpp`): gemma4 escapes spaces to `▁` and merges the whole text with no regex split or
byte-level step, and falls back to `<0xHH>`. `find_bpe_rank` allows gemma4's merges with raw newlines.

**Check:** a lab program tokenized five texts with both engines' tokenizers on gemma4:12b's GGUF. The token
ids match exactly on every text:

| Text | Tokens (both engines) |
|---|--:|
| the speech prompt | 857 |
| `farewell.txt` | 7,096 |
| code, tabs, CRLF, CJK, emoji, NBSP, zero-width space | 58 |
| `a\nb` | 3 |
| one 47 KB line | 11,200 |

## 2. Long-line crash

The regex split ran through `std::regex`, which recurses once per character and overflowed the stack on lines
over ~37 KB. gemma4 no longer uses a regex split, so the crash is gone: on `ollama.service` a 47 KB
single-line prompt reads as 11,230 tokens and gets a normal reply.

## 3. Load crash: "exit status 2"

Captured with a temporary log drop-in: the runner died with **SIGFPE in `clip_graph`** while loading the
audio projector. `clip_hparams::patch_size` is never initialized, the audio projector never sets it, and
`clip_graph` divides by it. Whenever that memory held 0 the load failed, so the failure came and went with
whatever memory the runner got.

**Fix** (`clip.cpp`): the gemma4 audio projector sets `patch_size = 1`. Its graph doesn't use it.
**Check:** six loads in a row at 4k and 12k context, no failure. Before the fix, 4 of 6 failed.

## 4. VRAM estimate

`GraphSize` sized all 48 layers as full-context layers at head dim 512. 40 of them are sliding-window
layers, which cache only the 1,024-token window at head dim 256 with 8 KV heads; the 8 global layers have 1
KV head. **Fix** (`fs/ggml/ggml.go`): gemma4's sliding-window layers are sized by window, head size and
heads, as llama.cpp allocates them. Only the llama.cpp path uses this estimate.

| gemma4:12b | Estimate before | Estimate after | Actual (nvidia-smi) | After ÷ actual |
|---|--:|--:|--:|--:|
| context 4,096 | 17.3 GiB | 15.1 GiB | 13.3 GiB | 1.14× |
| context 12,288 | 26.6 GiB | 15.2 GiB | 14.0 GiB | 1.09× |

The model now loads on 2 dies instead of 3. The remaining over-estimate at 4k is in the graph term, which
the llama.cpp path multiplies by 2.5 for every model (`llm/memory.go`).

## On the host

`farewell.txt` at 12,288 context reads as 7,121 tokens (the other gemma4 models read 7,139–7,140), fits
the window, and gemma4:12b summarizes it from the opening paragraphs.

## CI

**Pipeline** [37968154650](https://github.com/dogkeeper886/ollama37/actions/runs/37968154650): build,
deploy and canary pass. The models test fails 1 of 29: gemma4:12b's audio test. The model answered "la"
on `main` and on this branch, after thinking through the same steps. The judge sees only "la" and passed it
on `main` but flagged it here. The model's thinking on `main` began with a leaked raw `<|channel>thought`
token; on this branch it doesn't.

**Standard throughput** [37976473036](https://github.com/dogkeeper886/ollama37/actions/runs/37976473036)
against [37924854227](https://github.com/dogkeeper886/ollama37/actions/runs/37924854227): all 25 replies
pass the check and the judge.

| gemma4:12b | Before | After |
|---|--:|--:|
| Prompt tokens | 1,714 | 873 |
| Prefill tok/s | 67.3 | 71.4 |
| Decode tok/s | 10.77 | 11.16 |

- **Prompt tokens:** gemma4:12b now reads the same 873 tokens as gemma4:26b, and 872 for gemma4:31b and e2b.
- **gemma4:12b's reply changes,** as it should now that its prompt is tokenized as the model was trained, and
  reads coherently.
- **The other 24 replies are word for word identical.**

## Limits

- **Other audio projectors** (whisper family, ultravox) leave `patch_size` unset too; none of the CI models
  uses them.
- **One CI run per row.**
