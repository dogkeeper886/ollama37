# K80 model sizing reference

Which model to run on the Tesla K80 testbed, by how much GPU memory you have.

## The hardware

Each K80 board is two GK210 dies, ~11.4 GiB usable each; this host has two boards
(4 dies). Models split across dies by layer, so the practical question is **how many
dies a model needs** — weights plus KV cache and context. Three tiers matter here;
nothing in the current set needs a fourth die.

| Tier | Dies | ~VRAM | Weight budget |
|---|:--:|--:|---|
| Single die | 1 | 11.4 GiB | ≲ 10 GB |
| Two dies | 2 | 22.8 GiB | ~13–20 GB |
| Three dies | 3 | 34.2 GiB | ~24 GB |

The K80 is a for-fun target — even the two- and three-die models are slow. This table
is for coverage and comparison, not production throughput.

## How to read it

One representative model per family per tier — **Model · Size · what it Represents**.
Sizes are download size (Q4). Two comparison axes are kept intact:

- **Generation** — Gemma 3 vs Gemma 4, Qwen 3.5 vs Qwen 3.6 (`⟷` marks the pair).
- **Like-size** — the same nominal size across generations (Gemma 12b↔12b, Qwen
  27b↔27b and 35b↔35b).

Code-path-only models (`qwen3-vl`, `ornith`, `functiongemma`, qwen3-coder) and
redundant footprints are left out — this is a sizing reference, not a code-path
inventory.

## Single die — ~11.4 GiB

| Model | Size | Represents |
|---|--:|---|
| `gemma3:4b` | 3.3 GB | Gemma 3 — small |
| `gemma3:12b` | 8.1 GB | Gemma 3 — 12B `⟷` |
| `gemma4:12b` | 7.6 GB | Gemma 4 — 12B `⟷` · vision + audio |
| `gemma4:e4b` | 9.6 GB | Gemma 4 — small (native vision) |
| `qwen3.5:9b` | 6.6 GB | Qwen 3.5 — small |
| `deepseek-r1:14b` | 9.0 GB | DeepSeek-R1 — largest 1-die |
| `ministral-3:14b` | 9.1 GB | Ministral-3 — largest 1-die |
| `lfm2.5:8b` | 4.8 GB | LFM2.5 — fun |

## Two dies — ~22.8 GiB

| Model | Size | Represents |
|---|--:|---|
| `gemma3:27b` | 17 GB | Gemma 3 — large `⟷` |
| `gemma4:31b` | 20 GB | Gemma 4 — large `⟷` |
| `qwen3.5:27b` | 17 GB | Qwen 3.5 — 27B `⟷` |
| `qwen3.6:27b` | 17 GB | Qwen 3.6 — 27B `⟷` |
| `deepseek-r1:32b` | 20 GB | DeepSeek-R1 — large |
| `gpt-oss:20b` | 13 GB | gpt-oss — fun |

## Three dies — ~34.2 GiB

| Model | Size | Represents |
|---|--:|---|
| `qwen3.5:35b` | 24 GB | Qwen 3.5 — XL `⟷` |
| `qwen3.6:35b` | 24 GB | Qwen 3.6 — XL `⟷` · showcase |

## Notes

- **gemma4 is not one size ladder.** `12b` (vision **and** audio, separate clip-mmproj
  projector) and `e4b` (native vision only) are different capabilities, not
  interchangeable — both are kept at the single-die tier.
- **Qwen 3.6 has no small model** — its smallest is `27b` (two dies). `qwen3.5:9b`
  stands in for "small Qwen".
- **The three-die tier is Qwen-only** — the `35b` (24 GB) pair. No Gemma or DeepSeek
  reaches it; their largest (~20 GB) fits two dies.
- **Sizes are download size, not runtime VRAM.** Actual dies used include KV cache and
  context and can round a model up a tier at long context — see the measured
  `docs/reports/k80-vram-by-family-*.md` snapshots.
