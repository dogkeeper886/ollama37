# muse-glimmer on the K80 — throughput (#481)

| | |
|---|---|
| Build | branch `issue-481-muse-glimmer`, built with `build-local.sh`, version 2.3.3 |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470 |
| Measured | 2026-10-05, `cicd/tests/throughput.ts` and `models.ts`, default context (4096) |

| Prompt | Prefill tok/s | Decode tok/s | In / out tokens | GPU | VRAM |
|---|--:|--:|---|--:|---|
| `speech-rewrite` (throughput.ts) | 31.84 | 3.91 | 924 / 100 | 100 % | 18,700 MiB, 2 dies |
| `short-answer` (models.ts) | 18.73 | 4.07 | 69 / 131 | 100 % | 2 dies |
| `image`, 224 px (models.ts) | 20.98 | 4.05 | 138 / 54 | 100 % | 2 dies |

A dense 28B on the K80 decodes at about 4 tok/s, near `gemma3:27b` (4.45) and above
`qwen3.8:27b` (3.4). An image at the 1024-token cap (1200 × 900 px) peaked at 10.2 GB on
the output die.
