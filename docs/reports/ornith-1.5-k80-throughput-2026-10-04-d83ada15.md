# ornith-1.5 on the K80 — throughput (#480)

| | |
|---|---|
| Build | `d83ada15` (branch `issue-480-ornith-1.5`), built with `build-local.sh`, version 2.3.2 |
| Host | 4 × Tesla K80 dies (sm_37), CUDA 11.4, driver 470 |
| Measured | 2026-10-04, `cicd/tests/throughput.ts`, the `speech-rewrite` prompt, 100 output tokens, default context |

| Model | Arch | Prefill tok/s | Decode tok/s | In / out tokens | GPU | VRAM |
|---|---|--:|--:|---|--:|---|
| `ornith-1.5:9b` | qwen35 | 110.79 | 9.11 | 879 / 100 | 100 % | 7,620 MiB, 1 die |
| `ornith-1.5:35b` | qwen35moe | 85.73 | 10.89 | 879 / 100 | 100 % | 23,631 MiB, 3 dies |
| `ornith:35b` (1.0, baseline) | qwen35moe | 85.27 | 10.77 | 937 / 100 | 100 % | 22,748 MiB, 3 dies |

`ornith-1.5:35b` runs at `ornith:35b`'s speed: same architecture, same three-die split.
All three replies were checked and judged coherent. At 100 output tokens every reply is
still reasoning, so the judge read the thinking, not an answer.
