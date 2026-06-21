# Which GPUs *could* this project support? (Driver 470 arch map)

**Related**: [#223](https://github.com/dogkeeper886/ollama37/issues/223) (K80 + P100 mixed-GPU crash)
**Date**: 2026-06-21
**Status**: Largely realized. The 9-arch build shipped in [#341](https://github.com/dogkeeper886/ollama37/issues/341) — the `CUDA 11 K80` preset now compiles native cubin for `sm_37`–`sm_86` (Maxwell→Ampere). Only the K80 is hardware-validated; Pascal→Ampere are built-but-unvalidated (tracked in [#342](https://github.com/dogkeeper886/ollama37/issues/342)). This remains the reference map for that work.

## TL;DR

The project is pinned to the NVIDIA **470 driver** and **CUDA 11.4** because that's the
only combination that still runs the Tesla K80. That same combination can *physically*
run a lot more than the K80 — everything from Kepler up to Ampere (A100-class cards).

But "the card can run code" and "the project runs *correctly* on the card" are two
different things. There are **two layers**, and both have to line up:

1. **Build layer** — did we compile machine code for this GPU's architecture? If not,
   the first kernel launch crashes with XID 43 (this is exactly what happened to the
   P100 in #223).
2. **Code layer** — the kernels contain `if (compute_capability >= X)` switches that
   pick a fast path or a fallback. Compiling for a new card *turns those switches on*,
   and some of those paths have never been exercised on this project's one tested card
   (the K80). So each new architecture needs its switches traced and its output checked.

So: yes, in theory we can support every card the 470 driver supports. The build half is
easy. The code half — the "sm gates" — is the real work.

## The two layers, in one picture

```
  YOU WANT TO SUPPORT A NEW CARD (say, a P100)
                 |
   ┌─────────────┴──────────────┐
   |                            |
 LAYER 1: BUILD               LAYER 2: CODE GATES
 "does machine code           "is the code that runs
  exist for this card?"        actually right for it?"

 CMAKE_CUDA_ARCHITECTURES     cc >= GGML_CUDA_CC_*  (in the CUDA kernels)
 in CMakePresets.json         FlashAttentionSupported() (in ml/device.go)
   |                            |
   | miss the arch →            | wrong branch taken →
   v                            v
   XID 43, instant crash        runs, but maybe slow
   (the #223 P100 bug)          or subtly wrong output
```

Layer 1 is a one-line list. Layer 2 is where you have to think.

## What the 470 + CUDA 11.4 stack can run

The limit is the **overlap** of two things:

- **Driver 470** is the *last* driver that still supports the K80 (Kepler). It runs
  everything from Kepler up through Ampere.
- **CUDA 11.4** is the newest toolkit that pairs with the 470 driver, and it can *compile*
  for those same architectures (sm_37 up to sm_87).

Newer cards — Ada (RTX 40-series) and Hopper (H100) — need a newer driver and CUDA 12+,
which would **drop the K80**. That trade is the whole reason this project sits where it does.

```
   OLDER ───────────────────────────────────────────────────────► NEWER
   Kepler   Maxwell   Pascal   Volta   Turing   Ampere │ Ada   Hopper
   3.5 3.7  5.0 5.2   6.0 6.1  7.0     7.5      8.0 8.6 │ 8.9   9.0
   K40 K80  M40 M60   P100     V100    T4       A100    │ RTX40 H100
            GTX9xx    P40                       A40     │ L4
   └──────────── 470 driver + CUDA 11.4 cover this ─────┘ └─ needs 520+/CUDA12 ─┘
                                                            (would drop the K80)
```

| Architecture | Compute cap (sm) | Example data-center cards | In range? |
|---|---|---|---|
| Kepler | 3.5 / 3.7 | K40, **K80** | ✅ (the one we test) |
| Maxwell | 5.0 / 5.2 | M40, M60 | ✅ |
| Pascal | 6.0 | **P100** | ✅ |
| Pascal | 6.1 | **P40, P4** (+ GTX 10-series) | ✅ |
| Volta | 7.0 | **V100** | ✅ |
| Turing | 7.5 | **T4** | ✅ |
| Ampere | 8.0 | **A100** | ✅ |
| Ampere | 8.6 | **A40, A10, A16, A2** | ✅ |
| Ada / Hopper | 8.9 / 9.0 | RTX 40-series, L4, H100 | ❌ needs newer driver |

(Embedded Tegra/Jetson parts — 5.3, 6.2, 7.2, 8.7 — are technically in the toolkit's
range but aren't relevant to a desktop 470 driver, so they're left off.)

## The code gates ("sm gates") you'd have to trace

These are the `if (compute_capability >= X)` switches inside the CUDA kernels. Compiling
for a card decides which side of each switch it lands on. A ✅ means the hardware feature
is present; the support question is whether the project's code handles that crossing.

| Gate (where it lives) | Turns on at | K80 3.7 | Maxwell 5.x | P100 6.0 | P40/10xx 6.1 | V100 7.0 | T4 7.5 | Ampere 8.x |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FP16 math (`common.cuh:250`) | 6.0 | ✗ | ✗ | ✅ | ✅ | ✅ | ✅ | ✅ |
| *Fast* FP16 (`common.cuh:254,289`) | 6.0 but **not 6.1** | ✗ | ✗ | ✅ | ✗* | ✅ | ✅ | ✅ |
| dp4a int8 dot (`common.cuh:520`) | 6.1 | ✗ | ✗ | ✗ | ✅ | ✅ | ✅ | ✅ |
| FP16 tensor cores (`common.cuh:295`) | 7.0 | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ | ✅ |
| int8 tensor cores (`common.cuh:308`) | 7.5 | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ |
| Ampere MMA / async copy (`common.cuh:312`) | 8.0 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ |
| Flash attention (`ml/device.go:440`) | 7.0 (not 7.2) | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ | ✅ |

\* sm_6.1 has FP16 hardware but it's deliberately rate-limited, so upstream excludes it
from the "fast" path on purpose. Not a bug — a real subtlety to preserve.

The flash-attention gate is now pure-upstream (`cc >= 7.0`, ≠7.2): FA is **off** for the
K80 and Pascal, **on** for Volta and up. The fork once carried a hand-written K80
special-case (issues #108 / #112) but it was removed in #350 after a benchmark showed FA
is a net slowdown on the K80 — so no sub-7.0 card, including the P100, gets FA.

## What this means in plain terms

The gates sort the cards into three buckets:

- **Volta and newer (7.0+) — basically free.** These are the cards upstream ggml targets
  every day, so the tensor-core and flash-attention paths are well-worn. Add them to the
  build list and they mostly just work. The K80 patches don't get in the way — they all
  live behind `cc < 7.0` branches.

- **Pascal (6.0 and 6.1) — the real gap.** These switch on FP16 and dp4a code paths that
  the **K80-only build never even compiles**, so nothing here has run them. (The #223 P100
  crash itself was the build layer — no sm_60 cubin — not a code gate.) Pascal also sits
  below the pure-upstream FA gate, so it runs FA-off like the K80. This is the bucket that
  needs careful tracing and a real test on the hardware.

- **Maxwell (5.x) — easy but dull.** It sits below every feature gate, so it rides the
  exact same plain-vanilla code paths the K80 already uses. Low risk, low reward.

### One landmine to remember

The flash-attention gate excludes compute **7.2** on purpose
(`!(major == 7 && minor == 2)` in `ml/device.go:440`). That's the Jetson Xavier. If
embedded parts ever come into scope, that exclusion will bite — it looks like a card that
*should* qualify (it's ≥ 7.0) but is intentionally turned off.

## How you'd actually add a card (the recipe)

1. **Build:** add its `sm` number to `CMAKE_CUDA_ARCHITECTURES` (keep it **real**, not
   `-virtual`, so there's no slow first-run JIT). → check: the card boots a small model
   without XID 43.
2. **Trace the gates:** walk the table above for that card's compute capability and note
   every gate it newly crosses. Each crossing is a code path that hasn't run here before.
3. **Validate on real hardware:** run a known prompt, compare output against a card we
   already trust (or against CPU). Bit-exact or clearly-equivalent output = the crossed
   paths are sound. This is the step the maintainer can't do without owning the card —
   which is why support is a hardware commitment, not just a code change.

## References

- Build presets: `CMakePresets.json` (the `CUDA 11 K80` preset compiles `37-real;…;86-real` as of #341)
- Arch fallback logic: `ml/backend/ggml/ggml/src/ggml-cuda/CMakeLists.txt:8-44`
- Code gates: `ml/backend/ggml/ggml/src/ggml-cuda/common.cuh:80-85` (thresholds),
  `:250-314` (the feature switches), `:520` (dp4a)
- Flash-attention gate: `ml/device.go:436-453`
- Runtime "does this GPU work" filter: `discover/runner.go:154`
- K80 background: `docs/research/k80-quant-audit.md`, `docs/research/k80-fusion-audit.md`
- Toolchain pins: `docker/builder/Dockerfile` (CUDA 11.4, GCC 10), `docker/runtime/Dockerfile`
