# Audit: do the K80-specific workarounds stay correct on the 9-arch build?

**Issue**: [#343](https://github.com/dogkeeper886/ollama37/issues/343) · part of STORY-014 · plan [#340](https://github.com/dogkeeper886/ollama37/issues/340)
**Date**: 2026-06-21
**Status**: Complete. All three workarounds verified **safe as-is** on non-K80 and mixed-arch boxes. No code change required.

## TL;DR

[#341](https://github.com/dogkeeper886/ollama37/issues/341) widened the build from sm_37-only to the full 470
datacenter sweep (sm_37/50/52/60/61/70/75/80/86). Three workarounds were written
assuming "the only GPU is a K80 (Kepler, compute 3.7; each board = 2 dies of ~12 GB)."
This audit checked whether any of them mis-fires now that the same binary can run on
Pascal→Ampere cards, alone or mixed with a K80.

**Verdict: all three are safe.** None hardcodes K80 geometry; each keys on a per-device
property (compute capability, queried allocation granularity, or measured per-GPU layer
data). No fix is needed for non-K80 / mixed-box correctness.

## 1. CUBLAS two-tier fallback — `d9489265`

**What it does.** The K80 returns `CUBLAS_STATUS_ARCH_MISMATCH` from modern CUBLAS calls.
Two tiers fix it in `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`:
- **Tier 1** (`:1434-1437`, `:2141-2146`): pick `CUBLAS_GEMM_DEFAULT` instead of
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for pre-Volta devices.
- **Tier 2** (`:2171-2190`, `:2232-2251`): for the FP32 batched case, swap to the legacy
  `cublasSgemm*Batched` calls that work on Kepler.

**Gate.** Per-device, `GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA` (cc < 700),
read from the *currently active* device on every call. **Not** `cc == 370`.

**Why it's safe across the sweep.** The boundary sits exactly at Volta (700), the first
tensor-core architecture:

| Card | cc | Path | Correct? |
|---|---|---|---|
| Pascal 60/61, Maxwell 50/52 | <700 | `GEMM_DEFAULT` + legacy batched | ✅ these genuinely lack tensor cores |
| Volta 70 / Turing 75 / Ampere 80/86 | ≥700 | upstream `TENSOR_OP` + `*Ex` | ✅ untouched |

No tensor-core card is demoted; no pre-Volta card is asked to run a kernel it can't. The
predicate is a pure function of the local device's `cc` with no `static`/global arch flag
(the only `static` added is a `GGML_CUDA_DEBUG` logging toggle), so on a **mixed** K80 +
newer box each multiply selects its path per-device — the K80 cannot drag the newer card
onto the legacy route. (The Go-file changes in the commit are comments only.)

**Verdict: safe as-is.**

## 2. VMM pool granularity alignment — `46213c58`

**What it did (historical).** Granularity-aligned the VMM pool's VA reservation to avoid
`cuMemAddressReserve` returning `CUDA_ERROR_INVALID_VALUE` on the K80, plus a memory cap
and an OOM clamp.

**Key finding: it was fully reverted.** The upstream-sync commit `ef14fb5b` backed out
every line `46213c58` added. The current allocator is **stock upstream** — there is no
K80-specific VMM code left to mis-fire. Confirmed: `max_pool_size`, `total_mem * 0.9`,
and the OOM-guard string are all absent (`grep` → no matches); `CUDA_POOL_VMM_MAX_SIZE`
is back to the static `1<<35` (32 GB).

**Why it's safe.** The surviving granularity use is upstream's own per-device round-up:
`cuMemGetAllocationGranularity(...)` is queried per device at init (`ggml-cuda.cu:320`),
stored in a per-device array, and read by each pool (`:557`) — every card aligns to its
*own* granularity. The 32 GB VA ceiling is itself a multiple of any standard granularity.

**Verdict: safe as-is** (nothing K80-specific remains).

**Caveat (not a non-K80 risk):** the revert also dropped the workaround's K80 OOM guard
and crash fix. If a K80 in this fleet ever hits the original `CUDA_ERROR_INVALID_VALUE`,
that crash is back — but the 32 GB reservation is already granularity-aligned, which is
likely why upstream's design doesn't trip it. Re-confirming is a **K80-runtime** test, not
a code risk on the other architectures.

## 3. Ghost-GPU layout + vision reservation — `695557de` (#138/#141)

Two logic changes (the other two files in the commit are debug logging only):

**Ghost-GPU layout** (`llm/server.go:965-974`). `buildLayout` pins a floor on GPU count so
re-measurement can't shrink below GPUs already in use. The bug: a transient 93 MiB
*compute graph* spilling onto a 3rd K80 die ratcheted the floor permanently
("ghost-allocated" memory on a die holding no layers). The fix pins the floor only when a
die holds real **layer data** (`Weights[k] != 0 || Cache[k] != 0`), not a graph-only touch.

**Vision reservation** (`runner/ollamarunner/runner.go:1072-1075`). The worst-case vision
graph was reserved at 2048px when FA was on, else 512px. Vision towers always run the
manual (non-FA) attention path, so the full attention matrix materialized; on K80 the
batched-cuBLAS kernel exceeds the 1024 threads/block cap → launch failure. The fix always
caps at 512px (unless `OLLAMA_VISION_MAX_PIXELS` is set).

**Gate.** Neither is gated on arch / cc / K80 / die-count / VRAM size. Ghost-GPU keys on
per-device `Weights`/`Cache`; the vision cap keys on `envconfig.VisionMaxPixels()==0`.
Confirmed: no `k80`/`3.7`/`sm_37`/`12 GiB`/die-count constant in any executable line.

**Why it's safe.** All per-GPU math reads per-device fields (`FreeMemory`,
`MinimumMemory()`, per-index `Graph`/`Weights`/`Cache`), so heterogeneous VRAM sizes
account correctly. The vision cap is strictly *more conservative* on a big card than the
old 2048px path — it cannot over-reserve or crash.

**Verdict: safe as-is.**

**Follow-up worth considering (enhancement, not a fix — tracked in [#355](https://github.com/dogkeeper886/ollama37/issues/355)):**
the 512px vision default now applies to *every* architecture, so an FA-capable Ampere box
that previously defaulted to 2048px now defaults to 512px (escapable via
`OLLAMA_VISION_MAX_PIXELS`). If big cards should keep a higher default, the right gate is
**device capability** (max threads/block, or cc) rather than the old `flashAttention` flag — and note that after [#350](https://github.com/dogkeeper886/ollama37/issues/350)
removed K80 FA, the old `flashAttention`-based gate would no longer have distinguished K80
from Pascal anyway. Out of scope for this correctness audit.

## Summary

| Workaround | Gate | Non-K80 / mixed | Verdict |
|---|---|---|---|
| CUBLAS two-tier (`d9489265`) | per-device `cc < VOLTA` | correct (boundary = tensor cores) | safe |
| VMM granularity (`46213c58`) | reverted; stock upstream per-device query | correct | safe |
| Ghost-GPU + vision (`695557de`) | per-device memory data; env flag | correct (more conservative) | safe |

No code change required for #343. Two optional follow-ups noted — neither is a
non-K80/mixed-box correctness defect: the K80 OOM-guard regression (K80-runtime only), and
the capability-gated vision default (tracked in [#355](https://github.com/dogkeeper886/ollama37/issues/355)).
