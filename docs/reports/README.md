# `docs/reports/` — K80 fleet characterization

Measured reports on how models behave on the 4 × Tesla K80, grouped by model family and swept over
context length. Two paired axes plus the kept #352 baseline.

## Versioning

Each sweep produces a **new immutable snapshot** named
`<report>-<YYYY-MM-DD>-<build-sha>.md` — so every report is tied to the exact ollama build it
measured. **Diff two snapshots of the same axis to see what changed between builds** (a throughput
or VRAM delta is then provably a code change, not measurement drift). Snapshots are never edited in
place or mixed across builds; the pre-versioning (mixed-build) reports live in git history.

> The `<build-sha>` is the ollama commit that was deployed and measured. Until the build embeds its
> own SHA (tracked follow-up — `/api/version` currently returns a static `2.1.0`), the SHA is
> recovered from the deploying pipeline run and recorded in each snapshot's `Build` row.

## Reports

### Speed/fit axis — VRAM / throughput

| Snapshot | Build | Swept | Notes |
|---|---|---|---|
| [`…-2026-07-10-d83061de.md`](./k80-vram-by-family-2026-07-10-d83061de.md) | `d83061de` | 2026-07-10 | **Latest.** 14 models (+`lfm2.5:8b`, new `lfm2` family, ~33 tok/s). No-harm vs `c282ba37`: 49/51 rows within ±3 %. gemma4:26b @8k+ now spreads to 4 dies (−9 %) — gemma4/KV-graph effect, not lfm2. |
| [`…-2026-06-29-c282ba37.md`](./k80-vram-by-family-2026-06-29-c282ba37.md) | `c282ba37` | 2026-06-29 | First clean single-build sweep (all 13 models). gemma4:12b now on GPU; `swa_full` windowed cache reclaims its 16k VRAM. e4b@2k cold-load artifact published + flagged. |

### Capability axis — MCP tool-call

| Snapshot | Build | Swept | Notes |
|---|---|---|---|
| [`…-2026-07-10-d83061de.md`](./k80-mcp-by-family-2026-07-10-d83061de.md) | `d83061de` | 2026-07-10 | **Latest — T1 @8k only** (16k thermal-aborted at 81 °C, twice). 14 models (+`lfm2.5:8b`). 0 verdict regressions vs `c282ba37`; `deepseek-r1:8b-tools` ❌→✅; `lfm2.5:8b` ✅; 9/14 PASS. |
| [`…-2026-06-29-c282ba37.md`](./k80-mcp-by-family-2026-06-29-c282ba37.md) | `c282ba37` | 2026-06-29 | **T1 only.** gemma4:12b: fails-to-load (main) → loads + 16k T1 ✅ (8k over-explores the menu). 0 verdict changes vs main; 15/26 PASS. |
| [`k80-mcp-by-family.md`](./k80-mcp-by-family.md) | pre-versioning (mixed) | 2026-06-23 era | The older T1+**T2** report (gemma4:12b shown as fails-to-load). Kept for T2 until a clean T1+T2 re-sweep produces a versioned snapshot. |

The two family axes are meant to be **read together**: an MCP fail at high ctx while GPU% is 100% in
the VRAM report is a capability/truncation result, not a speed/offload artifact.

### Baseline

| [`deepseek-r1-k80-vram-throughput.md`](./deepseek-r1-k80-vram-throughput.md) | The #352 baseline — DeepSeek-R1 VRAM/throughput before/after the `graphSafetyMultiplier` change. Cited by `llm/memory.go`. |
|---|---|

## How they're produced

The sweep is one CI workflow (loops every model × context serially on the single K80, thermal-guarded,
reusing the `bench-throughput` / `test-mcp` harness):

```
gh workflow run test-report-sweep.yml -f suite=both
```

It uploads a TSV + a paste-ready markdown artifact; a maintainer commits it as the next
`<report>-<date>-<sha>.md` snapshot. (The per-model `test-throughput.yml` / `test-mcp.yml` workflows
still exist for one-off measurements.)

## Related

K80 deep-dive audits live in [`../research/`](../research/): flash-attention coverage
(`k80-fa-model-coverage.md`), kernel fusion (`k80-fusion-audit.md`), quantization defaults
(`k80-quant-audit.md`), and the multi-arch workarounds audit (`k80-workarounds-multiarch-audit.md`).
