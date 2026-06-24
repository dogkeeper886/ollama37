# `docs/reports/` — K80 fleet characterization

Measured reports on how models behave on the 4 × Tesla K80, grouped by model family and swept over
context length. Two paired axes plus the kept #352 baseline.

## Reports

| File | What it is |
|---|---|
| [`k80-vram-by-family.md`](./k80-vram-by-family.md) | **Speed/fit axis** — per-die VRAM, GPU vs CPU placement, throughput, swept 2k/4k/8k/16k. |
| [`k80-mcp-by-family.md`](./k80-mcp-by-family.md) | **Capability axis** — MCP tool-call verdicts (tool·query·content stages), swept 8k/16k. |
| [`deepseek-r1-k80-vram-throughput.md`](./deepseek-r1-k80-vram-throughput.md) | The #352 baseline — DeepSeek-R1 VRAM/throughput before/after the `graphSafetyMultiplier` change. Cited by `llm/memory.go`. |

The two family reports are meant to be **read together**: an MCP fail at high ctx while GPU% is 100%
in the VRAM report is a capability/truncation result, not a speed/offload artifact.

## How they're produced

Both are filled from CI workflow runs (one model per run, via the GPU temp guard):

```
# VRAM / throughput
gh workflow run test-throughput.yml -f models=<model> -f num_predict=16 -f context_size=<2048|4096|8192|16384>

# MCP tool-call capability (verify-live, testlink + playwright menu)
gh workflow run test-mcp.yml -f models=<model> -f num_ctx=<8192|16384> -f judge_mode=verify-live \
  -f verify_allow="list_projects,list_test_suites"
```

Each run's JSON artifact fills one row; markers are defined in each report's own legend.

## Related

K80 deep-dive audits live in [`../research/`](../research/): flash-attention coverage
(`k80-fa-model-coverage.md`), kernel fusion (`k80-fusion-audit.md`), quantization defaults
(`k80-quant-audit.md`), and the multi-arch workarounds audit (`k80-workarounds-multiarch-audit.md`).
