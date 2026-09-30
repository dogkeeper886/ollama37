#!/usr/bin/env python3
"""Aggregate report-sweep JSONs (tp_<ctx>.json, mcp_<test>_<ctx>.json) into a TSV
and ready-to-paste report-row markdown. Used by .github/workflows/test-report-sweep.yml.

Usage: aggregate-sweep.py <input_dir> <summary_md_out> <tsv_out>
"""
import glob
import json
import os
import re
import sys


def ctx_label(ctx):
    n = int(ctx)
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def load(path):
    try:
        return json.load(open(path)).get("results", []) or []
    except Exception as e:
        print(f"  warn: cannot parse {path}: {e}", file=sys.stderr)
        return []


def main():
    in_dir, md_out, tsv_out = sys.argv[1], sys.argv[2], sys.argv[3]
    tsv_rows = []          # suite-tagged flat rows for the TSV
    tp = {}                # model -> ctx -> dict
    mcp = {}               # model -> (test,ctx) -> dict
    fitmap = {}            # model -> (ctx,nb) -> dict  (STORY-022 num_batch x context)

    for path in sorted(glob.glob(os.path.join(in_dir, "tp_*.json"))):
        m = re.match(r"tp_(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        ctx = m.group(1)
        for r in load(path):
            model = r.get("model", "?")
            v = (r.get("vram_used_mib") or []) + [0, 0, 0, 0]
            row = {
                "gpu": r.get("gpu_offload_pct", 0),
                "vram": v[:4],
                "tps": r.get("eval_tps", 0),
                "done": r.get("done_reason", ""),
            }
            tp.setdefault(model, {})[ctx] = row
            tsv_rows.append(f"throughput\t{model}\t{ctx_label(ctx)}\t{row['gpu']}\t"
                            f"{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\t{row['tps']}\t{row['done']}")

    for path in sorted(glob.glob(os.path.join(in_dir, "mcp_*.json"))):
        m = re.match(r"mcp_(\w+)_(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        test, ctx = m.group(1), m.group(2)
        for r in load(path):
            model = r.get("model", "?")
            sup = r.get("supported")
            op = (r.get("check") or {}).get("overall_pass")
            verdict = "NOSUP" if not sup else ("PASS" if op else "FAIL")
            calls = ",".join(c.get("name", "") for c in (r.get("tool_calls") or [])) or "-"
            row = {
                "verdict": verdict,
                "calls": calls,
                "ptok": r.get("max_prompt_tokens", 0),
                "secs": r.get("total_duration_s", 0),
                "tps": r.get("eval_tps", 0),
            }
            mcp.setdefault(model, {})[(test, ctx)] = row
            tsv_rows.append(f"mcp\t{model}\t{test}\t{ctx_label(ctx)}\t{verdict}\t"
                            f"{row['ptok']}\t{row['secs']}\t{row['tps']}\t{calls}")

    # Fit-map cells: one model per file, ctx + num_batch in the name (fitmap_<ctx>_<nb>_<model>.json).
    # The fit verdict comes from the GPU snapshot (#445), independent of the correctness judge.
    for path in sorted(glob.glob(os.path.join(in_dir, "fitmap_*.json"))):
        m = re.match(r"fitmap_(\d+)_(\d+)_.+\.json$", os.path.basename(path))
        if not m:
            continue
        ctx, nb = m.group(1), m.group(2)
        for r in load(path):
            model = r.get("model", "?")
            gpu = r.get("gpu") or {}
            offload = gpu.get("offloadPct")
            # Window saturation (#449): a round's prompt+generated tokens reached num_ctx, so
            # the KV cache filled and decode ran under eviction — tok/s is not a clean figure.
            # Prefer the per-round flag the host now emits; fall back to the count predicate
            # for pre-#449 JSONs. Orthogonal to fit (a saturated cell can still be 100% on GPU),
            # but it invalidates the SPEED, so it takes display precedence to warn the reader.
            saturated = r.get("saturated")
            if saturated is None:
                saturated = (r.get("max_prompt_tokens", 0) + r.get("out_tokens", 0)) >= int(ctx)
            # ✂️ saturated = tok/s invalid; ✅ fully on GPU; ⚠️ CPU spill; ❌ never resident
            # (OOM or a load/round error — indistinguishable here, so "not resident", not "OOM").
            if not gpu or offload is None:
                fit = "NOFIT"
            elif saturated:
                fit = "SAT"
            elif offload >= 100:
                fit = "FIT"
            else:
                fit = "SPILL"
            per_die = ",".join(str(g.get("usedMib", 0)) for g in (gpu.get("perDie") or [])) or "-"
            op = (r.get("check") or {}).get("overall_pass")
            # A saturated cell FAILs regardless of the answer — its tok/s measurement is invalid.
            verdict = "PASS" if (op and not saturated) else "FAIL"
            row = {
                "fit": fit,
                "verdict": verdict,
                "tps": r.get("eval_tps", 0),
                "secs": r.get("total_duration_s", 0),
                "vram_g": round(gpu.get("totalMib", 0) / 1024, 1) if gpu else 0,
                "dies": gpu.get("activeDies", 0) if gpu else 0,
                "offload": offload if offload is not None else 0,
                "per_die": per_die,
            }
            fitmap.setdefault(model, {})[(ctx, nb)] = row
            tsv_rows.append(f"fitmap\t{model}\t{ctx_label(ctx)}\t{nb}\t{fit}\t{verdict}\t"
                            f"{row['tps']}\t{row['secs']}\t{row['vram_g']}\t{row['dies']}\t{row['offload']}\t{per_die}")

    with open(tsv_out, "w") as f:
        f.write("\n".join(tsv_rows) + "\n")

    # markdown: per-family-style rows the maintainer can paste into the report
    lines = ["# Report sweep results\n"]
    if tp:
        lines.append("## Throughput (`k80-vram-by-family.md`)\n")
        lines.append("| Model | ctx | GPU% | d0 | d1 | d2 | d3 | tok/s |")
        lines.append("|---|---|:--:|--:|--:|--:|--:|--:|")
        for model in sorted(tp):
            for ctx in sorted(tp[model], key=int):
                r = tp[model][ctx]
                v = r["vram"]
                gpu = f"{r['gpu']}%" if r["gpu"] else "—"
                lines.append(f"| `{model}` | {ctx_label(ctx)} | {gpu} | "
                             f"{v[0]} | {v[1]} | {v[2]} | {v[3]} | {r['tps']} |")
        lines.append("")
    if mcp:
        lines.append("## MCP (`k80-mcp-by-family.md`)\n")
        lines.append("| Model | test | ctx | verdict | prompt_tok | time_s | tok/s | calls |")
        lines.append("|---|---|---|:--:|--:|--:|--:|---|")
        for model in sorted(mcp):
            for (test, ctx) in sorted(mcp[model]):
                r = mcp[model][(test, ctx)]
                mark = {"PASS": "✅", "FAIL": "❌", "NOSUP": "—"}.get(r["verdict"], r["verdict"])
                lines.append(f"| `{model}` | {test} | {ctx_label(ctx)} | {mark} | "
                             f"{r['ptok']} | {r['secs']} | {r['tps']} | {r['calls']} |")
        lines.append("")

    if fitmap:
        lines.append("## Fit map (`num_batch` x context, STORY-022)\n")
        lines.append("Fit: ✅ fully on GPU · ⚠️ CPU spill · ❌ not resident (OOM or a load/round error) · "
                     "✂️ saturated = prompt+output reached num_ctx so decode ran under KV eviction "
                     "(tok/s invalid; the VRAM/dies/offload columns are still valid). "
                     "Numbers are decode tok/s · total_s · total VRAM · active dies · offload%.\n")
        mark = {"FIT": "✅", "SPILL": "⚠️", "NOFIT": "❌", "SAT": "✂️"}
        for model in sorted(fitmap):
            lines.append(f"### `{model}`\n")
            lines.append("| ctx | num_batch | fit | verdict | tok/s | total_s | VRAM (G) | dies | offload% | per-die used (MiB) |")
            lines.append("|---|--:|:--:|:--:|--:|--:|--:|:--:|--:|---|")
            for (ctx, nb) in sorted(fitmap[model], key=lambda k: (int(k[0]), -int(k[1]))):
                r = fitmap[model][(ctx, nb)]
                lines.append(f"| {ctx_label(ctx)} | {nb} | {mark.get(r['fit'], r['fit'])} | {r['verdict']} | "
                             f"{r['tps']} | {r['secs']} | {r['vram_g']} | {r['dies']} | {r['offload']}% | {r['per_die']} |")
            lines.append("")

    with open(md_out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"aggregated {len(tsv_rows)} rows -> {tsv_out}, {md_out}")


if __name__ == "__main__":
    main()
