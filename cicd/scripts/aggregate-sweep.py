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

    with open(md_out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"aggregated {len(tsv_rows)} rows -> {tsv_out}, {md_out}")


if __name__ == "__main__":
    main()
