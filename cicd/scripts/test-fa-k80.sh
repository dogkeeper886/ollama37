#!/usr/bin/env bash
# test-fa-k80.sh — K80 Flash Attention regression test + benchmark.
#
# Boots a fresh ollama37 container per configuration (FA off/on × KV cache
# type), runs a deterministic inference, captures VRAM + KV cache size +
# tokens/sec + container logs, then judges each response via the shared
# LLM-judge helper. Production ollama37 stop/start is handled by the
# caller (the workflow YAML).
#
# Two modes:
#   --benchmark            : run three configs (off-f16, on-f16, on-q8_0)
#                            and emit a markdown comparison table
#   (default, "single")    : run one config with optional FA-off baseline
#
# Output:
#   /tmp/test-fa-k80-results.json     — combined per-config metrics + judge
#   /tmp/fa-test/<config>/...         — per-config artifacts (response.json,
#                                       container-logs.log, metrics.json,
#                                       fa-related-logs.log, judgment.json)
#   /tmp/fa-test/benchmark-table.md   — (benchmark mode) markdown table
#   /tmp/fa-test/judge-table.md       — judge verdicts as markdown
#
# Exit status: non-zero if inference fails or any judge verdict is FAIL.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "$0")" && pwd)/lib"
# shellcheck source=lib/simple_check.sh
source "$LIB_DIR/simple_check.sh"
# shellcheck source=lib/judge_response.sh
source "$LIB_DIR/judge_response.sh"

# Defaults
MODEL="gemma3:4b"
PROMPT="Write a detailed multi-paragraph explanation of how large language models work. Cover tokenization, attention mechanisms, training procedures, and inference. Be thorough."
NUM_PREDICT=256
NUM_CTX=4096
MODE="single"      # "single" | "benchmark"
KV_CACHE_TYPE=""
BASELINE=false
JUDGE_HOST="${JUDGE_HOST:-http://localhost:11435}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma3:12b-judge}"

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --num-predict) NUM_PREDICT="$2"; shift 2 ;;
        --num-ctx) NUM_CTX="$2"; shift 2 ;;
        --benchmark) MODE="benchmark"; shift ;;
        --kv-cache-type) KV_CACHE_TYPE="$2"; shift 2 ;;
        --baseline-compare) BASELINE=true; shift ;;
        --judge-host) JUDGE_HOST="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Fresh artifact directory
rm -rf /tmp/fa-test
mkdir -p /tmp/fa-test
chmod 777 /tmp/fa-test

# Reusable: start a test container with given FA + KV settings, run a
# deterministic inference, capture response + logs + metrics. Args:
#   $1 = label (off-f16, on-f16, on-q8_0, baseline, fa)
#   $2 = enable_fa (0|1)
#   $3 = kv_cache_type ("" or "q8_0"/"q4_0")
# Writes: /tmp/fa-test/<label>/{response.json, container-logs.log, metrics.json, fa-related-logs.log}
run_inference() {
    local label="$1" enable_fa="$2" kv_type="$3"
    local out_dir="/tmp/fa-test/${label}"
    mkdir -p "$out_dir"

    echo ""
    echo "=========================================="
    echo "=== Run: $label (enable_fa=$enable_fa, kv_cache_type=${kv_type:-f16}, num_ctx=$NUM_CTX) ==="
    echo "=========================================="

    # Force OLLAMA_FLASH_ATTENTION explicitly. Models like Gemma3 default
    # f.FlashAttention()=true, so leaving it unset enables FA via model
    # default + #117's K80 gate. Explicit "0" or "1" guarantees behavior.
    local fa_args
    if [ "$enable_fa" = "1" ]; then
        fa_args="-e OLLAMA_FLASH_ATTENTION=1"
    else
        fa_args="-e OLLAMA_FLASH_ATTENTION=0"
    fi

    local kv_args=""
    if [ -n "$kv_type" ]; then
        kv_args="-e OLLAMA_KV_CACHE_TYPE=$kv_type"
    fi

    local cname="ollama37-fa-test-${label}"
    docker rm -f "$cname" 2>/dev/null || true

    # shellcheck disable=SC2086 -- fa_args/kv_args intentionally word-split
    docker run -d --rm \
        --name "$cname" \
        --runtime nvidia \
        --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e OLLAMA_HOST=0.0.0.0:11434 \
        -e OLLAMA_DEBUG=1 \
        $fa_args $kv_args \
        -v ollama-data:/root/.ollama \
        -p 11434:11434 \
        ollama37:latest

    # Wait for the API to come up (max 60s)
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/ >/dev/null 2>&1; then
            echo "ollama API ready after ${i}s"
            break
        fi
        sleep 1
    done

    if ! curl -sf http://localhost:11434/ >/dev/null 2>&1; then
        echo "::error::API never came up for $label"
        docker logs "$cname" > "$out_dir/container-logs.log" 2>&1 || true
        docker stop --time 30 "$cname" || true
        return 1
    fi

    # Deterministic inference (temp=0, seed=42, explicit num_ctx)
    curl -s http://localhost:11434/api/generate \
        -d "$(jq -nc \
            --arg model "$MODEL" \
            --arg prompt "$PROMPT" \
            --argjson num_predict "$NUM_PREDICT" \
            --argjson num_ctx "$NUM_CTX" \
            '{model: $model, prompt: $prompt, stream: false,
              options: {num_predict: $num_predict, num_ctx: $num_ctx,
                        temperature: 0, seed: 42}}')" \
        > "$out_dir/response.json"

    # Snapshot total VRAM with model + KV + activations live
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        > "$out_dir/vram-mib.txt"
    local vram
    vram=$(head -1 "$out_dir/vram-mib.txt" | tr -d ' ')

    # Capture container logs
    docker logs "$cname" > "$out_dir/container-logs.log" 2>&1
    echo "[$label] Log size: $(wc -l < "$out_dir/container-logs.log") lines"

    # Verify the response field exists
    if ! jq -e '.response' "$out_dir/response.json" >/dev/null 2>&1; then
        echo "::error::[$label] inference call did not return a 'response' field"
        echo "--- response.json ---"
        cat "$out_dir/response.json"
        echo "--- last 50 log lines ---"
        tail -50 "$out_dir/container-logs.log"
        docker stop --time 30 "$cname" || true
        return 1
    fi

    # Extract KV cache size from the container log
    local kv_mib
    kv_mib=$(grep -oE '"kv cache" device=[^ ]+ size="[0-9.]+ MiB"' "$out_dir/container-logs.log" \
                | head -1 | grep -oE '[0-9.]+ MiB' | grep -oE '[0-9.]+') || kv_mib="unknown"

    # All keys quoted defensively against jq reserved words.
    jq -n \
        --arg config "$label" \
        --arg model "$MODEL" \
        --argjson enable_fa "$enable_fa" \
        --arg kv_type "${kv_type:-f16}" \
        --argjson num_ctx "$NUM_CTX" \
        --argjson num_predict "$NUM_PREDICT" \
        --slurpfile resp "$out_dir/response.json" \
        --arg vram "$vram" \
        --arg kv_mib "$kv_mib" \
        '{
            "config": $config,
            "model": $model,
            "enable_fa": ($enable_fa == 1),
            "kv_cache_type": $kv_type,
            "num_ctx": $num_ctx,
            "num_predict": $num_predict,
            "eval_count": ($resp[0].eval_count // 0),
            "eval_duration_ns": ($resp[0].eval_duration // 0),
            "prompt_eval_duration_ns": ($resp[0].prompt_eval_duration // 0),
            "tokens_per_sec": (if ($resp[0].eval_duration // 0) > 0
                               then ($resp[0].eval_count / ($resp[0].eval_duration / 1000000000))
                               else 0 end),
            "vram_used_mib": ($vram | tonumber),
            "kv_cache_mib": $kv_mib,
            "response_preview": ($resp[0].response // "" | .[0:120])
        }' \
        > "$out_dir/metrics.json"

    echo "[$label] metrics:"
    cat "$out_dir/metrics.json"

    # Surface FA-related log lines for quick scanning
    grep -iE "flash[_ ]?att|fattn|FLASH_ATTN|kv.cache" "$out_dir/container-logs.log" \
        > "$out_dir/fa-related-logs.log" || true

    docker stop --time 30 "$cname" || true

    # Wait for VRAM to release before the next run
    for _ in $(seq 1 30); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "$used" -lt 1024 ]; then
            break
        fi
        sleep 1
    done
}

# Drive the configured run(s)
if [ "$MODE" = "benchmark" ]; then
    run_inference "off-f16" "0" ""
    run_inference "on-f16"  "1" ""
    run_inference "on-q8_0" "1" "q8_0"

    echo ""
    echo "=========================================="
    echo "=== Benchmark comparison ($MODEL, num_ctx=$NUM_CTX, num_predict=$NUM_PREDICT) ==="
    echo "=========================================="

    # Build a markdown comparison table from per-config metrics.json files.
    jq -s '. as $rows
           | "| Config | KV cache (MiB) | Total VRAM (MiB) | tokens/sec | eval (s) |\n|---|---|---|---|---|\n"
             + ($rows | map(
               "| \(.config) | \(.kv_cache_mib) | \(.vram_used_mib) | \(.tokens_per_sec | . * 100 | round / 100) | \(.eval_duration_ns / 1000000000 | . * 100 | round / 100) |"
             ) | join("\n"))' \
        /tmp/fa-test/off-f16/metrics.json \
        /tmp/fa-test/on-f16/metrics.json \
        /tmp/fa-test/on-q8_0/metrics.json \
        -r | tee /tmp/fa-test/benchmark-table.md
else
    if [ "$BASELINE" = "true" ]; then
        run_inference "baseline" "0" "$KV_CACHE_TYPE"
    fi
    run_inference "fa" "1" "$KV_CACHE_TYPE"

    if [ "$BASELINE" = "true" ]; then
        echo ""
        echo "=== Output comparison: baseline vs FA ==="
        BASELINE_RESP=$(jq -r '.response' /tmp/fa-test/baseline/response.json)
        FA_RESP=$(jq -r '.response' /tmp/fa-test/fa/response.json)
        if [ "$BASELINE_RESP" = "$FA_RESP" ]; then
            echo "::notice::Outputs match exactly"
        else
            echo "::warning::Outputs differ between baseline and FA"
        fi
        jq -n --arg b "$BASELINE_RESP" --arg f "$FA_RESP" \
            '{baseline: $b, fa: $f, match: ($b == $f)}' \
            > /tmp/fa-test/comparison.json
    fi
fi

# Judge captured responses via the shared LLM-judge helper. The judge is
# scoped to "is this response meaningful for the prompt?" per CLAUDE.md →
# "LLM Judge Scope" — that's exactly what we want here.
FAILED=0
{
    echo ""
    echo "## K80 FA LLM judge verdicts"
    echo ""
    echo "| Config | Verdict | Reason |"
    echo "|---|---|---|"
} > /tmp/fa-test/judge-table.md

if ! ls /tmp/fa-test/*/response.json >/dev/null 2>&1; then
    echo "::error::No inference responses captured under /tmp/fa-test/ — nothing to judge"
    exit 1
fi

for resp_file in /tmp/fa-test/*/response.json; do
    [ -f "$resp_file" ] || continue
    config_dir=$(dirname "$resp_file")
    config_name=$(basename "$config_dir")

    echo ""
    echo "=========================================="
    echo "=== Judging $config_name ==="
    echo "=========================================="

    RESPONSE_TEXT=$(jq -r '.response // ""' "$resp_file")
    THINKING_TEXT=$(jq -r '.thinking // ""' "$resp_file")

    # Simple check first — empty output is a hard fail.
    SIMPLE_VERDICT=$(simple_check "$RESPONSE_TEXT" "$THINKING_TEXT") || true
    SIMPLE_PASS=$(echo "$SIMPLE_VERDICT" | jq -r '.pass')

    if [ "$SIMPLE_PASS" != "true" ]; then
        reason=$(echo "$SIMPLE_VERDICT" | jq -r '.reason')
        echo "[$config_name] FAIL (simple) - $reason"
        echo "| $config_name | FAIL | $reason |" >> /tmp/fa-test/judge-table.md
        echo "$SIMPLE_VERDICT" > "$config_dir/judgment.json"
        FAILED=1
        continue
    fi

    # Combine thinking + response into the judge's single `output` field.
    JUDGE_OUTPUT="${THINKING_TEXT}${THINKING_TEXT:+

}${RESPONSE_TEXT}"

    VERDICT=$(judge_response "$JUDGE_HOST" "$JUDGE_MODEL" "$MODEL" "$PROMPT" "$JUDGE_OUTPUT")
    echo "$VERDICT" > "$config_dir/judgment.json"
    pass=$(echo "$VERDICT" | jq -r '.pass')
    reason=$(echo "$VERDICT" | jq -r '.reason' | tr '\n|' '  ' | cut -c1-200)

    if [ "$pass" = "true" ]; then
        verdict="PASS"
    else
        verdict="FAIL"
        FAILED=1
    fi

    echo "[$config_name] $verdict - $reason"
    echo "| $config_name | $verdict | $reason |" >> /tmp/fa-test/judge-table.md
done

# Aggregate results JSON
jq -s '{
    mode: $mode,
    model: $model,
    num_ctx: ($num_ctx | tonumber),
    num_predict: ($num_predict | tonumber),
    configs: .
}' \
    --arg mode "$MODE" \
    --arg model "$MODEL" \
    --arg num_ctx "$NUM_CTX" \
    --arg num_predict "$NUM_PREDICT" \
    /tmp/fa-test/*/metrics.json \
    > /tmp/test-fa-k80-results.json

echo ""
echo "Results written to /tmp/test-fa-k80-results.json"

if [ "$FAILED" -ne 0 ]; then
    echo "::error::One or more FA configs failed validation"
    exit 1
fi
echo "All configs passed validation"
