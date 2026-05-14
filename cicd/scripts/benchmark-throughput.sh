#!/bin/bash
# benchmark-throughput.sh - Measure model inference speed AND validate output.
#
# Two responsibilities, separated by mode:
#   - perf: tok/s, durations, VRAM (always)
#   - output check: simple (response non-empty) or dual (also LLM-judge meaningful)
#
# Usage:
#   ./benchmark-throughput.sh [options] model1 [model2 ...]
#
# Examples:
#   ./benchmark-throughput.sh gemma3:4b
#   ./benchmark-throughput.sh --llm --judge-model gemma3:12b-judge gemma3:4b
#   ./benchmark-throughput.sh -n 256 gemma3:4b qwen3:4b
#
# Options:
#   -H, --host URL          Ollama host (default: $OLLAMA_HOST or http://localhost:11434)
#   -n, --num-predict N     Max tokens to generate (default: 128)
#   -c, --context N         Context size (default: 2048)
#   -o, --output FILE       Save JSON report to file
#       --llm               Also run LLM judge on each response (dual mode)
#       --judge-model M     Judge model name (default: gemma3:12b-judge)
#       --judge-host URL    Judge ollama host (default: $JUDGE_HOST or same as --host)
#   -h, --help              Show this help
#
# Exit status: non-zero if any model fails the output check (simple or dual).

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
NUM_PREDICT=128
NUM_CTX=2048
OUTPUT=""
LLM_JUDGE=0
JUDGE_MODEL="${JUDGE_MODEL:-gemma3:12b-judge}"
JUDGE_HOST="${JUDGE_HOST:-}"
MODELS=()

PROMPT="Explain how a computer works to a curious 10-year-old. Be fun and use analogies."

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -H|--host) OLLAMA_HOST="$2"; shift 2 ;;
        -n|--num-predict) NUM_PREDICT="$2"; shift 2 ;;
        -c|--context) NUM_CTX="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        --llm) LLM_JUDGE=1; shift ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-host) JUDGE_HOST="$2"; shift 2 ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) MODELS+=("$1"); shift ;;
    esac
done

# Default judge host falls back to the model host if unset.
JUDGE_HOST="${JUDGE_HOST:-$OLLAMA_HOST}"

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models specified." >&2
    usage
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check Ollama is reachable
if ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_HOST}" >&2
    exit 1
fi

# Get per-GPU info from nvidia-smi
get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free \
            --format=csv,noheader,nounits 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Get GPU offload % from Ollama /api/ps while model is loaded
get_gpu_offload() {
    local model="$1"
    local ps_response
    ps_response=$(curl -sf "${OLLAMA_HOST}/api/ps" 2>/dev/null) || { echo "0"; return; }

    echo "$ps_response" | jq -r --arg m "$model" '
        .models[]? | select(.name == $m or .model == $m) |
        if .size > 0 then
            (.size_vram / .size * 100 | round)
        else
            0
        end
    ' 2>/dev/null | head -1 || echo "0"
}

# Run benchmark for one model, return JSON
bench_model() {
    local model="$1"

    # Warmup (loads model into GPU)
    curl -sf "${OLLAMA_HOST}/api/generate" \
        -d "$(jq -n --arg m "$model" --argjson ctx "$NUM_CTX" \
            '{model:$m, prompt:"Hi", stream:false, options:{num_predict:1, num_ctx:$ctx}}')" \
        > /dev/null 2>&1 || true

    # Snapshot GPU state while model is loaded
    local gpu_info gpu_pct
    gpu_info=$(get_gpu_info)
    gpu_pct=$(get_gpu_offload "$model")

    # Actual benchmark
    local response
    response=$(curl -sf "${OLLAMA_HOST}/api/generate" \
        -d "$(jq -n \
            --arg model "$model" \
            --arg prompt "$PROMPT" \
            --argjson num_predict "$NUM_PREDICT" \
            --argjson num_ctx "$NUM_CTX" \
            '{
                model: $model,
                prompt: $prompt,
                stream: false,
                options: {
                    temperature: 0,
                    seed: 42,
                    num_predict: $num_predict,
                    num_ctx: $num_ctx
                }
            }'
        )" 2>/dev/null)

    if [ -z "$response" ]; then
        echo "ERROR: No response from ${model}" >&2
        return 1
    fi

    # Build per-model result. `response` and `thinking` are both preserved:
    # thinking models (qwen3.5, qwen3-vl, deepseek-r1) can produce coherent
    # output entirely inside `.thinking` while leaving `.response` empty when
    # num_predict caps generation before the final answer.
    echo "$response" | jq \
        --arg model "$model" \
        --arg gpu_info "$gpu_info" \
        --argjson gpu_pct "${gpu_pct:-0}" \
        --argjson num_ctx "$NUM_CTX" \
        '{
            model: $model,
            in_tokens: .prompt_eval_count,
            out_tokens: .eval_count,
            prompt_eval_tps: (if .prompt_eval_duration > 0 then (.prompt_eval_count / (.prompt_eval_duration / 1e9) * 100 | round / 100) else 0 end),
            eval_tps: (if .eval_duration > 0 then (.eval_count / (.eval_duration / 1e9) * 100 | round / 100) else 0 end),
            total_duration_s: ((.total_duration // 0) / 1e9 * 100 | round / 100),
            load_duration_s: ((.load_duration // 0) / 1e9 * 100 | round / 100),
            gpu_offload_pct: $gpu_pct,
            num_ctx: $num_ctx,
            done_reason: (.done_reason // ""),
            response: (.response // ""),
            thinking: (.thinking // ""),
            gpu_info: $gpu_info
        }'

    # Unload model
    curl -sf "${OLLAMA_HOST}/api/generate" \
        -d "{\"model\":\"${model}\",\"keep_alive\":0}" > /dev/null 2>&1 || true
}

# Run the LLM judge against (model, prompt, response, thinking). Returns a
# JSON object with `pass` and `reason`. Falls back to pass=false if the judge
# endpoint is unreachable or returns malformed JSON — a missing judge must
# not silently pass a broken response.
judge_response() {
    local model="$1"
    local prompt_text="$2"
    local response_text="$3"
    local thinking_text="${4:-}"

    local judge_prompt
    judge_prompt=$(jq -nc \
        --arg model "$model" \
        --arg prompt "$prompt_text" \
        --arg response "$response_text" \
        --arg thinking "$thinking_text" \
        '{
            role: "Decide whether the model output is meaningful for the prompt. A meaningful output addresses the prompt with coherent text in the right language and topic. Reject empty, garbled, repetitive nonsense, off-topic, or error-message output.",
            rules: [
                "Thinking models emit two fields: `thinking` (internal reasoning) and `response` (final answer). Either is acceptable evidence of meaningful output — judge them together.",
                "Truncation at the token limit is fine if the visible text is coherent and on-topic. Many models exhaust the budget inside `thinking` and leave `response` empty; that is not a failure.",
                "Accept reasonable variations in phrasing.",
                "An API-level error message in either field is FAIL."
            ],
            model_under_test: $model,
            prompt: $prompt,
            response: $response,
            thinking: $thinking,
            respond: {
                format: "Respond with a single JSON object",
                fields: {
                    pass: "true if the output is meaningful and on-topic (consider response and thinking together), false otherwise",
                    reason: "Brief explanation"
                }
            }
        }')

    local judge_payload judge_raw verdict_text
    judge_payload=$(jq -nc \
        --arg model "$JUDGE_MODEL" \
        --arg prompt "$judge_prompt" \
        '{model:$model, prompt:$prompt, stream:false, format:"json", options:{temperature:0.1, num_predict:512}}')

    judge_raw=$(curl -sf --max-time 300 "${JUDGE_HOST}/api/generate" -d "$judge_payload" 2>/dev/null || true)
    if [ -z "$judge_raw" ]; then
        echo '{"pass":false,"reason":"judge endpoint unreachable or timed out"}'
        return
    fi

    verdict_text=$(echo "$judge_raw" | jq -r '.response // ""' 2>/dev/null)
    if [ -z "$verdict_text" ] || ! echo "$verdict_text" | jq empty 2>/dev/null; then
        echo '{"pass":false,"reason":"judge returned non-JSON response"}'
        return
    fi

    # Normalize to {pass, reason}.
    echo "$verdict_text" | jq -c '{pass: (.pass // false), reason: (.reason // "no reason given")}'
}

# Simple coherence check: either response or thinking must be non-empty after
# trimming whitespace. Thinking models can produce coherent output entirely
# in `.thinking` when num_predict caps generation before the final answer.
# Returns 0 if pass, 1 if fail; writes "{pass, reason, source}" JSON to stdout.
simple_check() {
    local response_text="$1"
    local thinking_text="${2:-}"
    local r_trim t_trim
    r_trim=$(echo -n "$response_text" | tr -d '[:space:]')
    t_trim=$(echo -n "$thinking_text" | tr -d '[:space:]')
    if [ -n "$r_trim" ]; then
        echo '{"pass":true,"reason":"non-empty response","source":"response"}'
        return 0
    fi
    if [ -n "$t_trim" ]; then
        echo '{"pass":true,"reason":"non-empty thinking (response empty)","source":"thinking"}'
        return 0
    fi
    echo '{"pass":false,"reason":"both response and thinking are empty/whitespace","source":"none"}'
    return 1
}

if [ "$LLM_JUDGE" -eq 1 ]; then
    JUDGE_MODE_LABEL="dual (simple + LLM)"
else
    JUDGE_MODE_LABEL="simple (non-empty response)"
fi

echo "" >&2
echo "=== Throughput Benchmark ===" >&2
echo "Models:      ${MODELS[*]}" >&2
echo "NumPredict:  ${NUM_PREDICT}" >&2
echo "Context:     ${NUM_CTX}" >&2
echo "Host:        ${OLLAMA_HOST}" >&2
echo "Judge mode:  ${JUDGE_MODE_LABEL}" >&2
if [ "$LLM_JUDGE" -eq 1 ]; then
    echo "Judge model: ${JUDGE_MODEL}" >&2
    echo "Judge host:  ${JUDGE_HOST}" >&2
fi
echo "Git SHA:     ${GIT_SHA}" >&2
echo "" >&2

# GPU info before any model is loaded
echo "--- GPU Status ---" >&2
GPU_BEFORE=$(get_gpu_info)
if [ -n "$GPU_BEFORE" ]; then
    echo "$GPU_BEFORE" | while IFS=', ' read -r idx name used total free; do
        echo "  GPU ${idx}: ${name} | ${used}/${total} MiB | ${free} MiB free" >&2
    done
else
    echo "  No NVIDIA GPU detected" >&2
fi
echo "" >&2

# Run each model
RESULTS="[]"
FAILED_CHECKS=0
for model in "${MODELS[@]}"; do
    echo "--- ${model} ---" >&2

    # Ensure model is available
    if ! curl -sf "${OLLAMA_HOST}/api/show" -d "{\"name\":\"${model}\"}" > /dev/null 2>&1; then
        echo "  Pulling ${model}..." >&2
        curl -sf "${OLLAMA_HOST}/api/pull" -d "{\"name\":\"${model}\",\"stream\":false}" > /dev/null 2>&1 || {
            echo "  ERROR: Failed to pull ${model}, skipping" >&2
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            continue
        }
    fi

    echo "  Warming up & loading..." >&2
    RESULT=$(bench_model "$model") || {
        echo "  ERROR: Benchmark failed, skipping" >&2
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        continue
    }

    # Print per-model perf summary
    GPU_PCT=$(echo "$RESULT" | jq -r '.gpu_offload_pct')
    IN_TOK=$(echo "$RESULT" | jq -r '.in_tokens')
    OUT_TOK=$(echo "$RESULT" | jq -r '.out_tokens')
    PROMPT_TPS=$(echo "$RESULT" | jq -r '.prompt_eval_tps')
    EVAL_TPS=$(echo "$RESULT" | jq -r '.eval_tps')

    # Per-GPU VRAM while this model is loaded
    GPU_LINE=$(echo "$RESULT" | jq -r '.gpu_info')
    if [ -n "$GPU_LINE" ] && [ "$GPU_LINE" != "null" ]; then
        echo "$GPU_LINE" | while IFS=', ' read -r idx name used total free; do
            echo "  GPU ${idx}: ${used}/${total} MiB" >&2
        done
    fi
    echo "  GPU offload: ${GPU_PCT}% | In: ${IN_TOK} tok | Out: ${OUT_TOK} tok" >&2
    echo "  Prompt: ${PROMPT_TPS} tok/s | Generate: ${EVAL_TPS} tok/s" >&2

    # Output validation. Perf numbers without a coherence check are misleading
    # — a model can output 128 tokens of garbage at 50 tok/s. Simple mode
    # rejects empty/whitespace; dual also runs an LLM judge. Both consider
    # `.thinking` alongside `.response` for thinking models.
    RESPONSE_TEXT=$(echo "$RESULT" | jq -r '.response // ""')
    THINKING_TEXT=$(echo "$RESULT" | jq -r '.thinking // ""')
    DONE_REASON=$(echo "$RESULT" | jq -r '.done_reason // ""')
    RESPONSE_LEN=${#RESPONSE_TEXT}
    THINKING_LEN=${#THINKING_TEXT}
    R_PREVIEW=$(printf '%s' "$RESPONSE_TEXT" | head -c 200)
    echo "  Response (${RESPONSE_LEN} chars, done_reason=${DONE_REASON}): ${R_PREVIEW}" >&2
    if [ "$THINKING_LEN" -gt 0 ]; then
        T_PREVIEW=$(printf '%s' "$THINKING_TEXT" | head -c 200)
        echo "  Thinking (${THINKING_LEN} chars): ${T_PREVIEW}" >&2
    fi

    SIMPLE_VERDICT=$(simple_check "$RESPONSE_TEXT" "$THINKING_TEXT") || true
    SIMPLE_PASS=$(echo "$SIMPLE_VERDICT" | jq -r '.pass')

    LLM_VERDICT='null'
    LLM_PASS_BOOL='null'
    if [ "$LLM_JUDGE" -eq 1 ]; then
        if [ "$SIMPLE_PASS" = "true" ]; then
            echo "  Running LLM judge (${JUDGE_MODEL})..." >&2
            LLM_VERDICT=$(judge_response "$model" "$PROMPT" "$RESPONSE_TEXT" "$THINKING_TEXT")
            LLM_PASS_BOOL=$(echo "$LLM_VERDICT" | jq -r '.pass')
            REASON=$(echo "$LLM_VERDICT" | jq -r '.reason')
            echo "  LLM judge: ${LLM_PASS_BOOL} — ${REASON}" >&2
        else
            echo "  LLM judge: skipped (simple check already failed)" >&2
        fi
    fi

    # Overall pass = simple AND (no llm judge OR llm judge pass)
    if [ "$SIMPLE_PASS" = "true" ] && { [ "$LLM_JUDGE" -eq 0 ] || [ "$LLM_PASS_BOOL" = "true" ]; }; then
        OVERALL_PASS=true
        echo "  Output check: PASS" >&2
    else
        OVERALL_PASS=false
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        REASON=$(echo "$SIMPLE_VERDICT" | jq -r '.reason')
        if [ "$LLM_JUDGE" -eq 1 ] && [ "$SIMPLE_PASS" = "true" ]; then
            REASON=$(echo "$LLM_VERDICT" | jq -r '.reason')
        fi
        echo "  Output check: FAIL — ${REASON}" >&2
    fi

    # Attach verdict block to the result.
    RESULT=$(echo "$RESULT" | jq \
        --argjson simple "$SIMPLE_VERDICT" \
        --argjson llm "$LLM_VERDICT" \
        --argjson overall "$OVERALL_PASS" \
        '. + {check: {overall_pass: $overall, simple: $simple, llm: $llm}}')

    RESULTS=$(echo "$RESULTS" | jq --argjson r "$RESULT" '. + [$r]')
    echo "" >&2
done

# GPU info after all models unloaded
GPU_AFTER=$(get_gpu_info)

# Build report
REPORT=$(jq -n \
    --arg git_sha "$GIT_SHA" \
    --arg timestamp "$TIMESTAMP" \
    --argjson num_predict "$NUM_PREDICT" \
    --argjson num_ctx "$NUM_CTX" \
    --arg gpu_before "$GPU_BEFORE" \
    --arg gpu_after "$GPU_AFTER" \
    --argjson results "$RESULTS" \
    '{
        git_sha: $git_sha,
        timestamp: $timestamp,
        config: { num_predict: $num_predict, num_ctx: $num_ctx },
        gpu: { before: $gpu_before, after: $gpu_after },
        results: $results
    }'
)

# Output JSON
if [ -n "$OUTPUT" ]; then
    echo "$REPORT" > "$OUTPUT"
    echo "Results written to ${OUTPUT}" >&2
else
    echo "$REPORT"
fi

# Detect GPU count and total VRAM from the first result's gpu_info
GPU_COUNT=0
GPU_TOTAL=""
GPU_NAME=""
if [ "$(echo "$RESULTS" | jq 'length')" -gt 0 ]; then
    FIRST_GPU_INFO=$(echo "$RESULTS" | jq -r '.[0].gpu_info // ""')
    if [ -n "$FIRST_GPU_INFO" ] && [ "$FIRST_GPU_INFO" != "null" ]; then
        # Parse with jq to handle GPU names with spaces (e.g., "Tesla K80")
        GPU_COUNT=$(echo "$FIRST_GPU_INFO" | grep -c .)
        GPU_NAME=$(echo "$FIRST_GPU_INFO" | head -1 | cut -d',' -f2 | sed 's/^ *//;s/ *$//')
        GPU_TOTAL=$(echo "$FIRST_GPU_INFO" | head -1 | cut -d',' -f4 | sed 's/^ *//;s/ *$//')
    fi
fi

# Print comparison table
echo "=== Results ===" >&2
echo "" >&2

# Print GPU info header (constant across all models)
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_LABELS=$(for i in $(seq 0 $((GPU_COUNT - 1))); do printf "GPU%d" "$i"; [ "$i" -lt $((GPU_COUNT - 1)) ] && printf ", " || true; done)
    echo "GPU: ${GPU_COUNT}x ${GPU_NAME} | VRAM: ${GPU_TOTAL} MiB each (${GPU_LABELS})" >&2
    echo "" >&2
fi

# Build dynamic VRAM column headers
VRAM_HDR=""
VRAM_SEP=""
for i in $(seq 0 $((GPU_COUNT - 1))); do
    VRAM_HDR="${VRAM_HDR}$(printf " %5s" "GPU$i")"
    VRAM_SEP="${VRAM_SEP}$(printf " %5s" "-----")"
done

printf "%-25s %5s %6s %7s %12s %10s %5s %s\n" \
    "MODEL" "CHECK" "IN" "OUT" "PROMPT tok/s" "GEN tok/s" "GPU%" "VRAM Used (MiB)" >&2
printf "%-25s %5s %6s %7s %12s %10s %5s" \
    "-------------------------" "-----" "------" "-------" "------------" "----------" "-----" >&2
printf "%s\n" "$VRAM_SEP" >&2

echo "$RESULTS" | jq -r '.[] |
    (.gpu_info | split("\n") | map(select(length > 0)) |
        if length > 0 then
            map(split(",") | map(gsub("^\\s+|\\s+$"; "")) | .[2])
            | join("|")
        else "n/a"
        end
    ) as $vram |
    ((.check.overall_pass // false) | if . then "PASS" else "FAIL" end) as $check |
    "\(.model)|\($check)|\(.in_tokens)|\(.out_tokens)|\(.prompt_eval_tps)|\(.eval_tps)|\(.gpu_offload_pct)|\($vram)"
' | while IFS='|' read -r model check in_tok out_tok prompt_tps eval_tps gpu_pct vram_rest; do
    # Build per-GPU used columns
    VRAM_COLS=""
    IFS='|' read -ra USED_VALS <<< "$vram_rest"
    for val in "${USED_VALS[@]}"; do
        VRAM_COLS="${VRAM_COLS}$(printf " %5s" "$val")"
    done
    printf "%-25s %5s %6s %7s %12s %10s %4s%%%s\n" \
        "$model" "$check" "$in_tok" "$out_tok" "$prompt_tps" "$eval_tps" "$gpu_pct" "$VRAM_COLS" >&2
done
echo "" >&2

# Surface the overall verdict so callers can rely on the exit code instead of
# parsing the JSON to find failed coherence checks.
if [ "$FAILED_CHECKS" -gt 0 ]; then
    echo "OUTPUT CHECK FAILED: ${FAILED_CHECKS} model(s) did not pass" >&2
    exit 1
fi
echo "OUTPUT CHECK PASSED: ${#MODELS[@]} model(s) produced valid responses" >&2
