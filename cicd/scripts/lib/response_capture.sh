#!/usr/bin/env bash
# response_capture.sh — call ollama /api/generate and emit an enriched
# JSON record with response, thinking, perf metrics, and done_reason.
#
# Source it:
#   source "$(dirname "$0")/lib/response_capture.sh"
#   result=$(response_capture "$HOST" "$MODEL" "$PROMPT" 128 2048)
#
# Or self-test:
#   bash response_capture.sh --self-test
#
# Output JSON shape (per call):
#   {
#     model, in_tokens, out_tokens,
#     prompt_eval_tps, eval_tps,
#     total_duration_s, load_duration_s,
#     done_reason, response, thinking
#   }

# response_capture HOST MODEL PROMPT [num_predict] [num_ctx]
#
# Performs: warmup (1-token prime) → benchmark generate → unload model.
# Returns the enriched per-model JSON record on stdout, status 0 on success.
# On failure (unreachable host, no response), writes an error to stderr and
# returns non-zero with no stdout.
response_capture() {
    local host="$1" model="$2" prompt="$3"
    local num_predict="${4:-128}" num_ctx="${5:-2048}"

    # Warmup (loads model into GPU, primes any caches)
    curl -sf "${host}/api/generate" \
        -d "$(jq -nc --arg m "$model" --argjson ctx "$num_ctx" \
            '{model:$m, prompt:"Hi", stream:false, options:{num_predict:1, num_ctx:$ctx}}')" \
        > /dev/null 2>&1 || true

    # Benchmark call (temperature=0, fixed seed for stability)
    local raw
    raw=$(curl -sf "${host}/api/generate" \
        -d "$(jq -nc \
            --arg model "$model" \
            --arg prompt "$prompt" \
            --argjson num_predict "$num_predict" \
            --argjson num_ctx "$num_ctx" \
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

    if [ -z "$raw" ]; then
        echo "response_capture: no response from ${model} at ${host}" >&2
        return 1
    fi

    # Project the API response into a stable schema. Keep `response` and
    # `thinking` separate — thinking models (qwen3.5, qwen3-vl, deepseek-r1,
    # gemma4) can produce coherent output entirely inside `.thinking` while
    # leaving `.response` empty when num_predict caps generation early.
    echo "$raw" | jq \
        --arg model "$model" \
        '{
            model: $model,
            in_tokens: .prompt_eval_count,
            out_tokens: .eval_count,
            prompt_eval_tps: (if .prompt_eval_duration > 0 then (.prompt_eval_count / (.prompt_eval_duration / 1e9) * 100 | round / 100) else 0 end),
            eval_tps: (if .eval_duration > 0 then (.eval_count / (.eval_duration / 1e9) * 100 | round / 100) else 0 end),
            total_duration_s: ((.total_duration // 0) / 1e9 * 100 | round / 100),
            load_duration_s: ((.load_duration // 0) / 1e9 * 100 | round / 100),
            done_reason: (.done_reason // ""),
            response: (.response // ""),
            thinking: (.thinking // "")
        }'

    # Unload model so the next caller starts from a clean state
    curl -sf "${host}/api/generate" \
        -d "{\"model\":\"${model}\",\"keep_alive\":0}" > /dev/null 2>&1 || true
}

# Direct invocation: --help or --self-test
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        --help|-h|"")
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            ;;
        --self-test)
            # Self-test only verifies the function is defined and the jq
            # template parses with placeholder values. Real API calls require
            # a running ollama instance — exercised by the script that sources
            # this helper.
            if ! type response_capture > /dev/null 2>&1; then
                echo "FAIL: response_capture not defined" >&2
                exit 1
            fi
            # Verify the jq projection template parses against a synthetic
            # /api/generate response.
            synthetic='{"prompt_eval_count":10,"eval_count":20,"prompt_eval_duration":1000000000,"eval_duration":2000000000,"total_duration":3000000000,"load_duration":500000000,"done_reason":"stop","response":"hi","thinking":""}'
            echo "$synthetic" | jq --arg model "test" '{
                model:$model, in_tokens:.prompt_eval_count, out_tokens:.eval_count,
                prompt_eval_tps:(if .prompt_eval_duration > 0 then (.prompt_eval_count / (.prompt_eval_duration / 1e9) * 100 | round / 100) else 0 end),
                eval_tps:(if .eval_duration > 0 then (.eval_count / (.eval_duration / 1e9) * 100 | round / 100) else 0 end),
                total_duration_s:((.total_duration // 0) / 1e9 * 100 | round / 100),
                load_duration_s:((.load_duration // 0) / 1e9 * 100 | round / 100),
                done_reason:(.done_reason // ""),
                response:(.response // ""),
                thinking:(.thinking // "")
            }' > /dev/null || { echo "FAIL: jq projection failed" >&2; exit 1; }
            echo "PASS: response_capture defined and jq projection parses"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Try --help or --self-test" >&2
            exit 1
            ;;
    esac
fi
