#!/usr/bin/env bash
# judge_response.sh — send (prompt, output) to the LLM judge and parse verdict.
#
# Source it:
#   source "$(dirname "$0")/lib/judge_response.sh"
#   verdict=$(judge_response "$JUDGE_HOST" "$JUDGE_MODEL" "$MODEL" "$PROMPT" "$OUTPUT")
#
# Or self-test:
#   bash judge_response.sh --self-test
#
# Per CLAUDE.md → "LLM Judge Scope", the judge is ONLY for "is this LLM
# response meaningful for its prompt?" — not for static / log / exit-code
# checks. Those stay deterministic.
#
# The judge sees a single `output` field (combined response + thinking), not
# a {response, thinking} split. Empirically, judges anchor on "empty
# response = fail" when given the split with a rule that thinking counts, and
# ignore the rule. Hiding the field-name distinction eliminates the bias.

# judge_response JUDGE_HOST JUDGE_MODEL MODEL_UNDER_TEST PROMPT OUTPUT
#
# Verdict JSON: {pass: bool, reason: string}
# Fails closed: returns pass=false if the judge endpoint is unreachable, times
# out, or returns non-JSON. A missing judge must not silently pass a broken
# response.
judge_response() {
    local judge_host="$1" judge_model="$2"
    local model_under_test="$3" prompt_text="$4" output_text="$5"

    local judge_prompt
    judge_prompt=$(jq -nc \
        --arg model "$model_under_test" \
        --arg prompt "$prompt_text" \
        --arg output "$output_text" \
        '{
            role: "Decide whether the model output is meaningful for the prompt. A meaningful output addresses the prompt with coherent text in the right language and topic. Reject empty, garbled, repetitive nonsense, off-topic, or error-message output.",
            rules: [
                "Truncation at the token limit is fine if the visible text is coherent and on-topic.",
                "Accept reasonable variations in phrasing.",
                "An API-level error message in the output is FAIL."
            ],
            model_under_test: $model,
            prompt: $prompt,
            output: $output,
            respond: {
                format: "Respond with a single JSON object",
                fields: {
                    pass: "true if the output is meaningful and on-topic, false otherwise",
                    reason: "Brief explanation"
                }
            }
        }')

    local judge_payload judge_raw verdict_text
    judge_payload=$(jq -nc \
        --arg model "$judge_model" \
        --arg prompt "$judge_prompt" \
        '{model:$model, prompt:$prompt, stream:false, format:"json", options:{temperature:0.1, num_predict:512}}')

    judge_raw=$(curl -sf --max-time 300 "${judge_host}/api/generate" -d "$judge_payload" 2>/dev/null || true)
    if [ -z "$judge_raw" ]; then
        echo '{"pass":false,"reason":"judge endpoint unreachable or timed out"}'
        return
    fi

    verdict_text=$(echo "$judge_raw" | jq -r '.response // ""' 2>/dev/null)
    if [ -z "$verdict_text" ] || ! echo "$verdict_text" | jq empty 2>/dev/null; then
        echo '{"pass":false,"reason":"judge returned non-JSON response"}'
        return
    fi

    # Normalize the judge's output to exactly {pass, reason}. Coerce non-bool
    # `.pass` values (e.g. "yes") to false to fail closed.
    echo "$verdict_text" | jq -c '{pass: (.pass == true), reason: (.reason // "no reason given")}'
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        --help|-h|"")
            sed -n '2,21p' "$0" | sed 's/^# \?//'
            ;;
        --self-test)
            # Self-test verifies the function is defined, the judge prompt
            # template parses with placeholders, and the unreachable-fallback
            # works. Live judge calls need a running judge endpoint and are
            # exercised by the script that sources this helper.
            if ! type judge_response > /dev/null 2>&1; then
                echo "FAIL: judge_response not defined" >&2
                exit 1
            fi
            # Verify both jq templates render
            jq -nc --arg m test --arg p hi --arg o world '{
                role: "x", rules: ["a","b","c"],
                model_under_test: $m, prompt: $p, output: $o,
                respond: {format: "x", fields: {pass: "x", reason: "x"}}
            }' > /dev/null || { echo "FAIL: judge prompt template" >&2; exit 1; }
            jq -nc --arg m test --arg p hi '{model:$m, prompt:$p, stream:false, format:"json", options:{temperature:0.1, num_predict:512}}' \
                > /dev/null || { echo "FAIL: judge payload template" >&2; exit 1; }
            # Verify unreachable-host produces a pass=false verdict
            v=$(judge_response "http://localhost:1" "test" "m" "p" "o")
            actual_pass=$(echo "$v" | jq -r '.pass')
            if [ "$actual_pass" = "false" ]; then
                echo "PASS: unreachable judge falls back to pass=false"
            else
                echo "FAIL: expected pass=false, got: $v" >&2
                exit 1
            fi
            echo "PASS: judge_response self-test"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Try --help or --self-test" >&2
            exit 1
            ;;
    esac
fi
