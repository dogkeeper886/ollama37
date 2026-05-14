#!/usr/bin/env bash
# simple_check.sh — deterministic non-empty-output check.
#
# Source it:
#   source "$(dirname "$0")/lib/simple_check.sh"
#   verdict=$(simple_check "$RESPONSE_TEXT" "$THINKING_TEXT") || echo "fail"
#
# Or self-test:
#   bash simple_check.sh --self-test
#
# Returns 0 + {pass:true, ...} JSON if either field is non-empty after
# whitespace trim. Returns 1 + {pass:false, ...} JSON if both are empty.
# Used to decide whether to even invoke the LLM judge — an empty model
# output is a hard fail regardless of judge opinion.

# simple_check RESPONSE [THINKING]
#
# Verdict JSON: {pass: bool, reason: string, source: "response" | "thinking" | "none"}
# Exit code: 0 if pass, 1 if fail.
simple_check() {
    local response_text="${1:-}"
    local thinking_text="${2:-}"
    local r_trim t_trim
    r_trim=$(printf '%s' "$response_text" | tr -d '[:space:]')
    t_trim=$(printf '%s' "$thinking_text" | tr -d '[:space:]')
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

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        --help|-h|"")
            sed -n '2,15p' "$0" | sed 's/^# \?//'
            ;;
        --self-test)
            pass=0; fail=0
            run() {
                local name="$1" expected_pass="$2"; shift 2
                local v rc
                v=$(simple_check "$@") && rc=0 || rc=1
                local actual_pass
                actual_pass=$(echo "$v" | jq -r '.pass')
                if [ "$actual_pass" = "$expected_pass" ] && [ "$rc" -eq "$([[ "$expected_pass" = true ]] && echo 0 || echo 1)" ]; then
                    echo "PASS: $name"
                    pass=$((pass+1))
                else
                    echo "FAIL: $name — got $v (rc=$rc)" >&2
                    fail=$((fail+1))
                fi
            }
            run "both empty"                false  ""        ""
            run "response only"             true   "hello"   ""
            run "thinking only"             true   ""        "reasoning..."
            run "both present"              true   "answer"  "thinking..."
            run "whitespace-only response"  true   "   "     "real thinking"
            run "whitespace-only both"      false  "   "     "  "
            echo "--- $pass passed, $fail failed ---"
            [ "$fail" -eq 0 ] || exit 1
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Try --help or --self-test" >&2
            exit 1
            ;;
    esac
fi
