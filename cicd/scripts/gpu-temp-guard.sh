#!/bin/bash
# gpu-temp-guard.sh - run a command under a K80 GPU temperature guard.
#
# Runs <command>, polling nvidia-smi for the hottest GPU die. If it crosses the
# threshold, the command (and its whole process group) is killed and the script
# exits 75 - a distinct "overheat" code the caller can act on. Otherwise it exits
# with the command's own code. Either way the peak temperature is appended to
# $GITHUB_STEP_SUMMARY (and stderr), so every run records how hot it got.
#
# Usage: gpu-temp-guard.sh [--threshold N] [--interval S] -- <command> [args...]
#
# Knobs (flags override env):
#   --threshold / GPU_TEMP_LIMIT     hard abort, °C   (default 80)
#   --interval  / GPU_TEMP_INTERVAL  poll period, s   (default 10)
set -u

OVERHEAT_RC=75
MAX_UNREADABLE=3        # consecutive unreadable polls -> abort (a hung driver / GPU off the bus is
                        # exactly what an overheat causes; blindly continuing defeats the guard)
threshold="${GPU_TEMP_LIMIT:-80}"
interval="${GPU_TEMP_INTERVAL:-10}"

usage() {
    echo "Usage: $0 [--threshold N] [--interval S] -- <command> [args...]" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --threshold) [ $# -ge 2 ] || { echo "gpu-temp-guard: --threshold needs a value" >&2; usage; exit 2; }; threshold="$2"; shift 2 ;;
        --interval)  [ $# -ge 2 ] || { echo "gpu-temp-guard: --interval needs a value"  >&2; usage; exit 2; }; interval="$2";  shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "gpu-temp-guard: unknown option '$1'" >&2; usage; exit 2 ;;
    esac
done
if [ $# -eq 0 ]; then
    echo "gpu-temp-guard: no command given" >&2; usage; exit 2
fi

# Record a line in the step summary (CI) and on stderr (always visible). stdout is
# left untouched so the wrapped command's `| tee` / `> file.json` redirects are clean.
note() {
    echo "$1" >&2
    [ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$1" >> "$GITHUB_STEP_SUMMARY"
    return 0
}

# No NVIDIA tooling (e.g. a dev box without a card): run unguarded rather than fail.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu-temp-guard: nvidia-smi not found - running WITHOUT a temperature guard" >&2
    "$@"
    exit $?
fi

max_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
        | sort -rn | head -1
}

set -m                  # job control: the background job gets its own process group
"$@" &
cmd_pid=$!
set +m

peak=0
had_reading=0
unreadable=0
abort_reason=""
while kill -0 "$cmd_pid" 2>/dev/null; do
    t="$(max_temp)"
    t="${t%%.*}"                       # tolerate a decimal reading, e.g. "82.0" -> "82"
    if [[ "$t" =~ ^[0-9]+$ ]]; then
        had_reading=1
        unreadable=0
        [ "$t" -gt "$peak" ] && peak="$t"
        if [ "$t" -gt "$threshold" ]; then
            abort_reason="OVERHEAT - peak ${peak}°C exceeded ${threshold}°C"
            break
        fi
    else
        # nvidia-smi gave no usable reading. A few in a row (a hung driver / GPU off the
        # bus) means we can't verify safety, so fail closed rather than run blind.
        unreadable=$((unreadable + 1))
        if [ "$unreadable" -ge "$MAX_UNREADABLE" ]; then
            abort_reason="temperature UNREADABLE for ${unreadable} polls (nvidia-smi failing) - cannot verify"
            break
        fi
    fi
    sleep "$interval"
done

if [ -n "$abort_reason" ]; then
    kill -TERM -"$cmd_pid" 2>/dev/null     # negative pid = the whole process group
    sleep 2
    kill -KILL -"$cmd_pid" 2>/dev/null
fi
wait "$cmd_pid" 2>/dev/null
cmd_rc=$?

if [ -n "$abort_reason" ]; then
    note "🌡️ GPU GUARD ABORT - ${abort_reason}; run killed"
    exit "$OVERHEAT_RC"
fi
if [ "$had_reading" -eq 0 ]; then
    note "🌡️ Peak GPU temp: unknown (nvidia-smi returned no valid reading)"
else
    note "🌡️ Peak GPU temp: ${peak}°C (limit ${threshold}°C)"
fi
exit "$cmd_rc"
