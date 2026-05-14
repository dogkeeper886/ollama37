#!/usr/bin/env bash
# container_log_snip.sh — extract a slice of a container's logs.
#
# Source it:
#   source "$(dirname "$0")/lib/container_log_snip.sh"
#   slice=$(container_log_snip ollama37 "===TEST:START===" "===TEST:END===")
#
# Or self-test:
#   bash container_log_snip.sh --self-test
#
# Two modes:
#   - Marker mode: snip logs between two literal markers (inclusive).
#     The script-under-test is expected to write the markers via
#     `docker exec <container> sh -c "echo MARKER"` or similar before/after
#     the action of interest.
#   - Tail mode: grab the last N seconds of logs via `docker logs --since`.
#     Less precise (no test boundary), simpler for one-off captures.

# container_log_snip CONTAINER START_MARKER END_MARKER
#
# Captures container stdout+stderr from the START_MARKER line through the
# END_MARKER line (inclusive). Returns the slice on stdout; empty if either
# marker is not found.
container_log_snip() {
    local container="$1" start_marker="$2" end_marker="$3"
    docker logs "$container" 2>&1 | \
        sed -n "/$(printf '%s' "$start_marker" | sed 's:[][\\/.*^$]:\\&:g')/,/$(printf '%s' "$end_marker" | sed 's:[][\\/.*^$]:\\&:g')/p"
}

# container_log_tail CONTAINER SECONDS
#
# Returns logs from the last N seconds. Coarser than the marker mode but
# simpler when the caller doesn't need test boundaries.
container_log_tail() {
    local container="$1" seconds="${2:-60}"
    docker logs --since "${seconds}s" "$container" 2>&1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        --help|-h|"")
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            ;;
        --self-test)
            if ! type container_log_snip > /dev/null 2>&1; then
                echo "FAIL: container_log_snip not defined" >&2
                exit 1
            fi
            if ! type container_log_tail > /dev/null 2>&1; then
                echo "FAIL: container_log_tail not defined" >&2
                exit 1
            fi
            # Verify the sed escaping handles markers with regex metacharacters
            # without crashing (we don't have docker available locally for a
            # live test).
            test_input=$(printf 'noise\n===START.x===\nbody\n===END.y===\ntrailing\n')
            sliced=$(echo "$test_input" | sed -n "/$(printf '%s' '===START.x===' | sed 's:[][\\/.*^$]:\\&:g')/,/$(printf '%s' '===END.y===' | sed 's:[][\\/.*^$]:\\&:g')/p")
            if [ "$sliced" = "===START.x===
body
===END.y===" ]; then
                echo "PASS: marker escaping handles regex metacharacters"
            else
                echo "FAIL: expected 3-line slice, got: $sliced" >&2
                exit 1
            fi
            echo "PASS: container_log_snip self-test"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Try --help or --self-test" >&2
            exit 1
            ;;
    esac
fi
