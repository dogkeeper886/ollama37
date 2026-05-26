#!/usr/bin/env bash
# test-mlx-smoke.sh — Phase B link-surface probe for the MLX-on-K80 port (#188).
#
# Builds libmlx.a + qwen_smoke (the on-path forward-surface probe exe) in the
# builder image, then runs qwen_smoke in the PRODUCTION RUNTIME image with the
# K80 GPUs exposed. The split is intentional: building in the builder image
# masks runtime gaps (libstdc++ delta, missing CUDA toolkit libs, etc.); only
# running in the runtime image surfaces what the eventual mlx runner exe will
# actually hit. See docs/traces/mlx-cuda-runtime-path.md for context.
#
# Production ollama37 stop/start is handled by the caller (the workflow YAML)
# so the K80 GPUs are free.
#
# Output:
#   /tmp/test-mlx-smoke-results.json — structured result with build + run sections
#   /tmp/test-mlx-smoke-build/        — build artifacts (libmlx.a, qwen_smoke, logs)
#
# Exit codes (load-bearing — workflow red/green follows):
#   0 = build OK + qwen_smoke ran and printed the success line
#   1 = build failed (compile or link error in libmlx.a / qwen_smoke)
#   2 = build OK but qwen_smoke errored at runtime (linker resolve / throw)
#   3 = environment issue (missing images, no GPU, etc.)
#
# Usage: test-mlx-smoke.sh [--build-image IMG] [--runtime-image IMG]
#                          [--build-dir DIR]   [--results-json PATH]
#                          [--skip-build]

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-ollama37-builder:latest}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ollama37:latest}"
BUILD_DIR="${BUILD_DIR:-/tmp/test-mlx-smoke-build}"
RESULTS_JSON="${RESULTS_JSON:-/tmp/test-mlx-smoke-results.json}"
SRC_DIR="${SRC_DIR:-$(git rev-parse --show-toplevel)}"
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-image) BUILD_IMAGE="$2"; shift 2 ;;
        --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --results-json) RESULTS_JSON="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done

GIT_SHA="$(cd "$SRC_DIR" && git rev-parse HEAD)"

# Result fields, initialized to "unknown / not reached" defaults. Each phase
# updates the ones it owns; trap on EXIT serializes whatever's in scope.
BUILD_STATUS="not_run"
BUILD_DURATION=0
BUILD_LOG_TAIL=""
LIBMLX_SIZE=0
LIBMLX_OBJECT_COUNT=0
EXE_SIZE=0
EXE_PATH=""

RUN_STATUS="skipped"
RUN_EXIT_CODE=-1
RUN_DURATION=0
RUN_STDOUT=""
RUN_STDERR=""
LDD_OUTPUT=""
LDD_MISSING=""

GPU_DEVICE_COUNT=0
GPU_NAMES=""

FINAL_EXIT=0

# shellcheck disable=SC2329 # invoked via `trap` below
write_results() {
    # Always run on exit; emits whatever was filled in. jq object keys are
    # quoted to avoid reserved-word collisions (per lint-ci skill).
    jq -n \
        --arg sha "$GIT_SHA" \
        --arg build_image "$BUILD_IMAGE" \
        --arg runtime_image "$RUNTIME_IMAGE" \
        --arg build_status "$BUILD_STATUS" \
        --argjson build_duration "$BUILD_DURATION" \
        --arg build_log_tail "$BUILD_LOG_TAIL" \
        --argjson libmlx_size "$LIBMLX_SIZE" \
        --argjson libmlx_objs "$LIBMLX_OBJECT_COUNT" \
        --argjson exe_size "$EXE_SIZE" \
        --arg exe_path "$EXE_PATH" \
        --arg run_status "$RUN_STATUS" \
        --argjson run_exit "$RUN_EXIT_CODE" \
        --argjson run_duration "$RUN_DURATION" \
        --arg run_stdout "$RUN_STDOUT" \
        --arg run_stderr "$RUN_STDERR" \
        --arg ldd "$LDD_OUTPUT" \
        --arg ldd_missing "$LDD_MISSING" \
        --argjson gpu_count "$GPU_DEVICE_COUNT" \
        --arg gpu_names "$GPU_NAMES" \
        --argjson final_exit "$FINAL_EXIT" \
        '{
          "git_sha": $sha,
          "test": "mlx-smoke",
          "final_exit": $final_exit,
          "build": {
            "image": $build_image,
            "status": $build_status,
            "duration_sec": $build_duration,
            "libmlx_size_bytes": $libmlx_size,
            "libmlx_object_count": $libmlx_objs,
            "exe_size_bytes": $exe_size,
            "exe_path": $exe_path,
            "log_tail": $build_log_tail
          },
          "run": {
            "image": $runtime_image,
            "status": $run_status,
            "exit_code": $run_exit,
            "duration_sec": $run_duration,
            "stdout": $run_stdout,
            "stderr": $run_stderr,
            "ldd_output": $ldd,
            "ldd_missing": $ldd_missing
          },
          "gpu": {
            "device_count": $gpu_count,
            "names": $gpu_names
          }
        }' > "$RESULTS_JSON"
    echo "results: $RESULTS_JSON"
}
trap write_results EXIT

# ------------------------------------------------------------- preflight ----
docker image inspect "$BUILD_IMAGE" >/dev/null 2>&1 || {
    echo "::error::build image not found: $BUILD_IMAGE" >&2
    FINAL_EXIT=3; exit 3
}
docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || {
    echo "::error::runtime image not found: $RUNTIME_IMAGE" >&2
    FINAL_EXIT=3; exit 3
}
GPU_DEVICE_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
GPU_NAMES=$(nvidia-smi -L 2>/dev/null | head -3 | tr '\n' ';' || true)
if [ "$GPU_DEVICE_COUNT" -eq 0 ]; then
    echo "::error::no GPUs visible via nvidia-smi" >&2
    FINAL_EXIT=3; exit 3
fi

# ----------------------------------------------------------------- build ----
if [ "$SKIP_BUILD" = true ] && [ -x "$BUILD_DIR/qwen_smoke" ]; then
    echo "skip_build: reusing $BUILD_DIR/qwen_smoke"
    BUILD_STATUS="skipped"
    EXE_PATH="$BUILD_DIR/qwen_smoke"
    EXE_SIZE=$(stat -c '%s' "$EXE_PATH" 2>/dev/null || echo 0)
    if [ -f "$BUILD_DIR/mlx_build/libmlx.a" ]; then
        LIBMLX_SIZE=$(stat -c '%s' "$BUILD_DIR/mlx_build/libmlx.a" 2>/dev/null || echo 0)
        LIBMLX_OBJECT_COUNT=$(ar t "$BUILD_DIR/mlx_build/libmlx.a" 2>/dev/null | wc -l || echo 0)
    fi
else
    rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
    BUILD_STATUS="running"
    BUILD_START=$(date +%s)

    set +e
    docker run --rm \
        -v "$SRC_DIR:/src:ro" \
        -v "$BUILD_DIR:/build" \
        "$BUILD_IMAGE" \
        bash -c '
            set -euo pipefail
            cmake -S /src/x/mlxrunner -B /build -DCMAKE_BUILD_TYPE=Release
            cmake --build /build -j"$(nproc)"
        ' > "$BUILD_DIR/build.log" 2>&1
    BUILD_RC=$?
    set -e
    BUILD_DURATION=$(( $(date +%s) - BUILD_START ))
    # tail -c keeps the bytes finite for JSON embedding
    BUILD_LOG_TAIL=$(tail -c 4000 "$BUILD_DIR/build.log" || true)

    if [ $BUILD_RC -ne 0 ] || [ ! -x "$BUILD_DIR/qwen_smoke" ]; then
        BUILD_STATUS="failed"
        echo "::error::build failed (rc=$BUILD_RC); see $BUILD_DIR/build.log"
        FINAL_EXIT=1; exit 1
    fi

    BUILD_STATUS="success"
    EXE_PATH="$BUILD_DIR/qwen_smoke"
    EXE_SIZE=$(stat -c '%s' "$EXE_PATH" 2>/dev/null || echo 0)
    if [ -f "$BUILD_DIR/mlx_build/libmlx.a" ]; then
        LIBMLX_SIZE=$(stat -c '%s' "$BUILD_DIR/mlx_build/libmlx.a" 2>/dev/null || echo 0)
        LIBMLX_OBJECT_COUNT=$(ar t "$BUILD_DIR/mlx_build/libmlx.a" 2>/dev/null | wc -l || echo 0)
    fi
    echo "build OK: libmlx.a ${LIBMLX_SIZE}B ${LIBMLX_OBJECT_COUNT} objs / qwen_smoke ${EXE_SIZE}B"
fi

# ------------------------------------------------------------------- run ----
# Run qwen_smoke in the PRODUCTION RUNTIME image (not the builder), with K80
# GPUs exposed. Surfaces real ld gaps that builder-side runs would mask.
RUN_STATUS="running"
RUN_START=$(date +%s)
set +e
docker run --rm --gpus all \
    -v "$BUILD_DIR:/build:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint bash \
    "$RUNTIME_IMAGE" \
    -c '
        set +e
        ldd /build/qwen_smoke 2>&1
        echo "--- ldd-end ---"
        /build/qwen_smoke
        echo "--- exit=$? ---"
    ' > "$BUILD_DIR/run.stdout" 2> "$BUILD_DIR/run.stderr"
RUN_RC=$?
set -e
RUN_DURATION=$(( $(date +%s) - RUN_START ))

# Split ldd portion out of stdout; tail -c keeps embedded bytes bounded.
LDD_OUTPUT=$(sed -n '1,/--- ldd-end ---/p' "$BUILD_DIR/run.stdout" \
             | head -n -1 | tail -c 4000 || true)
RUN_STDOUT=$(sed -n '/--- ldd-end ---/,$p' "$BUILD_DIR/run.stdout" \
             | tail -n +2 | tail -c 4000 || true)
RUN_STDERR=$(tail -c 4000 "$BUILD_DIR/run.stderr" || true)
LDD_MISSING=$(echo "$LDD_OUTPUT" | grep -E "not found|undefined" | head -c 2000 || true)
RUN_EXIT_CODE=$RUN_RC

if [ $RUN_RC -eq 0 ] && echo "$RUN_STDOUT" | grep -q "qwen_smoke: forward-surface link OK"; then
    RUN_STATUS="success"
    echo "run OK: qwen_smoke completed in ${RUN_DURATION}s"
    FINAL_EXIT=0
else
    RUN_STATUS="failed"
    echo "::error::qwen_smoke run failed (rc=$RUN_RC, duration=${RUN_DURATION}s)"
    if [ -n "$LDD_MISSING" ]; then
        echo "::error::missing libs (ldd): $LDD_MISSING"
    fi
    FINAL_EXIT=2
fi

exit $FINAL_EXIT
