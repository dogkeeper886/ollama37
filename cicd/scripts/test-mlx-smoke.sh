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
# MODEL_DIR: optional; if set + exists + qwen_load exe present, run the
# load-side smoke against shard 1. Default points where the qwen3.6:35b-mlx
# weights were pulled (~19 GB; pulled offline per #187 office-hour rule).
MODEL_DIR="${MODEL_DIR:-/home/jack/models/qwen3.6-35b-a3b-4bit}"
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-image) BUILD_IMAGE="$2"; shift 2 ;;
        --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --results-json) RESULTS_JSON="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
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

# qwen_load (Phase C load-surface probe) — runs only if MODEL_DIR resolves.
LOAD_STATUS="skipped"
LOAD_EXIT_CODE=-1
LOAD_DURATION=0
LOAD_STDOUT=""
LOAD_STDERR=""
LOAD_MODEL_DIR=""
LOAD_SHARD=""

# k80_p2p_probe (Phase D spike) — pure-CUDA multi-die probe. Different from
# qwen_smoke/qwen_load: NO CUDA_VISIBLE_DEVICES restriction so it sees all
# K80 dies, and NO model dir mount.
P2P_STATUS="skipped"
P2P_EXIT_CODE=-1
P2P_DURATION=0
P2P_STDOUT=""
P2P_STDERR=""

# qwen_load_go (Phase D.2 step 1) — Go cgo binary that drives libmlx.a via
# the C ABI shim. Mirrors qwen_load's first line ("loaded N tensors") to
# prove the cgo plumbing works end-to-end.
CGO_STATUS="skipped"
CGO_EXIT_CODE=-1
CGO_DURATION=0
CGO_STDOUT=""
CGO_STDERR=""

# qwen_runner (Phase D.3 step 1) — pure-Go exe; parses config.json and
# prints the language-model config block. No cgo, no MLX, no GPU. This
# is the entry point that future steps build the model assembly on.
RUNNER_STATUS="skipped"
RUNNER_EXIT_CODE=-1
RUNNER_DURATION=0
RUNNER_STDOUT=""
RUNNER_STDERR=""

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
        --arg load_status "$LOAD_STATUS" \
        --argjson load_exit "$LOAD_EXIT_CODE" \
        --argjson load_duration "$LOAD_DURATION" \
        --arg load_stdout "$LOAD_STDOUT" \
        --arg load_stderr "$LOAD_STDERR" \
        --arg load_model_dir "$LOAD_MODEL_DIR" \
        --arg load_shard "$LOAD_SHARD" \
        --arg p2p_status "$P2P_STATUS" \
        --argjson p2p_exit "$P2P_EXIT_CODE" \
        --argjson p2p_duration "$P2P_DURATION" \
        --arg p2p_stdout "$P2P_STDOUT" \
        --arg p2p_stderr "$P2P_STDERR" \
        --arg cgo_status "$CGO_STATUS" \
        --argjson cgo_exit "$CGO_EXIT_CODE" \
        --argjson cgo_duration "$CGO_DURATION" \
        --arg cgo_stdout "$CGO_STDOUT" \
        --arg cgo_stderr "$CGO_STDERR" \
        --arg runner_status "$RUNNER_STATUS" \
        --argjson runner_exit "$RUNNER_EXIT_CODE" \
        --argjson runner_duration "$RUNNER_DURATION" \
        --arg runner_stdout "$RUNNER_STDOUT" \
        --arg runner_stderr "$RUNNER_STDERR" \
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
          "load": {
            "status": $load_status,
            "exit_code": $load_exit,
            "duration_sec": $load_duration,
            "stdout": $load_stdout,
            "stderr": $load_stderr,
            "model_dir": $load_model_dir,
            "shard": $load_shard
          },
          "p2p": {
            "status": $p2p_status,
            "exit_code": $p2p_exit,
            "duration_sec": $p2p_duration,
            "stdout": $p2p_stdout,
            "stderr": $p2p_stderr
          },
          "cgo": {
            "status": $cgo_status,
            "exit_code": $cgo_exit,
            "duration_sec": $cgo_duration,
            "stdout": $cgo_stdout,
            "stderr": $cgo_stderr
          },
          "runner": {
            "status": $runner_status,
            "exit_code": $runner_exit,
            "duration_sec": $runner_duration,
            "stdout": $runner_stdout,
            "stderr": $runner_stderr
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
            # Phase D.2 step 1 (#189): Go cgo binary that drives libmlx.a
            # via the C ABI shim. Skip cleanly if anything errors (most
            # commonly: libmlx_cabi.a archive missing -> means cmake didn'\''t
            # finish; reported by the cmake build above).
            if [ -f /build/libmlx_cabi.a ] && [ -f /build/mlx_build/libmlx.a ]; then
              # /usr/local/cuda-11.4/lib64/stubs holds libcuda.so (the
              # driver lib stub for link-time resolution; runtime uses the
              # real libcuda.so.1 from /usr/local/nvidia/lib64 via the
              # nvidia-container mount).
              export CGO_LDFLAGS="-L/build -L/build/mlx_build -L/usr/local/cuda-11.4/lib64 -L/usr/local/cuda-11.4/lib64/stubs"
              export GOCACHE=/tmp/gocache
              # GOFLAGS env applies before any go subcommand, including
              # module-load-time VCS discovery (cmdline -buildvcs=false alone
              # isn'\''t enough — first CI run still hit "error obtaining VCS
              # status" despite the flag). Belt-and-suspenders: also tell git
              # the bind-mounted source is safe to read (the .git dir is
              # owned by the host user, root inside the container).
              export GOFLAGS="-buildvcs=false"
              mkdir -p "$GOCACHE"
              git config --global --add safe.directory /src || true
              (cd /src/x/mlxrunner/go && go build -buildvcs=false -o /build/qwen_load_go ./cmd/qwen_load_go) \
                  || echo "::warning::Go cgo build of qwen_load_go failed; see build.log"
              # Phase D.3 step 2 (#189): qwen_runner imports the mlx Go
              # package, so its build is now also gated on libmlx_cabi
              # and libmlx.a (same conditional as qwen_load_go).
              (cd /src/x/mlxrunner/go && go build -buildvcs=false -o /build/qwen_runner ./cmd/qwen_runner) \
                  || echo "::warning::Go build of qwen_runner failed; see build.log"
            else
              echo "qwen_load_go: skipped Go build (mlx_cabi or mlx archive missing)"
              echo "qwen_runner: skipped Go build (mlx_cabi or mlx archive missing)"
            fi
            # Pure-Go package tests always run; no libmlx dependency.
            export GOCACHE="${GOCACHE:-/tmp/gocache}"
            export GOFLAGS="${GOFLAGS:--buildvcs=false}"
            mkdir -p "$GOCACHE"
            git config --global --add safe.directory /src || true
            (cd /src/x/mlxrunner/go && go test -buildvcs=false ./models/qwen3_6_a3b/...) \
                || echo "::warning::Go tests for qwen3_6_a3b package failed; see build.log"
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

# -------------------------------------------------------------- qwen_load ----
# Phase C load-surface probe. Runs only if (a) qwen_load exe was built, and
# (b) MODEL_DIR resolves to a directory containing shard 1 of qwen3.6-35b-mlx.
# Failure here doesn't override an existing FINAL_EXIT=2 from the run step,
# but it does turn a green smoke into a load-failed exit-2.
LOAD_SHARD_FILE="model-00001-of-00004.safetensors"
if [ -x "$BUILD_DIR/qwen_load" ] && [ -f "$MODEL_DIR/$LOAD_SHARD_FILE" ]; then
    LOAD_MODEL_DIR="$MODEL_DIR"
    LOAD_SHARD="$LOAD_SHARD_FILE"
    LOAD_STATUS="running"
    LOAD_START=$(date +%s)
    set +e
    docker run --rm --gpus all \
        -v "$BUILD_DIR:/build:ro" \
        -v "$MODEL_DIR:/models:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint bash \
        "$RUNTIME_IMAGE" \
        -c "/build/qwen_load /models/$LOAD_SHARD_FILE" \
        > "$BUILD_DIR/load.stdout" 2> "$BUILD_DIR/load.stderr"
    LOAD_RC=$?
    set -e
    LOAD_DURATION=$(( $(date +%s) - LOAD_START ))
    LOAD_STDOUT=$(tail -c 4000 "$BUILD_DIR/load.stdout" || true)
    LOAD_STDERR=$(tail -c 4000 "$BUILD_DIR/load.stderr" || true)
    LOAD_EXIT_CODE=$LOAD_RC
    # Match both old ("load + reduce OK") and new ("load + reduce + take OK")
    # success strings so the script keeps working as qwen_load gains more probes.
    if [ $LOAD_RC -eq 0 ] && echo "$LOAD_STDOUT" | grep -qE "qwen_load: load \+ reduce.* OK"; then
        LOAD_STATUS="success"
        echo "load OK: qwen_load completed in ${LOAD_DURATION}s"
    else
        LOAD_STATUS="failed"
        echo "::error::qwen_load failed (rc=$LOAD_RC, duration=${LOAD_DURATION}s)"
        if [ "$FINAL_EXIT" -eq 0 ]; then
            FINAL_EXIT=2
        fi
    fi
else
    if [ ! -f "$MODEL_DIR/$LOAD_SHARD_FILE" ]; then
        echo "qwen_load: skipped (no $MODEL_DIR/$LOAD_SHARD_FILE)"
    fi
    if [ ! -x "$BUILD_DIR/qwen_load" ]; then
        echo "qwen_load: skipped (exe not built)"
    fi
fi

# ------------------------------------------------------- qwen_load_go ----
# Phase D.2 step 1 (#189) — Go cgo plumbing probe. Runs only if the Go
# binary built AND MODEL_DIR resolves to a shard. Mirrors qwen_load's
# behavior via cgo; informational (failure doesn't override FINAL_EXIT
# since the cgo runner is still in its earliest stage).
if [ -x "$BUILD_DIR/qwen_load_go" ] && [ -f "$MODEL_DIR/$LOAD_SHARD_FILE" ]; then
    CGO_STATUS="running"
    CGO_START=$(date +%s)
    set +e
    docker run --rm --gpus all \
        -v "$BUILD_DIR:/build:ro" \
        -v "$MODEL_DIR:/models:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint bash \
        "$RUNTIME_IMAGE" \
        -c "/build/qwen_load_go /models/$LOAD_SHARD_FILE" \
        > "$BUILD_DIR/cgo.stdout" 2> "$BUILD_DIR/cgo.stderr"
    CGO_RC=$?
    set -e
    CGO_DURATION=$(( $(date +%s) - CGO_START ))
    CGO_STDOUT=$(tail -c 4000 "$BUILD_DIR/cgo.stdout" || true)
    CGO_STDERR=$(tail -c 4000 "$BUILD_DIR/cgo.stderr" || true)
    CGO_EXIT_CODE=$CGO_RC
    if [ $CGO_RC -eq 0 ] && echo "$CGO_STDOUT" | grep -q "qwen_load_go: cgo OK"; then
        CGO_STATUS="success"
        echo "cgo OK: qwen_load_go completed in ${CGO_DURATION}s"
    else
        CGO_STATUS="failed"
        echo "::warning::qwen_load_go failed (rc=$CGO_RC) — informational at this stage"
    fi
else
    if [ ! -x "$BUILD_DIR/qwen_load_go" ]; then
        echo "qwen_load_go: skipped (Go cgo exe not built)"
    fi
fi

# ------------------------------------------------------- qwen_runner -----
# Phase D.3 step 1 (#189) — config parser; pure Go.
# Phase D.3 step 2 (#189) — also loads shard 1 via the mlx Go wrapper
# (now cgo + GPU required). Runs only if the exe built and the model
# dir has both config.json and shard 1. Failure DOES override
# FINAL_EXIT (load-bearing for D.3, unlike the still-experimental cgo
# block).
if [ -x "$BUILD_DIR/qwen_runner" ] && [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/$LOAD_SHARD_FILE" ]; then
    RUNNER_STATUS="running"
    RUNNER_START=$(date +%s)
    set +e
    docker run --rm --gpus all \
        -v "$BUILD_DIR:/build:ro" \
        -v "$MODEL_DIR:/models:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint bash \
        "$RUNTIME_IMAGE" \
        -c "/build/qwen_runner -model /models -shard /models/$LOAD_SHARD_FILE -load-weights -probe-ops" \
        > "$BUILD_DIR/runner.stdout" 2> "$BUILD_DIR/runner.stderr"
    RUNNER_RC=$?
    set -e
    RUNNER_DURATION=$(( $(date +%s) - RUNNER_START ))
    RUNNER_STDOUT=$(tail -c 4000 "$BUILD_DIR/runner.stdout" || true)
    RUNNER_STDERR=$(tail -c 4000 "$BUILD_DIR/runner.stderr" || true)
    RUNNER_EXIT_CODE=$RUNNER_RC
    if [ $RUNNER_RC -eq 0 ] && echo "$RUNNER_STDOUT" | grep -q "qwen_runner: OK"; then
        RUNNER_STATUS="success"
        echo "runner OK: qwen_runner completed in ${RUNNER_DURATION}s"
    else
        RUNNER_STATUS="failed"
        echo "::error::qwen_runner failed (rc=$RUNNER_RC)"
        if [ "$FINAL_EXIT" -eq 0 ]; then
            FINAL_EXIT=2
        fi
    fi
else
    if [ ! -x "$BUILD_DIR/qwen_runner" ]; then
        echo "qwen_runner: skipped (exe not built)"
    fi
    if [ ! -f "$MODEL_DIR/config.json" ]; then
        echo "qwen_runner: skipped (no $MODEL_DIR/config.json)"
    fi
    if [ ! -f "$MODEL_DIR/$LOAD_SHARD_FILE" ]; then
        echo "qwen_runner: skipped (no $MODEL_DIR/$LOAD_SHARD_FILE)"
    fi
fi

# --------------------------------------------------------- k80_p2p_probe ----
# Phase D spike. NO CUDA_VISIBLE_DEVICES restriction (probe needs to see all
# K80 dies). NO model dir mount (pure CUDA, no MLX dep). Failure does NOT
# override existing exit codes — this is informational; we surface the
# capability data and let the human decide whether multi-die sharding
# is viable.
if [ -x "$BUILD_DIR/k80_p2p_probe" ]; then
    P2P_STATUS="running"
    P2P_START=$(date +%s)

    # v3 fix for the v2.1 P8-idle artifact: wake the GPUs out of P8 before
    # the bandwidth measurement runs. Two levers, both host-level (survive
    # the docker-run boundary since they're driver state):
    #   (a) persistence mode keeps the driver loaded -> faster upshift +
    #       no cold-start penalty.
    #   (b) application clocks lock pins the SM / mem clocks at boost
    #       so the GPU stays in P0 even if the workload has gaps.
    # Both may fail silently without root; warm-up in the probe itself
    # catches the case where they didn't take effect.
    set +e
    nvidia-smi -pm 1 > /dev/null 2>&1
    # K80 boost: mem 2505 / graphics 875.  -ac wants "mem,graphics".
    nvidia-smi -ac 2505,875 > /dev/null 2>&1
    set -e

    set +e
    # Match production ollama37 container exactly: --runtime=nvidia,
    # NVIDIA_VISIBLE_DEVICES=all, NVIDIA_DRIVER_CAPABILITIES=compute,utility.
    # Avoids the --gpus shortcut ambiguity and ensures the bandwidth numbers
    # are valid for the actual deployment environment.
    docker run --rm \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -v "$BUILD_DIR:/build:ro" \
        --entrypoint bash \
        "$RUNTIME_IMAGE" \
        -c "/build/k80_p2p_probe" \
        > "$BUILD_DIR/p2p.stdout" 2> "$BUILD_DIR/p2p.stderr"
    P2P_RC=$?

    # Restore default clock state so the host isn't left with locked clocks
    # after the test. Persistence mode is harmless to leave on; the clock
    # lock should be released so other workloads can downclock when idle.
    nvidia-smi -rac > /dev/null 2>&1 || true
    set -e
    P2P_DURATION=$(( $(date +%s) - P2P_START ))
    P2P_STDOUT=$(tail -c 4000 "$BUILD_DIR/p2p.stdout" || true)
    P2P_STDERR=$(tail -c 4000 "$BUILD_DIR/p2p.stderr" || true)
    P2P_EXIT_CODE=$P2P_RC
    if [ $P2P_RC -eq 0 ] && echo "$P2P_STDOUT" | grep -q "k80_p2p: probe OK"; then
        P2P_STATUS="success"
        echo "p2p OK: k80_p2p_probe completed in ${P2P_DURATION}s"
    else
        P2P_STATUS="failed"
        echo "::warning::k80_p2p_probe failed (rc=$P2P_RC) — informational only, not overriding FINAL_EXIT"
    fi
else
    echo "k80_p2p_probe: skipped (exe not built)"
fi

exit $FINAL_EXIT
