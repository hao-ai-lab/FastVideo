#!/usr/bin/env bash
# Canonical Slurm performance lane. Reports are written outside the checkout
# so the trusted host driver can upload them after untrusted code exits.
set -uo pipefail

export PERFORMANCE_TRACKING_ROOT=/tmp/perf-tracking
export PERF_REPORTS_DIR=/workspace/artifacts/performance
mkdir -p "$PERF_REPORTS_DIR"

if [[ ${BUILDKITE_PULL_REQUEST:-false} =~ ^[1-9][0-9]*$ ]]; then
  export PERF_RUN_SOURCE=pr
  export PERF_UPLOAD_POLICY=pass
elif [ "${BUILDKITE_BRANCH:-}" = main ] \
  && { [ "${BUILDKITE_SOURCE:-}" = schedule ] || [ "${TEST_SCOPE:-}" = full ]; }; then
  export PERF_RUN_SOURCE=scheduled_main
  export PERF_UPLOAD_POLICY=always
elif [ "${TEST_SCOPE:-}" = direct ]; then
  export PERF_RUN_SOURCE=unknown
  export PERF_UPLOAD_POLICY=pass
else
  export PERF_RUN_SOURCE=unknown
  export PERF_UPLOAD_POLICY=never
fi

nvidia-smi \
  --query-gpu=index,timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu \
  --format=csv -l 10 > "$PERF_REPORTS_DIR/gpu_telemetry.csv" 2>/dev/null &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

pytest ./fastvideo/tests/performance -vs
pytest_rc=$?
compare_rc=0
if [ "$pytest_rc" -eq 0 ] || [ "$PERF_UPLOAD_POLICY" = always ]; then
  PERF_PYTEST_RC=$pytest_rc python ./fastvideo/tests/performance/compare_baseline.py
  compare_rc=$?
fi
python ./fastvideo/tests/performance/dashboard.py || true
cp -f fastvideo/tests/performance/results/*.json "$PERF_REPORTS_DIR/" 2>/dev/null || true

echo "--- GPU telemetry (clocks.sm vs clocks.max.sm reveals capped hosts) ---"
cat "$PERF_REPORTS_DIR/gpu_telemetry.csv" || true

final_rc=$pytest_rc
if [ "$final_rc" -eq 0 ]; then
  final_rc=$compare_rc
fi
exit "$final_rc"
