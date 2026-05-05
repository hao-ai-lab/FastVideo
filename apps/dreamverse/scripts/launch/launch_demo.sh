#!/usr/bin/env bash
# One-command Dreamverse demo launcher.
#
# Spawns the backend and frontend, polls health, prints URLs, and
# stops both children on Ctrl-C.
#
# Defaults match internal/ui:
#   * BE = dreamverse-server (8009) — full FE compatibility
#   * FE = Next.js dev:devtools (5274)
#
# Switch BE to fastvideo serve --config (typed-only path):
#   BE_FLAVOR=fastvideo bash launch_demo.sh
#
# Other env knobs:
#   BE_PORT=8010 FE_PORT=5274 bash launch_demo.sh
#   NO_FRONTEND=1 bash launch_demo.sh        # backend only
#   NO_BROWSER=1 bash launch_demo.sh         # skip xdg-open

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${DREAMVERSE_ROOT}/logs"
mkdir -p "${LOG_DIR}"

BE_FLAVOR="${BE_FLAVOR:-dreamverse}"
BE_PORT="${BE_PORT:-8009}"
FE_PORT="${FE_PORT:-5274}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-2400}"

case "${BE_FLAVOR}" in
  dreamverse)
    BE_SCRIPT="${SCRIPT_DIR}/launch_backend_dreamverse.sh"
    BE_PORT_ARGS=(--port "${BE_PORT}")
    HEALTH_PATH="/healthz"
    READY_PATH="/readyz"
    ;;
  fastvideo)
    BE_SCRIPT="${SCRIPT_DIR}/launch_backend_fastvideo.sh"
    # fastvideo serve takes dotted overrides on top of the YAML; the
    # nested ServerConfig path is server.port not --port.
    BE_PORT_ARGS=(--server.port "${BE_PORT}")
    # bare fastvideo serve only exposes /health; treat it as both the
    # liveness and readiness probe.
    HEALTH_PATH="/health"
    READY_PATH="/health"
    ;;
  *)
    echo "error: BE_FLAVOR must be 'dreamverse' or 'fastvideo' (got '${BE_FLAVOR}')" >&2
    exit 1
    ;;
esac

BE_URL="http://localhost:${BE_PORT}"
FE_URL="http://localhost:${FE_PORT}"
BE_LOG="${LOG_DIR}/demo-be.log"
FE_LOG="${LOG_DIR}/demo-fe.log"

cleanup() {
  local rc=$?
  echo
  echo "[launch-demo] shutting down children"
  if [[ -n "${BE_PID:-}" ]] && kill -0 "${BE_PID}" 2>/dev/null; then
    kill "${BE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FE_PID:-}" ]] && kill -0 "${FE_PID}" 2>/dev/null; then
    kill "${FE_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT INT TERM

probe() {
  local url="$1"
  curl -fsS -m 2 -o /dev/null "${url}"
}

wait_for() {
  local url="$1"
  local label="$2"
  local timeout="$3"
  local started end
  started="$(date +%s)"
  end="$((started + timeout))"
  while (( $(date +%s) < end )); do
    if probe "${url}"; then
      echo "[launch-demo] ${label} ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "error: ${label} did not respond within ${timeout}s at ${url}" >&2
  return 1
}

# --- backend ---
echo "[launch-demo] backend logs: ${BE_LOG}"
( "${BE_SCRIPT}" "${BE_PORT_ARGS[@]}" >"${BE_LOG}" 2>&1 ) &
BE_PID=$!
echo "[launch-demo] backend PID ${BE_PID} (flavor=${BE_FLAVOR}, port=${BE_PORT})"

if ! wait_for "${BE_URL}${HEALTH_PATH}" "backend health" "${HEALTH_TIMEOUT_SECONDS}"; then
  echo "------ backend log (tail) ------" >&2
  tail -n 60 "${BE_LOG}" >&2 || true
  exit 1
fi

if [[ "${HEALTH_PATH}" != "${READY_PATH}" ]]; then
  echo "[launch-demo] waiting for backend readiness (warmup may take a few minutes)…"
  if ! wait_for "${BE_URL}${READY_PATH}" "backend ready" "${READY_TIMEOUT_SECONDS}"; then
    echo "------ backend log (tail) ------" >&2
    tail -n 100 "${BE_LOG}" >&2 || true
    exit 1
  fi
fi

# --- frontend ---
if [[ "${NO_FRONTEND:-0}" != "1" ]]; then
  echo "[launch-demo] frontend logs: ${FE_LOG}"
  ( "${SCRIPT_DIR}/launch_frontend.sh" >"${FE_LOG}" 2>&1 ) &
  FE_PID=$!
  echo "[launch-demo] frontend PID ${FE_PID} (port ${FE_PORT})"

  if ! wait_for "${FE_URL}/" "frontend" 60; then
    echo "------ frontend log (tail) ------" >&2
    tail -n 60 "${FE_LOG}" >&2 || true
    exit 1
  fi

  if [[ "${NO_BROWSER:-0}" != "1" ]] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${FE_URL}/" >/dev/null 2>&1 || true
  fi
fi

cat <<EOF

[launch-demo] stack is up. Press Ctrl-C to stop.
  backend : ${BE_URL}  (${BE_FLAVOR}, log: ${BE_LOG})
  frontend: ${FE_URL}  (log: ${FE_LOG})

EOF

wait "${BE_PID}"
