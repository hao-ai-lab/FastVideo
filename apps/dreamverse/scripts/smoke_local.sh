#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${BACKEND_HOST:-127.0.0.1}"
PORT="${BACKEND_PORT:-8009}"
BACKEND_ORIGIN="http://${HOST}:${PORT}"
TIMEOUT_SECONDS="${DREAMVERSE_SMOKE_TIMEOUT_SECONDS:-240}"
POLL_INTERVAL_SECONDS="${DREAMVERSE_SMOKE_POLL_SECONDS:-2}"
BACKEND_LOG_PATH="${DREAMVERSE_SMOKE_LOG_PATH:-${ROOT_DIR}/outputs/smoke-local-backend.log}"
START_BACKEND="${DREAMVERSE_SMOKE_START_BACKEND:-1}"

backend_pid=""
started_backend=0

cleanup() {
  if [[ "${started_backend}" == "1" && -n "${backend_pid}" ]]; then
    kill "${backend_pid}" >/dev/null 2>&1 || true
    wait "${backend_pid}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

probe_json() {
  local path="$1"
  curl --silent --show-error --fail "${BACKEND_ORIGIN}${path}"
}

wait_for_endpoint() {
  local path="$1"
  local label="$2"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if probe_json "${path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
  done
  echo "Timed out waiting for ${label} at ${BACKEND_ORIGIN}${path}" >&2
  return 1
}

require_command curl

mkdir -p "$(dirname "${BACKEND_LOG_PATH}")"

if ! probe_json "/healthz" >/dev/null 2>&1; then
  if [[ "${START_BACKEND}" != "1" ]]; then
    echo "Dreamverse backend is not reachable at ${BACKEND_ORIGIN} and auto-start is disabled." >&2
    exit 1
  fi

  require_command dreamverse-server

  echo "Starting Dreamverse backend on ${BACKEND_ORIGIN}..."
  (
    cd "${ROOT_DIR}"
    exec dreamverse-server --host "${HOST}" --port "${PORT}"
  ) >"${BACKEND_LOG_PATH}" 2>&1 &
  backend_pid=$!
  started_backend=1
else
  echo "Dreamverse backend already running on ${BACKEND_ORIGIN}."
fi

echo "Waiting for /healthz..."
wait_for_endpoint "/healthz" "healthz"
echo "Waiting for /readyz..."
wait_for_endpoint "/readyz" "readyz"

status_payload="$(probe_json "/status")"
echo "Dreamverse local smoke check passed."
echo "Backend URL: ${BACKEND_ORIGIN}"
echo "Status: ${status_payload}"

if [[ "${started_backend}" == "1" ]]; then
  trap - EXIT
  echo "Backend PID: ${backend_pid}"
  echo "Backend log: ${BACKEND_LOG_PATH}"
  echo "Backend is still running so you can launch the frontend:"
  echo "  cd ${ROOT_DIR}/web"
  echo "  BACKEND_HOST=${HOST} BACKEND_PORT=${PORT} pnpm run dev"
else
  echo "You can now launch the frontend:"
  echo "  cd ${ROOT_DIR}/web"
  echo "  BACKEND_HOST=${HOST} BACKEND_PORT=${PORT} pnpm run dev"
fi
