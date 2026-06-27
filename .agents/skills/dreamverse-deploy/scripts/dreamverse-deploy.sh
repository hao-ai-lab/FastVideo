#!/usr/bin/env bash
# See ../SKILL.md for usage and safety notes.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  dreamverse-deploy.sh [FLAGS] <GPU> [BACKEND_PORT] [FRONTEND_PORT]
  dreamverse-deploy.sh --stop [BACKEND_PORT] [FRONTEND_PORT]

Defaults: backend 8009, frontend 5299, warmup off, torch.compile off.

Flags:
  --warmup / --no-warmup
  --torch-compile / --no-torch-compile
  --nvenc / --no-nvenc
  --force-gpu-kill   Kill every compute process on the selected GPU.
  --dry-run          Print the resolved plan without changing processes.
  --stop             Stop the managed stack and listeners on selected ports.
  -h, --help

Environment:
  DREAMVERSE_WARMUP, DREAMVERSE_TORCH_COMPILE, DREAMVERSE_NVENC
  DREAMVERSE_BACKEND_PORT, DREAMVERSE_FRONTEND_PORT
  DREAMVERSE_REPO_ROOT, DREAMVERSE_DEPLOY_DIR
  NPM                       npm executable override
  DREAMVERSE_DEPLOY_READY_TIMEOUT_SECONDS (default 2400)
  DREAMVERSE_DEPLOY_FRONTEND_TIMEOUT_SECONDS (default 120)
USAGE
}

fail() {
  echo "error: $*" >&2
  exit 1
}

normalize_bool() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|on) echo 1 ;;
    0|false|no|off) echo 0 ;;
    *) fail "expected a boolean value, got '$1'" ;;
  esac
}

validate_port() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || fail "${name} must be an integer from 1 to 65535 (got '${value}')"
  (( value >= 1 && value <= 65535 )) \
    || fail "${name} must be an integer from 1 to 65535 (got '${value}')"
}

MODE=deploy
DRY_RUN=0
FORCE_GPU_KILL=0
WARMUP_OVERRIDE=""
TORCH_COMPILE_OVERRIDE=""
NVENC_OVERRIDE=""
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --warmup) WARMUP_OVERRIDE=1 ;;
    --no-warmup) WARMUP_OVERRIDE=0 ;;
    --torch-compile) TORCH_COMPILE_OVERRIDE=1 ;;
    --no-torch-compile) TORCH_COMPILE_OVERRIDE=0 ;;
    --nvenc) NVENC_OVERRIDE=1 ;;
    --no-nvenc) NVENC_OVERRIDE=0 ;;
    --force-gpu-kill) FORCE_GPU_KILL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --stop) MODE=stop ;;
    -h|--help) usage; exit 0 ;;
    --) shift; POSITIONAL+=("$@"); break ;;
    -*) fail "unknown flag '$1'" ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_ROOT="${DREAMVERSE_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
BACKEND_PORT_DEFAULT="${DREAMVERSE_BACKEND_PORT:-8009}"
FRONTEND_PORT_DEFAULT="${DREAMVERSE_FRONTEND_PORT:-5299}"

if [[ "${MODE}" == stop ]]; then
  (( ${#POSITIONAL[@]} <= 2 )) || fail "--stop accepts only [BACKEND_PORT] [FRONTEND_PORT]"
  BACKEND_PORT="${POSITIONAL[0]:-${BACKEND_PORT_DEFAULT}}"
  FRONTEND_PORT="${POSITIONAL[1]:-${FRONTEND_PORT_DEFAULT}}"
  GPU=""
  (( FORCE_GPU_KILL == 0 )) || fail "--force-gpu-kill requires deploy mode and a GPU"
else
  (( ${#POSITIONAL[@]} >= 1 && ${#POSITIONAL[@]} <= 3 )) || {
    usage >&2
    exit 2
  }
  GPU="${POSITIONAL[0]}"
  BACKEND_PORT="${POSITIONAL[1]:-${BACKEND_PORT_DEFAULT}}"
  FRONTEND_PORT="${POSITIONAL[2]:-${FRONTEND_PORT_DEFAULT}}"
  [[ "${GPU}" =~ ^[0-9]+$ ]] || fail "GPU must be a non-negative integer (got '${GPU}')"
fi

validate_port BACKEND_PORT "${BACKEND_PORT}"
validate_port FRONTEND_PORT "${FRONTEND_PORT}"
[[ "${BACKEND_PORT}" != "${FRONTEND_PORT}" ]] || fail "backend and frontend ports must differ"

WARMUP="${WARMUP_OVERRIDE:-$(normalize_bool "${DREAMVERSE_WARMUP:-false}")}"
TORCH_COMPILE="${TORCH_COMPILE_OVERRIDE:-$(normalize_bool "${DREAMVERSE_TORCH_COMPILE:-false}")}"
NVENC="${NVENC_OVERRIDE:-$(normalize_bool "${DREAMVERSE_NVENC:-false}")}"

DEPLOY_ROOT="${DREAMVERSE_DEPLOY_DIR:-${REPO_ROOT}/.agents/tmp/dreamverse-deploy}"
INSTANCE_DIR="${DEPLOY_ROOT}/be${BACKEND_PORT}-fe${FRONTEND_PORT}"
PID_FILE="${INSTANCE_DIR}/launcher.pid"
STACK_LOG="${INSTANCE_DIR}/stack.log"
LAUNCHER="${REPO_ROOT}/apps/dreamverse/scripts/launch/launch_demo.sh"
NPM="${NPM:-npm}"

print_plan() {
  cat <<PLAN
Dreamverse ${MODE} plan
  repo:         ${REPO_ROOT}
  gpu:          ${GPU:-<not used>}
  backend:      http://127.0.0.1:${BACKEND_PORT}
  frontend:     http://127.0.0.1:${FRONTEND_PORT}
  warmup:       ${WARMUP}
  compile:      ${TORCH_COMPILE}
  nvenc:        ${NVENC}
  force GPU kill: ${FORCE_GPU_KILL}
  artifacts:    ${INSTANCE_DIR}
PLAN
}

if (( DRY_RUN == 1 )); then
  print_plan
  exit 0
fi

command -v curl >/dev/null 2>&1 || fail "curl is required"
if ! command -v lsof >/dev/null 2>&1 && ! command -v fuser >/dev/null 2>&1; then
  fail "lsof or fuser is required to identify port listeners"
fi

is_pid_alive() {
  kill -0 "$1" 2>/dev/null
}

terminate_pid() {
  local pid="$1" label="$2"
  [[ -n "${pid}" && "${pid}" != "$$" ]] || return 0
  is_pid_alive "${pid}" || return 0
  echo "stopping ${label}"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in {1..20}; do
    is_pid_alive "${pid}" || return 0
    sleep 0.25
  done
  kill -KILL "${pid}" 2>/dev/null || true
}

terminate_managed_group() {
  local pid args
  [[ -f "${PID_FILE}" ]] || return 0
  read -r pid < "${PID_FILE}" || true
  [[ "${pid:-}" =~ ^[0-9]+$ ]] || return 0
  if ! is_pid_alive "${pid}"; then
    return 0
  fi
  args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  if [[ "${args}" != *"${LAUNCHER}"* ]]; then
    echo "warn: refusing to stop stale PID ${pid}; it is not the Dreamverse launcher" >&2
    return 0
  fi
  echo "stopping managed Dreamverse process group ${pid}"
  kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in {1..40}; do
    is_pid_alive "${pid}" || return 0
    sleep 0.25
  done
  kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

list_port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
  else
    fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
  fi
}

stop_port_listeners() {
  local port="$1" pid
  while read -r pid; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    terminate_pid "${pid}" "listener on port ${port} (PID ${pid})"
  done < <(list_port_pids "${port}")
}

mkdir -p "${INSTANCE_DIR}"
terminate_managed_group
stop_port_listeners "${BACKEND_PORT}"
stop_port_listeners "${FRONTEND_PORT}"

if [[ "${MODE}" == stop ]]; then
  rm -f "${PID_FILE}"
  echo "Dreamverse stopped on ports ${BACKEND_PORT}/${FRONTEND_PORT}."
  exit 0
fi

[[ -x "${LAUNCHER}" ]] || fail "launcher is missing or not executable: ${LAUNCHER}"
for command_name in dreamverse-server nvidia-smi setsid; do
  command -v "${command_name}" >/dev/null 2>&1 || fail "${command_name} is required"
done
command -v "${NPM}" >/dev/null 2>&1 || fail "npm command not found: ${NPM}"

GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
  | awk -F', ' -v gpu="${GPU}" '$1 == gpu {print $2}')"
[[ -n "${GPU_UUID}" ]] || fail "GPU ${GPU} was not reported by nvidia-smi"

gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
    | awk -F', ' -v uuid="${GPU_UUID}" '$2 == uuid {print $1}'
}

GPU_PIDS=""
for _ in {1..5}; do
  GPU_PIDS="$(gpu_compute_pids)"
  [[ -n "${GPU_PIDS}" ]] || break
  sleep 1
done
if [[ -n "${GPU_PIDS}" ]]; then
  if (( FORCE_GPU_KILL == 0 )); then
    printf 'error: GPU %s is still used by compute PID(s): %s\n' \
      "${GPU}" "$(echo "${GPU_PIDS}" | tr '\n' ' ')" >&2
    echo "Inspect them with nvidia-smi; rerun with --force-gpu-kill only if they are disposable." >&2
    exit 1
  fi
  for pid in ${GPU_PIDS}; do
    terminate_pid "${pid}" "GPU ${GPU} compute process (PID ${pid})"
  done
  for _ in {1..30}; do
    [[ -z "$(gpu_compute_pids)" ]] && break
    sleep 1
  done
  GPU_PIDS="$(gpu_compute_pids)"
  [[ -z "${GPU_PIDS}" ]] \
    || fail "GPU ${GPU} did not release compute PID(s): $(echo "${GPU_PIDS}" | tr '\n' ' ')"
fi

FFMPEG_ENV="${REPO_ROOT}/apps/dreamverse/scripts/ffmpeg-env.sh"
if [[ -z "${FASTVIDEO_FFMPEG_BIN:-}" && -f "${FFMPEG_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${FFMPEG_ENV}"
fi

if (( NVENC == 1 )); then
  FFMPEG_BIN="$(command -v "${FASTVIDEO_FFMPEG_BIN:-ffmpeg}" 2>/dev/null || true)"
  [[ -n "${FFMPEG_BIN}" ]] || fail "--nvenc requires ffmpeg"
  "${FFMPEG_BIN}" -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc \
    || fail "${FFMPEG_BIN} was not built with h264_nvenc"
  if ! CUDA_VISIBLE_DEVICES="${GPU}" "${FFMPEG_BIN}" -hide_banner -loglevel error -y \
      -f lavfi -i 'color=red:size=64x64:rate=24:duration=0.2' \
      -c:v h264_nvenc -f null - >/dev/null 2>&1; then
    fail "GPU ${GPU} could not open an NVENC session"
  fi
  export FASTVIDEO_FFMPEG_BIN="${FFMPEG_BIN}"
  export FASTVIDEO_VIDEO_CODEC=h264_nvenc
else
  export FASTVIDEO_VIDEO_CODEC="${FASTVIDEO_VIDEO_CODEC:-libx264}"
fi

: > "${STACK_LOG}"
setsid env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  FASTVIDEO_ENABLE_DEVTOOLS=1 \
  FASTVIDEO_ENABLE_STARTUP_WARMUP="${WARMUP}" \
  FASTVIDEO_GPU_COUNT=1 \
  ENABLE_TORCH_COMPILE="${TORCH_COMPILE}" \
  BE_PORT="${BACKEND_PORT}" \
  FE_PORT="${FRONTEND_PORT}" \
  NO_BROWSER=1 \
  NPM="${NPM}" \
  DREAMVERSE_LOG_DIR="${INSTANCE_DIR}" \
  "${LAUNCHER}" > "${STACK_LOG}" 2>&1 < /dev/null &
LAUNCHER_PID=$!
printf '%s\n' "${LAUNCHER_PID}" > "${PID_FILE}"

wait_for_url() {
  local url="$1" label="$2" timeout="$3"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 -o /dev/null "${url}" 2>/dev/null; then
      echo "${label} ready: ${url}"
      return 0
    fi
    if ! is_pid_alive "${LAUNCHER_PID}"; then
      echo "error: Dreamverse launcher exited while waiting for ${label}" >&2
      return 1
    fi
    sleep 2
  done
  echo "error: timed out after ${timeout}s waiting for ${label}: ${url}" >&2
  return 1
}

READY_TIMEOUT="${DREAMVERSE_DEPLOY_READY_TIMEOUT_SECONDS:-2400}"
FRONTEND_TIMEOUT="${DREAMVERSE_DEPLOY_FRONTEND_TIMEOUT_SECONDS:-120}"
validate_port DREAMVERSE_DEPLOY_READY_TIMEOUT_SECONDS "${READY_TIMEOUT}"
validate_port DREAMVERSE_DEPLOY_FRONTEND_TIMEOUT_SECONDS "${FRONTEND_TIMEOUT}"

if ! wait_for_url "http://127.0.0.1:${BACKEND_PORT}/readyz" backend "${READY_TIMEOUT}" \
    || ! wait_for_url "http://127.0.0.1:${FRONTEND_PORT}/" frontend "${FRONTEND_TIMEOUT}"; then
  tail -n 100 "${STACK_LOG}" >&2 || true
  [[ ! -f "${INSTANCE_DIR}/demo-be.log" ]] || tail -n 100 "${INSTANCE_DIR}/demo-be.log" >&2
  [[ ! -f "${INSTANCE_DIR}/demo-fe.log" ]] || tail -n 100 "${INSTANCE_DIR}/demo-fe.log" >&2
  terminate_managed_group
  exit 1
fi

cat <<SUMMARY
Dreamverse deploy ready
  backend:  http://127.0.0.1:${BACKEND_PORT}
  frontend: http://127.0.0.1:${FRONTEND_PORT}
  GPU:      ${GPU}
  PID:      ${LAUNCHER_PID}
  logs:     ${INSTANCE_DIR}
  stop:     $0 --stop ${BACKEND_PORT} ${FRONTEND_PORT}
SUMMARY
