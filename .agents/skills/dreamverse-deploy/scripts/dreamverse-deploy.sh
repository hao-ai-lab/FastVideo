#!/usr/bin/env bash
# See ../SKILL.md for full usage.

set -euo pipefail

if [[ "${1:-}" == "--stop" ]]; then
  for pat in 'apps/dreamverse/server/main.py' 'main.py --host 0.0.0.0 --port' 'next dev --port' 'next-server (v'; do
    pkill -KILL -f "${pat}" 2>/dev/null || true
  done
  if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
    gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="${2}" '$1==g {print $2}')"
    if [[ -n "${gpu_uuid}" ]]; then
      for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
                    | awk -F', ' -v u="${gpu_uuid}" '$2==u {print $1}'); do
        kill -9 "${pid}" 2>/dev/null || true
      done
    fi
  fi
  sleep 2
  echo "stopped: ports may take a few seconds to free"
  exit 0
fi

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]] || [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
  cat <<USAGE
Usage: $(basename "$0") <GPU> [BACKEND_PORT] [FRONTEND_PORT]
       $(basename "$0") --stop

Args:
  GPU            Physical GPU index (required), e.g. 4
  BACKEND_PORT   default 8009
  FRONTEND_PORT  default 5274

Env overrides:
  DREAMVERSE_WARMUP    'true'|'false' (default false)
  DREAMVERSE_REPO_ROOT default: \$(git rev-parse --show-toplevel)
  DREAMVERSE_LOG_DIR   default: /tmp/opencode/dreamverse-deploy
USAGE
  exit "$([[ "${1:-}" =~ ^(-h|--help)$ ]] && echo 0 || echo 2)"
fi

GPU="${1}"
BACKEND_PORT="${2:-8009}"
FRONTEND_PORT="${3:-5274}"

if ! [[ "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "error: GPU must be a non-negative integer (got '${GPU}')" >&2
  exit 2
fi

WARMUP="${DREAMVERSE_WARMUP:-false}"
case "${WARMUP}" in
  true|false) ;;
  *) echo "error: DREAMVERSE_WARMUP must be 'true' or 'false' (got '${WARMUP}')" >&2; exit 2 ;;
esac

REPO_ROOT="${DREAMVERSE_REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
LOG_DIR="${DREAMVERSE_LOG_DIR:-/tmp/opencode/dreamverse-deploy}"

# ---------------------------------------------------------------------------
# Prereq checks
# ---------------------------------------------------------------------------

bail() { echo "error: $*" >&2; exit 3; }

[[ -d "${REPO_ROOT}/apps/dreamverse" ]] \
  || bail "REPO_ROOT '${REPO_ROOT}' does not contain apps/dreamverse/. Are you on a migration branch?"
[[ -x "${REPO_ROOT}/apps/dreamverse/scripts/dreamverse-server" ]] \
  || bail "wrapper script missing or not executable: apps/dreamverse/scripts/dreamverse-server"
[[ -x "${REPO_ROOT}/.venv/bin/python" ]] \
  || bail "FastVideo .venv missing at ${REPO_ROOT}/.venv"
"${REPO_ROOT}/.venv/bin/python" -c 'import flashinfer' 2>/dev/null \
  || bail "flashinfer-python not installed in .venv (uv pip install flashinfer-python --no-build-isolation)"

PNPM="${PNPM:-/home/william5lin/.local/share/pnpm/pnpm}"
command -v "${PNPM}" >/dev/null 2>&1 || PNPM="$(command -v pnpm 2>/dev/null || true)"
[[ -n "${PNPM}" ]] && [[ -x "${PNPM}" ]] || bail "pnpm not found in PATH or at /home/william5lin/.local/share/pnpm/pnpm"

GCC13=/usr/bin/gcc-13
GPP13=/usr/bin/g++-13
[[ -x "${GCC13}" ]] || bail "${GCC13} not executable (needed for nvcc workaround)"
[[ -x "${GPP13}" ]] || bail "${GPP13} not executable (needed for nvcc workaround)"

[[ -f "${HOME}/.env" ]] || echo "warn: ${HOME}/.env missing — provider API keys may be unset" >&2

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Teardown anything on target ports
# ---------------------------------------------------------------------------

echo "[1/8] killing any existing deploy on ports ${BACKEND_PORT}/${FRONTEND_PORT} and GPU ${GPU}..."

kill_port_pid() {
  local port="$1"
  local pid
  pid="$(ss -tlnp 2>/dev/null | grep -E ":${port}\b" | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -1 || true)"
  if [[ -n "${pid}" ]]; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

for pat in "main.py --host 0.0.0.0 --port ${BACKEND_PORT}" "next dev --port ${FRONTEND_PORT}" "NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 next dev --port ${FRONTEND_PORT}"; do
  pkill -KILL -f "${pat}" 2>/dev/null || true
done
kill_port_pid "${BACKEND_PORT}"
kill_port_pid "${FRONTEND_PORT}"

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="${GPU}" '$1==g {print $2}')"
if [[ -n "${gpu_uuid}" ]]; then
  for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
                | awk -F', ' -v u="${gpu_uuid}" '$2==u {print $1}'); do
    if [[ -n "${pid}" ]] && [[ "${pid}" != "$$" ]]; then
      cmd="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
      kill -9 "${pid}" 2>/dev/null && echo "        killed GPU${GPU} pid=${pid} (${cmd:-?})" || true
    fi
  done
fi

for i in $(seq 1 30); do
  free_be=true
  free_fe=true
  ss -tln 2>/dev/null | grep -qE ":${BACKEND_PORT}\b" && free_be=false
  ss -tln 2>/dev/null | grep -qE ":${FRONTEND_PORT}\b" && free_fe=false
  gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU + 1))p" || echo 99999)"
  if "${free_be}" && "${free_fe}" && [[ "${gpu_mem}" -lt 1000 ]]; then
    break
  fi
  sleep 1
done

gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU + 1))p" || echo 0)"
echo "        ports cleared; GPU${GPU} at ${gpu_mem} MiB"

# ---------------------------------------------------------------------------
# Launch backend
# ---------------------------------------------------------------------------

echo "[2/8] launching backend on GPU ${GPU} port ${BACKEND_PORT} (warmup=${WARMUP})..."

backend_log="${LOG_DIR}/backend-gpu${GPU}.log"
: > "${backend_log}"

setsid bash -c "
  set -a
  if [[ -f \"${HOME}/.env\" ]]; then
    source \"${HOME}/.env\"
  fi
  set +a
  export CUDA_VISIBLE_DEVICES=${GPU}
  export FASTVIDEO_ENABLE_DEVTOOLS=1
  export FASTVIDEO_ENABLE_STARTUP_WARMUP=${WARMUP}
  export FASTVIDEO_GPU_COUNT=1
  export CC=${GCC13}
  export CXX=${GPP13}
  export CUDAHOSTCXX=${GPP13}
  export NVCC_PREPEND_FLAGS=\"-ccbin ${GCC13} -allow-unsupported-compiler\"
  cd \"${REPO_ROOT}\"
  exec ./apps/dreamverse/scripts/dreamverse-server --host 0.0.0.0 --port ${BACKEND_PORT}
" > "${backend_log}" 2>&1 < /dev/null &
disown

# Wait briefly, then resolve actual python PID (the inner process, not the
# wrapper bash).
sleep 4
backend_pid="$(pgrep -f "main.py --host 0.0.0.0 --port ${BACKEND_PORT}" | head -1 || true)"

if [[ -z "${backend_pid}" ]]; then
  echo "error: backend failed to spawn. Last 30 lines of log:" >&2
  tail -30 "${backend_log}" >&2
  exit 4
fi

echo "        backend pid=${backend_pid} log=${backend_log}"

# Poll /readyz (allow up to 5 min for first compile + model load)
echo "[3/8] polling http://127.0.0.1:${BACKEND_PORT}/readyz ..."
ready=0
for i in $(seq 1 50); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${BACKEND_PORT}/readyz" 2>/dev/null || echo 000)"
  if [[ "${code}" == "200" ]]; then
    ready=1
    break
  fi
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo "error: backend pid ${backend_pid} died. Last 50 lines:" >&2
    tail -50 "${backend_log}" >&2
    exit 5
  fi
  sleep 6
done

if [[ "${ready}" != "1" ]]; then
  echo "error: backend did not become /readyz=200 within 5 min. Last 50 lines:" >&2
  tail -50 "${backend_log}" >&2
  exit 5
fi

echo "[4/8] backend /readyz OK"

# ---------------------------------------------------------------------------
# Launch frontend
# ---------------------------------------------------------------------------

echo "[5/8] launching frontend on port ${FRONTEND_PORT}..."

frontend_log="${LOG_DIR}/frontend-port${FRONTEND_PORT}.log"
: > "${frontend_log}"

# Resolve dev script: dev:devtools forces port 5274 + devtools env. If the
# requested port differs, run `next dev --port` directly with devtools env.
fe_cmd="run dev:devtools"
if [[ "${FRONTEND_PORT}" != "5274" ]]; then
  fe_cmd="exec next dev --port ${FRONTEND_PORT}"
fi

setsid bash -c "
  cd \"${REPO_ROOT}/apps/dreamverse/web\"
  export NEXT_PUBLIC_INCLUDE_DEVTOOLS=1
  export BACKEND_URL=http://127.0.0.1:${BACKEND_PORT}
  export BACKEND_HOST=127.0.0.1
  export BACKEND_PORT=${BACKEND_PORT}
  exec '${PNPM}' ${fe_cmd}
" > "${frontend_log}" 2>&1 < /dev/null &
disown

sleep 4
frontend_pid="$(pgrep -f "next dev --port ${FRONTEND_PORT}" | head -1 || true)"
if [[ -z "${frontend_pid}" ]]; then
  echo "error: frontend failed to spawn. Last 30 lines:" >&2
  tail -30 "${frontend_log}" >&2
  exit 6
fi

echo "        frontend pid=${frontend_pid} log=${frontend_log}"

# Poll FE root
echo "[6/8] polling http://127.0.0.1:${FRONTEND_PORT}/ ..."
fe_ready=0
for i in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${FRONTEND_PORT}/" 2>/dev/null || echo 000)"
  if [[ "${code}" == "200" ]]; then
    fe_ready=1
    break
  fi
  if ! kill -0 "${frontend_pid}" 2>/dev/null; then
    echo "error: frontend pid ${frontend_pid} died. Last 30 lines:" >&2
    tail -30 "${frontend_log}" >&2
    exit 7
  fi
  sleep 2
done

if [[ "${fe_ready}" != "1" ]]; then
  echo "error: frontend did not respond 200 within 60s. Last 30 lines:" >&2
  tail -30 "${frontend_log}" >&2
  exit 7
fi

echo "[7/8] frontend / OK"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

cwd="$(readlink "/proc/${backend_pid}/cwd" 2>/dev/null || echo unknown)"
gpu_mem_now="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU + 1))p" || echo 0)"

cat <<SUMMARY
[8/8] redeploy OK

  Frontend  : http://localhost:${FRONTEND_PORT}    (PID ${frontend_pid})
  Backend   : http://localhost:${BACKEND_PORT}     (PID ${backend_pid})
              cwd=${cwd}
              gpu=${GPU} mem=${gpu_mem_now} MiB

  Logs      : ${backend_log}
              ${frontend_log}

  Stop      : ./.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh --stop

  E2E       : cd apps/dreamverse/web && \\
                PLAYWRIGHT_SKIP_WEBSERVER=1 \\
                BACKEND_URL=http://127.0.0.1:${BACKEND_PORT} \\
                PLAYWRIGHT_BASE_URL=http://127.0.0.1:${FRONTEND_PORT} \\
                NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \\
                pnpm exec playwright test
SUMMARY
