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

usage() {
  cat <<USAGE
Usage: $(basename "$0") [FLAGS] <GPU> [BACKEND_PORT] [FRONTEND_PORT]
       $(basename "$0") --stop [GPU]

Positional:
  GPU            Physical GPU index (required), e.g. 4
  BACKEND_PORT   default 8009
  FRONTEND_PORT  default 5274

Flags (override env vars when both set):
  --warmup / --no-warmup            run GPU warmup at boot (default off)
  --torch-compile / --no-torch-compile
                                     enable max-autotune torch.compile
                                     (default off — first segment ~3-4min
                                     when on, ~45s when off)
  -h, --help                        show this help

Env overrides:
  DREAMVERSE_WARMUP                 'true'|'false' (default false)
  DREAMVERSE_TORCH_COMPILE          'true'|'false' (default false)
  DREAMVERSE_REPO_ROOT              default: \$(git rev-parse --show-toplevel)
  DREAMVERSE_LOG_DIR                default: /tmp/opencode/dreamverse-deploy
  DREAMVERSE_REQUIRE_NATIVE_FFMPEG  'true'|'false' (default false)
USAGE
}

WARMUP_OVERRIDE=""
TORCH_COMPILE_OVERRIDE=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              usage; exit 0 ;;
    --warmup)               WARMUP_OVERRIDE=true; shift ;;
    --no-warmup)            WARMUP_OVERRIDE=false; shift ;;
    --torch-compile)        TORCH_COMPILE_OVERRIDE=true; shift ;;
    --no-torch-compile)     TORCH_COMPILE_OVERRIDE=false; shift ;;
    --)                     shift; while [[ $# -gt 0 ]]; do POSITIONAL+=("$1"); shift; done ;;
    -*)                     echo "error: unknown flag '$1'" >&2; usage >&2; exit 2 ;;
    *)                      POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

GPU="${1}"
BACKEND_PORT="${2:-8009}"
FRONTEND_PORT="${3:-5274}"

if ! [[ "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "error: GPU must be a non-negative integer (got '${GPU}')" >&2
  exit 2
fi

WARMUP="${WARMUP_OVERRIDE:-${DREAMVERSE_WARMUP:-false}}"
case "${WARMUP}" in
  true|false) ;;
  *) echo "error: warmup must be 'true' or 'false' (got '${WARMUP}')" >&2; exit 2 ;;
esac

TORCH_COMPILE="${TORCH_COMPILE_OVERRIDE:-${DREAMVERSE_TORCH_COMPILE:-false}}"
case "${TORCH_COMPILE}" in
  true|false) ;;
  *) echo "error: torch-compile must be 'true' or 'false' (got '${TORCH_COMPILE}')" >&2; exit 2 ;;
esac
TORCH_COMPILE_FLAG=$([[ "${TORCH_COMPILE}" == "true" ]] && echo 1 || echo 0)

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

NATIVE_FFMPEG_BIN="${HOME}/opt/ffmpeg-native/bin/ffmpeg"
NATIVE_VIDEO_CODEC=libx264
REQUIRE_NATIVE_FFMPEG="${DREAMVERSE_REQUIRE_NATIVE_FFMPEG:-false}"
case "${REQUIRE_NATIVE_FFMPEG}" in
  true|false) ;;
  *) bail "DREAMVERSE_REQUIRE_NATIVE_FFMPEG must be 'true' or 'false' (got '${REQUIRE_NATIVE_FFMPEG}')" ;;
esac
if [[ -x "${NATIVE_FFMPEG_BIN}" ]]; then
  echo "        native ffmpeg: ${NATIVE_FFMPEG_BIN} (codec=${NATIVE_VIDEO_CODEC})"
elif [[ "${REQUIRE_NATIVE_FFMPEG}" == "true" ]]; then
  bail "DREAMVERSE_REQUIRE_NATIVE_FFMPEG=true but ${NATIVE_FFMPEG_BIN} missing. Run: bash apps/dreamverse/scripts/install_native_ffmpeg.sh"
else
  echo "warn: ${NATIVE_FFMPEG_BIN} missing — backend will fall back to system ffmpeg (\$(command -v ffmpeg))." >&2
  echo "      Build native ffmpeg with: bash apps/dreamverse/scripts/install_native_ffmpeg.sh" >&2
fi

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

echo "[2/8] launching backend on GPU ${GPU} port ${BACKEND_PORT} (warmup=${WARMUP} torch_compile=${TORCH_COMPILE})..."

backend_log="${LOG_DIR}/backend-gpu${GPU}.log"
: > "${backend_log}"

setsid bash -c "
  set -a
  if [[ -f \"${HOME}/.env\" ]]; then
    source \"${HOME}/.env\"
  fi
  set +a
  if [[ -x \"${NATIVE_FFMPEG_BIN}\" ]]; then
    export FASTVIDEO_FFMPEG_BIN=\"${NATIVE_FFMPEG_BIN}\"
    export FASTVIDEO_VIDEO_CODEC=\"${NATIVE_VIDEO_CODEC}\"
  fi
  export CUDA_VISIBLE_DEVICES=${GPU}
  export FASTVIDEO_ENABLE_DEVTOOLS=1
  export FASTVIDEO_ENABLE_STARTUP_WARMUP=${WARMUP}
  export FASTVIDEO_GPU_COUNT=1
  export ENABLE_TORCH_COMPILE=${TORCH_COMPILE_FLAG}
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
ffmpeg_in_use="$(tr '\0' '\n' < "/proc/${backend_pid}/environ" 2>/dev/null | sed -n 's/^FASTVIDEO_FFMPEG_BIN=//p' | head -1)"
[[ -z "${ffmpeg_in_use}" ]] && ffmpeg_in_use="$(command -v ffmpeg 2>/dev/null || echo '<not found>') (system fallback)"

cat <<SUMMARY
[8/8] redeploy OK

  Frontend  : http://localhost:${FRONTEND_PORT}    (PID ${frontend_pid})
  Backend   : http://localhost:${BACKEND_PORT}     (PID ${backend_pid})
              cwd=${cwd}
              gpu=${GPU} mem=${gpu_mem_now} MiB
              ffmpeg=${ffmpeg_in_use}

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
