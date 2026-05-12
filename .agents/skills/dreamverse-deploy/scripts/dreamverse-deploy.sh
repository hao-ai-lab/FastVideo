#!/usr/bin/env bash
# See ../SKILL.md for full usage.

set -euo pipefail

is_pid_alive() {
  kill -0 "$1" 2>/dev/null
}

terminate_pid() {
  local pid="$1"
  local label="${2:-pid=${pid}}"

  [[ -n "${pid}" ]] && [[ "${pid}" != "$$" ]] || return 0
  is_pid_alive "${pid}" || return 0

  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    is_pid_alive "${pid}" || return 0
    sleep 0.5
  done

  if is_pid_alive "${pid}"; then
    kill -9 "${pid}" 2>/dev/null && echo "        force-killed ${label}" || true
  fi
}

terminate_pattern() {
  local pattern="$1"
  local pid

  if ! command -v pgrep >/dev/null 2>&1; then
    pkill -TERM -f "${pattern}" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "${pattern}" 2>/dev/null || true
    return 0
  fi

  for pid in $(pgrep -f -- "${pattern}" 2>/dev/null || true); do
    terminate_pid "${pid}" "pattern='${pattern}' pid=${pid}"
  done
}

list_port_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return 0
  fi

  ss -tlnp 2>/dev/null | awk -v port=":${port}" '
    $0 ~ port {
      while (match($0, /pid=[0-9]+/)) {
        print substr($0, RSTART + 4, RLENGTH - 4)
        $0 = substr($0, RSTART + RLENGTH)
      }
    }
  ' || true
}

if [[ "${1:-}" == "--stop" ]]; then
  for pat in 'apps/dreamverse/server/main.py' 'main.py --host 0.0.0.0 --port' 'next dev --port' 'next-server (v'; do
    terminate_pattern "${pat}"
  done
  if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
    gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="${2}" '$1==g {print $2}')"
    if [[ -n "${gpu_uuid}" ]]; then
      for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
                    | awk -F', ' -v u="${gpu_uuid}" '$2==u {print $1}'); do
        terminate_pid "${pid}" "GPU${2} pid=${pid}"
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
  --nvenc / --no-nvenc              use h264_nvenc hardware encoder (default
                                     off — uses libx264 software encoder).
                                     Requires native ffmpeg built with NVENC.
  -h, --help                        show this help

Env overrides:
  DREAMVERSE_WARMUP                 'true'|'false' (default false)
  DREAMVERSE_TORCH_COMPILE          'true'|'false' (default false)
  DREAMVERSE_NVENC                  'true'|'false' (default false)
  DREAMVERSE_REPO_ROOT              default: \$(git rev-parse --show-toplevel)
  DREAMVERSE_LOG_DIR                default: /tmp/opencode/dreamverse-deploy
  DREAMVERSE_REQUIRE_NATIVE_FFMPEG  'true'|'false' (default false)
USAGE
}

WARMUP_OVERRIDE=""
TORCH_COMPILE_OVERRIDE=""
NVENC_OVERRIDE=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              usage; exit 0 ;;
    --warmup)               WARMUP_OVERRIDE=true; shift ;;
    --no-warmup)            WARMUP_OVERRIDE=false; shift ;;
    --torch-compile)        TORCH_COMPILE_OVERRIDE=true; shift ;;
    --no-torch-compile)     TORCH_COMPILE_OVERRIDE=false; shift ;;
    --nvenc)                NVENC_OVERRIDE=true; shift ;;
    --no-nvenc)             NVENC_OVERRIDE=false; shift ;;
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

NVENC="${NVENC_OVERRIDE:-${DREAMVERSE_NVENC:-false}}"
case "${NVENC}" in
  true|false) ;;
  *) echo "error: nvenc must be 'true' or 'false' (got '${NVENC}')" >&2; exit 2 ;;
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

CONDA_ENV_PYTHON="${DREAMVERSE_PYTHON:-${HOME}/miniconda3/envs/fv-main/bin/python}"
[[ -x "${CONDA_ENV_PYTHON}" ]] \
  || bail "conda env python missing at ${CONDA_ENV_PYTHON} (set DREAMVERSE_PYTHON to override)"
"${CONDA_ENV_PYTHON}" -c 'import flashinfer' 2>/dev/null \
  || bail "flashinfer-python not installed in ${CONDA_ENV_PYTHON} (run: ${CONDA_ENV_PYTHON} -m pip install flashinfer-python --no-build-isolation)"

PNPM="${PNPM:-}"
if [[ -n "${PNPM}" ]]; then
  PNPM_REQUESTED="${PNPM}"
  PNPM="$(command -v "${PNPM}" 2>/dev/null || true)"
  [[ -n "${PNPM}" ]] || bail "pnpm not executable or not in PATH: ${PNPM_REQUESTED} (set PNPM to override)"
elif [[ -x "${HOME}/.local/share/pnpm/pnpm" ]]; then
  PNPM="${HOME}/.local/share/pnpm/pnpm"
else
  PNPM="$(command -v pnpm 2>/dev/null || true)"
fi
[[ -n "${PNPM}" ]] && [[ -x "${PNPM}" ]] || bail "pnpm not found. Set PNPM, install at ${HOME}/.local/share/pnpm/pnpm, or add pnpm to PATH"

GCC13="$(command -v "${GCC13:-gcc-13}" 2>/dev/null || true)"
GPP13="$(command -v "${GPP13:-g++-13}" 2>/dev/null || true)"
[[ -n "${GCC13}" ]] && command -v "${GCC13}" >/dev/null 2>&1 \
  || bail "gcc-13 not found or not executable (needed for nvcc workaround). Set GCC13 or install gcc-13 in PATH"
[[ -n "${GPP13}" ]] && command -v "${GPP13}" >/dev/null 2>&1 \
  || bail "g++-13 not found or not executable (needed for nvcc workaround). Set GPP13 or install g++-13 in PATH"

[[ -f "${HOME}/.env" ]] || echo "warn: ${HOME}/.env missing — provider API keys may be unset" >&2

NATIVE_FFMPEG_BIN="${HOME}/opt/ffmpeg-native/bin/ffmpeg"
if [[ "${NVENC}" == "true" ]]; then
  NATIVE_VIDEO_CODEC=h264_nvenc
else
  NATIVE_VIDEO_CODEC=libx264
fi
REQUIRE_NATIVE_FFMPEG="${DREAMVERSE_REQUIRE_NATIVE_FFMPEG:-false}"
case "${REQUIRE_NATIVE_FFMPEG}" in
  true|false) ;;
  *) bail "DREAMVERSE_REQUIRE_NATIVE_FFMPEG must be 'true' or 'false' (got '${REQUIRE_NATIVE_FFMPEG}')" ;;
esac
if [[ -x "${NATIVE_FFMPEG_BIN}" ]]; then
  if [[ "${NVENC}" == "true" ]]; then
    encoder_list="$("${NATIVE_FFMPEG_BIN}" -hide_banner -encoders 2>/dev/null || true)"
    if [[ "${encoder_list}" != *h264_nvenc* ]]; then
      bail "--nvenc requested but ${NATIVE_FFMPEG_BIN} was not built with NVENC. Rebuild: bash apps/dreamverse/scripts/install_native_ffmpeg.sh (with ENABLE_NVENC=1, the default)"
    fi
    if ! "${NATIVE_FFMPEG_BIN}" -hide_banner -loglevel error -y \
            -f lavfi -i 'color=red:size=64x64:rate=24:duration=0.2' \
            -c:v h264_nvenc -f null - >/dev/null 2>&1; then
      bail "--nvenc requested but the GPU on this host has no NVENC silicon (probe failed: 'OpenEncodeSessionEx unsupported device'). Datacenter Blackwell (B200) and some H100 SKUs ship without NVENC; --nvenc only works on hosts with NVENC-capable GPUs (RTX 50-series, T4, A10, etc.)."
    fi
  fi
  echo "        native ffmpeg: ${NATIVE_FFMPEG_BIN} (codec=${NATIVE_VIDEO_CODEC})"
elif [[ "${REQUIRE_NATIVE_FFMPEG}" == "true" ]] || [[ "${NVENC}" == "true" ]]; then
  bail "${NATIVE_FFMPEG_BIN} missing (required by --nvenc or DREAMVERSE_REQUIRE_NATIVE_FFMPEG=true). Run: bash apps/dreamverse/scripts/install_native_ffmpeg.sh"
else
  echo "warn: ${NATIVE_FFMPEG_BIN} missing — backend will fall back to system ffmpeg (\$(command -v ffmpeg))." >&2
  echo "      Build native ffmpeg with: bash apps/dreamverse/scripts/install_native_ffmpeg.sh" >&2
fi
echo "        python:        ${CONDA_ENV_PYTHON}"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Teardown anything on target ports
# ---------------------------------------------------------------------------

echo "[1/8] killing any existing deploy on ports ${BACKEND_PORT}/${FRONTEND_PORT} and GPU ${GPU}..."

kill_port_pid() {
  local port="$1"
  local pid

  for pid in $(list_port_pids "${port}"); do
    terminate_pid "${pid}" "port=${port} pid=${pid}"
  done
}

for pat in "main.py --host 0.0.0.0 --port ${BACKEND_PORT}" "next dev --port ${FRONTEND_PORT}" "NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 next dev --port ${FRONTEND_PORT}"; do
  terminate_pattern "${pat}"
done
kill_port_pid "${BACKEND_PORT}"
kill_port_pid "${FRONTEND_PORT}"

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="${GPU}" '$1==g {print $2}')"
if [[ -n "${gpu_uuid}" ]]; then
  for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
                | awk -F', ' -v u="${gpu_uuid}" '$2==u {print $1}'); do
    if [[ -n "${pid}" ]] && [[ "${pid}" != "$$" ]]; then
      cmd="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
      terminate_pid "${pid}" "GPU${GPU} pid=${pid} (${cmd:-?})"
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

echo "[2/8] launching backend on GPU ${GPU} port ${BACKEND_PORT} (warmup=${WARMUP} torch_compile=${TORCH_COMPILE} nvenc=${NVENC})..."

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
  export DREAMVERSE_PYTHON=\"${CONDA_ENV_PYTHON}\"
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

# Poll /readyz. Deadline scales with warmup + torch.compile flags
# because warmup runs two synthetic segments before /readyz=200, and
# torch.compile max-autotune adds ~3-4min cold start to the first
# segment. Empirical worst case (warmup=true, torch_compile=true):
# ~7 min on B200; we budget 15 min for safety.
if [[ "${WARMUP}" == "true" ]] && [[ "${TORCH_COMPILE}" == "true" ]]; then
  READYZ_BUDGET_SECONDS=900
elif [[ "${WARMUP}" == "true" ]] || [[ "${TORCH_COMPILE}" == "true" ]]; then
  READYZ_BUDGET_SECONDS=480
else
  READYZ_BUDGET_SECONDS=300
fi
READYZ_POLL_INTERVAL=6
READYZ_MAX_ITERS=$(( READYZ_BUDGET_SECONDS / READYZ_POLL_INTERVAL ))

echo "[3/8] polling http://127.0.0.1:${BACKEND_PORT}/readyz (budget=${READYZ_BUDGET_SECONDS}s) ..."
ready=0
for i in $(seq 1 ${READYZ_MAX_ITERS}); do
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
  sleep ${READYZ_POLL_INTERVAL}
done

if [[ "${ready}" != "1" ]]; then
  echo "error: backend did not become /readyz=200 within ${READYZ_BUDGET_SECONDS}s. Last 50 lines:" >&2
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
